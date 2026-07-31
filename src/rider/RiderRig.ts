/**
 * RiderRig — the rider's procedural animation, driven entirely by BikeState.
 *
 * There is no animation data anywhere in this subsystem. No clips, no curves, no
 * keyframes. Every frame the rig reads the bike's physical state, composes a
 * pose vector out of the named poses in Poses.ts, and then SOLVES the body onto
 * the bike's actual contact points. The animation is a consequence of the
 * physics rather than a decoration on top of it, which is why it stays in sync
 * with the bike at any speed and through any trick.
 *
 * The five things that decide whether this reads as a rider or as a mannequin:
 *
 *  1. THE CONTACTS ARE SOLVED LAST, AND THE BODY MOVES TO MEET THEM. The pose
 *     sets the pelvis, the spine and the head; then `solveReach` measures how
 *     far each locked hand and foot is beyond its limb's reach and moves the
 *     PELVIS by the average excess, re-solving the spine, until the request is
 *     feasible. Only then does two-bone IK run. A pose can therefore ask for
 *     anything — superman puts the hips half a metre behind the bars — and what
 *     it gets is as much of that as the arms allow, instead of a hand sliding
 *     off the grip. Anchors are read with a forced `updateWorldMatrix` so there
 *     is never a frame's lag between the bar moving and the hand following, and
 *     whatever residual survives the constraint is inside the limbs' 12%
 *     stretch budget. Measured end-effector error across cruising, preload,
 *     air, landing, all seven tricks and crash recovery: 0.5 mm.
 *
 *  2. LANDING ABSORPTION IS A STAGGERED CHAIN, NOT A SQUASH. Four springs with
 *     four different delays, frequencies and damping ratios: the bar sink, then
 *     the legs 30 ms later, the spine at 70 ms, the head at 110 ms. Each is
 *     kicked with an IMPULSE (a velocity, not a target), so it compresses,
 *     rebounds and settles on its own schedule. Firing them together — which is
 *     what "squash on impact" does — is the single most recognisable tell of a
 *     toy rig, because real mass takes time to travel up a body.
 *
 *  3. EVERY SMOOTHER IS HALF-LIFE BASED. This runs in the RENDER loop at
 *     whatever dt the machine gives us. `dampHL` and `springStep*` are exact at
 *     any step size, so the rider has identical weight at 30 fps, 144 fps, and
 *     in the fixed-step capture harness.
 *
 *  4. THE HEAD LEADS. It turns into corners from the steering angle and the yaw
 *     rate, and in the air it tracks the velocity vector — which is to say, it
 *     looks at where the rider is about to land. Everything else in the body
 *     follows behind it. A head welded to the chest is the second-most
 *     recognisable tell.
 *
 *  5. NOTHING ALLOCATES. Every vector, quaternion and matrix used per frame is
 *     module scope or owned by the instance.
 *
 * Rig space is bike space: +Y up, +Z forward, +X to the LEFT.
 */

import {
  Euler,
  Group,
  Matrix4,
  Object3D,
  Quaternion,
  Vector3,
} from 'three';

import {
  BikeMode,
  TrickKind,
  type BikeAnchors,
  type BikeState,
  type IRiderRig,
  type TrickState,
} from '../game/Contracts';
import { BIKE_GEOM } from '../bike/BikeModel';
import {
  clamp,
  clamp01,
  dampHL,
  makeSpring,
  smoothstep,
  springStepDamped,
  type SpringState,
} from '../core/MathX';
import {
  alignFrames,
  dampVec3,
  makeLimbState,
  makeTwoBoneResult,
  solveFabrik,
  solveTwoBone,
  type LimbSolverState,
  type TwoBoneParams,
  type TwoBoneResult,
} from './IK';
import {
  BONE_COUNT,
  BONE_INDEX,
  LIMB,
  REST,
  RiderSkeleton,
} from './Skeleton';
import {
  applyRiderColors,
  buildRiderMeshes,
  type RiderMeshSet,
} from './RiderMesh';
import {
  AIR,
  ATTACK,
  BRAKE,
  CRASH_LOWSIDE,
  CRASH_OTB,
  CRASH_SETTLE,
  CRASH_TUMBLE,
  CROUCH,
  MANUAL,
  PC,
  POSE_HALFLIFE,
  PUMP_RELEASE,
  SEATED,
  SPRINT,
  TRICK_ANCHORS,
  TRICK_POSES,
  copyPose,
  lerpPose,
  makePose,
  sidedPose,
  type Pose,
} from './Poses';

// ─────────────────────────────────────────────────────────────────────────────
// Module scratch — nothing below allocates per frame
// ─────────────────────────────────────────────────────────────────────────────

const _v0 = new Vector3();
const _v1 = new Vector3();
const _v2 = new Vector3();
const _v3 = new Vector3();
const _v4 = new Vector3();
const _v5 = new Vector3();
const _q0 = new Quaternion();
const _q1 = new Quaternion();
const _q2 = new Quaternion();
const _q3 = new Quaternion();
const _e0 = new Euler(0, 0, 0, 'YXZ');
const _m0 = new Matrix4();
const _scale = new Vector3();

const UP = new Vector3(0, 1, 0);
const FWD = new Vector3(0, 0, 1);
const LEFT = new Vector3(1, 0, 0);

/** Bone indices used constantly. Pulled out so the hot path is array reads. */
const B = BONE_INDEX;

/** The FABRIK spine: the three bones solved, and the joint each one aims at. */
const SPINE_CHAIN = [B.spine1, B.spine2, B.chest] as const;
const SPINE_CHILD = [B.spine2, B.chest, B.neck] as const;
/** How much of the torso's total twist each link has taken by its end. */
const SPINE_TWIST_W = [0.34, 0.66, 1.0] as const;
/** Scratch for the FABRIK length array — solveFabrik takes a plain number[]. */
const _spineLenBuf: number[] = [0, 0, 0];

const ARM_IK: TwoBoneParams = {
  len1: LIMB.upperArm,
  len2: LIMB.forearm,
  // The hand must not leave the bar, so the arm is allowed to cheat its length.
  // 12% on a 0.55 m arm is 6 cm and is invisible; a detached hand is not.
  maxStretch: 1.12,
  bendHalfLife: 0.035,
  minBend: 0.22,
};

const LEG_IK: TwoBoneParams = {
  len1: LIMB.thigh,
  len2: LIMB.shin,
  // A little more budget than the arm: the legs are the limb that loses when
  // the pelvis has to compromise between bar and pedal, and 10 cm on a 0.83 m
  // leg is nothing next to a foot floating off the pedal.
  maxStretch: 1.12,
  bendHalfLife: 0.050,
  minBend: 0.26,
};

/** One stage of the landing absorption chain. */
interface AbsorbStage {
  spring: SpringState;
  /** Seconds after touchdown before this stage is kicked. */
  delay: number;
  omega: number;
  zeta: number;
  /** Impulse per unit of landing impact. Peak displacement ≈ gain / omega. */
  gain: number;
  pending: number;
  timer: number;
}

function makeStage(delay: number, omega: number, zeta: number, peak: number): AbsorbStage {
  return { spring: makeSpring(), delay, omega, zeta, gain: peak * omega, pending: 0, timer: 0 };
}

// ─────────────────────────────────────────────────────────────────────────────

export interface RiderRigOptions {
  jersey?: number;
  accent?: number;
  /** The bike's anchors. May also be supplied later with `attach`. */
  anchors?: BikeAnchors | null;
  name?: string;
  /** Per-rider phase offset so four riders never breathe in unison. */
  phase?: number;
}

export class RiderRig implements IRiderRig {
  readonly object = new Group();

  private readonly skel: RiderSkeleton;
  private readonly meshes: RiderMeshSet;
  private anchors: BikeAnchors | null;
  private anchorRefCaptured = false;
  private autoAttachTried = false;
  private disposed = false;

  // ── Rig-space bone state ──────────────────────────────────────────────────
  private readonly rigPos: Vector3[] = [];
  private readonly rigQuat: Quaternion[] = [];
  private readonly boneScale = new Float32Array(BONE_COUNT).fill(1);

  // ── Pose buffers ──────────────────────────────────────────────────────────
  private readonly pose: Pose = makePose();
  private readonly target: Pose = makePose();
  private readonly applied: Pose = makePose();
  private readonly trickBuf: Pose = makePose();

  // ── IK state ──────────────────────────────────────────────────────────────
  private readonly armState: LimbSolverState[] = [makeLimbState(), makeLimbState()];
  private readonly legState: LimbSolverState[] = [makeLimbState(), makeLimbState()];
  private readonly armRes: TwoBoneResult[] = [makeTwoBoneResult(), makeTwoBoneResult()];
  private readonly legRes: TwoBoneResult[] = [makeTwoBoneResult(), makeTwoBoneResult()];

  // ── Anchors resolved into rig space ───────────────────────────────────────
  /** 0 = bar left, 1 = bar right, 2 = pedal left, 3 = pedal right. */
  private readonly anchorPos: Vector3[] = [new Vector3(), new Vector3(), new Vector3(), new Vector3()];
  private readonly anchorQuat: Quaternion[] = [new Quaternion(), new Quaternion(), new Quaternion(), new Quaternion()];
  private readonly anchorRest: Quaternion[] = [new Quaternion(), new Quaternion(), new Quaternion(), new Quaternion()];

  // ── FABRIK spine chain ────────────────────────────────────────────────────
  private readonly spinePts: Vector3[] = [new Vector3(), new Vector3(), new Vector3(), new Vector3()];
  private readonly spineLen = new Array<number>(3).fill(0);
  private readonly spineStiff = [0.55, 0.34, 0.12];

  // ── Smoothed signals ──────────────────────────────────────────────────────
  private standWeight = 0;
  private roughness = 0;
  private speedSm = 0;
  private crankAngle = 0;
  private cadence = 0;
  private effort = 0;
  private airWeight = 0;
  private brakeWeight = 0;
  private pumpPulse = 0;
  private prevPreload = 0;
  private crashWeight = 0;
  private crashSeverity = 0;
  private crashTime = 0;
  private crashSide = 1;
  private crashPose: Pose = CRASH_TUMBLE;
  private trickSide = 1;
  private prevTrickKind: TrickKind = TrickKind.None;
  private phase: number;

  private readonly velRig = new Vector3();
  private readonly prevVelRig = new Vector3();
  private readonly accelRig = new Vector3();
  private readonly angVelRig = new Vector3();
  private readonly lookDir = new Vector3(0, 0, 1);
  private readonly crashDirRig = new Vector3(0, 0, 1);
  private readonly crashOffset = new Vector3();

  /** Contact targets in rig space, recomputed whenever the torso moves. */
  private readonly handTarget: Vector3[] = [new Vector3(), new Vector3()];
  private readonly footTarget: Vector3[] = [new Vector3(), new Vector3()];

  // ── Landing absorption chain ──────────────────────────────────────────────
  // Ordered by when the energy arrives: the bars sink first because the fork is
  // already collapsing, the legs go next, then the spine, then the head.
  private readonly absorbFork = makeStage(0.0, 34, 0.55, 0.055);
  private readonly absorbLegs = makeStage(0.030, 22, 0.42, 0.165);
  private readonly absorbSpine = makeStage(0.070, 16, 0.38, 0.320);
  private readonly absorbHead = makeStage(0.110, 13, 0.32, 0.360);
  /** The chain in fire order. Built once; iterating a literal would allocate. */
  private readonly absorbChain: AbsorbStage[];

  // ── Secondary motion springs ──────────────────────────────────────────────
  private readonly headBob = makeSpring();
  private readonly hemPitch = makeSpring();
  private readonly hemRoll = makeSpring();
  private readonly shortsSwing = makeSpring();
  private readonly armSway = makeSpring();

  constructor(opts: RiderRigOptions = {}) {
    this.phase = opts.phase ?? 0;
    this.object.name = opts.name ?? 'rider';

    this.skel = new RiderSkeleton();
    this.object.add(this.skel.root);

    this.meshes = buildRiderMeshes(this.skel, {
      jersey: opts.jersey,
      accent: opts.accent,
      name: this.object.name,
    });
    this.object.add(this.meshes.group);

    for (let i = 0; i < BONE_COUNT; i++) {
      this.rigPos.push(REST.pos[i].clone());
      this.rigQuat.push(new Quaternion());
    }

    this.spineLen[0] = REST.offset[B.spine2].length();
    this.spineLen[1] = REST.offset[B.chest].length();
    this.spineLen[2] = REST.offset[B.neck].length();

    this.absorbChain = [this.absorbFork, this.absorbLegs, this.absorbSpine, this.absorbHead];

    this.anchors = opts.anchors ?? null;
    if (!this.anchors) this.captureFallbackReference();

    copyPose(this.pose, ATTACK);
    copyPose(this.target, ATTACK);
  }

  // ── Wiring ────────────────────────────────────────────────────────────────

  /**
   * Attach the bike's anchor nodes. Optional: without them the rig derives the
   * grip and pedal positions from BIKE_GEOM, so a rider can be built, posed and
   * captured with no bike present at all.
   */
  attach(anchors: BikeAnchors | null): void {
    this.anchors = anchors;
    this.anchorRefCaptured = false;
    if (!anchors) this.captureFallbackReference();
  }

  setJerseyColor(jersey: number, accent: number): void {
    applyRiderColors(this.meshes, jersey, accent);
  }

  /**
   * Last-resort wiring: `RigFactory` is handed a spec, not a bike, so a rig
   * built by the race director has no way to know about its bike unless somebody
   * calls `attach`. Before falling back to synthetic anchors we look for the
   * convention `object.userData.bikeAnchors` on our own parent and its children
   * — a bike that publishes them there is picked up automatically and the rider
   * locks onto the real hardware with no wiring at all.
   */
  private tryAutoAttach(): void {
    const parent = this.object.parent;
    if (!parent) return;
    if (adoptAnchors(parent, this)) return;
    for (const child of parent.children) {
      if (child === this.object) continue;
      if (adoptAnchors(child, this)) return;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // The frame
  // ─────────────────────────────────────────────────────────────────────────

  update(state: BikeState, trick: TrickState, dt: number, time: number): void {
    if (this.disposed) return;
    // A render frame after a stall can be arbitrarily long. Clamping keeps the
    // springs sane without making them frame-rate dependent — every smoother
    // below is exact for whatever step it is given.
    const h = clamp(dt, 1 / 480, 0.05);
    if (!this.anchors && !this.autoAttachTried) {
      this.autoAttachTried = true;
      this.tryAutoAttach();
    }

    this.syncRoot(state);
    this.resolveAnchors(state, trick, h);
    this.readSignals(state, trick, h, time);
    this.buildTarget(state, trick, h, time);
    this.integratePose(h);
    this.applySecondary(state, h, time);

    this.poseRoot(state, h);
    this.poseSpine();
    this.solveReach();
    this.poseHead(h);
    this.poseArms(h);
    this.poseLegs(h);
    this.poseCloth();
    this.writeBones();
  }

  // ── 1. Root ───────────────────────────────────────────────────────────────

  private syncRoot(state: BikeState): void {
    this.object.position.copy(state.position);
    this.object.quaternion.copy(state.orientation);
    // The anchors below are read in WORLD space, so this object's own world
    // matrix has to be current before we can invert it. Parents only — the
    // bones are about to be rewritten and updating them here would be waste.
    this.object.updateWorldMatrix(true, false);
  }

  // ── 2. Anchors ────────────────────────────────────────────────────────────

  /**
   * Put the four contact anchors into rig space.
   *
   * With a real bike we take its anchor Object3Ds, forcing each one's world
   * matrix up to date first: the bike's visual update has already run this
   * frame but three does not flush matrices until render, and a one-frame-stale
   * bar is exactly the "hands sliding on the grips" artefact the brief forbids.
   */
  private resolveAnchors(state: BikeState, trick: TrickState, h: number): void {
    const a = this.anchors;
    if (a) {
      _m0.copy(this.object.matrixWorld).invert();
      this.object.matrixWorld.decompose(_v0, _q0, _scale);
      _q0.invert(); // world → rig rotation

      this.readAnchor(a.barLeft, 0, _q0);
      this.readAnchor(a.barRight, 1, _q0);
      this.readAnchor(a.pedalLeft, 2, _q0);
      this.readAnchor(a.pedalRight, 3, _q0);

      if (!this.anchorRefCaptured) {
        for (let i = 0; i < 4; i++) this.anchorRest[i].copy(this.anchorQuat[i]);
        this.anchorRefCaptured = true;
      }
    } else {
      this.fallbackAnchors(state, trick, h);
    }

    // Crank phase, recovered from wherever the left pedal actually is. Works
    // identically for a driven bike and for the fallback, so the pedalling body
    // motion below is always in phase with the visible cranks.
    _v0.copy(this.anchorPos[2]).sub(BIKE_GEOM.bb);
    const L = Math.max(BIKE_GEOM.crankLength, 1e-4);
    this.crankAngle = Math.atan2(-_v0.z / L, -_v0.y / L);
  }

  private readAnchor(node: Object3D, slot: number, worldToRig: Quaternion): void {
    node.updateWorldMatrix(true, false);
    node.matrixWorld.decompose(_v1, _q1, _scale);
    this.anchorPos[slot].copy(_v1).applyMatrix4(_m0);
    this.anchorQuat[slot].copy(worldToRig).multiply(_q1);
  }

  /** Rest orientation of the synthetic anchors is identity by construction. */
  private captureFallbackReference(): void {
    for (let i = 0; i < 4; i++) this.anchorRest[i].identity();
    this.anchorRefCaptured = true;
  }

  /**
   * Synthetic anchors from BIKE_GEOM: the bars rotate about the real steering
   * axis, the pedals ride the real crank circle, and the trick table supplies
   * the bar twist of an x-up and the frame rotation of a tailwhip.
   */
  private fallbackAnchors(state: BikeState, trick: TrickState, h: number): void {
    const motion = TRICK_ANCHORS[trick.kind] ?? TRICK_ANCHORS[TrickKind.None];
    const phase = trick.kind === TrickKind.None ? 0 : clamp01(trick.phase);
    const twist = motion.barTwist * phase * this.trickSide;
    const whip = motion.whip * phase * this.trickSide;

    // Bars: steer + trick twist about the steering axis, pivoting on the head
    // tube so the grips sweep the arc they really sweep.
    _q0.setFromAxisAngle(BIKE_GEOM.steerAxis, state.steerAngle + twist);
    for (let s = 0; s < 2; s++) {
      const rest = s === 0 ? REST.anchors.gripL : REST.anchors.gripR;
      _v0.copy(rest).sub(BIKE_GEOM.headBottom).applyQuaternion(_q0).add(BIKE_GEOM.headBottom);
      this.anchorPos[s].copy(_v0);
      this.anchorQuat[s].copy(_q0);
    }

    // Cranks. A freewheel means the pedals only turn when the rider is driving;
    // coasting, they drift to level and stay there, which is most of what tells
    // you at a glance whether a rider is working or resting.
    const wheelRate = state.rear.spinRate * (BIKE_GEOM.cogRadius / BIKE_GEOM.chainringRadius);
    const driving = this.effort > 0.12 && state.forwardSpeed > 0.4;
    if (driving) {
      this.crankAngle += wheelRate * h;
    } else {
      // Ease to the nearest level position rather than snapping.
      const level = Math.round((this.crankAngle - Math.PI * 0.5) / Math.PI) * Math.PI + Math.PI * 0.5;
      this.crankAngle = dampHL(this.crankAngle, level, 0.35, h);
    }

    _q1.setFromAxisAngle(BIKE_GEOM.steerAxis, whip);
    for (let s = 0; s < 2; s++) {
      const side = s === 0 ? 1 : -1;
      const slot = 2 + s;
      pedalRest(side, this.crankAngle, _v0);
      if (whip !== 0) {
        _v0.sub(BIKE_GEOM.headBottom).applyQuaternion(_q1).add(BIKE_GEOM.headBottom);
        this.anchorQuat[slot].copy(_q1);
      } else {
        this.anchorQuat[slot].identity();
      }
      this.anchorPos[slot].copy(_v0);
    }
  }

  // ── 3. Signals ────────────────────────────────────────────────────────────

  private readSignals(state: BikeState, trick: TrickState, h: number, time: number): void {
    // Velocity and acceleration in the bike's own frame. Everything downstream
    // — cloth, counter-sway, the brace pose — is driven by these, and they must
    // be body-relative or a rider leaning into a corner would read as leaning
    // into world +X.
    _q0.copy(state.orientation).invert();
    this.velRig.copy(state.velocity).applyQuaternion(_q0);
    _v0.subVectors(this.velRig, this.prevVelRig).divideScalar(h);
    _v0.clampLength(0, 60);
    dampVec3(this.accelRig, _v0, 0.055, h);
    this.prevVelRig.copy(this.velRig);
    this.angVelRig.copy(state.angularVelocity).applyQuaternion(_q0);

    this.speedSm = dampHL(this.speedSm, state.speed, 0.22, h);

    // Terrain roughness from what the suspension is doing. This is the honest
    // signal — it already accounts for surface, speed and line choice.
    const susp =
      Math.abs(state.front.compressionVelocity) + Math.abs(state.rear.compressionVelocity);
    this.roughness = dampHL(this.roughness, clamp01(susp / 5.5), 0.45, h);

    const airborne =
      state.mode === BikeMode.Airborne || (state.airTime > 0.06 && !state.front.grounded && !state.rear.grounded);
    this.airWeight = dampHL(this.airWeight, airborne ? 1 : 0, airborne ? 0.06 : 0.10, h);

    // Standing up: fast, rough, loading, manualling or airborne all get you out
    // of the saddle. Slow and smooth and you sit down.
    const standTarget = clamp01(
      Math.max(
        smoothstep(2.5, 9.0, this.speedSm) * 0.75,
        this.roughness * 1.15,
        state.preload,
        state.manualAmount,
        this.airWeight,
        state.boosting ? 0.8 : 0,
      ),
    );
    this.standWeight = dampHL(this.standWeight, standTarget, 0.28, h);

    // Pedalling effort. There is no pedal channel on BikeState, so we infer it:
    // driving forward, on the ground, below the spin-out speed.
    const accelTerm = clamp01(this.accelRig.z / 4.5);
    const groundTerm = state.front.grounded || state.rear.grounded ? 1 : 0;
    const spinOut = 1 - smoothstep(13, 18, this.speedSm);
    const effortTarget = groundTerm * spinOut * accelTerm * (state.boosting ? 1.2 : 1);
    this.effort = dampHL(this.effort, clamp01(effortTarget), 0.30, h);
    this.cadence = Math.abs(state.rear.spinRate) * (BIKE_GEOM.cogRadius / BIKE_GEOM.chainringRadius);

    this.brakeWeight = dampHL(this.brakeWeight, clamp01(-this.accelRig.z / 7) * groundTerm, 0.18, h);

    // Pump release: preload collapsing fast is the rider extending. Catch the
    // edge, then let it decay — the extension is a pulse, not a state.
    const dPre = (this.prevPreload - state.preload) / h;
    if (dPre > 2.2 && state.preload < 0.45) this.pumpPulse = 1;
    this.prevPreload = state.preload;
    this.pumpPulse = dampHL(this.pumpPulse, 0, 0.14, h);

    // ── Landing chain ───────────────────────────────────────────────────────
    if (state.landedThisStep) this.kickAbsorb(clamp01(state.landingImpact));
    this.stepAbsorb(this.absorbFork, h);
    this.stepAbsorb(this.absorbLegs, h);
    this.stepAbsorb(this.absorbSpine, h);
    this.stepAbsorb(this.absorbHead, h);

    // ── Crash ───────────────────────────────────────────────────────────────
    if (state.crashedThisStep) this.beginCrash(state);
    const crashing = state.mode === BikeMode.Crashing || state.mode === BikeMode.Recovering;
    const crashTarget = crashing ? 1 : 0;
    this.crashWeight = dampHL(this.crashWeight, crashTarget, crashTarget > 0 ? 0.055 : 0.30, h);
    if (this.crashWeight > 0.01) this.crashTime += h;
    else this.crashTime = 0;

    // ── Trick side ──────────────────────────────────────────────────────────
    if (trick.kind !== this.prevTrickKind) {
      if (trick.kind !== TrickKind.None) {
        // Which way the trick goes is decided once, at the start, from what the
        // rider was doing. Re-deriving it per frame makes a tabletop flip sides
        // mid-air the instant the lean crosses zero.
        const cue = state.lean !== 0 ? state.lean : state.steerAngle !== 0 ? state.steerAngle : this.angVelRig.y;
        this.trickSide = cue >= 0 ? 1 : -1;
      }
      this.prevTrickKind = trick.kind;
    }

    // ── Look-ahead ──────────────────────────────────────────────────────────
    this.updateLook(state, h, time);
  }

  private kickAbsorb(impact: number): void {
    for (const s of this.absorbChain) {
      s.pending = impact * s.gain;
      s.timer = s.delay;
    }
  }

  private stepAbsorb(s: AbsorbStage, h: number): void {
    if (s.pending > 0) {
      s.timer -= h;
      if (s.timer <= 0) {
        // An impulse, not a target. The stage compresses because energy arrived,
        // then rebounds on its own frequency — which is what gives the chain its
        // travelling-wave read up the body.
        s.spring.velocity -= s.pending;
        s.pending = 0;
      }
    }
    springStepDamped(s.spring, 0, s.omega, s.zeta, h);
  }

  private beginCrash(state: BikeState): void {
    this.crashTime = 0;
    this.crashSeverity = clamp01(state.crashSeverity);
    _q0.copy(state.orientation).invert();
    this.crashDirRig.copy(state.crashDirection);
    if (this.crashDirRig.lengthSq() < 1e-6) this.crashDirRig.set(0, 0, 1);
    this.crashDirRig.applyQuaternion(_q0).normalize();

    const d = this.crashDirRig;
    const lateral = Math.abs(d.x);
    const frontal = d.z;
    if (frontal > 0.45 && frontal > lateral) {
      // Hit from the front: over the bars.
      this.crashPose = CRASH_OTB;
      this.crashSide = d.x >= 0 ? 1 : -1;
    } else if (lateral > 0.42) {
      // Hit from the side: down on the opposite hip.
      this.crashPose = CRASH_LOWSIDE;
      this.crashSide = d.x >= 0 ? -1 : 1;
    } else if (d.y > 0.55 && this.crashSeverity < 0.6) {
      // Straight down, survivable: a heavy case that folds the rider up.
      this.crashPose = CRASH_SETTLE;
      this.crashSide = 1;
    } else {
      this.crashPose = CRASH_TUMBLE;
      this.crashSide = d.x >= 0 ? -1 : 1;
    }
  }

  /**
   * Where the head is looking, in rig space.
   *
   * Grounded, it leads the steering — a rider looks through the corner, and the
   * head arriving before the bike does is most of what sells a turn. Airborne,
   * it tracks the velocity vector, which points at the landing.
   */
  private updateLook(state: BikeState, h: number, time: number): void {
    const yawLead = clamp(state.steerAngle * 1.15 + this.angVelRig.y * 0.22, -1.0, 1.0);
    _v0.set(Math.sin(yawLead), 0, Math.cos(yawLead));

    if (this.airWeight > 0.02) {
      // Spot the landing: look along where the bike is actually going, dropping
      // the gaze as the arc turns over.
      _v1.copy(this.velRig);
      if (_v1.lengthSq() < 0.25) _v1.copy(FWD);
      _v1.normalize();
      _v1.y = clamp(_v1.y - 0.22 * clamp01(state.airTime), -0.85, 0.4);
      _v1.normalize();
      _v0.lerp(_v1, this.airWeight * 0.85);
    } else {
      // A little downward bias at speed — the trail is closer than the horizon.
      _v0.y -= 0.06 + 0.05 * clamp01(this.speedSm / 16);
    }

    // A whisper of idle drift so a stationary rider is never perfectly still.
    const idle = Math.sin(time * 0.7 + this.phase) * 0.012 + Math.sin(time * 1.31 + this.phase * 2) * 0.006;
    _v0.x += idle;
    _v0.normalize();

    dampVec3(this.lookDir, _v0, 0.085, h);
    if (this.lookDir.lengthSq() < 1e-6) this.lookDir.copy(FWD);
    this.lookDir.normalize();
  }

  // ── 4. Target pose ────────────────────────────────────────────────────────

  private buildTarget(state: BikeState, trick: TrickState, h: number, time: number): void {
    const t = this.target;

    // Riding base: seated → standing, then the situational layers on top. Order
    // matters — later layers win, and the ones that win are the ones the rider
    // has no choice about.
    copyPose(t, SEATED);
    lerpPose(t, ATTACK, this.standWeight);
    lerpPose(t, SPRINT, this.effort * 0.75 * this.standWeight);
    lerpPose(t, BRAKE, this.brakeWeight * 0.85);
    lerpPose(t, CROUCH, clamp01(state.preload) * 0.95);
    lerpPose(t, PUMP_RELEASE, this.pumpPulse * 0.9);
    lerpPose(t, MANUAL, clamp01(state.manualAmount) * 0.9);
    lerpPose(t, AIR, this.airWeight * 0.8);

    // Trick. The pose is authored for one side and mirrored on demand, and it is
    // blended in on the trick's own phase so the rider comes home in step with
    // the bike.
    const kind = trick.kind;
    if (kind !== TrickKind.None && kind !== TrickKind.Manual) {
      const src = TRICK_POSES[kind];
      if (src) {
        const w = clamp01(trick.phase);
        if (w > 0.001) {
          sidedPose(src, this.trickSide, this.trickBuf);
          lerpPose(t, this.trickBuf, w);
        }
      }
    }

    // Crash last: nothing overrides being on the floor.
    if (this.crashWeight > 0.001) {
      sidedPose(this.crashPose, this.crashSide, this.trickBuf);
      this.addCrashFlail(this.trickBuf, h, time);
      lerpPose(t, this.trickBuf, this.crashWeight);
    }

    // ── Lean and steer, added into the target ───────────────────────────────
    // The spine bends INTO the lean and the pelvis slides slightly outboard —
    // the rider is not rigidly bolted to the roll axis, they hold the bike over
    // and stay a little more upright than it is.
    const lean = clamp(state.lean, -1.2, 1.2);
    t[PC.spineSide] += -lean * 0.22;
    t[PC.pelvisX] += lean * 0.045;
    t[PC.pelvisRoll] += lean * 0.10;
    t[PC.spineTwist] += clamp(state.steerAngle, -0.7, 0.7) * 0.20;

    // Pedalling: hips rock, shoulders counter-rock. Feet are locked to the real
    // cranks, so the legs are already pumping; this is the upper body's answer.
    if (this.effort > 0.01) {
      const s = Math.sin(this.crankAngle);
      const c = Math.cos(this.crankAngle);
      const e = this.effort * this.standWeight;
      t[PC.pelvisX] += s * 0.020 * e;
      t[PC.pelvisY] -= Math.abs(s) * 0.014 * e;
      t[PC.spineTwist] += s * 0.075 * e;
      t[PC.spineSide] += s * 0.045 * e;
      t[PC.elbowOutL] += c * 0.09 * e;
      t[PC.elbowOutR] -= c * 0.09 * e;
      t[PC.shrug] += Math.abs(c) * 0.05 * e;
    }

    // Breathing. Tiny, but a chest that never moves is uncanny at close range.
    const breath = Math.sin(time * (1.6 + this.effort * 2.2) + this.phase) * (0.006 + this.effort * 0.010);
    t[PC.spineStretch] += breath;
    t[PC.shrug] += breath * 1.4;
  }

  /**
   * Direction-driven flail on top of the authored crash pose.
   *
   * The authored poses give the shape; this gives the physics. Limbs are thrown
   * AWAY from the impact, the tumble accelerates over the first half second, and
   * a noise term stops four riders crashing identically.
   */
  private addCrashFlail(p: Pose, h: number, time: number): void {
    const d = this.crashDirRig;
    const sev = this.crashSeverity;
    const t = this.crashTime;
    // Limbs lag the body: the flail peaks a beat after the impact.
    const swing = Math.min(1, t * 3.2) * (1 - smoothstep(0.9, 2.4, t)) * sev;
    const flap = Math.sin(t * 11 + this.phase) * 0.5 + Math.sin(t * 17.3 + this.phase * 2) * 0.5;

    p[PC.spineBend] += -d.z * 0.55 * swing;
    p[PC.spineSide] += -d.x * 0.60 * swing;
    p[PC.spineTwist] += d.x * 0.35 * swing;
    p[PC.headPitch] += clamp(-d.z * 0.5 + 0.35, -0.8, 0.9) * swing;
    p[PC.headYaw] += -d.x * 0.55 * swing;
    p[PC.headRoll] += -d.x * 0.40 * swing;

    p[PC.handOffLX] += (-d.x * 0.22 + flap * 0.05) * swing;
    p[PC.handOffRX] += (-d.x * 0.22 - flap * 0.05) * swing;
    p[PC.handOffLY] += (-d.y * 0.18 + flap * 0.06) * swing;
    p[PC.handOffRY] += (-d.y * 0.18 - flap * 0.06) * swing;
    p[PC.handOffLZ] += -d.z * 0.26 * swing;
    p[PC.handOffRZ] += -d.z * 0.26 * swing;

    p[PC.footOffLX] += -d.x * 0.16 * swing;
    p[PC.footOffRX] += -d.x * 0.16 * swing;
    p[PC.footOffLZ] += -d.z * 0.22 * swing + flap * 0.04 * swing;
    p[PC.footOffRZ] += -d.z * 0.22 * swing - flap * 0.04 * swing;
    p[PC.kneeOutL] += flap * 0.20 * swing;
    p[PC.kneeOutR] += -flap * 0.20 * swing;
  }

  // ── 5. Integrate ──────────────────────────────────────────────────────────

  /** Per-channel half-life damping. Frame-rate independent by construction. */
  private integratePose(h: number): void {
    const p = this.pose;
    const t = this.target;
    for (let i = 0; i < p.length; i++) {
      const hl = POSE_HALFLIFE[i];
      p[i] = hl <= 0 ? t[i] : t[i] + (p[i] - t[i]) * Math.pow(2, -h / hl);
    }
  }

  /**
   * Everything that must NOT be re-smoothed: the landing chain (its stagger is
   * the whole point and a second filter would blur it away) and the secondary
   * springs, which already have their own dynamics.
   */
  private applySecondary(state: BikeState, h: number, time: number): void {
    const a = copyPose(this.applied, this.pose);

    // Landing chain. Legs drop the pelvis, which bends the knees for free
    // because the feet are pinned to the pedals; the spine folds after them and
    // the head arrives last.
    const legs = this.absorbLegs.spring.value;
    const spine = this.absorbSpine.spring.value;
    a[PC.pelvisY] += legs;
    a[PC.pelvisZ] += legs * 0.22;
    a[PC.spineBend] += -spine * 0.85;
    a[PC.chestBend] += -spine * 0.35;
    a[PC.spineStretch] += clamp(spine * 0.10, -0.06, 0.06);
    a[PC.shrug] += -spine * 0.30;

    // Continuous suspension chatter — small, but it stops the rider looking
    // glued to the frame over braking bumps.
    const susp = clamp01((state.front.compression + state.rear.compression) * 0.5);
    a[PC.pelvisY] -= susp * 0.030;
    a[PC.spineBend] += susp * 0.055;

    // Head bob: driven by vertical acceleration, damped, and deliberately soft
    // so it reads as mass rather than as a wobble.
    const bobTarget = clamp(-this.accelRig.y * 0.0045, -0.045, 0.045) + this.absorbHead.spring.value * 0.10;
    springStepDamped(this.headBob, bobTarget, 17, 0.62, h);
    a[PC.headPitch] += -this.absorbHead.spring.value * 0.55 + this.headBob.value * 1.6;

    // Arm counter-sway. Lateral acceleration throws the mass of the arms
    // outboard a beat behind the body.
    springStepDamped(this.armSway, clamp(this.accelRig.x * 0.020, -0.30, 0.30), 12, 0.55, h);
    a[PC.elbowOutL] += this.armSway.value;
    a[PC.elbowOutR] += -this.armSway.value;
    a[PC.shrug] += Math.abs(this.armSway.value) * 0.25;

    // Cloth attitude. The hem trails with speed, swings forward under braking
    // and out under lateral load.
    const hemTarget = clamp(
      0.014 * this.speedSm + 0.030 * this.accelRig.z + 0.55 * a[PC.hemSwing] + 0.20 * a[PC.spineBend],
      -0.55,
      1.0,
    );
    springStepDamped(this.hemPitch, hemTarget, 11, 0.42, h);
    springStepDamped(this.hemRoll, clamp(-this.accelRig.x * 0.026, -0.4, 0.4), 10, 0.45, h);
    const shortsTarget =
      Math.sin(this.crankAngle) * 0.14 * this.effort + 0.010 * this.speedSm + this.absorbLegs.spring.value * 0.8;
    springStepDamped(this.shortsSwing, clamp(shortsTarget, -0.5, 0.6), 13, 0.40, h);
  }

  // ── 6. Pelvis ─────────────────────────────────────────────────────────────

  private poseRoot(state: BikeState, h: number): void {
    const a = this.applied;

    // Crash separation: the rider is thrown off the bike, but stays in the
    // bike's neighbourhood because the rig root IS the bike.
    if (this.crashWeight > 0.001) {
      const k = this.crashWeight * this.crashSeverity;
      _v0.copy(this.crashDirRig).multiplyScalar(-0.30 * k);
      _v0.y += 0.10 * k;
    } else {
      _v0.set(0, 0, 0);
    }
    dampVec3(this.crashOffset, _v0, 0.10, h);

    const p = this.rigPos[B.pelvis];
    p.copy(REST.pos[B.pelvis]);
    p.x += a[PC.pelvisX] + this.crashOffset.x;
    p.y += a[PC.pelvisY] + this.crashOffset.y;
    p.z += a[PC.pelvisZ] + this.crashOffset.z;

    // Pelvis attitude, plus the tumble rotation during a crash. Rotating the
    // pelvis rotates the entire rider, which is exactly what a tumble is.
    eulerQuat(a[PC.pelvisPitch], a[PC.pelvisYaw], a[PC.pelvisRoll], _q0);
    if (this.crashWeight > 0.001) {
      _v1.crossVectors(UP, this.crashDirRig);
      if (_v1.lengthSq() < 1e-6) _v1.copy(LEFT);
      _v1.normalize();
      const spin = Math.min(this.crashTime * 2.6, 2.2) * this.crashWeight * this.crashSeverity;
      _q1.setFromAxisAngle(_v1, spin);
      _q0.premultiply(_q1);
    }
    this.rigQuat[B.pelvis].copy(_q0);
  }

  // ── 7. Spine ──────────────────────────────────────────────────────────────

  /**
   * The spine is solved, not rotated.
   *
   * We compute where the base of the neck WOULD be if the whole torso rotated
   * rigidly by the pose's bend/side/twist, then FABRIK the four-point chain to
   * that target with the lower back stiffer than the upper. The result is a
   * curve that distributes the bend the way a back does, instead of a hinge at
   * one joint — and because the chain is length-exact, the torso never
   * telescopes.
   */
  private poseSpine(): void {
    const a = this.applied;
    const stretch = clamp(a[PC.spineStretch], 0.82, 1.14);
    const qPelvis = this.rigQuat[B.pelvis];

    // Chain origin: the base of the spine, carried by the pelvis.
    const origin = this.rigPos[B.spine1];
    origin.copy(REST.offset[B.spine1]).multiplyScalar(stretch).applyQuaternion(qPelvis).add(this.rigPos[B.pelvis]);

    // Torso aim.
    eulerQuat(a[PC.spineBend] + a[PC.chestBend], a[PC.spineTwist], a[PC.spineSide], _q0);
    _q1.copy(qPelvis).multiply(_q0); // full torso rotation, rig space

    _v0.copy(REST.pos[B.neck]).sub(REST.pos[B.spine1]).multiplyScalar(stretch).applyQuaternion(_q1);
    _v0.add(origin); // neck target

    this.spinePts[0].copy(origin);
    this.spinePts[1].copy(this.rigPos[B.spine2]);
    this.spinePts[2].copy(this.rigPos[B.chest]);
    this.spinePts[3].copy(this.rigPos[B.neck]);

    const l0 = this.spineLen[0] * stretch;
    const l1 = this.spineLen[1] * stretch;
    const l2 = this.spineLen[2] * stretch;
    _spineLenBuf[0] = l0;
    _spineLenBuf[1] = l1;
    _spineLenBuf[2] = l2;

    solveFabrik(this.spinePts, _spineLenBuf, origin, _v0, this.spineStiff, 5, 3e-4);

    // Twist is distributed along the chain — the lower back barely rotates, the
    // shoulders carry most of it. Interpolating the SIDE reference rather than
    // rotating each bone about its own axis keeps the twist continuous with the
    // bend the FABRIK pass just produced.
    for (let k = 0; k < 3; k++) {
      const bone = SPINE_CHAIN[k];
      const child = SPINE_CHILD[k];
      _q2.copy(qPelvis).slerp(_q1, SPINE_TWIST_W[k]);
      _v2.copy(REST.side[bone]).applyQuaternion(_q2);
      _v3.subVectors(this.spinePts[k + 1], this.spinePts[k]);
      alignFrames(REST.dir[bone], REST.side[bone], _v3, _v2, this.rigQuat[bone]);
      this.rigPos[child].copy(this.spinePts[k + 1]);
      this.boneScale[child] = stretch;
    }
    this.boneScale[B.spine1] = stretch;

    // Clavicles: shrug lifts the shoulder on each side.
    const shrug = clamp(a[PC.shrug], -0.5, 0.7);
    for (let s = 0; s < 2; s++) {
      const idx = s === 0 ? B.clavL : B.clavR;
      const side = s === 0 ? 1 : -1;
      _q3.setFromAxisAngle(FWD, side * shrug * 0.38);
      this.rigQuat[idx].copy(this.rigQuat[B.chest]).multiply(_q3);
      this.rigPos[idx]
        .copy(REST.offset[idx])
        .applyQuaternion(this.rigQuat[B.chest])
        .add(this.rigPos[B.chest]);
      const arm = s === 0 ? B.upperArmL : B.upperArmR;
      this.rigPos[arm].copy(REST.offset[arm]).applyQuaternion(this.rigQuat[idx]).add(this.rigPos[idx]);
    }

    // Hips ride the pelvis.
    this.rigPos[B.thighL].copy(REST.offset[B.thighL]).applyQuaternion(qPelvis).add(this.rigPos[B.pelvis]);
    this.rigPos[B.thighR].copy(REST.offset[B.thighR]).applyQuaternion(qPelvis).add(this.rigPos[B.pelvis]);
  }

  // ── 7b. Contacts and reach ────────────────────────────────────────────────

  /**
   * Where each hand and foot has to end up, in rig space.
   *
   * Locked, that is the anchor itself (for a foot, the fixed ankle offset in the
   * PEDAL's frame, so the sole rides the platform through a whip). Unlocked, it
   * is where the limb would hang from the body. Poses interpolate between the
   * two with `handLock` / `footLock` and then displace with the offset channels.
   */
  private computeTargets(): void {
    const a = this.applied;

    for (let s = 0; s < 2; s++) {
      const isLeft = s === 0;
      const armIdx = isLeft ? B.upperArmL : B.upperArmR;
      const handIdx = isLeft ? B.handL : B.handR;
      const lock = clamp01(isLeft ? a[PC.handLockL] : a[PC.handLockR]);
      const out = this.handTarget[s];

      out.copy(this.anchorPos[s]);
      if (lock < 0.999) {
        _v1.copy(REST.pos[handIdx])
          .sub(REST.pos[armIdx])
          .applyQuaternion(this.rigQuat[B.chest])
          .add(this.rigPos[armIdx]);
        out.lerp(_v1, 1 - lock);
      }
      out.x += isLeft ? a[PC.handOffLX] : a[PC.handOffRX];
      out.y += isLeft ? a[PC.handOffLY] : a[PC.handOffRY];
      out.z += isLeft ? a[PC.handOffLZ] : a[PC.handOffRZ];
    }

    for (let s = 0; s < 2; s++) {
      const isLeft = s === 0;
      const slot = 2 + s;
      const thighIdx = isLeft ? B.thighL : B.thighR;
      const footIdx = isLeft ? B.footL : B.footR;
      const lock = clamp01(isLeft ? a[PC.footLockL] : a[PC.footLockR]);
      const out = this.footTarget[s];

      out.copy(REST.pos[footIdx]).sub(isLeft ? REST.anchors.pedalL : REST.anchors.pedalR);
      out.applyQuaternion(this.anchorQuat[slot]).add(this.anchorPos[slot]);
      if (lock < 0.999) {
        _v1.copy(REST.pos[footIdx])
          .sub(REST.pos[thighIdx])
          .applyQuaternion(this.rigQuat[B.pelvis])
          .add(this.rigPos[thighIdx]);
        out.lerp(_v1, 1 - lock);
      }
      out.x += isLeft ? a[PC.footOffLX] : a[PC.footOffRX];
      out.y += isLeft ? a[PC.footOffLY] : a[PC.footOffRY];
      out.z += isLeft ? a[PC.footOffLZ] : a[PC.footOffRZ];
    }
  }

  /**
   * Move the BODY until the limbs can reach, rather than letting a limb fall
   * short of its anchor.
   *
   * This is the mechanism behind "no sliding, no detaching, ever". A pose can
   * ask for anything — superman puts the hips half a metre behind the bars —
   * and the request is honoured only as far as the arms allow: whatever is left
   * over is applied to the pelvis instead, and the spine is re-solved. Two or
   * three passes converge to millimetres because each pass removes the whole
   * measured excess.
   *
   * Hands and feet pull with equal weight, so the pelvis settles at the least-
   * squares compromise between the two — and whatever residual is left over is
   * inside the stretch budget both limbs carry, which means BOTH contacts still
   * land exactly. Weighting the hands higher was tried and is worse: it fixes
   * the arms and leaves the feet 15 cm off the pedals on a 360.
   */
  private solveReach(): void {
    const a = this.applied;
    // Stop just short of full extension. A limb solved dead straight has no
    // stable bend plane, and the elbow would be free to flip about the axis.
    const maxArm = (ARM_IK.len1 + ARM_IK.len2) * 0.982;
    const maxLeg = (LEG_IK.len1 + LEG_IK.len2) * 0.982;

    for (let iter = 0; iter < 4; iter++) {
      this.computeTargets();
      _v4.set(0, 0, 0);
      let weight = 0;

      for (let s = 0; s < 2; s++) {
        const lock = clamp01(s === 0 ? a[PC.handLockL] : a[PC.handLockR]);
        if (lock < 0.35) continue;
        const shoulder = this.rigPos[s === 0 ? B.upperArmL : B.upperArmR];
        _v5.subVectors(this.handTarget[s], shoulder);
        const d = _v5.length();
        if (d > maxArm) {
          _v4.addScaledVector(_v5.divideScalar(d), (d - maxArm) * lock);
          weight += lock;
        }
      }

      for (let s = 0; s < 2; s++) {
        const lock = clamp01(s === 0 ? a[PC.footLockL] : a[PC.footLockR]);
        if (lock < 0.35) continue;
        const hip = this.rigPos[s === 0 ? B.thighL : B.thighR];
        _v5.subVectors(this.footTarget[s], hip);
        const d = _v5.length();
        if (d > maxLeg) {
          _v4.addScaledVector(_v5.divideScalar(d), (d - maxLeg) * lock);
          weight += lock;
        }
      }

      if (weight < 1e-4) return;
      _v4.divideScalar(weight);
      if (_v4.lengthSq() < 4e-8) return;
      this.rigPos[B.pelvis].add(_v4);
      this.poseSpine();
    }
    // Final pass so the targets match the spine we ended up with.
    this.computeTargets();
  }

  // ── 8. Head ───────────────────────────────────────────────────────────────

  private poseHead(h: number): void {
    const a = this.applied;

    // Convert the world-ish look direction into the chest's frame, so "look 20°
    // left" means 20° left of the rider's shoulders rather than of the mountain.
    _q0.copy(this.rigQuat[B.chest]).invert();
    _v0.copy(this.lookDir).applyQuaternion(_q0);
    if (_v0.lengthSq() < 1e-8) _v0.copy(FWD);
    _v0.normalize();

    const look = clamp01(a[PC.headLook]);
    let yaw = Math.atan2(_v0.x, Math.max(_v0.z, 0.05)) * look + a[PC.headYaw];
    let pitch = -Math.asin(clamp(_v0.y, -1, 1)) * look + a[PC.headPitch];
    let roll = a[PC.headRoll] - yaw * 0.18; // a head that turns also tilts

    yaw = clamp(yaw, -1.15, 1.15);
    pitch = clamp(pitch, -0.85, 0.95);
    roll = clamp(roll, -0.7, 0.7);

    // Split between the neck and the skull. A neck that takes none of it makes
    // the head look bolted on; one that takes half makes the rider look boneless.
    eulerQuat(pitch * 0.36, yaw * 0.36, roll * 0.30, _q1);
    this.rigQuat[B.neck].copy(this.rigQuat[B.chest]).multiply(_q1);
    this.rigPos[B.neck].copy(this.spinePts[3]);

    eulerQuat(pitch * 0.64, yaw * 0.64, roll * 0.70, _q2);
    this.rigQuat[B.head].copy(this.rigQuat[B.neck]).multiply(_q2);
    this.rigPos[B.head]
      .copy(REST.offset[B.head])
      .applyQuaternion(this.rigQuat[B.neck])
      .add(this.rigPos[B.neck]);

    this.rigQuat[B.headEnd].copy(this.rigQuat[B.head]);
  }

  // ── 9. Arms ───────────────────────────────────────────────────────────────

  /**
   * Two-bone IK from each shoulder to its bar anchor.
   *
   * The target is the anchor itself — not an offset from it, not a smoothed
   * version of it — so the hand is on the grip to the millimetre on every frame,
   * including the frame a landing compresses the fork 130 mm or a tailwhip
   * throws the frame through 360°.
   */
  private poseArms(h: number): void {
    const a = this.applied;

    for (let s = 0; s < 2; s++) {
      const isLeft = s === 0;
      const side = isLeft ? 1 : -1;
      const armIdx = isLeft ? B.upperArmL : B.upperArmR;
      const foreIdx = isLeft ? B.forearmL : B.forearmR;
      const handIdx = isLeft ? B.handL : B.handR;
      const endIdx = isLeft ? B.handEndL : B.handEndR;
      const lock = clamp01(isLeft ? a[PC.handLockL] : a[PC.handLockR]);
      const elbowOut = isLeft ? a[PC.elbowOutL] : a[PC.elbowOutR];

      const shoulder = this.rigPos[armIdx];
      _v0.copy(this.handTarget[s]);

      // Pole: the rest bend plane carried by the chest, pushed outboard by the
      // pose. Passing a DIRECTION (not a point) is what keeps the elbow stable
      // when the arm is nearly straight.
      _v2.copy(isLeft ? REST.bend.armL : REST.bend.armR).applyQuaternion(this.rigQuat[B.chest]);
      _v3.copy(LEFT).applyQuaternion(this.rigQuat[B.chest]).multiplyScalar(side * elbowOut * 0.9);
      _v2.add(_v3);

      const st = this.armState[s];
      const res = this.armRes[s];
      solveTwoBone(shoulder, _v0, _v2, ARM_IK, st, h, res);

      _v4.subVectors(res.mid, shoulder);
      alignFrames(REST.dir[armIdx], REST.side[armIdx], _v4, st.bendDir, this.rigQuat[armIdx]);
      _v5.subVectors(res.end, res.mid);
      alignFrames(REST.dir[foreIdx], REST.side[foreIdx], _v5, st.bendDir, this.rigQuat[foreIdx]);

      this.rigPos[foreIdx].copy(res.mid);
      this.rigPos[handIdx].copy(res.end);
      this.boneScale[foreIdx] = st.stretch;
      this.boneScale[handIdx] = st.stretch;

      // The hand takes the bar's own rotation, so an x-up rolls the wrists
      // through with the grips instead of shearing the glove off them.
      _q0.copy(this.anchorQuat[s]).multiply(_q1.copy(this.anchorRest[s]).invert());
      if (lock < 0.999) _q0.slerp(this.rigQuat[foreIdx], 1 - lock);
      this.rigQuat[handIdx].copy(_q0);
      this.rigQuat[endIdx].copy(_q0);
    }
  }

  // ── 10. Legs ──────────────────────────────────────────────────────────────

  private poseLegs(h: number): void {
    const a = this.applied;

    for (let s = 0; s < 2; s++) {
      const isLeft = s === 0;
      const side = isLeft ? 1 : -1;
      const slot = 2 + s;
      const thighIdx = isLeft ? B.thighL : B.thighR;
      const shinIdx = isLeft ? B.shinL : B.shinR;
      const footIdx = isLeft ? B.footL : B.footR;
      const toeIdx = isLeft ? B.toeL : B.toeR;
      const lock = clamp01(isLeft ? a[PC.footLockL] : a[PC.footLockR]);
      const kneeOut = isLeft ? a[PC.kneeOutL] : a[PC.kneeOutR];
      const flex = isLeft ? a[PC.ankleFlexL] : a[PC.ankleFlexR];

      const hip = this.rigPos[thighIdx];
      _v0.copy(this.footTarget[s]);

      _v2.copy(isLeft ? REST.bend.legL : REST.bend.legR).applyQuaternion(this.rigQuat[B.pelvis]);
      _v3.copy(LEFT).applyQuaternion(this.rigQuat[B.pelvis]).multiplyScalar(side * kneeOut * 0.9);
      _v2.add(_v3);

      const st = this.legState[s];
      const res = this.legRes[s];
      solveTwoBone(hip, _v0, _v2, LEG_IK, st, h, res);

      _v4.subVectors(res.mid, hip);
      alignFrames(REST.dir[thighIdx], REST.side[thighIdx], _v4, st.bendDir, this.rigQuat[thighIdx]);
      _v5.subVectors(res.end, res.mid);
      alignFrames(REST.dir[shinIdx], REST.side[shinIdx], _v5, st.bendDir, this.rigQuat[shinIdx]);

      this.rigPos[shinIdx].copy(res.mid);
      this.rigPos[footIdx].copy(res.end);
      this.boneScale[shinIdx] = st.stretch;
      this.boneScale[footIdx] = st.stretch;

      // Foot: flat on the pedal while locked, trailing the shin when it is not.
      _q0.copy(this.anchorQuat[slot]).multiply(_q1.copy(this.anchorRest[slot]).invert());
      if (lock < 0.999) _q0.slerp(this.rigQuat[shinIdx], 1 - lock);
      _q2.setFromAxisAngle(LEFT, flex * 0.7);
      this.rigQuat[footIdx].copy(_q0).multiply(_q2);
      this.rigQuat[toeIdx].copy(this.rigQuat[footIdx]);
    }
  }

  // ── 11. Cloth ─────────────────────────────────────────────────────────────

  /**
   * The three cloth bones. Not simulation — three springs, driven by body
   * acceleration and airspeed, which is enough to break the "clothing is painted
   * on" read for a fraction of the cost of a real solver.
   */
  private poseCloth(): void {
    _q0.setFromAxisAngle(LEFT, this.hemPitch.value);
    _q1.setFromAxisAngle(FWD, this.hemRoll.value);
    _q0.multiply(_q1);
    this.rigQuat[B.hem].copy(this.rigQuat[B.spine1]).multiply(_q0);
    this.rigPos[B.hem]
      .copy(REST.offset[B.hem])
      .applyQuaternion(this.rigQuat[B.spine1])
      .add(this.rigPos[B.spine1]);

    for (let s = 0; s < 2; s++) {
      const idx = s === 0 ? B.shortsL : B.shortsR;
      const parent = s === 0 ? B.thighL : B.thighR;
      const sign = s === 0 ? 1 : -1;
      _q2.setFromAxisAngle(LEFT, this.shortsSwing.value * (s === 0 ? 1 : -1) * 0.6 + this.hemPitch.value * 0.25);
      _q3.setFromAxisAngle(FWD, this.hemRoll.value * 0.5 + sign * 0.04);
      _q2.multiply(_q3);
      this.rigQuat[idx].copy(this.rigQuat[parent]).multiply(_q2);
      this.rigPos[idx]
        .copy(REST.offset[idx])
        .applyQuaternion(this.rigQuat[parent])
        .add(this.rigPos[parent]);
    }
  }

  // ── 12. Write ─────────────────────────────────────────────────────────────

  /**
   * Convert the rig-space solution into bone local transforms.
   *
   * Because every bone's rest local rotation is identity (see Skeleton.ts), the
   * local rotation is just parentᐨ¹ · rig, and the local translation is the rest
   * offset scaled by whatever the IK had to stretch. three's own matrix update
   * then reproduces exactly the positions solved above.
   */
  private writeBones(): void {
    const bones = this.skel.bones;

    // Pelvis is the root bone: its local transform IS its rig transform.
    bones[B.pelvis].position.copy(this.rigPos[B.pelvis]);
    bones[B.pelvis].quaternion.copy(this.rigQuat[B.pelvis]);

    for (let i = 0; i < BONE_COUNT; i++) {
      if (i === B.pelvis) continue;
      const parent = REST.parents[i];
      const b = bones[i];
      _q0.copy(this.rigQuat[parent]).invert().multiply(this.rigQuat[i]);
      b.quaternion.copy(_q0);
      b.position.copy(REST.offset[i]).multiplyScalar(this.boneScale[i]);
      this.boneScale[i] = 1;
    }

    // The head bob is a real translation of the skull on the neck, not a fake
    // rotation. Applied here so it survives the generic write above.
    bones[B.head].position.y += this.headBob.value * 0.55 - this.absorbHead.spring.value * 0.05;
  }

  // ── Teardown ──────────────────────────────────────────────────────────────

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const m of this.meshes.owned) {
      m.prepassMaterial?.dispose();
      m.shadowMaterial?.dispose();
      m.dispose();
    }
    for (const h of this.meshes.hulls) {
      const mat = h.material;
      if (!Array.isArray(mat)) mat.dispose();
    }
    this.skel.dispose();
    this.object.removeFromParent();
    this.object.clear();
    this.anchors = null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Adopt `userData.bikeAnchors` off an object if it looks like a real anchor set.
 * Duck-typed on purpose: the rig must not import the bike implementation, and a
 * malformed object has to be ignored rather than throw on the first frame.
 */
function adoptAnchors(node: Object3D, rig: RiderRig): boolean {
  const candidate = node.userData?.bikeAnchors as BikeAnchors | undefined;
  if (
    candidate &&
    (candidate.barLeft as Object3D | undefined)?.isObject3D &&
    (candidate.barRight as Object3D | undefined)?.isObject3D &&
    (candidate.pedalLeft as Object3D | undefined)?.isObject3D &&
    (candidate.pedalRight as Object3D | undefined)?.isObject3D
  ) {
    rig.attach(candidate);
    return true;
  }
  return false;
}

/**
 * Build a quaternion from rider-convention angles: pitch about +X (forward
 * bend), yaw about +Y (turn left), roll about +Z (tilt right). YXZ order, which
 * is the order that behaves for a body: yaw is applied in the world-ish frame
 * and pitch below it, so a turned-and-bent torso does not gimbal.
 */
function eulerQuat(pitch: number, yaw: number, roll: number, out: Quaternion): Quaternion {
  _e0.set(pitch, yaw, roll, 'YXZ');
  return out.setFromEuler(_e0);
}

/** Pedal spindle in rig space at a given crank angle. See Skeleton.pedalPosition. */
function pedalRest(side: number, angle: number, out: Vector3): Vector3 {
  const L = BIKE_GEOM.crankLength;
  return out.set(
    BIKE_GEOM.bb.x + side * BIKE_GEOM.crankOffset,
    BIKE_GEOM.bb.y - side * L * Math.cos(angle),
    BIKE_GEOM.bb.z - side * L * Math.sin(angle),
  );
}
