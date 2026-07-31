/**
 * BikePhysics — arcade handling with real weight.
 *
 * A rigid body with two raycast wheels. The wheels (Wheel.ts) own the
 * suspension and the tyre model; this file owns the body, the control laws and
 * the state machine. It runs at a FIXED 120 Hz from the engine's accumulator —
 * never at frame rate — so handling cannot change with the frame rate.
 *
 * The four decisions that define how this bike feels:
 *
 *  1. LEAN IS SERVOED, NOT FREE. A 92 kg mass on a 1.08 m wheelbase with a
 *     1.02 m centre of mass is genuinely unstable: simulated honestly it falls
 *     over the instant you stop steering. Real riders stabilise it with a
 *     continuous balance loop, so the bike does too — a critically damped PD
 *     controller drives the roll angle (measured against the CONTACT NORMAL,
 *     not against world up) toward a target set by steering and speed. Because
 *     the reference is the contact normal, the same controller also lays the
 *     bike over onto a berm and stands it up again on the exit for free.
 *
 *  2. THE LEAN IS WHAT TURNS THE BIKE. The steer angle exists and matters at
 *     walking pace, but above ~8 m/s the cornering force comes overwhelmingly
 *     from camber thrust in the tyre model. This is why braking mid-corner
 *     pushes you wide: the friction circle takes the lateral budget away.
 *
 *  3. AIR CONTROL IS AUTHORITY, NOT ATTITUDE. Airborne, input adds angular
 *     acceleration toward a target rate; releasing input does NOT null the
 *     rotation. A flip you started keeps going, which is the only way a
 *     backflip can be a commitment rather than a button.
 *
 *  4. THE PUMP IS A TIMING WINDOW. Crouching stores charge; releasing spends it
 *     as an impulse along the contact normal. If the release lands within
 *     PUMP_WINDOW of leaving the ground, the stored charge is also added to the
 *     vertical velocity directly. That double payoff is deliberate — it is what
 *     makes a well-timed lip release feel categorically different from an
 *     early one rather than just slightly better.
 *
 * Body space: +Y up, +Z forward, +X to the LEFT. `position` is the body origin:
 * mid-wheelbase, on the axle line, suspension topped out.
 */

import { Object3D, Quaternion, Vector3 } from 'three';

import {
  BikeMode,
  type BikeInput,
  type BikeState,
  type SurfaceProperties,
} from '../game/Contracts';
import { BIKE } from '../game/WorldConstants';
import {
  DEFAULT_SURFACE,
  FRONT_TYRE,
  REAR_TYRE,
  Wheel,
  createFallbackTerrain,
  type BikeTerrain,
  type SuspendResult,
} from './Wheel';
import { BIKE_GEOM, FRONT_MOUNT, REAR_MOUNT } from './BikeModel';
import { clamp, clamp01, dampHL, lerp, moveTowards } from '../core/MathX';

// ─────────────────────────────────────────────────────────────────────────────
// Tuning that belongs to the body rather than to a wheel or the world
// ─────────────────────────────────────────────────────────────────────────────

export const BODY_TUNE = {
  /** Diagonal inertia in body space: about X (pitch), Y (yaw), Z (roll). */
  inertiaPitch: 33,
  inertiaYaw: 23,
  inertiaRoll: 16,

  /** Lean controller. omega_n = sqrt(kP); zeta = kD / (2*sqrt(kP)). */
  leanKp: 148,
  leanKd: 25,
  /** How fast the bars move to the demanded angle, radians per second. */
  steerRate: 5.4,
  /** Speed at which lean fully replaces steer as the cornering mechanism. */
  leanAuthoritySpeed: 9.5,

  /** Pitch: manual/endo authority and the passive stabiliser toward the slope. */
  manualTorque: 470,
  endoTorque: 330,
  pitchStabiliserKp: 26,
  pitchStabiliserKd: 7.5,
  /** Yaw damping — kills the residual spin a two-wheeled body accumulates. */
  yawDamp: 1.6,

  /** Airborne. */
  airAuthority: 13.5,
  airIdleDamp: 0.24,
  /** Above this the bike auto-levels toward the slope it is falling onto. */
  airAssistStart: 0.55,
  airAssist: 1.05,

  /** Preload and pump. */
  pumpChargeTime: 0.42,
  pumpDecayTime: 0.55,
  pumpImpulse: 3.05,      // m/s along the contact normal at full charge
  pumpWindow: 0.14,       // s — the release-to-takeoff window worth learning
  pumpAirBonus: 2.70,     // m/s added on top if the window is hit
  hopImpulse: 3.55,

  /** Crash. */
  crashMinTime: 1.15,
  crashSettleSpeed: 2.6,
  recoverTime: 0.85,
  /** Lean beyond this while grounded and moving is a low-side. */
  lowSideLean: 1.28,
  /** Deceleration above this in one step is a wall strike, in m/s. */
  wallImpactSpeed: 7.5,

  /** Rolling resistance scale — the surface table supplies the per-surface part. */
  rollingResistance: 26,
  /** Downforce-ish term so the bike does not float over crests at speed. */
  crestHold: 0.34,
};

// ─────────────────────────────────────────────────────────────────────────────
// Scratch
// ─────────────────────────────────────────────────────────────────────────────
const _f = new Vector3();
const _t = new Vector3();
const _r = new Vector3();
const _n = new Vector3();
const _up = new Vector3();
const _fwd = new Vector3();
const _left = new Vector3();
const _v = new Vector3();
const _v2 = new Vector3();
const _q = new Quaternion();
const _q2 = new Quaternion();
const _omega = new Vector3();
const _susp: SuspendResult = { force: 0, penetration: 0, clearance: 0 };
const _susp2: SuspendResult = { force: 0, penetration: 0, clearance: 0 };

const UP = new Vector3(0, 1, 0);
const FWD = new Vector3(0, 0, 1);

const ZERO_INPUT: BikeInput = {
  steer: 0,
  pedal: 0,
  brakeRear: 0,
  brakeFront: 0,
  crouch: 0,
  pitchLean: 0,
  airPitch: 0,
  airYaw: 0,
  airRoll: 0,
  wantBoost: false,
  wantHop: false,
};

export interface BikePhysicsOptions {
  terrain?: BikeTerrain;
  mass?: number;
  /** Extra grip and stability for the AI, so the pack does not fall over. */
  stabilityBias?: number;
}

export class BikePhysics {
  readonly front: Wheel;
  readonly rear: Wheel;
  readonly state: BikeState;

  /** Where the rider rig, camera and FX read interpolated transforms from. */
  readonly object = new Object3D();

  terrain: BikeTerrain;
  mass: number;
  private stability: number;

  /** Previous physics transform, for the render-side interpolation. */
  readonly prevPosition = new Vector3();
  readonly prevOrientation = new Quaternion();
  private prevFrontSpin = 0;
  private prevRearSpin = 0;
  private prevFrontCompression = 0;
  private prevRearCompression = 0;

  /** Centre of mass in world space, recomputed every step. */
  readonly com = new Vector3();

  // ── Control state ─────────────────────────────────────────────────────────
  private steerDemand = 0;
  private targetLean = 0;
  private crouchPrev = 0;
  private pumpFired = -1e3;
  private pumpFiredCharge = 0;
  private hopPrev = false;
  private modeClock = 0;
  private crashClock = 0;
  private groundedGrace = 0;
  private lastSpeed = 0;

  /** Accumulators, cleared every step. */
  private force = new Vector3();
  private torque = new Vector3();

  constructor(opts: BikePhysicsOptions = {}) {
    this.terrain = opts.terrain ?? createFallbackTerrain();
    this.mass = opts.mass ?? BIKE.mass;
    this.stability = opts.stabilityBias ?? 0;

    this.front = new Wheel({
      mountLocal: FRONT_MOUNT.clone(),
      travel: BIKE.forkTravel,
      stiffness: BIKE.forkStiffness,
      damping: BIKE.forkDamping,
      radius: BIKE.wheelRadius,
      isFront: true,
      tyre: FRONT_TYRE,
      compressionDampScale: 0.86,
      reboundDampScale: 1.34,
    });

    this.rear = new Wheel({
      mountLocal: REAR_MOUNT.clone(),
      travel: BIKE.shockTravel,
      stiffness: BIKE.shockStiffness,
      damping: BIKE.shockDamping,
      radius: BIKE.wheelRadius,
      isFront: false,
      tyre: REAR_TYRE,
      compressionDampScale: 0.92,
      reboundDampScale: 1.42,
    });

    this.state = {
      position: new Vector3(),
      velocity: new Vector3(),
      orientation: new Quaternion(),
      angularVelocity: new Vector3(),
      forwardSpeed: 0,
      speed: 0,
      lean: 0,
      steerAngle: 0,
      pitch: 0,
      front: this.front,
      rear: this.rear,
      mode: BikeMode.Grounded,
      modeTime: 0,
      airHeight: 0,
      airTime: 0,
      peakAirHeight: 0,
      preload: 0,
      pumpCharge: 0,
      manualling: false,
      manualAmount: 0,
      boost: 0,
      boosting: false,
      landedThisStep: false,
      landingImpact: 0,
      landingQuality: 1,
      crashedThisStep: false,
      crashDirection: new Vector3(0, 0, 1),
      crashSeverity: 0,
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // The step
  // ───────────────────────────────────────────────────────────────────────────

  step(inputRaw: BikeInput, dt: number): void {
    const s = this.state;
    const input = s.mode === BikeMode.Crashing || s.mode === BikeMode.Finished ? ZERO_INPUT : inputRaw;

    // Latched-for-one-step flags belong to the consumer of the PREVIOUS step,
    // so they are cleared at the top, not the bottom.
    s.landedThisStep = false;
    s.crashedThisStep = false;

    this.prevPosition.copy(s.position);
    this.prevOrientation.copy(s.orientation);
    this.prevFrontSpin = this.front.spinUnwrapped;
    this.prevRearSpin = this.rear.spinUnwrapped;
    this.prevFrontCompression = this.front.compression;
    this.prevRearCompression = this.rear.compression;

    this.force.set(0, 0, 0);
    this.torque.set(0, 0, 0);

    // ── Frames ──────────────────────────────────────────────────────────────
    _up.copy(UP).applyQuaternion(s.orientation);
    _fwd.copy(FWD).applyQuaternion(s.orientation);
    _left.copy(_up).cross(_fwd).normalize();
    this.com.copy(s.position).addScaledVector(_up, BIKE.comHeight);

    s.speed = s.velocity.length();
    s.forwardSpeed = s.velocity.dot(_fwd);

    // ── Controls ────────────────────────────────────────────────────────────
    this.updateSteering(input, dt);
    this.updatePreload(input, dt);
    this.front.steer = s.steerAngle;
    this.rear.steer = 0;

    // ── Suspension ──────────────────────────────────────────────────────────
    this.front.suspend(s.position, s.orientation, this.com, s.velocity, s.angularVelocity, this.terrain, dt, _susp);
    this.rear.suspend(s.position, s.orientation, this.com, s.velocity, s.angularVelocity, this.terrain, dt, _susp2);

    this.applyNormalForce(this.front, _susp);
    this.applyNormalForce(this.rear, _susp2);

    const wasAirborne = s.mode === BikeMode.Airborne;
    const grounded = this.front.grounded || this.rear.grounded;

    // ── The ground reference frame ──────────────────────────────────────────
    // Everything about balance is measured against this, not against world up.
    _n.set(0, 0, 0);
    if (this.front.grounded) _n.addScaledVector(this.front.contactNormal, 0.42);
    if (this.rear.grounded) _n.addScaledVector(this.rear.contactNormal, 0.58);
    if (_n.lengthSq() < 1e-6) {
      // Airborne: aim at the ground we are actually going to land on.
      const h = this.terrain.heightAt(s.position.x, s.position.z);
      this.terrain.normalAt(s.position.x, s.position.z, _n);
      s.airHeight = Math.max(0, s.position.y - h);
    } else {
      s.airHeight = 0;
    }
    _n.normalize();

    // Signed roll of the body about its own forward axis, relative to the
    // ground normal. Positive is leaning right.
    _v.copy(_n).cross(_up);
    s.lean = Math.atan2(_v.dot(_fwd), clamp(_n.dot(_up), -1, 1));

    // Signed pitch: how far the bike's forward axis is above the ground plane.
    s.pitch = Math.asin(clamp(-_fwd.dot(_n), -1, 1));

    // ── Drive, brakes, tyres ────────────────────────────────────────────────
    this.solveDrive(input, dt);

    // ── Body forces ─────────────────────────────────────────────────────────
    this.force.y -= this.mass * BIKE.gravity;

    // Aerodynamic drag, plus the surface's own drag term.
    if (s.speed > 0.05) {
      const surfDrag = this.rear.grounded ? this.rear.surface.drag : 0;
      const k = BIKE.dragK + surfDrag * 0.5;
      this.force.addScaledVector(s.velocity, -k * s.speed);
    }

    // Crest hold. A bike cresting a roller at speed leaves the ground a frame
    // before the geometry says it should, which reads as floaty; a small pull
    // along the contact normal keeps the wheels down over rollers without
    // affecting a genuine lip.
    if (grounded && s.speed > 6) {
      const curve = clamp01((_n.y - _up.y) * 4);
      this.force.addScaledVector(_n, -this.mass * BIKE.gravity * BODY_TUNE.crestHold * curve);
    }

    // ── Attitude control ────────────────────────────────────────────────────
    if (s.mode === BikeMode.Crashing) {
      this.updateCrash(dt);
    } else if (grounded) {
      this.controlGrounded(input, dt, _n);
    } else {
      this.controlAirborne(input, dt, _n);
    }

    // ── Integrate ───────────────────────────────────────────────────────────
    this.integrate(dt);

    // ── Mode machine ────────────────────────────────────────────────────────
    this.updateMode(grounded, wasAirborne, dt, _n);

    s.speed = s.velocity.length();
    _fwd.copy(FWD).applyQuaternion(s.orientation);
    s.forwardSpeed = s.velocity.dot(_fwd);
    s.modeTime += dt;
    this.lastSpeed = s.speed;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Controls
  // ───────────────────────────────────────────────────────────────────────────

  private updateSteering(input: BikeInput, dt: number): void {
    const s = this.state;
    const speed = Math.abs(s.forwardSpeed);

    // Steering authority falls with speed. Without this the bars are a
    // catastrophe at 20 m/s and useless in a switchback at 4.
    const falloff = 1 / (1 + BIKE.steerSpeedFalloff * speed);
    this.steerDemand = input.steer * BIKE.maxSteer * falloff;
    s.steerAngle = moveTowards(s.steerAngle, this.steerDemand, BODY_TUNE.steerRate * dt);

    // Above `leanAuthoritySpeed` the turn is carried by lean, below it by the
    // bars. The crossover is what makes slow technical sections feel like
    // steering and fast sections feel like carving.
    const leanAuthority = clamp01(speed / BODY_TUNE.leanAuthoritySpeed);
    const grip = this.rear.grounded ? this.rear.surface.grip : 1;
    this.targetLean = input.steer * BIKE.maxLean * leanAuthority * lerp(0.7, 1, grip);

    // Braking stands the bike up — you cannot hold full lean on the brakes.
    const braking = clamp01(input.brakeRear * 0.6 + input.brakeFront);
    this.targetLean *= 1 - braking * 0.34;
  }

  private updatePreload(input: BikeInput, dt: number): void {
    const s = this.state;
    // Asymmetric: compressing takes ~90 ms, releasing ~35 ms. The release has
    // to be near-instant or the pump timing window is unlearnable.
    const rate = input.crouch > s.preload ? 0.09 : 0.035;
    s.preload = dampHL(s.preload, input.crouch, rate, dt);

    const grounded = this.front.grounded || this.rear.grounded;
    if (input.crouch > 0.55 && grounded) {
      s.pumpCharge = Math.min(1, s.pumpCharge + dt / BODY_TUNE.pumpChargeTime);
    } else if (input.crouch < 0.35) {
      s.pumpCharge = Math.max(0, s.pumpCharge - dt / BODY_TUNE.pumpDecayTime);
    }

    // Release edge: spend the charge as an impulse along the contact normal.
    const released = this.crouchPrev > 0.6 && input.crouch < 0.35;
    if (released && s.pumpCharge > 0.12) {
      this.pumpFired = 0;
      this.pumpFiredCharge = s.pumpCharge;
      if (grounded) {
        _n.copy(this.rear.grounded ? this.rear.contactNormal : this.front.contactNormal);
        s.velocity.addScaledVector(_n, BODY_TUNE.pumpImpulse * s.pumpCharge);
      }
      s.pumpCharge = 0;
    } else if (this.pumpFired >= 0) {
      this.pumpFired += dt;
      if (this.pumpFired > BODY_TUNE.pumpWindow) this.pumpFired = -1e3;
    }
    this.crouchPrev = input.crouch;

    // Bunny hop, on the button edge, from a compressed stance.
    if (input.wantHop && !this.hopPrev && grounded) {
      _n.copy(this.rear.grounded ? this.rear.contactNormal : this.front.contactNormal);
      const scale = 0.62 + s.preload * 0.55;
      s.velocity.addScaledVector(_n, BODY_TUNE.hopImpulse * scale);
      // A hop is a rear-first lift: nose up slightly so it looks like a hop and
      // not a lift on a string.
      _t.copy(_left).multiplyScalar(-BODY_TUNE.manualTorque * 0.6);
      this.torque.add(_t);
    }
    this.hopPrev = input.wantHop;
  }

  private solveDrive(input: BikeInput, dt: number): void {
    const s = this.state;

    // Pedalling. Force falls to zero at spin-out; you cannot pedal a BMX to
    // 20 m/s on the flat, and the descent is supposed to supply the speed.
    const spinOut = clamp01(1 - Math.abs(s.forwardSpeed) / BIKE.spinOutSpeed);
    let drive = input.pedal * BIKE.pedalForce * spinOut * spinOut;

    // Boost spends the trick meter.
    s.boosting = false;
    if (input.wantBoost && s.boost > 0.02) {
      s.boost = Math.max(0, s.boost - BIKE.boostDrainPerSecond * dt);
      s.boosting = true;
      drive += BIKE.boostForce;
    }

    // Rolling resistance, applied as a brake so the friction circle sees it.
    const rrFront = this.front.grounded
      ? this.front.surface.rollingResistance * BODY_TUNE.rollingResistance
      : 0;
    const rrRear = this.rear.grounded
      ? this.rear.surface.rollingResistance * BODY_TUNE.rollingResistance
      : 0;

    const brakeRear = input.brakeRear * BIKE.brakeForceRear + rrRear;
    const brakeFront = input.brakeFront * BIKE.brakeForceFront + rrFront;

    this.front.solveTyre(
      s.orientation, this.com, s.velocity, s.angularVelocity,
      0, brakeFront, s.lean, this.mass, dt,
    );
    this.rear.solveTyre(
      s.orientation, this.com, s.velocity, s.angularVelocity,
      drive, brakeRear, s.lean, this.mass, dt,
    );

    this.applyContactForce(this.front);
    this.applyContactForce(this.rear);
  }

  private applyNormalForce(w: Wheel, out: SuspendResult): void {
    if (out.force <= 0) return;
    _f.copy(w.contactNormal).multiplyScalar(out.force);
    this.force.add(_f);
    _r.copy(w.contactPoint).sub(this.com);
    _t.copy(_r).cross(_f);
    this.torque.add(_t);

    // Penetration recovery: push straight out, no torque, so a wheel that ends
    // a step buried does not also get flicked.
    if (out.penetration > 0.001) {
      this.state.position.addScaledVector(w.contactNormal, Math.min(out.penetration, 0.12));
    }
  }

  private applyContactForce(w: Wheel): void {
    if (!w.grounded || w.tyreForce.lengthSq() < 1e-8) return;
    this.force.add(w.tyreForce);
    _r.copy(w.contactPoint).sub(this.com);
    _t.copy(_r).cross(w.tyreForce);
    this.torque.add(_t);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Attitude
  // ───────────────────────────────────────────────────────────────────────────

  private controlGrounded(input: BikeInput, dt: number, n: Vector3): void {
    const s = this.state;

    // ── Roll: the balance loop ──────────────────────────────────────────────
    const rollRate = s.angularVelocity.dot(_fwd);
    const leanErr = this.targetLean - s.lean;
    const kp = BODY_TUNE.leanKp * (1 + this.stability * 0.5);
    const kd = BODY_TUNE.leanKd * (1 + this.stability * 0.35);
    const rollTorque = BODY_TUNE.inertiaRoll * (leanErr * kp - rollRate * kd);
    this.torque.addScaledVector(_fwd, clamp(rollTorque, -9000, 9000));

    // ── Pitch: manual, endo, and a stabiliser toward the slope ──────────────
    // pitchLean is +1 back (manual) .. -1 forward (endo / over the bars).
    const back = Math.max(0, input.pitchLean);
    const fwdLean = Math.max(0, -input.pitchLean);

    // A manual needs speed and a loaded rear wheel; you cannot manual a
    // stationary bike, and pretending you can is the tell of a fake one.
    const manualAuth = clamp01((Math.abs(s.forwardSpeed) - 1.6) / 4) * clamp01(this.rear.load / 500);
    const manualDrive = back * BODY_TUNE.manualTorque * manualAuth;
    const endoDrive = fwdLean * BODY_TUNE.endoTorque * clamp01(this.front.load / 400);

    // Torque about the LEFT axis: positive raises the nose.
    this.torque.addScaledVector(_left, -manualDrive + endoDrive);

    // Passive stabiliser, only where the rider is not asking for attitude. It
    // is what stops the bike from slowly nosing into the ground on a long
    // descent, and it is deliberately weak so bumps still pitch the bike.
    const authority = 1 - clamp01(Math.abs(input.pitchLean));
    if (authority > 0.01) {
      const pitchRate = s.angularVelocity.dot(_left);
      const pitchErr = -s.pitch; // want the forward axis in the ground plane
      const stab =
        BODY_TUNE.inertiaPitch *
        (pitchErr * BODY_TUNE.pitchStabiliserKp - pitchRate * BODY_TUNE.pitchStabiliserKd) *
        authority;
      this.torque.addScaledVector(_left, clamp(stab, -2600, 2600));
    }

    // Manual readout for the rig and the trick system.
    const frontUp = !this.front.grounded && this.rear.grounded;
    s.manualAmount = dampHL(s.manualAmount, frontUp ? clamp01(0.35 + back * 0.65) : 0, 0.09, dt);
    s.manualling = frontUp && s.manualAmount > 0.35;

    // ── Yaw damping ─────────────────────────────────────────────────────────
    _v.copy(_up).multiplyScalar(-s.angularVelocity.dot(_up) * BODY_TUNE.yawDamp * BODY_TUNE.inertiaYaw);
    this.torque.add(_v);

    // Low-side. Past the lean limit with load on the wheels, the bike is down.
    if (Math.abs(s.lean) > BODY_TUNE.lowSideLean && s.speed > 5.5) {
      this.beginCrash(clamp01((Math.abs(s.lean) - BODY_TUNE.lowSideLean) * 2 + 0.3), _left);
    }

    // Wall strike: a big unexplained deceleration in one step.
    const decel = this.lastSpeed - s.speed;
    if (decel > BODY_TUNE.wallImpactSpeed * dt * 60 && this.lastSpeed > 9) {
      _v.copy(s.velocity).normalize();
      this.beginCrash(clamp01(decel / (BODY_TUNE.wallImpactSpeed * dt * 120)), _v);
    }
  }

  private controlAirborne(input: BikeInput, dt: number, n: Vector3): void {
    const s = this.state;
    s.manualling = false;
    s.manualAmount = dampHL(s.manualAmount, 0, 0.14, dt);

    // Authority, not attitude: input adds acceleration toward a target rate and
    // releasing input leaves the rotation alone (bar a whisper of damping), so
    // a flip you committed to keeps going.
    const targetPitch = input.airPitch * BIKE.airPitchRate;
    const targetYaw = input.airYaw * BIKE.airYawRate;
    const targetRoll = input.airRoll * BIKE.airRollRate;

    const curPitch = s.angularVelocity.dot(_left);
    const curYaw = s.angularVelocity.dot(_up);
    const curRoll = s.angularVelocity.dot(_fwd);

    const A = BODY_TUNE.airAuthority;
    const idle = BODY_TUNE.airIdleDamp;

    const dPitch = input.airPitch !== 0 ? (targetPitch - curPitch) * A : -curPitch * idle;
    const dYaw = input.airYaw !== 0 ? (targetYaw - curYaw) * A : -curYaw * idle;
    const dRoll = input.airRoll !== 0 ? (targetRoll - curRoll) * A : -curRoll * idle;

    _omega.copy(_left).multiplyScalar(dPitch * dt);
    _omega.addScaledVector(_up, dYaw * dt);
    _omega.addScaledVector(_fwd, dRoll * dt);
    s.angularVelocity.add(_omega);

    // Landing assist. Late in a fall, with no rotation input, the bike drifts
    // toward matching the slope it is about to hit. This is the difference
    // between a game where every jump ends in a wash-out and one where landing
    // well is a skill you can actually express. It is weak, and it turns off
    // completely the moment the player asks for rotation.
    const noInput = input.airPitch === 0 && input.airRoll === 0;
    const falling = s.velocity.y < -1.5;
    if (noInput && falling && s.airTime > BODY_TUNE.airAssistStart) {
      _v.copy(_up).cross(n);            // axis that rotates up toward n
      const err = Math.acos(clamp(_up.dot(n), -1, 1));
      if (_v.lengthSq() > 1e-8 && err > 0.02) {
        _v.normalize().multiplyScalar(err * BODY_TUNE.airAssist * dt * clamp01(s.airTime));
        s.angularVelocity.add(_v);
      }
    }
  }

  private updateCrash(dt: number): void {
    const s = this.state;
    this.crashClock += dt;

    // Tumble: bleed rotation slowly so it reads as a body with mass rolling to
    // a stop, not as a ragdoll being switched off.
    s.angularVelocity.multiplyScalar(Math.pow(0.62, dt));

    // Ground friction on whatever is touching.
    if (this.front.grounded || this.rear.grounded) {
      _v.copy(s.velocity);
      _v.y = 0;
      this.force.addScaledVector(_v, -this.mass * 2.4);
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Integration
  // ───────────────────────────────────────────────────────────────────────────

  private integrate(dt: number): void {
    const s = this.state;

    // Linear.
    _v.copy(this.force).multiplyScalar(dt / this.mass);
    s.velocity.add(_v);
    s.position.addScaledVector(s.velocity, dt);

    // Angular. The inertia tensor is diagonal in BODY space, so the torque is
    // rotated in, divided, and rotated back. The gyroscopic w x Iw term is
    // dropped: at these rates it contributes less than the integrator's own
    // error and it is a reliable source of instability.
    _q2.copy(s.orientation).invert();
    _v.copy(this.torque).applyQuaternion(_q2);
    _v.x /= BODY_TUNE.inertiaPitch;
    _v.y /= BODY_TUNE.inertiaYaw;
    _v.z /= BODY_TUNE.inertiaRoll;
    _v.applyQuaternion(s.orientation);
    s.angularVelocity.addScaledVector(_v, dt);

    // Hard clamp. Nothing legitimate spins faster than this, and an unclamped
    // angular velocity after a bad contact turns one bad frame into a NaN.
    const w = s.angularVelocity.length();
    if (w > 26) s.angularVelocity.multiplyScalar(26 / w);

    if (w > 1e-6) {
      _v.copy(s.angularVelocity).multiplyScalar(1 / w);
      _q.setFromAxisAngle(_v, w * dt);
      s.orientation.premultiply(_q).normalize();
    }

    // Terminal velocity, so a fall down the ravine cannot outrun the collision.
    const speed = s.velocity.length();
    if (speed > 62) s.velocity.multiplyScalar(62 / speed);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Mode machine
  // ───────────────────────────────────────────────────────────────────────────

  private updateMode(grounded: boolean, wasAirborne: boolean, dt: number, n: Vector3): void {
    const s = this.state;

    // A one-wheel-off bump is not a jump. Both wheels have to be clear for a
    // short grace period before the bike is airborne, or the state flickers
    // through every rock garden and every FX system flickers with it.
    if (grounded) this.groundedGrace = 0.085;
    else this.groundedGrace = Math.max(0, this.groundedGrace - dt);

    const airborne = !grounded && this.groundedGrace <= 0;

    switch (s.mode) {
      case BikeMode.Grounded: {
        s.airTime = 0;
        s.peakAirHeight = 0;
        if (airborne) this.setMode(BikeMode.Airborne);
        break;
      }

      case BikeMode.Airborne: {
        s.airTime += dt;
        s.peakAirHeight = Math.max(s.peakAirHeight, s.airHeight);

        // The pump window: a release that lands within PUMP_WINDOW of takeoff
        // pays out again, directly into vertical velocity.
        if (this.pumpFired >= 0 && this.pumpFired <= BODY_TUNE.pumpWindow && s.airTime < dt * 2) {
          s.velocity.addScaledVector(n, BODY_TUNE.pumpAirBonus * this.pumpFiredCharge);
          this.pumpFired = -1e3;
        }

        if (grounded) this.resolveLanding(n);
        break;
      }

      case BikeMode.Crashing: {
        const slow = s.speed < BODY_TUNE.crashSettleSpeed;
        if (this.crashClock > BODY_TUNE.crashMinTime && slow && grounded) {
          this.setMode(BikeMode.Recovering);
        }
        break;
      }

      case BikeMode.Recovering: {
        // Stand the bike back up over `recoverTime`, aligned to the slope and
        // pointing down the fall line it was travelling on.
        const k = clamp01(dt / Math.max(BODY_TUNE.recoverTime - s.modeTime, dt));
        _fwd.copy(FWD).applyQuaternion(s.orientation);
        _v.copy(_fwd).addScaledVector(n, -_fwd.dot(n));
        if (_v.lengthSq() < 1e-6) _v.set(0, 0, 1);
        _v.normalize();
        buildOrientation(_v, n, _q);
        s.orientation.slerp(_q, k).normalize();
        s.angularVelocity.multiplyScalar(Math.pow(0.05, dt));
        if (s.modeTime >= BODY_TUNE.recoverTime) this.setMode(BikeMode.Grounded);
        break;
      }

      case BikeMode.Finished:
      default:
        break;
    }
  }

  private resolveLanding(n: Vector3): void {
    const s = this.state;
    _up.copy(UP).applyQuaternion(s.orientation);

    // Quality is the angle between the bike's up and the slope's normal. Nail
    // the slope and it is 1; land flat on a steep landing and it collapses.
    const align = clamp(_up.dot(n), -1, 1);
    const mismatch = Math.acos(align);
    s.landingQuality = clamp01(1 - mismatch / BIKE.landingAngleTolerance);

    const closing = Math.max(0, -s.velocity.dot(n));
    s.landingImpact = clamp01(closing / BIKE.landingSpeedTolerance);
    s.landedThisStep = true;

    // A bad landing scrubs speed; a good one keeps almost all of it. This is
    // the entire reward for learning to match the slope.
    //
    // But the penalty has to be proportional to the IMPACT, not applied at full
    // strength to every touchdown. Speed is lost on landing because the vertical
    // component gets absorbed, so a 0.2 m hop over a stone should cost almost
    // nothing. Charging it the full mismatch penalty put the bike in a
    // catastrophic loop on rough ground: hop, land a few degrees nose-down, lose
    // 58% of the speed, hop again — 19 m/s bled to 8 m/s in four seconds of
    // DOWNHILL. The rider never did anything wrong; the trail was simply bumpy.
    const bite = clamp01(s.landingImpact / 0.55);
    const keep = lerp(1.0, lerp(0.42, 0.985, s.landingQuality * s.landingQuality), bite);
    _v.copy(s.velocity).addScaledVector(n, -s.velocity.dot(n)); // in-plane part
    const along = _v.length();
    _v.multiplyScalar(along > 1e-5 ? (along * keep) / along : 0);
    s.velocity.copy(_v);

    // Kill the rotation the air gave us; the suspension takes the rest.
    s.angularVelocity.multiplyScalar(0.22);

    const tooHard = s.landingImpact > 0.94;
    const tooCrooked = s.landingQuality < 0.16 && s.landingImpact > 0.30;
    if (tooHard || tooCrooked) {
      _v.copy(s.velocity).normalize();
      this.beginCrash(clamp01(Math.max(s.landingImpact, 1 - s.landingQuality)), _v);
    } else {
      this.setMode(BikeMode.Grounded);
    }
    s.airTime = 0;
  }

  private beginCrash(severity: number, direction: Vector3): void {
    const s = this.state;
    if (s.mode === BikeMode.Crashing || s.mode === BikeMode.Finished) return;
    this.setMode(BikeMode.Crashing);
    this.crashClock = 0;
    s.crashedThisStep = true;
    s.crashSeverity = clamp01(0.25 + severity * 0.75);
    s.crashDirection.copy(direction).normalize();
    if (s.crashDirection.lengthSq() < 0.5) s.crashDirection.set(0, 0, 1);

    // Give the tumble a real axis: the crash direction crossed with up, scaled
    // by severity. Deterministic, so a replay and a ghost agree.
    _v.copy(s.crashDirection).cross(UP).normalize();
    if (_v.lengthSq() < 0.5) _v.set(1, 0, 0);
    s.angularVelocity.addScaledVector(_v, 4.2 + s.crashSeverity * 6.5);
    s.angularVelocity.addScaledVector(UP, (s.crashDirection.x - s.crashDirection.z) * 2.4);
    s.velocity.multiplyScalar(lerp(0.86, 0.44, s.crashSeverity));
    s.velocity.y += 1.4 + s.crashSeverity * 2.2;

    s.pumpCharge = 0;
    s.preload = 0;
    s.manualling = false;
  }

  private setMode(m: BikeMode): void {
    if (this.state.mode === m) return;
    this.state.mode = m;
    this.state.modeTime = 0;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // External control
  // ───────────────────────────────────────────────────────────────────────────

  reset(position: Vector3, forward: Vector3): void {
    const s = this.state;
    this.terrain.normalAt(position.x, position.z, _n);
    if (_n.lengthSq() < 1e-6) _n.set(0, 1, 0);
    _n.normalize();

    _v.copy(forward).addScaledVector(_n, -forward.dot(_n));
    if (_v.lengthSq() < 1e-6) _v.set(0, 0, 1);
    _v.normalize();
    buildOrientation(_v, _n, s.orientation);

    s.position.copy(position);
    s.velocity.set(0, 0, 0);
    s.angularVelocity.set(0, 0, 0);
    s.forwardSpeed = 0;
    s.speed = 0;
    s.lean = 0;
    s.steerAngle = 0;
    s.pitch = 0;
    s.airTime = 0;
    s.airHeight = 0;
    s.peakAirHeight = 0;
    s.preload = 0;
    s.pumpCharge = 0;
    s.manualling = false;
    s.manualAmount = 0;
    s.boosting = false;
    s.landedThisStep = false;
    s.landingImpact = 0;
    s.landingQuality = 1;
    s.crashedThisStep = false;
    s.crashSeverity = 0;
    this.setMode(BikeMode.Grounded);
    this.steerDemand = 0;
    this.targetLean = 0;
    this.crouchPrev = 0;
    this.pumpFired = -1e3;
    this.hopPrev = false;
    this.crashClock = 0;
    this.groundedGrace = 0.1;
    this.lastSpeed = 0;

    this.front.reset();
    this.rear.reset();
    this.prevPosition.copy(position);
    this.prevOrientation.copy(s.orientation);
    this.prevFrontSpin = 0;
    this.prevRearSpin = 0;
    this.prevFrontCompression = 0;
    this.prevRearCompression = 0;
    this.com.copy(position).addScaledVector(_n, BIKE.comHeight);
  }

  /** Add boost meter, 0..1. Called by the trick system on a banked trick. */
  addBoost(amount: number): void {
    this.state.boost = clamp01(this.state.boost + amount);
  }

  /** Force the finished state — no input, but the bike keeps rolling out. */
  finish(): void {
    this.setMode(BikeMode.Finished);
  }

  /**
   * Put the bike down deliberately. The capture harness needs a crash it can
   * shoot on a known frame, and waiting for one to happen naturally makes the
   * review set non-deterministic — which defeats the point of a fixed-dt
   * harness. Also used by the race director for scripted AI mistakes.
   */
  forceCrash(severity = 0.7, direction?: Vector3): void {
    if (direction) {
      _v2.copy(direction);
    } else {
      _v2.copy(this.state.velocity);
      if (_v2.lengthSq() < 1e-4) _v2.set(0, 0, 1);
    }
    this.beginCrash(severity, _v2);
  }

  // ── Interpolation accessors, for the render side ──────────────────────────

  interpolatedPosition(alpha: number, out: Vector3): Vector3 {
    return out.copy(this.prevPosition).lerp(this.state.position, alpha);
  }

  interpolatedOrientation(alpha: number, out: Quaternion): Quaternion {
    return out.copy(this.prevOrientation).slerp(this.state.orientation, alpha);
  }

  interpolatedFrontSpin(alpha: number): number {
    return lerp(this.prevFrontSpin, this.front.spinUnwrapped, alpha);
  }

  interpolatedRearSpin(alpha: number): number {
    return lerp(this.prevRearSpin, this.rear.spinUnwrapped, alpha);
  }

  interpolatedFrontCompression(alpha: number): number {
    return lerp(this.prevFrontCompression, this.front.compression, alpha);
  }

  interpolatedRearCompression(alpha: number): number {
    return lerp(this.prevRearCompression, this.rear.compression, alpha);
  }

  /** The surface currently under the bike, for FX and audio. */
  get surface(): SurfaceProperties {
    if (this.rear.grounded) return this.rear.surface;
    if (this.front.grounded) return this.front.surface;
    return DEFAULT_SURFACE;
  }
}

/**
 * Build an orientation from a forward direction and an up direction, without
 * the Object3D.lookAt round trip (which allocates and flips handedness).
 * Bike space is +Z forward, +Y up, +X left.
 */
export function buildOrientation(forward: Vector3, up: Vector3, out: Quaternion): Quaternion {
  _v2.copy(forward).normalize();
  _v.copy(up).addScaledVector(_v2, -up.dot(_v2));
  if (_v.lengthSq() < 1e-8) _v.set(0, 1, 0).addScaledVector(_v2, -_v2.y);
  _v.normalize();
  // left = up x forward
  const lx = _v.y * _v2.z - _v.z * _v2.y;
  const ly = _v.z * _v2.x - _v.x * _v2.z;
  const lz = _v.x * _v2.y - _v.y * _v2.x;
  // Column-major 3x3 as [left, up, forward] -> quaternion.
  return quatFromBasis(lx, ly, lz, _v.x, _v.y, _v.z, _v2.x, _v2.y, _v2.z, out);
}

function quatFromBasis(
  m00: number, m10: number, m20: number,
  m01: number, m11: number, m21: number,
  m02: number, m12: number, m22: number,
  out: Quaternion,
): Quaternion {
  const trace = m00 + m11 + m22;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    out.set((m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s, 0.25 / s);
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    out.set(0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s);
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    out.set((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s);
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    out.set((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s);
  }
  return out.normalize();
}

/** Re-export so the wheel geometry constants travel with the physics. */
export { BIKE_GEOM };
