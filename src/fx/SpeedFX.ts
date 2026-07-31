/**
 * SpeedFX — making speed feel like speed.
 *
 * Three separate systems live here because they are three separate answers to
 * the same question, and a racing game needs all three:
 *
 *  1. SPEED LINES. The drawn convention — radial tapered wedges from a focus
 *     point. The shader already exists in ShaderChunks (speedLines()); what
 *     this module owns is the DRIVING of it, which is where it actually lives
 *     or dies. Two details matter far more than the shader: the ramp is
 *     non-linear so lines only appear when you are genuinely fast (below ~11
 *     m/s there are none at all, and the response is cubic-ish above that), and
 *     the focus point sits ~20m ahead along the TRAVEL direction rather than at
 *     screen centre. Dead-centre lines read as a screensaver; lines converging
 *     on the point you are actually heading toward read as motion, and they
 *     swing to the inside of a corner on their own.
 *
 *  2. GEOMETRY MOTION SMEAR. Not a post-process blur — a duplicate of the mesh
 *     whose trailing vertices are extruded backward along the LOCAL velocity of
 *     each vertex, drawn translucent with QUANTISED opacity. Local, not global:
 *     the extrusion is driven by linear velocity PLUS omega x r, so during a
 *     360 the outstretched limbs streak hard and the torso barely moves, which
 *     is exactly how a rotating figure is drawn. This is how animation draws a
 *     fast limb: a solid shape with a tail, not a blurred photograph. Post blur
 *     softens edges, and every other pixel in this game is trying to stay hard.
 *
 *  3. WHEEL SPIN SMEAR. A separate mechanism again, because neither of the
 *     above can express a wheel. A thin annulus in the wheel plane draws
 *     quantised radial streak wedges whose angular width opens up with spin
 *     rate — the drawn shorthand for a spinning wheel, one draw call each.
 */

import {
  BufferGeometry,
  Color,
  DoubleSide,
  GLSL3,
  InstancedMesh,
  Matrix4,
  Mesh,
  NormalBlending,
  Object3D,
  PerspectiveCamera,
  Quaternion,
  RingGeometry,
  ShaderMaterial,
  SkinnedMesh,
  Vector2,
  Vector3,
} from 'three';

import { BikeMode, type BikeState } from '../game/Contracts';
import { clamp, clamp01, dampHL } from '../core/MathX';
import { globalUniformBlock, POST_STATE } from '../npr/NprGlobals';
import { RAMPS, SUN_RIM_COLOR } from '../npr/Palette';
import { GLSL_COMMON, GLSL_FRAG_OUT, GLSL_GLOBAL_UNIFORMS, GLSL_VERTEX_TRANSFORM } from '../npr/ShaderChunks';

// ── Module scratch. Nothing below allocates in an update path. ───────────────
const _wp = new Vector3();
const _inst = new Vector3();
const _camDir = new Vector3();
const _toPoint = new Vector3();
const _focusPoint = new Vector3();
const _dir = new Vector3();
const _axis = new Vector3();
const _scale = new Vector3();
const _quat = new Quaternion();
const _UNIT_Z = new Vector3(0, 0, 1);
const _ZERO = new Vector3();

// ─────────────────────────────────────────────────────────────────────────────
// Tuning
// ─────────────────────────────────────────────────────────────────────────────

export const SPEED_TUNING = {
  /**
   * No speed lines at all below this, m/s.
   *
   * These three numbers used to put the onset at 39 km/h and full strength at
   * 97 km/h with a 2.3 exponent, which meant that at 54 km/h — squarely where
   * this course is actually ridden — the normalised term was 0.25 and the
   * exponent crushed it to 0.04. There was no speed cue anywhere in the
   * 30-65 km/h band the game spends almost all of its time in, so a scree
   * straight and a switchback were indistinguishable without reading the
   * speedo. The effect has to live where the speed lives.
   */
  lineFloor: 8.0,
  /** Full-strength reference speed, m/s. */
  lineCeiling: 24.0,
  /** Exponent on the normalised speed. Above 1 holds the effect back at the
   *  bottom of the range without deleting the middle of it. */
  lineExponent: 1.45,
  // Lowering the floor to 8 m/s put lines on screen across the real speed
  // range, but at the old 0.78 ceiling they wash over the riders and flatten
  // the frame. The effect has to sit BEHIND the subject in the read, not on
  // top of it.
  lineMax: 0.52,
  /** Extra intensity contributed by an active boost. */
  boostBonus: 0.55,
  /** Metres ahead of the rider the focus point sits. */
  focusLead: 20.0,
  /** Focus is clamped inside this box so it never leaves the frame. */
  focusBox: { x0: 0.20, x1: 0.80, y0: 0.26, y1: 0.78 },
  /** Seconds of hold after a smear() request before it starts fading. */
  smearHold: 0.22,
  /**
   * Wheel spin (rad/s) at which the spin smear starts and saturates.
   *
   * These were 26 and 88, which is not a speed range this bike ever visits.
   * 88 rad/s on a 0.33 m wheel is 105 km/h; the course is ridden at 30-80, so
   * the effect never rose above a third and the tyre knobs stayed individually
   * countable at 49 km/h — a 26" wheel turning 39 DEGREES between one rendered
   * frame and the next. A knob at the rim travels 0.22 m of arc in that frame,
   * tens of times its own width; there is no exposure, real or drawn, in which
   * it resolves.
   *
   * The honest thresholds come straight from that arithmetic. 14 rad/s is
   * ~17 km/h, where a knob first moves further than its own width per frame
   * and the pattern starts to break up. 45 rad/s is ~53 km/h, by which point
   * it is unambiguously a solid disc.
   */
  spinStart: 14,
  spinFull: 45,
  /** Metres of tail per (m/s) of local vertex speed. */
  smearMetresPerSpeed: 0.020,
  smearMaxLength: 0.55,
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// Motion smear material
// ─────────────────────────────────────────────────────────────────────────────

const SMEAR_VERT = /* glsl */ `
  precision highp float;
  ${GLSL_COMMON}
  ${GLSL_GLOBAL_UNIFORMS}
  ${GLSL_VERTEX_TRANSFORM}

  uniform vec3  uLinear;     // world linear velocity of this mesh, m/s
  uniform vec3  uOmega;      // world angular velocity of the subject, rad/s
  uniform vec3  uPivot;      // world point omega rotates about
  uniform float uMetresPerSpeed;
  uniform float uMaxLength;
  uniform float uAmount;     // 0..1 master

  out float vTrail;
  /** How far this vertex actually moved, as a fraction of uMaxLength. */
  out float vStretch;

  void main() {
    vec3 localPos = position;
    vec3 localNrm = normal;
    resolveLocal(localPos, localNrm);

    vec3 wpos, wnrm;
    toWorld(localPos, localNrm, wpos, wnrm);

    // PER-VERTEX velocity, not per-object. omega x r is what makes a whipping
    // limb streak while the hips barely move; a single object-space direction
    // would smear the whole rider sideways during a spin and read as a bug.
    vec3 vel = uLinear + cross(uOmega, wpos - uPivot);
    float speed = length(vel);
    float len = min(speed * uMetresPerSpeed, uMaxLength) * uAmount;

    vTrail = 0.0;
    vStretch = 0.0;
    if (len > 1e-4 && speed > 1e-4) {
      vec3 tdir = vel / speed;
      // Only the TRAILING half of the surface is extruded. Faces looking into
      // the direction of travel stay exactly where they are, so the duplicate
      // becomes a teardrop with a tail rather than a fattened copy — and the
      // un-extruded half is discarded in the fragment stage so it never
      // z-fights with the mesh it was cloned from.
      float trail = saturate1(-dot(normalize(wnrm), tdir));
      // Squared so the tail concentrates behind the shape instead of smearing
      // the whole silhouette sideways.
      float t = trail * trail;
      float stretch = len * t;
      wpos -= tdir * stretch;
      vTrail = t;
      // The DISTANCE this vertex travelled, not just which way it faces. A
      // vertex that did not move must contribute nothing, or the duplicate mesh
      // sits exactly on top of the source and lays a flat translucent wash over
      // the whole figure — which is what dissolved the rider into a pale ghost.
      vStretch = stretch / max(uMaxLength, 1e-4);
    }

    gl_Position = projectionMatrix * viewMatrix * vec4(wpos, 1.0);
  }
`;

const SMEAR_FRAG = /* glsl */ `
  precision highp float;
  ${GLSL_COMMON}
  ${GLSL_GLOBAL_UNIFORMS}
  ${GLSL_FRAG_OUT}

  uniform vec3  uColorNear;
  uniform vec3  uColorFar;
  uniform float uOpacity;

  in float vTrail;
  in float vStretch;

  void main() {
    // Kill the un-extruded shell entirely.
    float shape = smoothstep(0.10, 0.42, vTrail);

    // Gate on how far the vertex actually MOVED. Facing away from travel is not
    // enough on its own: the whole trailing half of a nearly-stationary mesh
    // passes that test, so the clone was drawn over the body at full strength.
    shape *= smoothstep(0.04, 0.40, vStretch);

    // ...and thin the very tip so the streak tapers instead of ending in a wall.
    float a = uOpacity * shape * (1.0 - vTrail * 0.42);

    // Three hard opacity levels. A smoothly fading ghost is a photographic
    // signal; a streak drawn at two or three flat values is an animated one.
    //
    // ROUNDED, not ceiled. ceil() promotes every surviving fragment to at least
    // 1/3 — so an alpha of 0.02 was drawn at 0.333, and the quantiser that was
    // supposed to make the streak graphic was instead multiplying it by 16x.
    a = floor(clamp(a, 0.0, 1.0) * 3.0 + 0.5) / 3.0;
    if (a <= 0.001) discard;

    vec3 col = mix(uColorNear, uColorFar, vTrail);
    fragColor = vec4(col, a);
  }
`;

function createSmearMaterial(colorNear: Color, colorFar: Color): ShaderMaterial {
  const m = new ShaderMaterial({
    glslVersion: GLSL3,
    uniforms: {
      ...globalUniformBlock(),
      uLinear: { value: new Vector3() },
      uOmega: { value: new Vector3() },
      uPivot: { value: new Vector3() },
      uMetresPerSpeed: { value: SPEED_TUNING.smearMetresPerSpeed },
      uMaxLength: { value: SPEED_TUNING.smearMaxLength },
      uAmount: { value: 0 },
      uColorNear: { value: colorNear.clone() },
      uColorFar: { value: colorFar.clone() },
      uOpacity: { value: 0 },
    },
    vertexShader: SMEAR_VERT,
    fragmentShader: SMEAR_FRAG,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: NormalBlending,
    side: DoubleSide,
  });
  m.name = 'fx:smear';
  return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Wheel spin smear
// ─────────────────────────────────────────────────────────────────────────────

const SPIN_VERT = /* glsl */ `
  precision highp float;
  ${GLSL_COMMON}
  ${GLSL_GLOBAL_UNIFORMS}

  out vec2 vLocal;

  void main() {
    vLocal = position.xy;
    vec3 wpos = (modelMatrix * vec4(position, 1.0)).xyz;
    gl_Position = projectionMatrix * viewMatrix * vec4(wpos, 1.0);
  }
`;

const SPIN_FRAG = /* glsl */ `
  precision highp float;
  ${GLSL_COMMON}
  ${GLSL_GLOBAL_UNIFORMS}
  ${GLSL_FRAG_OUT}

  uniform vec3  uSpokeColor;
  uniform vec3  uTreadColor;
  uniform float uSpin;     // 0..1
  uniform float uPhase;
  uniform float uArcs;
  uniform float uOpacity;

  in vec2 vLocal;

  void main() {
    if (uSpin <= 0.001) discard;

    float r = length(vLocal);
    if (r > 1.0 || r < 0.18) discard;

    float a = atan(vLocal.y, vLocal.x);
    float slot = fract(a / TAU * uArcs + uPhase);

    // The wedge OPENS with spin. At the threshold the streaks are thin spokes;
    // at full speed they have widened until the disc is nearly solid. That
    // widening is the whole read: it is what an animator draws instead of
    // blurring, and it maps directly onto how a real spoked wheel disappears.
    float w = mix(0.09, 0.47, uSpin);
    float m = 1.0 - smoothstep(w, w + 0.035, abs(slot - 0.5));

    // Two radial zones. The tread band goes solid almost immediately (a moving
    // tyre reads as a continuous ring); the spoke field stays streaky.
    float tread = smoothstep(0.86, 0.93, r);
    float spokes = 1.0 - tread;

    vec3 col = mix(uSpokeColor, uTreadColor, tread);
    float alpha = uSpin * (spokes * m * 0.62 + tread * 0.9);
    alpha = ceil(clamp(alpha * uOpacity, 0.0, 1.0) * 3.0) / 3.0;
    if (alpha <= 0.001) discard;

    fragColor = vec4(col, alpha);
  }
`;

let _spinGeometry: BufferGeometry | null = null;
function spinGeometry(): BufferGeometry {
  if (_spinGeometry) return _spinGeometry;
  // Unit annulus in the XY plane, normal +Z. Scaled to the wheel radius by the
  // handle's local matrix.
  const g = new RingGeometry(0.17, 1.0, 40, 1);
  g.name = 'fx:spin-smear';
  _spinGeometry = g;
  return g;
}

/**
 * One wheel's spin smear. The mesh follows an anchor Object3D's world matrix
 * without ever being parented to it, so registering a smear never mutates
 * another subsystem's scene graph.
 */
export class SpinSmear {
  readonly mesh: Mesh;
  private material: ShaderMaterial;
  private local = new Matrix4();
  private anchor: Object3D | null = null;
  private spin01 = 0;
  private phase = 0;
  private rate = 0;

  constructor(radius: number, axis: Vector3) {
    this.material = new ShaderMaterial({
      glslVersion: GLSL3,
      uniforms: {
        ...globalUniformBlock(),
        uSpokeColor: { value: RAMPS.metal.colors[2].clone() },
        uTreadColor: { value: RAMPS.tyre.colors[2].clone() },
        uSpin: { value: 0 },
        uPhase: { value: 0 },
        uArcs: { value: 15 },
        uOpacity: { value: 0.85 },
      },
      vertexShader: SPIN_VERT,
      fragmentShader: SPIN_FRAG,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: NormalBlending,
      side: DoubleSide,
    });
    this.material.name = 'fx:spin-smear';

    this.mesh = new Mesh(spinGeometry(), this.material);
    this.mesh.name = 'fx:spin-smear';
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;
    this.mesh.matrixWorldAutoUpdate = false;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    this.mesh.renderOrder = 17;
    this.mesh.visible = false;
    this.mesh.userData.fxTransparent = true;
    this.mesh.userData.nprSkipPrepass = true;

    this.setRadius(radius, axis);
  }

  /** Rebuild the local offset: rotate the ring's +Z normal onto `axis`, then scale. */
  setRadius(radius: number, axis: Vector3): void {
    _axis.copy(axis);
    if (_axis.lengthSq() < 1e-8) _axis.set(1, 0, 0);
    _axis.normalize();
    _quat.setFromUnitVectors(_UNIT_Z, _axis);
    this.local.makeRotationFromQuaternion(_quat);
    this.local.scale(_scale.set(radius, radius, radius));
  }

  setAnchor(o: Object3D | null): void {
    this.anchor = o;
  }

  /** `spinRate` in rad/s, signed. */
  setSpin(spinRate: number): void {
    this.rate = spinRate;
  }

  update(dt: number): void {
    const mag = Math.abs(this.rate);
    const target = clamp01((mag - SPEED_TUNING.spinStart) / (SPEED_TUNING.spinFull - SPEED_TUNING.spinStart));
    this.spin01 = dampHL(this.spin01, target, 0.05, dt);

    if (this.spin01 <= 0.004 || !this.anchor) {
      this.mesh.visible = false;
      return;
    }
    this.mesh.visible = true;

    // Streaks drift with the wheel, but at a fraction of the true rate — a
    // wheel spinning at 90 rad/s would strobe if the pattern tracked it, and
    // the widening wedge already carries the speed information.
    this.phase += ((this.rate * dt) / (Math.PI * 2)) * 0.18;
    if (this.phase > 1e6 || this.phase < -1e6) this.phase = 0;

    this.material.uniforms.uSpin.value = this.spin01;
    this.material.uniforms.uPhase.value = this.phase;

    this.mesh.matrixWorld.multiplyMatrices(this.anchor.matrixWorld, this.local);
  }

  dispose(): void {
    this.material.dispose();
    this.mesh.removeFromParent();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Smear bookkeeping
// ─────────────────────────────────────────────────────────────────────────────

interface SmearPart {
  mesh: Mesh;
  material: ShaderMaterial;
  /** The mesh this was cloned from — we track ITS world transform, not the root's. */
  src: Object3D;
  lastPos: Vector3;
  vel: Vector3;
  primed: boolean;
}

interface SmearEntry {
  target: Object3D;
  parts: SmearPart[];
  request: number;
  hold: number;
  current: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// SpeedFX
// ─────────────────────────────────────────────────────────────────────────────

export interface SpeedFXOptions {
  /** Cap on simultaneously-smeared targets. Each part costs one extra draw. */
  maxSmearTargets?: number;
  /** Cap on meshes cloned per target when a group is handed to smear(). */
  maxMeshesPerTarget?: number;
}

export class SpeedFX {
  readonly object: Object3D;

  private entries = new Map<Object3D, SmearEntry>();
  private spins: SpinSmear[] = [];
  private maxTargets: number;
  private maxMeshes: number;

  private colorNear = RAMPS.cloth.colors[2].clone();
  private colorFar = SUN_RIM_COLOR.clone();

  // Post-state drivers, all smoothed so nothing steps.
  private intensity = 0;
  private boostPunch = 0;
  private radial = 0;
  private chroma = 0;
  private focus = new Vector2(0.5, 0.52);
  private wasBoosting = false;

  /** Subject rotation, shared by every smear entry. See setSubjectSpin(). */
  private omega = new Vector3();
  private pivot = new Vector3();

  /** Objects auto-smeared from the subject's own speed, set by the facade. */
  private autoTargets: Object3D[] = [];

  constructor(opts: SpeedFXOptions = {}) {
    this.maxTargets = opts.maxSmearTargets ?? 6;
    this.maxMeshes = opts.maxMeshesPerTarget ?? 6;
    this.object = new Object3D();
    this.object.name = 'fx:speed';
    // Smear meshes carry world matrices copied straight off their sources, so
    // this container must never contribute a transform of its own.
    this.object.matrixAutoUpdate = false;
  }

  // ── Colours ───────────────────────────────────────────────────────────────

  /** Match the streak to a rider's jersey. Both colours must come from the palette. */
  setSmearColors(near: Color, far: Color): void {
    this.colorNear.copy(near);
    this.colorFar.copy(far);
    for (const e of this.entries.values()) {
      for (const p of e.parts) {
        (p.material.uniforms.uColorNear.value as Color).copy(near);
        (p.material.uniforms.uColorFar.value as Color).copy(far);
      }
    }
  }

  // ── Smear ─────────────────────────────────────────────────────────────────

  /**
   * Request motion smear on a target for a short window (SPEED_TUNING.smearHold
   * seconds), after which it fades. Call every frame to hold it on.
   *
   * The target is never re-parented and never modified: we reference its
   * geometry (no buffer duplication) from our own mesh and copy its world
   * matrix each frame. For a SkinnedMesh we bind the SAME skeleton, so the
   * smear deforms in lockstep with the rider without the rig knowing we exist.
   */
  smear(target: Object3D, amount: number): void {
    const a = clamp01(amount);
    if (a <= 0.005) return;

    let e = this.entries.get(target);
    if (!e) {
      if (this.entries.size >= this.maxTargets) return;
      const built = this.build(target);
      if (!built) return;
      e = built;
      this.entries.set(target, e);
    }
    if (a > e.request) e.request = a;
    e.hold = SPEED_TUNING.smearHold;
  }

  /**
   * The subject's angular velocity and the world point it rotates about.
   *
   * This is deliberately global rather than per-entry: in practice everything
   * being smeared belongs to the same rider, and the alternative is asking
   * every caller to supply a pivot they do not have. If a second racer is ever
   * smeared simultaneously its rotational component will be wrong; the linear
   * component still is not.
   */
  setSubjectSpin(omega: Vector3, pivot: Vector3): void {
    this.omega.copy(omega);
    this.pivot.copy(pivot);
  }

  private build(target: Object3D): SmearEntry | null {
    const parts: SmearPart[] = [];

    const consider = (o: Object3D): void => {
      if (parts.length >= this.maxMeshes) return;
      const m = o as Mesh;
      if (!m.isMesh || !m.geometry) return;
      // Never smear a hull — it would double every outline into the tail —
      // and never smear another FX surface.
      if (o.userData?.isHull || o.userData?.fxTransparent) return;
      const material = createSmearMaterial(this.colorNear, this.colorFar);
      const mesh = this.cloneFor(m, material);
      m.getWorldPosition(_wp);
      parts.push({
        mesh,
        material,
        src: m,
        lastPos: _wp.clone(),
        vel: new Vector3(),
        primed: false,
      });
    };

    consider(target);
    if (parts.length < this.maxMeshes) {
      target.traverse((o) => {
        if (o === target) return;
        consider(o);
      });
    }

    if (parts.length === 0) return null;
    for (const p of parts) this.object.add(p.mesh);

    return { target, parts, request: 0, hold: 0, current: 0 };
  }

  private cloneFor(src: Mesh, material: ShaderMaterial): Mesh {
    let out: Mesh;
    if ((src as SkinnedMesh).isSkinnedMesh) {
      const s = src as SkinnedMesh;
      const sm = new SkinnedMesh(s.geometry, material);
      sm.bind(s.skeleton, s.bindMatrix);
      out = sm;
    } else if ((src as InstancedMesh).isInstancedMesh) {
      const s = src as InstancedMesh;
      const im = new InstancedMesh(s.geometry, material, s.count);
      im.instanceMatrix = s.instanceMatrix;
      im.count = s.count;
      out = im;
    } else {
      out = new Mesh(src.geometry, material);
    }
    out.name = `${src.name || 'mesh'}:smear`;
    out.frustumCulled = false;
    out.castShadow = false;
    out.receiveShadow = false;
    out.renderOrder = 16;
    out.matrixAutoUpdate = false;
    out.matrixWorldAutoUpdate = false;
    out.visible = false;
    out.userData.fxTransparent = true;
    out.userData.nprSkipPrepass = true;
    return out;
  }

  /** Objects that get smear automatically from the subject's own speed. */
  setAutoSmearTargets(objects: Object3D[]): void {
    this.autoTargets = objects;
  }

  // ── Wheel spin ────────────────────────────────────────────────────────────

  /**
   * Register a wheel. `axis` is the spin axis in the ANCHOR's local space
   * (usually its local X for a bike wheel). Returns the handle so the bike or
   * rider system can push a spin rate into it every frame.
   */
  addSpinSmear(anchor: Object3D, radius: number, axis: Vector3): SpinSmear {
    const s = new SpinSmear(radius, axis);
    s.setAnchor(anchor);
    this.object.add(s.mesh);
    this.spins.push(s);
    return s;
  }

  // ── Frame ─────────────────────────────────────────────────────────────────

  /**
   * `dt` is the SCALED delta (smear should freeze with the world).
   * `state` may be null outside a race — the post dials then decay to zero
   * rather than sticking on whatever the last frame left them at.
   */
  update(dt: number, camera: PerspectiveCamera, state: BikeState | null): void {
    if (state) this.setSubjectSpin(state.angularVelocity, state.position);
    this.updateSmears(dt, state);
    for (const s of this.spins) s.update(dt);
    this.updatePost(dt, camera, state);
  }

  private updateSmears(dt: number, state: BikeState | null): void {
    // Automatic smear: the rider and bike streak once genuinely fast, and any
    // time the body is rotating hard through a trick — which is where the
    // effect earns its keep, because a 360 at 4 rad/s moves a limb faster
    // across the screen than the whole bike moves down the hill.
    if (state) {
      const spin = this.omega.length();
      // Onset at 15 m/s (54 km/h) rather than 19 (68 km/h). The old floor sat
      // above almost every speed the course is actually ridden at, so the
      // geometry smear — the one that streaks a limb during a trick — was
      // effectively dead code outside a full-tuck sprint.
      const fromSpeed = clamp01((state.speed - 15) / 9) * 0.68;
      const fromSpin = clamp01((spin - 4.2) / 6.5) * 0.9;
      const amount = Math.max(fromSpeed, fromSpin);
      if (amount > 0.02) {
        for (const o of this.autoTargets) this.smear(o, amount);
      }
    }

    if (this.entries.size === 0) return;
    const invDt = dt > 1e-5 ? 1 / dt : 0;
    const velBlend = 1 - Math.pow(2, -Math.max(dt, 0) / 0.045);

    for (const e of this.entries.values()) {
      if (e.hold > 0) {
        e.hold -= dt;
        e.current = dampHL(e.current, e.request, 0.035, dt);
      } else {
        e.request = 0;
        e.current = dampHL(e.current, 0, 0.07, dt);
      }
      const active = e.current > 0.02;

      for (const p of e.parts) {
        p.src.getWorldPosition(_wp);
        if (!p.primed) {
          p.lastPos.copy(_wp);
          p.primed = true;
        }
        _inst.subVectors(_wp, p.lastPos).multiplyScalar(invDt);
        // Heavy smoothing: raw per-frame position deltas are noisy enough to
        // make the tail flicker direction, which reads as a glitch, not motion.
        p.vel.lerp(_inst, velBlend);
        p.lastPos.copy(_wp);

        p.mesh.visible = active;
        if (!active) continue;

        p.mesh.matrixWorld.copy(p.src.matrixWorld);
        (p.material.uniforms.uLinear.value as Vector3).copy(p.vel);
        (p.material.uniforms.uOmega.value as Vector3).copy(this.omega);
        (p.material.uniforms.uPivot.value as Vector3).copy(this.pivot);
        p.material.uniforms.uAmount.value = e.current;
        p.material.uniforms.uOpacity.value = 0.55 * e.current;
      }
    }
  }

  private updatePost(dt: number, camera: PerspectiveCamera, state: BikeState | null): void {
    let targetIntensity = 0;
    let targetRadial = 0;

    if (state) {
      const v01 = clamp01(
        (state.speed - SPEED_TUNING.lineFloor) / (SPEED_TUNING.lineCeiling - SPEED_TUNING.lineFloor),
      );
      // Non-linear on purpose. A linear ramp puts half the effect on at half
      // speed, and the player never registers the difference between "quite
      // fast" and "flat out" — which is the only difference that matters.
      const shaped = Math.pow(v01, SPEED_TUNING.lineExponent);
      targetIntensity = shaped * SPEED_TUNING.lineMax;

      // Boost is not a scaling of the same curve — it is a separate, much
      // sharper event. Attack in ~45ms, release over ~280ms, with an instant
      // step on the rising edge so the punch lands on the frame it fires.
      const boosting = state.boosting;
      this.boostPunch = dampHL(this.boostPunch, boosting ? 1 : 0, boosting ? 0.045 : 0.28, dt);
      if (boosting && !this.wasBoosting) this.boostPunch = Math.max(this.boostPunch, 0.6);
      this.wasBoosting = boosting;

      targetIntensity += this.boostPunch * SPEED_TUNING.boostBonus;

      // Air is quiet. Losing the lines in flight is what makes them come back
      // as a punch on landing.
      if (state.mode === BikeMode.Airborne) targetIntensity *= 0.68;
      if (state.mode === BikeMode.Crashing) targetIntensity *= 0.12;

      targetRadial = Math.pow(v01, 3.0) * 0.30 + this.boostPunch * 0.42;

      this.updateFocus(dt, camera, state);
    } else {
      this.boostPunch = dampHL(this.boostPunch, 0, 0.2, dt);
      this.wasBoosting = false;
      this.focus.x = dampHL(this.focus.x, 0.5, 0.25, dt);
      this.focus.y = dampHL(this.focus.y, 0.52, 0.25, dt);
    }

    targetIntensity = clamp01(targetIntensity);
    this.intensity = dampHL(this.intensity, targetIntensity, 0.10, dt);
    this.radial = dampHL(this.radial, clamp01(targetRadial), 0.11, dt);
    // Chromatic fringing rides the same curve but far weaker — it is a
    // seasoning, and at any visible strength it starts reading as a lens.
    this.chroma = dampHL(this.chroma, this.intensity * 0.18 + this.boostPunch * 0.22, 0.12, dt);

    POST_STATE.speedLineIntensity = this.intensity;
    POST_STATE.speedLineFocus.set(this.focus.x, this.focus.y);
    POST_STATE.radialBlur = this.radial;
    POST_STATE.chromaticAberration = this.chroma;
  }

  private updateFocus(dt: number, camera: PerspectiveCamera, state: BikeState): void {
    _dir.copy(state.velocity);
    _dir.y *= 0.35; // flatten the vertical so a big air doesn't fling the focus
    if (_dir.lengthSq() < 1.0) {
      _dir.set(0, 0, 1).applyQuaternion(state.orientation);
    }
    if (_dir.lengthSq() < 1e-6) _dir.set(0, 0, 1);
    _dir.normalize();

    _focusPoint.copy(state.position).addScaledVector(_dir, SPEED_TUNING.focusLead);
    _focusPoint.y += 1.0;

    camera.getWorldDirection(_camDir);
    _toPoint.copy(_focusPoint).sub(camera.position);

    let fx = 0.5;
    let fy = 0.52;
    // Only project a point genuinely in front of the camera; behind it the
    // perspective divide mirrors and the focus jumps to the wrong side.
    if (_toPoint.dot(_camDir) > 1.0) {
      _focusPoint.project(camera);
      fx = _focusPoint.x * 0.5 + 0.5;
      fy = _focusPoint.y * 0.5 + 0.5;
    }
    const box = SPEED_TUNING.focusBox;
    fx = clamp(fx, box.x0, box.x1);
    fy = clamp(fy, box.y0, box.y1);

    this.focus.x = dampHL(this.focus.x, fx, 0.14, dt);
    this.focus.y = dampHL(this.focus.y, fy, 0.14, dt);
  }

  /** Drop a smear target and free its meshes. Safe to call for unknown objects. */
  release(target: Object3D): void {
    const e = this.entries.get(target);
    if (!e) return;
    for (const p of e.parts) {
      p.mesh.removeFromParent();
      p.material.dispose();
    }
    this.entries.delete(target);
  }

  reset(): void {
    for (const t of [...this.entries.keys()]) this.release(t);
    this.intensity = 0;
    this.boostPunch = 0;
    this.radial = 0;
    this.chroma = 0;
    this.omega.copy(_ZERO);
    this.focus.set(0.5, 0.52);
    POST_STATE.speedLineIntensity = 0;
    POST_STATE.radialBlur = 0;
    POST_STATE.chromaticAberration = 0;
  }

  dispose(): void {
    this.reset();
    for (const s of this.spins) s.dispose();
    this.spins.length = 0;
    this.object.removeFromParent();
  }
}
