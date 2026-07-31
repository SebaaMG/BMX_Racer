/**
 * Wheel — one raycast wheel: suspension, tyre, and everything the visual and
 * audio systems read off a contact patch.
 *
 * Two decisions in here are worth defending, because the obvious implementation
 * is wrong in both cases:
 *
 *  1. LATERAL force comes from a slip-angle curve with a real peak and a real
 *     fall-off past it, filtered through a relaxation length. LONGITUDINAL force
 *     comes from a direct force demand (pedal / brake) clamped by the friction
 *     circle. That asymmetry is deliberate. The lateral loop acts on a 92 kg
 *     body and is numerically soft — a slip curve there is stable and it is
 *     where all the feel lives. The longitudinal loop, if driven the same way,
 *     acts on a 0.13 kg·m² wheel: at 120 Hz a 900 N tyre force swings the slip
 *     ratio by more than 1.0 in a single step and the whole thing rings itself
 *     apart. Every arcade vehicle that "feels twitchy under braking" has that
 *     bug. So longitudinal is solved as a constraint, and the wheel's visible
 *     spin is driven FROM the solved state rather than feeding back into it.
 *
 *  2. Suspension force is applied along the CONTACT NORMAL, not along the body's
 *     down axis. On a 30° traverse those differ by 30° and the difference is
 *     exactly the force that has to be held by lateral grip — which is why a
 *     steep off-camber traverse is hard here and would be free if the force went
 *     straight down the fork.
 *
 * Everything in this file is allocation-free after construction. All scratch
 * vectors are module scope.
 */

import { Quaternion, Vector3 } from 'three';
import {
  SurfaceKind,
  type SurfaceProperties,
  type TerrainSample,
  type WheelState,
} from '../game/Contracts';
import { SURFACES } from '../game/WorldConstants';
import { clamp, clamp01 } from '../core/MathX';

// ─────────────────────────────────────────────────────────────────────────────
// Surfaces
// ─────────────────────────────────────────────────────────────────────────────

/**
 * SURFACES in WorldConstants carries the numbers but not `kind` or `audioTone`,
 * which SurfaceProperties requires. We complete the table here rather than
 * editing a shared file. The terrain system builds the same table from the same
 * source numbers, so the two agree by construction.
 */
export const SURFACE_TABLE: Record<SurfaceKind, SurfaceProperties> = {
  [SurfaceKind.Rock]: { kind: SurfaceKind.Rock, ...SURFACES.rock, audioTone: 'rock' },
  [SurfaceKind.Dirt]: { kind: SurfaceKind.Dirt, ...SURFACES.dirt, audioTone: 'hardpack' },
  [SurfaceKind.Grass]: { kind: SurfaceKind.Grass, ...SURFACES.grass, audioTone: 'grass' },
  [SurfaceKind.Scree]: { kind: SurfaceKind.Scree, ...SURFACES.scree, audioTone: 'gravel' },
  [SurfaceKind.Snow]: { kind: SurfaceKind.Snow, ...SURFACES.snow, audioTone: 'snow' },
  [SurfaceKind.Water]: { kind: SurfaceKind.Water, ...SURFACES.water, audioTone: 'water' },
  [SurfaceKind.Trail]: { kind: SurfaceKind.Trail, ...SURFACES.trail, audioTone: 'hardpack' },
};

export const DEFAULT_SURFACE = SURFACE_TABLE[SurfaceKind.Dirt];

/** The subset of ITerrain the bike actually needs. ITerrain satisfies this. */
export interface BikeTerrain {
  heightAt(x: number, z: number): number;
  sampleAt(x: number, z: number, out?: TerrainSample): TerrainSample;
  normalAt(x: number, z: number, out?: Vector3): Vector3;
}

/** Allocate a reusable TerrainSample for `out` parameters. */
export function makeTerrainSample(): TerrainSample {
  return {
    height: 0,
    normal: new Vector3(0, 1, 0),
    slope: 0,
    kind: SurfaceKind.Dirt,
    surface: DEFAULT_SURFACE,
  };
}

/**
 * A stand-in mountain for developing the bike before the real terrain lands.
 * Deterministic (pure trigonometry, no RNG), gently descending toward +Z, with
 * enough rolling shape and one clean lip to test pumping against.
 *
 * This is development scaffolding, not world content — it is never used when a
 * real ITerrain is supplied.
 */
export function createFallbackTerrain(): BikeTerrain {
  const _n = new Vector3();
  const h = (x: number, z: number): number => {
    // Base descent: ~11° down the +Z axis.
    let y = 60 - z * 0.195;
    // Rolling shape at two scales.
    y += Math.sin(x * 0.031) * 2.6 + Math.cos(z * 0.024 + 1.3) * 3.1;
    y += Math.sin(x * 0.11 + z * 0.07) * 0.62;
    // A tabletop lip at z = 40..56 so preload/pump has something to work with.
    const lip = Math.exp(-Math.pow((z - 48) / 9, 2)) * Math.exp(-Math.pow(x / 14, 2));
    y += lip * 3.4;
    return y;
  };
  return {
    heightAt: h,
    normalAt(x, z, out) {
      const o = out ?? _n;
      const e = 0.5;
      const dx = (h(x + e, z) - h(x - e, z)) / (2 * e);
      const dz = (h(x, z + e) - h(x, z - e)) / (2 * e);
      return o.set(-dx, 1, -dz).normalize();
    },
    sampleAt(x, z, out) {
      const o = out ?? makeTerrainSample();
      o.height = h(x, z);
      this.normalAt(x, z, o.normal);
      o.slope = Math.acos(clamp(o.normal.y, -1, 1));
      o.kind = SurfaceKind.Dirt;
      o.surface = SURFACE_TABLE[SurfaceKind.Dirt];
      return o;
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Tyre curve
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Normalised tyre force as a function of slip, peaking at 1.0 when |slip| equals
 * `peak` and decaying to `tail` well past it.
 *
 * The rise is a quarter sine rather than a line so the derivative goes to zero
 * exactly at the peak: a tyre parked right on the limit then produces a force
 * that does not chatter as the slip jitters by a fraction of a percent, which is
 * the difference between "on the edge of grip" and "buzzing".
 *
 * The fall-off uses a super-Gaussian so it is also C1 at the peak — no kink, and
 * therefore no audible/visible discontinuity as a slide develops. `tail` is what
 * makes the slide RECOVERABLE: a sliding tyre still generates 60–75% of peak, so
 * catching it is a matter of unwinding the slip angle, not of praying.
 */
export function slipCurve(slip: number, peak: number, tail: number): number {
  const a = Math.abs(slip) / Math.max(peak, 1e-4);
  let f: number;
  if (a <= 1) {
    f = Math.sin(a * Math.PI * 0.5);
  } else {
    f = tail + (1 - tail) * Math.exp(-Math.pow((a - 1) * 1.35, 1.6));
  }
  return slip < 0 ? -f : f;
}

export interface TyreConfig {
  /** Slip angle (rad) at peak lateral force, on a nominal (grip = 1) surface. */
  peakSlipAngle: number;
  /** Force retained deep into a lateral slide, as a fraction of peak. */
  tailLat: number;
  /** How much lean alone generates cornering force. This is what makes a lean turn. */
  camber: number;
  /** Grip anisotropy — bike tyres brake slightly better than they corner. */
  muLongScale: number;
  muLatScale: number;
  /** Lateral relaxation length, metres. Tyre carcass compliance. */
  relaxLat: number;
  /** Peak μ falls this fraction per unit of load above nominal. */
  loadSensitivity: number;
  nominalLoad: number;
  /** Sliding (kinetic) friction as a fraction of peak, for a locked wheel. */
  slidingRatio: number;
  /** Minimum lateral grip retained when longitudinal force saturates the circle. */
  ellipseFloor: number;
}

export const FRONT_TYRE: TyreConfig = {
  peakSlipAngle: 0.155,
  tailLat: 0.68,
  camber: 0.62,
  muLongScale: 1.04,
  muLatScale: 1.03,
  relaxLat: 0.34,
  loadSensitivity: 0.10,
  nominalLoad: 850,
  slidingRatio: 0.76,
  ellipseFloor: 0.14,
};

export const REAR_TYRE: TyreConfig = {
  peakSlipAngle: 0.175,
  tailLat: 0.72,
  camber: 0.58,
  muLongScale: 1.08,
  muLatScale: 0.96,
  relaxLat: 0.40,
  loadSensitivity: 0.11,
  nominalLoad: 1050,
  slidingRatio: 0.74,
  ellipseFloor: 0.12,
};

export interface WheelConfig {
  /** Suspension mount point in body space, at full extension the wheel hangs `travel` below. */
  mountLocal: Vector3;
  travel: number;
  stiffness: number;
  damping: number;
  radius: number;
  isFront: boolean;
  tyre: TyreConfig;
  /** Compression damping multiplier — a fork that takes the hit softly. */
  compressionDampScale: number;
  /** Rebound damping multiplier — and returns under control. */
  reboundDampScale: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scratch — module scope, never allocated in an update path
// ─────────────────────────────────────────────────────────────────────────────
const _mount = new Vector3();
const _down = new Vector3();
const _up = new Vector3();
const _planePt = new Vector3();
const _probe = new Vector3();
const _n = new Vector3();
const _r = new Vector3();
const _vel = new Vector3();
const _fwd = new Vector3();
const _left = new Vector3();
const _tmp = new Vector3();
const _tmp2 = new Vector3();
const _q = new Quaternion();

/** Surfaces steeper than this are walls, not ground — the suspension ignores them. */
const MAX_SUSPENSION_TILT_COS = Math.cos(1.08); // 62°

export interface SuspendResult {
  /** Normal force magnitude, newtons. Zero when airborne. */
  force: number;
  /** How far the wheel is buried below the surface, metres. Zero normally. */
  penetration: number;
  /** Distance from the tyre's lowest point to the ground when airborne, metres. */
  clearance: number;
}

/**
 * One raycast wheel. Owns its own WheelState fields directly so the bike can
 * hand `bike.state.front` straight out without a copy.
 */
export class Wheel implements WheelState {
  readonly config: WheelConfig;

  // ── WheelState ────────────────────────────────────────────────────────────
  readonly contactPoint = new Vector3();
  readonly contactNormal = new Vector3(0, 1, 0);
  grounded = false;
  compression = 0;
  compressionVelocity = 0;
  spin = 0;
  spinRate = 0;
  slipRatio = 0;
  lateralSlip = 0;
  surface: SurfaceProperties = DEFAULT_SURFACE;
  load = 0;

  // ── Internals ─────────────────────────────────────────────────────────────
  /** Unwrapped spin, for interpolation across the 2π seam. */
  spinUnwrapped = 0;
  /** Suspension extension in metres: `travel` = topped out, 0 = bottomed. */
  suspensionLength: number;
  /** World position of the wheel centre. */
  readonly centre = new Vector3();
  /** Rolling direction in the contact plane. */
  readonly forward = new Vector3(0, 0, 1);
  /** Left-hand lateral axis in the contact plane. */
  readonly left = new Vector3(1, 0, 0);
  /** Longitudinal / lateral contact velocity, m/s. */
  vLong = 0;
  vLat = 0;
  /** Filtered slip angle, radians. */
  slipAngle = 0;
  /** True while the brake demand exceeds available longitudinal grip. */
  locked = false;
  /** True while drive torque exceeds available longitudinal grip. */
  spinningUp = false;
  /** Applied tyre forces this step, world space. */
  readonly tyreForce = new Vector3();
  /** Terrain sample scratch owned by this wheel — reused every step. */
  private sample: TerrainSample = makeTerrainSample();
  /** Compression on the previous step, for the visual rate when the ray misses. */
  private prevCompression = 0;
  /** Bottom-out flag, latched for one step so the FX/audio can read it. */
  bottomedThisStep = false;
  /** Steer angle applied to this wheel, radians (positive = right). */
  steer = 0;

  constructor(config: WheelConfig) {
    this.config = config;
    this.suspensionLength = config.travel;
  }

  reset(): void {
    this.suspensionLength = this.config.travel;
    this.compression = 0;
    this.prevCompression = 0;
    this.compressionVelocity = 0;
    this.grounded = false;
    this.spin = 0;
    this.spinUnwrapped = 0;
    this.spinRate = 0;
    this.slipRatio = 0;
    this.lateralSlip = 0;
    this.slipAngle = 0;
    this.locked = false;
    this.spinningUp = false;
    this.load = 0;
    this.vLong = 0;
    this.vLat = 0;
    this.tyreForce.set(0, 0, 0);
    this.contactNormal.set(0, 1, 0);
    this.surface = DEFAULT_SURFACE;
  }

  /** World position of this wheel's suspension mount. */
  mountWorld(origin: Vector3, quat: Quaternion, out: Vector3): Vector3 {
    return out.copy(this.config.mountLocal).applyQuaternion(quat).add(origin);
  }

  /**
   * Cast the suspension ray and integrate the spring/damper.
   *
   * `origin` is the body origin (mid-wheelbase, at the topped-out axle line),
   * `com` the centre of mass, `bodyVel`/`bodyAngVel` the rigid body's motion.
   * Returns the normal force to apply at `contactPoint` along `contactNormal`.
   */
  suspend(
    origin: Vector3,
    quat: Quaternion,
    com: Vector3,
    bodyVel: Vector3,
    bodyAngVel: Vector3,
    terrain: BikeTerrain,
    dt: number,
    out: SuspendResult,
  ): SuspendResult {
    const cfg = this.config;
    out.force = 0;
    out.penetration = 0;
    out.clearance = 0;
    this.bottomedThisStep = false;
    this.prevCompression = this.compression;

    _up.set(0, 1, 0).applyQuaternion(quat);
    _down.copy(_up).negate();
    this.mountWorld(origin, quat, _mount);

    // Two-pass ground query. The first pass samples straight below the mount,
    // which is wrong by up to `t * sin(slope)` on a steep face; the second pass
    // samples under the resulting contact point and converges.
    const maxLen = cfg.travel + cfg.radius;
    let hit = false;
    let t = maxLen;
    _probe.copy(_mount);

    for (let pass = 0; pass < 2; pass++) {
      const s = terrain.sampleAt(_probe.x, _probe.z, this.sample);
      _n.copy(s.normal);
      if (_n.lengthSq() < 1e-8) _n.set(0, 1, 0);
      else _n.normalize();

      // Treat anything steeper than 62° as a wall. Suspension against a wall
      // launches the bike sideways; walls are handled as impacts instead.
      if (_n.y < MAX_SUSPENSION_TILT_COS) {
        _n.y = MAX_SUSPENSION_TILT_COS;
        _n.normalize();
      }

      _planePt.set(_probe.x, s.height, _probe.z);
      const denom = _down.dot(_n);
      if (denom > -0.2) {
        hit = false;
        break;
      }
      _tmp.copy(_planePt).sub(_mount);
      t = _tmp.dot(_n) / denom;
      hit = true;
      if (pass === 0) {
        // Advance the probe to the tentative contact and resample.
        _probe.copy(_down).multiplyScalar(clamp(t, 0, maxLen * 1.6)).add(_mount);
      }
      this.surface = s.surface ?? SURFACE_TABLE[s.kind] ?? DEFAULT_SURFACE;
    }

    if (!hit || t > maxLen) {
      // Airborne. The spring pushes the wheel to full extension against rebound
      // damping — that visible droop is what sells a jump.
      this.grounded = false;
      const target = cfg.travel;
      const rate = 2.9; // m/s of topping-out extension
      this.suspensionLength = Math.min(target, this.suspensionLength + rate * dt);
      this.compression = 1 - this.suspensionLength / cfg.travel;
      this.compressionVelocity = (this.compression - this.prevCompression) / dt;
      this.load = 0;
      out.clearance = hit ? t - maxLen : 4;
      this.centre.copy(_down).multiplyScalar(this.suspensionLength).add(_mount);
      this.contactPoint.copy(this.centre).addScaledVector(_down, cfg.radius);
      this.contactNormal.copy(_n);
      return out;
    }

    // ── Contact ──────────────────────────────────────────────────────────────
    this.grounded = true;
    this.contactNormal.copy(_n);

    let len = t - cfg.radius;
    if (len < 0) {
      out.penetration = -len;
      len = 0;
    }
    len = Math.min(len, cfg.travel);
    this.suspensionLength = len;
    this.compression = 1 - len / cfg.travel;

    this.centre.copy(_down).multiplyScalar(len).add(_mount);
    this.contactPoint.copy(_down).multiplyScalar(t).add(_mount);

    // Velocity of the mount, including the body's rotation about the COM.
    _r.copy(_mount).sub(com);
    _vel.copy(bodyAngVel).cross(_r).add(bodyVel);

    // d(suspensionLength)/dt from the mount's motion. See the derivation in the
    // header: t = ((P - m)·n) / (d·n), so ṫ = -(u·n)/(d·n).
    const denom = _down.dot(_n);
    const lenRate = -_vel.dot(_n) / denom;
    const compressRate = -lenRate; // m/s, positive = compressing

    this.compressionVelocity = compressRate / cfg.travel;

    // ── Spring + damper ──────────────────────────────────────────────────────
    const disp = cfg.travel - len; // metres of compression
    const harsh = this.surface.harshness;
    // Rock is harsh: less damping means the hit comes through. Loam is forgiving.
    const dampScale = 0.72 + harsh * 0.34;
    const dampCoeff =
      cfg.damping *
      dampScale *
      (compressRate > 0 ? cfg.compressionDampScale : cfg.reboundDampScale);

    let force = cfg.stiffness * disp + dampCoeff * compressRate;

    // Bottom-out: a progressive bumper over the last 12% of travel, plus a hard
    // extra damper. Without this a big landing punches straight through the
    // travel and the fork reads as a rigid stick at exactly the moment the
    // player is looking at it.
    const bumpStart = cfg.travel * 0.88;
    if (disp > bumpStart) {
      const over = disp - bumpStart;
      force += cfg.stiffness * 9.0 * over * over / Math.max(cfg.travel - bumpStart, 1e-4);
      if (compressRate > 0.35) {
        force += dampCoeff * 2.4 * compressRate;
        this.bottomedThisStep = true;
      }
    }

    // Never pull the bike down; never explode.
    force = clamp(force, 0, cfg.stiffness * cfg.travel * 3.4);
    this.load = force;
    out.force = force;
    return out;
  }

  /**
   * Solve the tyre forces at this contact.
   *
   * `driveForce` and `brakeForce` are demands in newtons at the contact patch;
   * `leanAngle` is the bike's roll relative to the contact normal, used for
   * camber thrust. Writes `tyreForce` (world) and all the slip diagnostics.
   */
  solveTyre(
    quat: Quaternion,
    com: Vector3,
    bodyVel: Vector3,
    bodyAngVel: Vector3,
    driveForce: number,
    brakeForce: number,
    leanAngle: number,
    mass: number,
    dt: number,
  ): Vector3 {
    this.tyreForce.set(0, 0, 0);
    const cfg = this.config;
    const tyre = cfg.tyre;

    if (!this.grounded || this.load <= 1) {
      // Freewheeling in the air: the spin coasts down very slowly.
      this.spinRate *= Math.pow(0.985, dt * 120);
      this.advanceSpin(dt);
      this.slipRatio = 0;
      this.lateralSlip = 0;
      this.locked = false;
      this.spinningUp = false;
      this.slipAngle *= Math.pow(0.5, dt / 0.2);
      return this.tyreForce;
    }

    // ── Contact frame ────────────────────────────────────────────────────────
    // The wheel's heading is the body forward yawed by the steer angle. Positive
    // steer is to the RIGHT, and a positive yaw about body +Y goes LEFT, so the
    // steer rotation is negated.
    _fwd.set(-Math.sin(this.steer), 0, Math.cos(this.steer)).applyQuaternion(quat);
    const n = this.contactNormal;
    _fwd.addScaledVector(n, -_fwd.dot(n));
    if (_fwd.lengthSq() < 1e-8) _fwd.set(0, 0, 1).addScaledVector(n, -n.z);
    _fwd.normalize();
    // With +Y up and +Z forward this right-handed basis has +X to the LEFT,
    // and n × forward reproduces exactly that.
    _left.copy(n).cross(_fwd).normalize();
    this.forward.copy(_fwd);
    this.left.copy(_left);

    _r.copy(this.contactPoint).sub(com);
    _vel.copy(bodyAngVel).cross(_r).add(bodyVel);
    const vLong = _vel.dot(_fwd);
    const vLat = _vel.dot(_left);
    this.vLong = vLong;
    this.vLat = vLat;
    this.lateralSlip = vLat;

    // ── Available grip ───────────────────────────────────────────────────────
    // Load sensitivity: a tyre crushed by weight transfer loses peak μ. Small
    // effect, but it is what makes trail-braking into a corner feel like a
    // trade rather than a free lunch.
    const loadRatio = this.load / tyre.nominalLoad;
    const loadFactor = 1 / (1 + tyre.loadSensitivity * Math.max(0, loadRatio - 1));
    const grip = this.surface.grip * loadFactor;
    const muLong = grip * tyre.muLongScale;
    const muLat = grip * tyre.muLatScale;
    const fxCap = muLong * this.load;
    const fyCap = muLat * this.load;

    // ── Longitudinal: solved as a demand, clamped by the circle ──────────────
    const vRef = Math.max(Math.abs(vLong), 0.001);
    let fx = driveForce;
    this.spinningUp = driveForce > fxCap * 1.02;

    this.locked = false;
    if (brakeForce > 1) {
      // The brake can never reverse the wheel's direction of travel: cap the
      // decelerating force at what would bring this contact to a standstill.
      const stopForce = (Math.abs(vLong) * mass) / Math.max(dt, 1e-4) * 0.5;
      const applied = Math.min(brakeForce, stopForce + fxCap);
      const dir = vLong >= 0 ? -1 : 1;
      fx += dir * applied;
      // Lock-up: the demand exceeds what the contact can transmit. This is the
      // whole rear-brake-slide mechanic — a locked wheel eats the entire
      // friction budget longitudinally, which leaves nothing for cornering and
      // steps the back end out on its own.
      this.locked = applied > fxCap * 0.97 && Math.abs(vLong) > 0.45;
    }

    if (this.locked) {
      // Sliding friction is below the static peak, which is why a locked wheel
      // both stops you less well AND is unstable.
      fx = (vLong >= 0 ? -1 : 1) * fxCap * tyre.slidingRatio;
    } else {
      fx = clamp(fx, -fxCap, fxCap);
    }

    // ── Lateral: slip-angle curve with relaxation ────────────────────────────
    const alphaTarget = Math.atan2(vLat, Math.max(Math.abs(vLong), 1.4));
    // Relaxation length: the carcass has to deform before the force appears.
    // Rate is bounded below so the tyre still responds at a standstill.
    const relaxRate = clamp01(
      (Math.abs(vLong) * dt) / Math.max(tyre.relaxLat, 1e-3) + dt * 5.5,
    );
    this.slipAngle += (alphaTarget - this.slipAngle) * relaxRate;

    // Loose surfaces reach peak later and hold more force past it: a scree slide
    // is broad and mushy, a hardpack slide is sharp.
    const looseness = 1 - clamp01(this.surface.grip);
    const peakAlpha = tyre.peakSlipAngle * (1 + looseness * 1.45);
    const tail = clamp01(tyre.tailLat + looseness * 0.16);

    // Camber thrust. A leaned wheel generates cornering force with zero slip
    // angle — this is the term that makes leaning actually turn the bike rather
    // than just tilting the model.
    const camberTerm = tyre.camber * Math.sin(clamp(leanAngle, -1.4, 1.4));
    const latDemand = clamp(slipCurve(this.slipAngle, peakAlpha, tail) + camberTerm, -1.35, 1.35);
    let fy = -fyCap * latDemand;

    // ── Friction circle ──────────────────────────────────────────────────────
    // Longitudinal usage eats the lateral budget. The floor keeps a fully locked
    // wheel from behaving like frictionless ice, which is unrecoverable and no
    // fun; 12–14% is enough to catch a slide with a deliberate steer input.
    const ux = clamp(fx / Math.max(fxCap, 1e-3), -1, 1);
    const budget = Math.max(Math.sqrt(Math.max(0, 1 - ux * ux)), tyre.ellipseFloor);
    const fyLimit = fyCap * budget;
    fy = clamp(fy, -fyLimit, fyLimit);

    this.tyreForce.copy(_fwd).multiplyScalar(fx).addScaledVector(_left, fy);

    // ── Wheel spin (visual + diagnostic) ─────────────────────────────────────
    const omegaFree = vLong / cfg.radius;
    let targetOmega = omegaFree;
    if (this.locked) {
      targetOmega = 0;
    } else if (this.spinningUp) {
      // Excess drive spins the wheel faster than the ground. Scale by how far
      // past the limit the demand is, capped so it never looks like a cartoon.
      targetOmega = omegaFree + Math.min(driveForce / Math.max(fxCap, 1) - 1, 1.2) * 9;
    }
    // 35 ms half-life: fast enough to read as a lockup, slow enough that the
    // wheel visibly winds down rather than snapping.
    this.spinRate += (targetOmega - this.spinRate) * (1 - Math.pow(2, -dt / 0.035));
    this.advanceSpin(dt);

    // Slip ratio diagnostic, in the contract's -1..1 with negative = skidding.
    if (this.locked) {
      this.slipRatio = -Math.min(1, Math.abs(vLong) / 3);
    } else if (this.spinningUp) {
      this.slipRatio = Math.min(1, (this.spinRate * cfg.radius - vLong) / Math.max(vRef, 2));
    } else {
      this.slipRatio = clamp(ux * 0.14, -1, 1);
    }

    return this.tyreForce;
  }

  private advanceSpin(dt: number): void {
    this.spinUnwrapped += this.spinRate * dt;
    // The public angle is wrapped; the unwrapped value is what the visual
    // interpolator lerps, so the wheel never flicks backwards across the seam.
    this.spin = this.spinUnwrapped % (Math.PI * 2);
    if (this.spin < 0) this.spin += Math.PI * 2;
  }

  /** How hard this tyre is sliding, 0..1. Drives dust rate and tyre noise. */
  get slideAmount(): number {
    if (!this.grounded) return 0;
    const lat = Math.min(1, Math.abs(this.lateralSlip) / 5.5);
    const lon = this.locked ? Math.min(1, Math.abs(this.vLong) / 6) : 0;
    const spinUp = this.spinningUp ? 0.55 : 0;
    return clamp01(Math.max(lat, lon, spinUp));
  }
}

/** Convenience for the physics: the wheel's contact velocity magnitude. */
export function contactSpeed(w: Wheel): number {
  return Math.hypot(w.vLong, w.vLat);
}

/** Rotate a body-space vector into world space without allocating. */
export function bodyToWorld(v: Vector3, quat: Quaternion, out: Vector3): Vector3 {
  return out.copy(v).applyQuaternion(quat);
}

/** Exposed so BikePhysics can reuse the same scratch discipline. */
export const wheelScratch = { _tmp, _tmp2, _q };
