/**
 * FX — the facade.
 *
 * createEffects() wires the five subsystems into one object that implements
 * IEffects, and then does the thing that actually matters for integration:
 * it DERIVES almost everything from BikeState on its own.
 *
 * Hand it the player's BikeState once per frame and it will emit wheel dust at
 * the right rate for the surface and the slip, throw gravel out of a skid,
 * splash the stream, detect the landing from the mode transition (not just the
 * single-physics-step flag, which a 120Hz sim can raise and clear between two
 * rendered frames), fire the dust burst, decide whether the landing earned an
 * impact hold, drive the speed lines, and smear the rider. The Game never has
 * to know what a puff is.
 *
 * Manual control is still there — every IEffects method does exactly what it
 * says — but the automatic path is the intended one, because the alternative
 * is a hundred lines of emission policy living in Game.ts where nobody will
 * ever tune it.
 *
 * TIME SCALE. Two systems can slow or stop the clock: the camera's big-air
 * hold and the impact freeze. The facade publishes the minimum of the two as
 * `timeScale`, and beginFrame() exists so the Game can read it BEFORE it scales
 * this frame's dt rather than a frame late — which matters enormously for a
 * one-frame freeze and not at all for a 300ms slow-mo.
 */

import {
  Object3D,
  PerspectiveCamera,
  Scene,
  Vector3,
  type Camera,
} from 'three';

import {
  SurfaceKind,
  type BikeState,
  type IEffects,
  type ITerrain,
  type SurfaceProperties,
  type WheelState,
} from '../game/Contracts';
import { clamp01 } from '../core/MathX';
import { Rng } from '../core/RNG';
import { BIKE, SURFACES } from '../game/WorldConstants';
import { HUD_PALETTE } from '../npr/Palette';

import { CameraDirector, CAMERA_TUNING, type CameraDirectorOptions } from './CameraDirector';
import { DustSystem } from './DustSystem';
import { DebrisSystem } from './Debris';
import { SpeedFX, SpinSmear } from './SpeedFX';
import { ImpactFrames, IMPACT_TUNING } from './ImpactFrames';

export { CameraDirector, CAMERA_TUNING } from './CameraDirector';
export { DustSystem, dustTintFor, puffAtlas, QuadParticlePool } from './DustSystem';
export { DebrisSystem, chipAtlas } from './Debris';
export { SpeedFX, SpinSmear, SPEED_TUNING } from './SpeedFX';
export { ImpactFrames, IMPACT_TUNING } from './ImpactFrames';
export type { CameraDirectorOptions } from './CameraDirector';

const _fallbackPos = new Vector3();
const _fallbackNrm = new Vector3(0, 1, 0);

/**
 * Used when the physics has not filled in a contact yet (first frame, or a
 * crash in mid-air). Dirt is the least surprising default and the numbers come
 * straight from WorldConstants.
 */
const DEFAULT_SURFACE: SurfaceProperties = {
  kind: SurfaceKind.Dirt,
  ...SURFACES.dirt,
  audioTone: 'hardpack',
};

/** Boost ignition flash colour, resolved once. */
const BOOST_FLASH_HEX = HUD_PALETTE.boost.getHex();

export interface EffectsDeps {
  /** The FX object is added to this scene by createEffects. */
  scene: Scene;
  /** The camera the CameraDirector will drive — pass the Engine's. */
  camera: PerspectiveCamera;
  terrain?: ITerrain | null;
  seed?: number | string;
  dustCapacity?: number;
  debrisCapacity?: number;
  /** Bike local forward axis, only used as a heading fallback. Default +Z. */
  forwardAxis?: Vector3;
  /**
   * Derive dust, debris and impact frames from the subject BikeState.
   * Off means every emission is a manual call.
   */
  autoEmit?: boolean;
  /** Extra options forwarded to the CameraDirector. */
  cameraOptions?: Partial<Omit<CameraDirectorOptions, 'camera' | 'terrain'>>;
}

export class Effects implements IEffects {
  readonly object: Object3D;
  readonly dust: DustSystem;
  readonly debris: DebrisSystem;
  readonly speed: SpeedFX;
  readonly impact: ImpactFrames;
  readonly cameraDirector: CameraDirector;

  private rng: Rng;
  private subject: BikeState | null = null;
  private autoEmit: boolean;
  private terrain: ITerrain | null;

  // Independent edge detection. The CameraDirector runs its own copy for its
  // own shake; keeping them separate means neither depends on the other's
  // update order, and each fires its own effects exactly once.
  private prevAirborne = false;
  private prevCrashing = false;
  private prevBoosting = false;
  private landCooldown = 0;
  private crashCooldown = 0;

  private impactSteppedThisFrame = false;

  constructor(deps: EffectsDeps) {
    this.rng = new Rng(deps.seed ?? 'fx');
    this.autoEmit = deps.autoEmit ?? true;
    this.terrain = deps.terrain ?? null;

    this.object = new Object3D();
    this.object.name = 'fx';
    this.object.matrixAutoUpdate = false;

    this.dust = new DustSystem({
      capacity: deps.dustCapacity ?? 1100,
      rng: this.rng.fork('dust'),
    });
    this.debris = new DebrisSystem({
      capacity: deps.debrisCapacity ?? 640,
      rng: this.rng.fork('debris'),
      terrain: this.terrain,
    });
    this.speed = new SpeedFX();
    this.impact = new ImpactFrames();

    this.cameraDirector = new CameraDirector({
      camera: deps.camera,
      terrain: this.terrain,
      forwardAxis: deps.forwardAxis,
      rng: this.rng.fork('camera'),
      ...(deps.cameraOptions ?? {}),
    });

    this.object.add(this.dust.object);
    this.object.add(this.debris.object);
    this.object.add(this.speed.object);
    deps.scene.add(this.object);
  }

  // ── Wiring ────────────────────────────────────────────────────────────────

  /** The BikeState everything automatic is derived from. Null outside a race. */
  setSubject(state: BikeState | null): void {
    this.subject = state;
    if (state) {
      this.prevAirborne = state.mode === 'airborne';
      this.prevCrashing = state.mode === 'crashing';
    }
  }

  setTerrain(t: ITerrain | null): void {
    this.terrain = t;
    this.debris.setTerrain(t);
    this.cameraDirector.setTerrain(t);
  }

  /** Objects that automatically smear at speed / during trick rotation. */
  setSmearTargets(objects: Object3D[]): void {
    this.speed.setAutoSmearTargets(objects);
  }

  /** Register a wheel for radial spin smear. See SpeedFX.addSpinSmear. */
  addSpinSmear(anchor: Object3D, radius: number, axis: Vector3): SpinSmear {
    return this.speed.addSpinSmear(anchor, radius, axis);
  }

  // ── Time ──────────────────────────────────────────────────────────────────

  /** Minimum of the camera's slow-mo and the impact freeze. */
  get timeScale(): number {
    return Math.min(this.cameraDirector.timeScale, this.impact.timeScale);
  }

  /**
   * Advance the impact-frame state machine and return the time scale to apply
   * to THIS frame. Call first, with the real (unscaled) frame delta.
   *
   * Optional — if you never call it, update() advances the impact machine
   * itself and the freeze simply lands one frame later.
   */
  beginFrame(realDt: number): number {
    this.impact.update(realDt);
    this.impactSteppedThisFrame = true;
    return this.timeScale;
  }

  // ── Frame ─────────────────────────────────────────────────────────────────

  /**
   * `dt` is the SCALED delta. `realDt` defaults to it; pass the unscaled delta
   * (or use beginFrame) so a freeze can end.
   */
  update(dt: number, _time: number, camera: Camera, realDt?: number): void {
    const rd = realDt ?? dt;
    if (!this.impactSteppedThisFrame) this.impact.update(rd);
    this.impactSteppedThisFrame = false;

    const cam = (camera as PerspectiveCamera).isPerspectiveCamera
      ? (camera as PerspectiveCamera)
      : this.cameraDirector.camera;

    const s = this.subject;
    if (s && this.autoEmit) {
      this.detectEvents(s, rd);
      if (dt > 0) this.emitFromState(s, dt);
    }

    this.dust.update(dt, cam);
    this.debris.update(dt, cam);
    this.speed.update(dt, cam, s);
  }

  // ── Automatic emission ────────────────────────────────────────────────────

  private detectEvents(s: BikeState, dt: number): void {
    if (this.landCooldown > 0) this.landCooldown -= dt;
    if (this.crashCooldown > 0) this.crashCooldown -= dt;

    const airborneNow = s.mode === 'airborne';
    const crashingNow = s.mode === 'crashing';

    const landed = s.landedThisStep || (this.prevAirborne && !airborneNow && !crashingNow);
    if (landed && this.landCooldown <= 0) {
      this.landCooldown = 0.08;
      this.notifyLanding(s);
    }

    const crashed = s.crashedThisStep || (!this.prevCrashing && crashingNow);
    if (crashed && this.crashCooldown <= 0) {
      this.crashCooldown = 0.40;
      this.notifyCrash(s);
    }

    // Boost ignition gets a flash but never a freeze — it happens far too
    // often to spend a hold on, and a hold would fight the acceleration.
    if (s.boosting && !this.prevBoosting) this.impact.flashOnly(0.32, BOOST_FLASH_HEX);
    this.prevBoosting = s.boosting;

    this.prevAirborne = airborneNow;
    this.prevCrashing = crashingNow;
  }

  private emitFromState(s: BikeState, dt: number): void {
    // The rear wheel does most of the visible work — it carries the drive, it
    // locks first under braking, and it is the one the camera is looking at.
    this.emitWheel(s.rear, s, dt, 1.0);
    this.emitWheel(s.front, s, dt, 0.58);
  }

  private emitWheel(w: WheelState, s: BikeState, dt: number, weight: number): void {
    if (!w || !w.grounded) return;
    const surf = w.surface ?? DEFAULT_SURFACE;
    const pos = w.contactPoint ?? s.position;
    const nrm = w.contactNormal ?? _fallbackNrm;

    const slip = Math.abs(w.lateralSlip ?? 0);
    const ratio = w.slipRatio ?? 0;
    const lock = clamp01(-ratio);
    const spinUp = clamp01(ratio);
    // Load normalised against roughly a single wheel's share of static weight,
    // so a wheel that has gone light through a compression stops throwing dust.
    const load01 = clamp01((w.load ?? 0) / (BIKE.mass * BIKE.gravity * 0.75));

    const roll = clamp01((s.speed - 4) / 16) * 0.42;
    const skid = clamp01(slip / 5.5) + lock * 0.75 + spinUp * 0.5;
    const rate = (roll + skid * 1.4) * 24 * weight * (0.35 + load01 * 0.9);
    if (rate > 0.5) this.dust.trail(pos, nrm, s.velocity, rate, dt, surf);

    if (surf.kind === SurfaceKind.Water) {
      if (s.speed > 2.5 && this.rng.next() < clamp01(dt * (4 + s.speed * 0.7))) {
        this.debris.splash(pos, nrm, s.velocity, clamp01(0.25 + s.speed / 22) * weight);
      }
    } else if (skid > 0.30 && surf.dustAmount > 0.45) {
      // Gravel is gated stochastically rather than accumulated, so it stays
      // correct in expectation without needing per-wheel carry state.
      if (this.rng.next() < clamp01(skid * dt * 7)) {
        this.debris.screeSpray(pos, nrm, s.velocity, clamp01(skid * 0.55) * weight, surf);
      }
    }
  }

  /** Public so a caller with exact physics-step timing can drive it instead. */
  notifyLanding(s: BikeState): void {
    const impact = clamp01(s.landingImpact);
    const w = s.rear?.grounded ? s.rear : s.front;
    const surf = w?.surface ?? DEFAULT_SURFACE;
    const pos = this.contactOrFallback(w, s);
    const nrm = w?.contactNormal ?? _fallbackNrm;

    // Landings always throw dust, even a soft one — the dust is the read that
    // the wheels touched. Only the SIZE tracks the impact.
    this.dust.burst(pos, nrm, s.velocity, Math.max(impact, 0.58), surf);
    if (impact > 0.12) this.debris.screeSpray(pos, nrm, s.velocity, impact * 0.85, surf);
    if (surf.kind === SurfaceKind.Water) {
      this.debris.splash(pos, nrm, s.velocity, 0.45 + impact * 0.55);
    }
    this.impact.trigger(impact);
  }

  notifyCrash(s: BikeState): void {
    const sev = clamp01(s.crashSeverity || 0.6);
    const w = s.rear?.grounded ? s.rear : s.front;
    const surf = w?.surface ?? DEFAULT_SURFACE;
    const pos = this.contactOrFallback(w, s);
    const nrm = w?.contactNormal ?? _fallbackNrm;

    this.dust.burst(pos, nrm, s.velocity, 0.85 + sev * 0.15, surf);
    this.debris.crashDebris(pos, s.velocity, s.crashDirection, sev, surf);
    this.impact.trigger(0.45 + sev * 0.55, undefined, true);
  }

  private contactOrFallback(w: WheelState | undefined, s: BikeState): Vector3 {
    const p = w?.contactPoint;
    if (p && Number.isFinite(p.x) && (p.lengthSq() > 1e-6 || s.position.lengthSq() < 1e-6)) return p;
    _fallbackPos.copy(s.position);
    if (this.terrain) _fallbackPos.y = this.terrain.heightAt(s.position.x, s.position.z);
    return _fallbackPos;
  }

  // ── IEffects ──────────────────────────────────────────────────────────────

  dustBurst(
    position: Vector3,
    normal: Vector3,
    velocity: Vector3,
    amount: number,
    surface: SurfaceProperties,
  ): void {
    this.dust.burst(position, normal, velocity, amount, surface ?? DEFAULT_SURFACE);
  }

  /**
   * Continuous dust. `rate` is puffs per second before the surface multiplier.
   *
   * The contract has no dt, so this uses a fixed nominal step of one 60Hz
   * frame. If you are calling it from the 120Hz physics step, halve the rate
   * or use `dust.trail(..., dt, ...)` directly — that overload takes a real dt.
   */
  dustTrail(
    position: Vector3,
    normal: Vector3,
    velocity: Vector3,
    rate: number,
    surface: SurfaceProperties,
  ): void {
    this.dust.trail(position, normal, velocity, rate, 1 / 60, surface ?? DEFAULT_SURFACE);
  }

  impactFrame(intensity: number, tint?: number): void {
    this.impact.trigger(intensity, tint);
  }

  smear(target: Object3D, amount: number): void {
    this.speed.smear(target, amount);
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /** Clear every live particle and post dial. Use on race restart. */
  reset(): void {
    this.debris.clear();
    this.speed.reset();
    this.impact.reset();
    this.landCooldown = 0;
    this.crashCooldown = 0;
    this.prevAirborne = false;
    this.prevCrashing = false;
    this.prevBoosting = false;
  }

  dispose(): void {
    this.dust.dispose();
    this.debris.dispose();
    this.speed.dispose();
    this.cameraDirector.dispose();
    this.object.removeFromParent();
  }
}

export function createEffects(deps: EffectsDeps): Effects {
  return new Effects(deps);
}

/** Build only the camera, for a Game that does not want the particle systems. */
export function createCameraDirector(opts: CameraDirectorOptions): CameraDirector {
  return new CameraDirector(opts);
}
