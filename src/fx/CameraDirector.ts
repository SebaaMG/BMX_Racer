/**
 * CameraDirector — the camera language.
 *
 * A downhill racing game is won or lost on the camera long before it is won on
 * the physics. The bike can be perfectly tuned and the whole thing will feel
 * inert if the camera is bolted rigidly behind it, because a rigid rig removes
 * the only cue the player has for lateral motion: nothing moves relative to
 * anything else, so a 60 km/h corner and a 20 km/h corner look identical.
 *
 * Everything here exists to break that rigidity in controlled ways.
 *
 * THE LAG. The chase anchor does not follow the bike's heading — it follows a
 * LAGGED heading, half-life 0.16–0.30s scaling with speed. Entering a corner
 * the camera is still aimed down the previous straight, so the bike sweeps
 * across the frame; leaving it, the anchor catches up and the camera swings
 * through the exit. On top of that the position spring is deliberately
 * UNDER-damped (zeta 0.68), so it overshoots and settles rather than arriving.
 * Those two things together are the whip.
 *
 * THE FOV. 62° cruising to 78° flat out, but on a 2.7 exponent, so almost all
 * of the change lives in the top third of the speed range. A linear FOV ramp
 * reads as a slow zoom and the player stops noticing it; a curve that does
 * nothing until you are genuinely fast and then opens hard reads as speed.
 *
 * THE SHAKE. Directional and impact-shaped, never a rumble. A primary
 * oscillation along the impact axis under an envelope with a visible rebound,
 * plus a decorrelated simplex wobble. Simplex rather than Math.random because
 * the capture harness compares builds frame for frame and a random camera would
 * make every diff a false positive.
 *
 * THE SWING. Above a threshold of air, rising, with enough hang time left, the
 * camera orbits ~66° to show the trick — and then starts coming back EARLY,
 * cancelling itself the moment the projected time-to-land drops below 0.55s.
 * A camera still orbiting when the wheels touch is worse than never orbiting.
 *
 * THE SLOW-MO. Rare by construction: 7m of air, past apex, 9s cooldown. It
 * exposes a timeScale the Game multiplies its dt by; the camera itself keeps
 * moving on the scaled clock so the whole world slows together.
 */

import { PerspectiveCamera, Quaternion, Vector3 } from 'three';

import {
  BikeMode,
  CameraMode,
  type BikeState,
  type ICameraDirector,
  type IReplayRecorder,
  type ITerrain,
  type ReplayFrame,
} from '../game/Contracts';
import {
  clamp,
  clamp01,
  dampAngleHL,
  dampHL,
  ease,
  lerp,
  makeSpring,
  shortAngle,
  smoothstep,
  springStep,
  springStepDamped,
  type SpringState,
} from '../core/MathX';
import { Noise2D } from '../core/Noise';
import { Rng } from '../core/RNG';
import { BIKE } from '../game/WorldConstants';

// ── Module scratch ───────────────────────────────────────────────────────────
const _flatVel = new Vector3();
const _fwd = new Vector3();
const _dirV = new Vector3();
const _rightV = new Vector3();
const _desired = new Vector3();
const _lookWanted = new Vector3();
const _camFinal = new Vector3();
const _lookFinal = new Vector3();
const _shakeDir = new Vector3();
const _tmp = new Vector3();
const _qa = new Quaternion();
const _qb = new Quaternion();
const UP = new Vector3(0, 1, 0);

/** Deterministic shake noise. Never Math.random — captures must be comparable. */
const SHAKE_NOISE = new Noise2D('camera-shake');

// ─────────────────────────────────────────────────────────────────────────────
// Tuning
// ─────────────────────────────────────────────────────────────────────────────

export const CAMERA_TUNING = {
  fovBase: 62,
  fovTop: 78,
  /** Exponent on normalised speed. High on purpose — see the header. */
  fovExponent: 2.7,
  /** Speed treated as "flat out", m/s. */
  referenceSpeed: 26,

  chaseDistance: 5.0,
  chaseDistanceSpeedGain: 2.1,
  chaseHeight: 1.90,
  chaseHeightSpeedGain: 0.55,
  /** Position spring: under-damped, which is where the whip comes from. */
  chaseOmega: 6.2,
  chaseZeta: 0.68,
  /** Look-at spring: critically damped and much stiffer, so framing stays solid. */
  lookOmega: 11.0,
  /** Heading lag half-life at rest and at reference speed, seconds. */
  lagHalfLifeSlow: 0.16,
  lagHalfLifeFast: 0.30,
  /** Metres of outward drift per unit of lateral acceleration. */
  cornerSwing: 0.11,
  cornerSwingMax: 2.6,
  /** Camera roll per unit of lateral acceleration, radians. */
  rollGain: 0.0085,
  rollMax: 0.12,

  /** Peak shake displacement in metres at amount 1.0. */
  shakeMetres: 0.42,
  /** Primary oscillation, rad/s (~10 Hz). */
  shakeFrequency: 62,

  airSwingArc: 1.15,
  airSwingMinAirTime: 0.55,
  airSwingBailout: 0.55,
  airSwingCooldown: 2.2,
  airSwingRise: 1.4,

  slowMoMinPeak: 7.0,
  slowMoScale: 0.38,
  slowMoAttack: 0.12,
  slowMoHold: 0.30,
  slowMoRelease: 0.32,
  slowMoCooldown: 9.0,

  collisionSamples: 5,
  collisionMargin: 1.35,
} as const;

function estimateAirRemaining(t: BikeState): number {
  // Ballistic time to return to the ground directly below, from the current
  // vertical velocity and height. Cheap, and good enough that the swing knows
  // when to start coming home.
  const g = BIKE.gravity;
  const vy = t.velocity.y;
  const h = Math.max(t.airHeight, 0);
  const disc = vy * vy + 2 * g * h;
  if (disc <= 0) return 0;
  return (vy + Math.sqrt(disc)) / g;
}

// ─────────────────────────────────────────────────────────────────────────────

export interface CameraDirectorOptions {
  /** The camera to drive. Pass the Engine's so aspect/resize keep working. */
  camera: PerspectiveCamera;
  terrain?: ITerrain | null;
  fovBase?: number;
  fovTop?: number;
  /**
   * The bike's forward axis in its own local space. Only used as a FALLBACK
   * when the bike is nearly stationary — the heading normally comes straight
   * from the velocity, which needs no convention at all.
   */
  forwardAxis?: Vector3;
  rng?: Rng;
  /** Detect landings/crashes from BikeState itself. Off if you drive them. */
  autoDetectEvents?: boolean;
}

export class CameraDirector implements ICameraDirector {
  readonly camera: PerspectiveCamera;
  mode: CameraMode = CameraMode.Chase;

  /** The Game multiplies its dt by this. 1 normally, <1 during a big-air hold. */
  timeScale = 1;

  /** Last state passed to update(). Null before the first frame. */
  subject: BikeState | null = null;

  private terrain: ITerrain | null;
  private forwardAxis: Vector3;
  private rng: Rng;
  private autoDetect: boolean;

  // Springs.
  private sx: SpringState = makeSpring();
  private sy: SpringState = makeSpring();
  private sz: SpringState = makeSpring();
  private lx: SpringState = makeSpring();
  private ly: SpringState = makeSpring();
  private lz: SpringState = makeSpring();
  private fovS: SpringState;

  // Composed each frame.
  private camPos = new Vector3(0, 5, 10);
  private lookPos = new Vector3();

  // Heading.
  private aimYaw = 0;
  private prevYaw = 0;
  private yawRate = 0;
  private roll = 0;
  private headingPrimed = false;

  // FOV.
  private fovBase: number;
  private fovTop: number;
  private kick = 0;
  private prevBoosting = false;

  // Shake.
  private shakeAmp = 0;
  private shakeT = 0;
  private shakeDur = 0;
  private shakeSeed = 0;
  private shakeDir = new Vector3(0, 1, 0);
  private shakeOffset = new Vector3();
  private shakeRoll = 0;

  // Air swing.
  private swingActive = false;
  private swingT = 0;
  private swingDur = 1;
  private swingDir = 1;
  private swingAmount = 0;
  private swingCooldown = 0;

  // Slow-mo.
  private slowActive = false;
  private slowT = 0;
  private slowCooldown = 0;

  // Collision.
  private collisionLift = 0;

  // Event edge detection.
  private prevAirborne = false;
  private prevCrashing = false;
  private landCooldown = 0;
  private crashCooldown = 0;

  // Cinematic.
  private cineAnchor = new Vector3();
  private cineValid = false;
  private cineSide = 1;

  // Orbit / free / fixed.
  private orbitYaw = 0.6;
  private orbitPitch = 0.22;
  private orbitDist = 9;
  private orbitSpin = 0.35;
  private freePos = new Vector3(0, 8, 20);
  private freeYaw = 0;
  private freePitch = 0;
  private freeInput = { forward: 0, strafe: 0, lift: 0, yaw: 0, pitch: 0, speed: 1 };
  private fixedPos = new Vector3(0, 5, 10);
  private fixedLook = new Vector3();

  // Replay.
  private replaySource: IReplayRecorder | null = null;
  private replayT = 0;
  private replayStart = 0;
  private replayEnd = 0;
  private replaySpeed = 0.55;
  private replayBaseYaw = 0;
  /** Sampled replay subject — the Game can pose a ghost bike from these. */
  readonly replayPosition = new Vector3();
  readonly replayOrientation = new Quaternion();
  replayProgress = 0;

  /** Fired on a detected landing / crash. The FX facade hangs dust off these. */
  onLandingEvent: ((state: BikeState, impact: number) => void) | null = null;
  onCrashEvent: ((state: BikeState, severity: number) => void) | null = null;

  constructor(opts: CameraDirectorOptions) {
    this.camera = opts.camera;
    this.terrain = opts.terrain ?? null;
    this.forwardAxis = (opts.forwardAxis ?? new Vector3(0, 0, 1)).clone().normalize();
    this.rng = opts.rng ?? new Rng('fx:camera');
    this.autoDetect = opts.autoDetectEvents ?? true;
    this.fovBase = opts.fovBase ?? CAMERA_TUNING.fovBase;
    this.fovTop = opts.fovTop ?? CAMERA_TUNING.fovTop;
    this.fovS = makeSpring(this.fovBase);
    this.camera.fov = this.fovBase;
    this.camera.updateProjectionMatrix();
    this.camPos.copy(this.camera.position);
    this.sx.value = this.camPos.x;
    this.sy.value = this.camPos.y;
    this.sz.value = this.camPos.z;
  }

  setTerrain(t: ITerrain | null): void {
    this.terrain = t;
  }

  get lookAtPoint(): Vector3 {
    return this.lookPos;
  }

  // ── Frame ─────────────────────────────────────────────────────────────────

  /**
   * `dt` is the SCALED delta — the camera slows with the world. `realDt` is
   * optional and defaults to `dt`; pass the unscaled delta so the slow-mo
   * envelope can end (driving it with its own output would never release).
   */
  update(target: BikeState, dt: number, time: number, realDt?: number): void {
    this.subject = target;
    const d = clamp(dt, 0, 0.1);
    const rd = clamp(realDt ?? dt, 0, 0.1);

    if (this.autoDetect) this.detectEvents(target, rd);

    switch (this.mode) {
      case CameraMode.Chase:
        this.updateChase(target, d);
        break;
      case CameraMode.Cinematic:
        this.updateCinematic(target, d, time);
        break;
      case CameraMode.Replay:
        this.updateReplay(d);
        break;
      case CameraMode.Orbit:
        this.updateOrbit(target, d);
        break;
      case CameraMode.Free:
        this.updateFree(d);
        break;
      case CameraMode.Fixed:
        this.camPos.copy(this.fixedPos);
        this.lookPos.copy(this.fixedLook);
        break;
    }

    this.updateFov(this.mode === CameraMode.Chase ? target : null, d);
    this.updateSlowMo(target, rd);
    this.compose(d);
  }

  // ── Chase ─────────────────────────────────────────────────────────────────

  private updateChase(t: BikeState, dt: number): void {
    _flatVel.copy(t.velocity);
    _flatVel.y = 0;
    const planar = _flatVel.length();

    let travelYaw = this.aimYaw;
    if (planar > 1.2) {
      travelYaw = Math.atan2(_flatVel.x, _flatVel.z);
    } else {
      // Fallback only. The heading normally comes from velocity, which needs
      // no assumption about which local axis the bike calls "forward".
      _fwd.copy(this.forwardAxis).applyQuaternion(t.orientation);
      _fwd.y = 0;
      if (_fwd.lengthSq() > 1e-5) travelYaw = Math.atan2(_fwd.x, _fwd.z);
    }

    if (!this.headingPrimed) {
      this.aimYaw = travelYaw;
      this.prevYaw = travelYaw;
      this.headingPrimed = true;
    }

    const spd = t.speed;
    const speed01 = clamp01(spd / CAMERA_TUNING.referenceSpeed);

    // Yaw rate from the TRUE heading, before the lag is applied — this is the
    // corner signal, and reading it off the lagged anchor would smear it.
    const dYaw = shortAngle(this.prevYaw, travelYaw);
    this.prevYaw = travelYaw;
    const instRate = dt > 1e-4 ? clamp(dYaw / dt, -6, 6) : this.yawRate;
    this.yawRate = dampHL(this.yawRate, instRate, 0.09, dt);

    const lagHL = lerp(CAMERA_TUNING.lagHalfLifeSlow, CAMERA_TUNING.lagHalfLifeFast, speed01);
    this.aimYaw = dampAngleHL(this.aimYaw, travelYaw, lagHL, dt);

    // Air framing: pull back and rise so the whole arc is legible.
    const airborne = t.mode === BikeMode.Airborne;
    const airH = airborne ? Math.max(t.airHeight, 0) : 0;
    const airLift = clamp(airH * 0.10, 0, 2.2);
    const airPull = clamp(airH * 0.20, 0, 4.0);

    this.updateAirSwing(t, dt);

    const dist = CAMERA_TUNING.chaseDistance + CAMERA_TUNING.chaseDistanceSpeedGain * speed01 + airPull;
    const height = CAMERA_TUNING.chaseHeight + CAMERA_TUNING.chaseHeightSpeedGain * speed01 + airLift;

    const anchorYaw = this.aimYaw + this.swingAmount * this.swingDir * CAMERA_TUNING.airSwingArc;
    _dirV.set(Math.sin(anchorYaw), 0, Math.cos(anchorYaw));
    // right = dir x up.
    _rightV.set(-_dirV.z, 0, _dirV.x);

    // Lateral acceleration proxy. Positive yawRate turns toward +X from +Z,
    // which is a LEFT turn, whose outside is +right — so the drift sign is
    // straight through with no negation.
    const latAccel = clamp(this.yawRate * spd, -34, 34);
    const swingLat = clamp(latAccel * CAMERA_TUNING.cornerSwing, -CAMERA_TUNING.cornerSwingMax, CAMERA_TUNING.cornerSwingMax);

    _desired
      .copy(t.position)
      .addScaledVector(_dirV, -dist)
      .addScaledVector(_rightV, swingLat)
      .addScaledVector(UP, height + this.swingAmount * CAMERA_TUNING.airSwingRise);

    springStepDamped(this.sx, _desired.x, CAMERA_TUNING.chaseOmega, CAMERA_TUNING.chaseZeta, dt);
    springStepDamped(this.sy, _desired.y, CAMERA_TUNING.chaseOmega * 1.25, 0.92, dt);
    springStepDamped(this.sz, _desired.z, CAMERA_TUNING.chaseOmega, CAMERA_TUNING.chaseZeta, dt);
    this.camPos.set(this.sx.value, this.sy.value, this.sz.value);

    // Look point: lead the rider a little so the frame shows where they are
    // going. Airborne, drop the aim so the landing stays on screen.
    _lookWanted.copy(t.position);
    _lookWanted.y += 1.05;
    _lookWanted.addScaledVector(t.velocity, 0.10);
    if (airborne) _lookWanted.y -= clamp(airH * 0.10, 0, 1.6);

    springStep(this.lx, _lookWanted.x, CAMERA_TUNING.lookOmega, dt);
    springStep(this.ly, _lookWanted.y, CAMERA_TUNING.lookOmega, dt);
    springStep(this.lz, _lookWanted.z, CAMERA_TUNING.lookOmega, dt);
    this.lookPos.set(this.lx.value, this.ly.value, this.lz.value);

    const targetRoll = clamp(-latAccel * CAMERA_TUNING.rollGain, -CAMERA_TUNING.rollMax, CAMERA_TUNING.rollMax);
    this.roll = dampHL(this.roll, targetRoll, 0.10, dt);
  }

  // ── Air swing ─────────────────────────────────────────────────────────────

  private updateAirSwing(t: BikeState, dt: number): void {
    if (this.swingCooldown > 0) this.swingCooldown -= dt;

    const airborne = t.mode === BikeMode.Airborne;

    if (
      !this.swingActive &&
      airborne &&
      this.swingCooldown <= 0 &&
      t.airTime > CAMERA_TUNING.airSwingMinAirTime &&
      t.velocity.y > 0.4
    ) {
      const rem = estimateAirRemaining(t);
      // Only commit if there is genuinely time to go out and come back.
      if (rem > 1.1) this.beginAirSwing(Math.min(rem * 0.85, 2.4));
    }

    if (!this.swingActive) {
      this.swingAmount = dampHL(this.swingAmount, 0, 0.14, dt);
      return;
    }

    this.swingT += dt;

    // Bail out early. Landing legibility beats showing off the trick, always.
    const rem = airborne ? estimateAirRemaining(t) : 0;
    if (!airborne || rem < CAMERA_TUNING.airSwingBailout) {
      this.swingDur = Math.min(this.swingDur, this.swingT + 0.18);
    }

    const u = clamp01(this.swingT / Math.max(this.swingDur, 0.2));
    let env: number;
    if (u < 0.42) env = ease.inOutCubic(u / 0.42);
    else if (u < 0.60) env = 1;
    else env = 1 - ease.inOutCubic((u - 0.60) / 0.40);

    this.swingAmount = dampHL(this.swingAmount, env, 0.05, dt);

    if (u >= 1) {
      this.swingActive = false;
      this.swingCooldown = CAMERA_TUNING.airSwingCooldown;
    }
  }

  beginAirSwing(duration: number): void {
    this.swingActive = true;
    this.swingT = 0;
    this.swingDur = Math.max(duration, 0.4);
    const t = this.subject;
    // Orbit AGAINST the spin: the relative rotation is larger, which is what
    // makes the trick read. Orbiting with it would cancel the spin out and the
    // rider would look like they were hanging still in the air.
    if (t && Math.abs(t.angularVelocity.y) > 0.6) {
      this.swingDir = t.angularVelocity.y > 0 ? -1 : 1;
    } else if (Math.abs(this.yawRate) > 0.15) {
      this.swingDir = this.yawRate > 0 ? 1 : -1;
    } else {
      this.swingDir = this.rng.chance(0.5) ? 1 : -1;
    }
  }

  // ── FOV ───────────────────────────────────────────────────────────────────

  private updateFov(t: BikeState | null, dt: number): void {
    let target = this.fovBase;

    if (t) {
      const s01 = clamp01(t.speed / CAMERA_TUNING.referenceSpeed);
      target = this.fovBase + (this.fovTop - this.fovBase) * Math.pow(s01, CAMERA_TUNING.fovExponent);
      // Narrowing slightly in the air makes the height read as height. The
      // instinct is to widen for drama; widening actually flattens the drop.
      if (t.mode === BikeMode.Airborne) target -= clamp(t.airHeight * 0.22, 0, 3.2);

      if (t.boosting) {
        if (!this.prevBoosting) this.fovKick(6.5);
        target += 2.4;
      }
      this.prevBoosting = t.boosting;
    }

    this.kick = dampHL(this.kick, 0, 0.20, dt);
    target += this.kick;

    springStep(this.fovS, target, 8.5, dt);
    const f = clamp(this.fovS.value, 40, 110);
    if (Math.abs(this.camera.fov - f) > 1e-3) {
      this.camera.fov = f;
      this.camera.updateProjectionMatrix();
    }
  }

  fovKick(amount: number): void {
    if (amount > this.kick) this.kick = amount;
  }

  // ── Slow-mo ───────────────────────────────────────────────────────────────

  private updateSlowMo(t: BikeState | null, realDt: number): void {
    if (this.slowCooldown > 0) this.slowCooldown -= realDt;

    if (
      t &&
      !this.slowActive &&
      this.slowCooldown <= 0 &&
      this.mode === CameraMode.Chase &&
      t.mode === BikeMode.Airborne &&
      t.peakAirHeight >= CAMERA_TUNING.slowMoMinPeak &&
      t.airTime > 0.85 &&
      Math.abs(t.velocity.y) < 1.8
    ) {
      // Fires at apex (vertical velocity through zero) on a genuinely big jump.
      this.slowActive = true;
      this.slowT = 0;
    }

    if (!this.slowActive) {
      this.timeScale = 1;
      return;
    }

    this.slowT += realDt;
    const A = CAMERA_TUNING.slowMoAttack;
    const H = CAMERA_TUNING.slowMoHold;
    const R = CAMERA_TUNING.slowMoRelease;

    let s: number;
    if (this.slowT < A) s = ease.inOutCubic(this.slowT / A);
    else if (this.slowT < A + H) s = 1;
    else s = 1 - ease.inOutCubic((this.slowT - A - H) / R);

    if (this.slowT >= A + H + R) {
      this.slowActive = false;
      this.slowCooldown = CAMERA_TUNING.slowMoCooldown;
      s = 0;
    }
    this.timeScale = lerp(1, CAMERA_TUNING.slowMoScale, clamp01(s));
  }

  /** Force a slow-mo hold. Ignores the trigger conditions but honours nothing else. */
  triggerSlowMo(): void {
    this.slowActive = true;
    this.slowT = 0;
  }

  // ── Shake ─────────────────────────────────────────────────────────────────

  shake(amount: number, duration: number): void {
    // Never downgrade a shake that is still stronger than the new request —
    // a small follow-up hit must not cut a big one short.
    const remaining = this.shakeDur > 0 ? clamp01(1 - this.shakeT / this.shakeDur) : 0;
    if (amount <= this.shakeAmp * remaining) return;
    this.shakeAmp = amount;
    this.shakeDur = Math.max(duration, 0.05);
    this.shakeT = 0;
    this.shakeSeed += 13.77;
  }

  /** Directional shake. `dir` is the world-space axis the impulse arrived along. */
  shakeFrom(dir: Vector3, amount: number, duration: number): void {
    _shakeDir.copy(dir);
    if (_shakeDir.lengthSq() < 1e-8) _shakeDir.set(0, 1, 0);
    else _shakeDir.normalize();
    const remaining = this.shakeDur > 0 ? clamp01(1 - this.shakeT / this.shakeDur) : 0;
    if (amount <= this.shakeAmp * remaining) return;
    this.shakeDir.copy(_shakeDir);
    this.shake(amount, duration);
  }

  private applyShake(dt: number): void {
    this.shakeOffset.set(0, 0, 0);
    this.shakeRoll = 0;
    if (this.shakeDur <= 0 || this.shakeT >= this.shakeDur) return;

    this.shakeT += dt;
    const u = clamp01(this.shakeT / this.shakeDur);

    // The envelope is the whole difference between an impact and a rumble.
    // A steep power decay gives the instantaneous peak; the cosine term puts
    // one visible rebound in the tail, which is what says "something struck
    // something" rather than "an engine is running".
    const env = Math.pow(1 - u, 2.4) * (1 + 0.38 * Math.cos(u * Math.PI * 3.0));
    const amp = this.shakeAmp * CAMERA_TUNING.shakeMetres;

    const primary = Math.sin(this.shakeT * CAMERA_TUNING.shakeFrequency) * env * amp;
    this.shakeOffset.copy(this.shakeDir).multiplyScalar(primary);

    const s = this.shakeSeed;
    const w = env * amp * 0.42;
    this.shakeOffset.x += SHAKE_NOISE.noise(this.shakeT * 13.0 + s, 0.0) * w;
    this.shakeOffset.y += SHAKE_NOISE.noise(this.shakeT * 11.3 + s, 5.7) * w;
    this.shakeOffset.z += SHAKE_NOISE.noise(this.shakeT * 15.1 + s, 11.3) * w;
    this.shakeRoll = SHAKE_NOISE.noise(this.shakeT * 8.6 + s, 21.1) * env * this.shakeAmp * 0.055;
  }

  // ── Events ────────────────────────────────────────────────────────────────

  private detectEvents(t: BikeState, dt: number): void {
    if (this.landCooldown > 0) this.landCooldown -= dt;
    if (this.crashCooldown > 0) this.crashCooldown -= dt;

    const airborneNow = t.mode === BikeMode.Airborne;
    const crashingNow = t.mode === BikeMode.Crashing;

    // Landings are detected from the mode TRANSITION as well as from the
    // one-step flag, because update() runs once per rendered frame while
    // physics runs at 120Hz — the flag can be set and cleared inside a single
    // frame's pair of steps and never be observed here.
    const landed = t.landedThisStep || (this.prevAirborne && !airborneNow && !crashingNow);
    if (landed && this.landCooldown <= 0) {
      this.landCooldown = 0.08;
      this.onLanding(t);
    }

    const crashed = t.crashedThisStep || (!this.prevCrashing && crashingNow);
    if (crashed && this.crashCooldown <= 0) {
      this.crashCooldown = 0.40;
      this.onCrash(t);
    }

    this.prevAirborne = airborneNow;
    this.prevCrashing = crashingNow;
  }

  /** Public so a caller with exact physics-step timing can drive it instead. */
  onLanding(t: BikeState): void {
    const impact = clamp01(t.landingImpact);
    if (impact < 0.04) return;

    // Direction: dominantly vertical, because the ground pushed up — plus a
    // component back along travel so a fast flat landing shoves the camera as
    // well as bouncing it.
    _flatVel.copy(t.velocity);
    _flatVel.y = 0;
    if (_flatVel.lengthSq() > 1e-6) _flatVel.normalize();
    _tmp.set(0, 1, 0).addScaledVector(_flatVel, -0.35);

    this.shakeFrom(_tmp, 0.30 + impact * 0.90, 0.28 + impact * 0.30);
    this.fovKick(1.6 + impact * 5.0);
    // The swing has no business continuing once the wheels are down.
    this.swingActive = false;
    this.onLandingEvent?.(t, impact);
  }

  onCrash(t: BikeState): void {
    const sev = clamp01(t.crashSeverity || 0.6);
    _tmp.copy(t.crashDirection).multiplyScalar(-1);
    _tmp.y += 0.7;
    this.shakeFrom(_tmp, 0.65 + sev * 0.85, 0.55 + sev * 0.45);
    this.swingActive = false;
    this.slowActive = false;
    this.onCrashEvent?.(t, sev);
  }

  // ── Cinematic ─────────────────────────────────────────────────────────────

  private updateCinematic(t: BikeState, dt: number, time: number): void {
    _tmp.copy(t.position).sub(this.cineAnchor);
    _tmp.y = 0;
    if (!this.cineValid || _tmp.length() > 62) this.pickCinematicAnchor(t);

    // A held shot still has to breathe or it reads as a still frame with a
    // moving subject pasted on.
    _desired.copy(this.cineAnchor);
    _desired.y += Math.sin(time * 0.21) * 0.9;
    _desired.x += Math.sin(time * 0.13) * 1.2;
    _desired.z += Math.cos(time * 0.11) * 1.2;

    // Much softer than the chase: a crane, not an arm.
    springStepDamped(this.sx, _desired.x, 2.2, 1.0, dt);
    springStepDamped(this.sy, _desired.y, 2.2, 1.0, dt);
    springStepDamped(this.sz, _desired.z, 2.2, 1.0, dt);
    this.camPos.set(this.sx.value, this.sy.value, this.sz.value);

    _lookWanted.copy(t.position);
    _lookWanted.y += 1.2;
    springStep(this.lx, _lookWanted.x, 5.0, dt);
    springStep(this.ly, _lookWanted.y, 5.0, dt);
    springStep(this.lz, _lookWanted.z, 5.0, dt);
    this.lookPos.set(this.lx.value, this.ly.value, this.lz.value);

    this.roll = dampHL(this.roll, 0, 0.4, dt);
  }

  private pickCinematicAnchor(t: BikeState): void {
    _flatVel.copy(t.velocity);
    _flatVel.y = 0;
    if (_flatVel.lengthSq() < 1e-4) _flatVel.copy(this.forwardAxis).applyQuaternion(t.orientation);
    _flatVel.y = 0;
    if (_flatVel.lengthSq() < 1e-6) _flatVel.set(0, 0, 1);
    _flatVel.normalize();
    _rightV.set(-_flatVel.z, 0, _flatVel.x);

    this.cineSide = this.rng.chance(0.5) ? 1 : -1;
    this.cineAnchor
      .copy(t.position)
      .addScaledVector(_flatVel, this.rng.range(26, 44))
      .addScaledVector(_rightV, this.cineSide * this.rng.range(14, 26));
    this.cineAnchor.y += this.rng.range(6, 14);

    if (this.terrain) {
      const h = this.terrain.heightAt(this.cineAnchor.x, this.cineAnchor.z);
      if (this.cineAnchor.y < h + 4) this.cineAnchor.y = h + 4;
    }
    this.cineValid = true;
  }

  // ── Orbit / free / fixed ──────────────────────────────────────────────────

  setOrbit(yaw: number, pitch: number, dist: number, spin = 0.35): void {
    this.orbitYaw = yaw;
    this.orbitPitch = pitch;
    this.orbitDist = dist;
    this.orbitSpin = spin;
  }

  private updateOrbit(t: BikeState, dt: number): void {
    this.orbitYaw += this.orbitSpin * dt;
    const cy = Math.cos(this.orbitPitch);
    this.lookPos.copy(t.position);
    this.lookPos.y += 1.1;
    this.camPos.set(
      this.lookPos.x + Math.sin(this.orbitYaw) * cy * this.orbitDist,
      this.lookPos.y + Math.sin(this.orbitPitch) * this.orbitDist,
      this.lookPos.z + Math.cos(this.orbitYaw) * cy * this.orbitDist,
    );
    this.roll = dampHL(this.roll, 0, 0.2, dt);
  }

  setFreeInput(forward: number, strafe: number, lift: number, yaw: number, pitch: number, speed = 1): void {
    this.freeInput.forward = forward;
    this.freeInput.strafe = strafe;
    this.freeInput.lift = lift;
    this.freeInput.yaw = yaw;
    this.freeInput.pitch = pitch;
    this.freeInput.speed = speed;
  }

  private updateFree(dt: number): void {
    const i = this.freeInput;
    this.freeYaw += i.yaw * dt;
    this.freePitch = clamp(this.freePitch + i.pitch * dt, -1.4, 1.4);

    const cp = Math.cos(this.freePitch);
    _dirV.set(Math.sin(this.freeYaw) * cp, Math.sin(this.freePitch), Math.cos(this.freeYaw) * cp).normalize();
    _rightV.set(-_dirV.z, 0, _dirV.x).normalize();

    const v = 18 * i.speed * dt;
    this.freePos.addScaledVector(_dirV, i.forward * v);
    this.freePos.addScaledVector(_rightV, i.strafe * v);
    this.freePos.y += i.lift * v;

    this.camPos.copy(this.freePos);
    this.lookPos.copy(this.freePos).add(_dirV);
    this.roll = 0;
  }

  snapTo(position: Vector3, lookAt: Vector3): void {
    this.camPos.copy(position);
    this.lookPos.copy(lookAt);
    this.fixedPos.copy(position);
    this.fixedLook.copy(lookAt);
    this.freePos.copy(position);

    this.sx.value = position.x; this.sx.velocity = 0;
    this.sy.value = position.y; this.sy.velocity = 0;
    this.sz.value = position.z; this.sz.velocity = 0;
    this.lx.value = lookAt.x; this.lx.velocity = 0;
    this.ly.value = lookAt.y; this.ly.velocity = 0;
    this.lz.value = lookAt.z; this.lz.velocity = 0;

    this.collisionLift = 0;
    this.shakeDur = 0;
    this.shakeT = 0;
    this.swingActive = false;
    this.swingAmount = 0;
    this.roll = 0;

    this.camera.position.copy(position);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(lookAt);
    this.camera.updateMatrixWorld();
  }

  /** Re-seat the chase rig on the subject with no spring travel. Use on reset. */
  resetTo(target: BikeState): void {
    this.headingPrimed = false;
    this.aimYaw = 0;
    this.yawRate = 0;
    _flatVel.copy(target.velocity);
    _flatVel.y = 0;
    if (_flatVel.lengthSq() < 1e-4) {
      _flatVel.copy(this.forwardAxis).applyQuaternion(target.orientation);
      _flatVel.y = 0;
    }
    if (_flatVel.lengthSq() < 1e-6) _flatVel.set(0, 0, 1);
    _flatVel.normalize();

    _desired
      .copy(target.position)
      .addScaledVector(_flatVel, -CAMERA_TUNING.chaseDistance)
      .addScaledVector(UP, CAMERA_TUNING.chaseHeight);
    _lookWanted.copy(target.position);
    _lookWanted.y += 1.05;
    this.snapTo(_desired, _lookWanted);
    this.subject = target;
    this.prevAirborne = target.mode === BikeMode.Airborne;
    this.prevCrashing = target.mode === BikeMode.Crashing;
    this.timeScale = 1;
    this.slowActive = false;
    this.kick = 0;
    this.fovS.value = this.fovBase;
    this.fovS.velocity = 0;
  }

  // ── Replay ────────────────────────────────────────────────────────────────

  setReplaySource(rec: IReplayRecorder | null): void {
    this.replaySource = rec;
  }

  /**
   * Drive the results-screen cinematic from the recorder's biggest-air window.
   * Returns false if there is nothing worth showing, so the results screen can
   * fall back to a static framing rather than playing an empty replay.
   */
  playBiggestAir(opts: { speed?: number; lead?: number; trail?: number } = {}): boolean {
    const rec = this.replaySource;
    if (!rec || rec.frames.length < 4) return false;
    const w = rec.getBiggestAir();
    if (!w) return false;

    const lead = opts.lead ?? 0.7;
    const trail = opts.trail ?? 1.0;
    this.replayStart = Math.max(rec.frames[0].t, w.start - lead);
    this.replayEnd = Math.min(rec.frames[rec.frames.length - 1].t, w.end + trail);
    if (this.replayEnd - this.replayStart < 0.35) return false;

    this.replayT = this.replayStart;
    this.replaySpeed = opts.speed ?? 0.55;
    this.sampleReplay(this.replayT);

    // Base the sweep on the direction of travel at the start of the window.
    _fwd.copy(this.forwardAxis).applyQuaternion(this.replayOrientation);
    _fwd.y = 0;
    this.replayBaseYaw = _fwd.lengthSq() > 1e-6 ? Math.atan2(_fwd.x, _fwd.z) : 0;

    this.mode = CameraMode.Replay;
    this.cineValid = false;
    return true;
  }

  private updateReplay(dt: number): void {
    if (!this.replaySource || this.replayEnd <= this.replayStart) return;

    this.replayT += dt * this.replaySpeed;
    if (this.replayT > this.replayEnd) this.replayT = this.replayStart;
    this.sampleReplay(this.replayT);

    const u = clamp01((this.replayT - this.replayStart) / (this.replayEnd - this.replayStart));
    this.replayProgress = u;

    // A single continuous arc: start wide and behind, sweep around and push in
    // through the apex, then ease back out as the rider comes down. One move,
    // no cuts — cutting inside a two-second clip reads as a bug.
    const yaw = this.replayBaseYaw + lerp(-0.85, 2.05, ease.inOutCubic(u));
    const dist = lerp(16, 7.5, smoothstep(0, 0.55, u)) + lerp(0, 5.5, smoothstep(0.72, 1, u));
    const height = lerp(3.4, 1.7, smoothstep(0.1, 0.62, u)) + lerp(0, 1.8, smoothstep(0.75, 1, u));

    const cy = Math.cos(0.12);
    _desired.set(
      this.replayPosition.x + Math.sin(yaw) * cy * dist,
      this.replayPosition.y + height,
      this.replayPosition.z + Math.cos(yaw) * cy * dist,
    );

    springStepDamped(this.sx, _desired.x, 5.0, 1.0, dt);
    springStepDamped(this.sy, _desired.y, 5.0, 1.0, dt);
    springStepDamped(this.sz, _desired.z, 5.0, 1.0, dt);
    this.camPos.set(this.sx.value, this.sy.value, this.sz.value);

    _lookWanted.copy(this.replayPosition);
    _lookWanted.y += 1.0;
    springStep(this.lx, _lookWanted.x, 9.0, dt);
    springStep(this.ly, _lookWanted.y, 9.0, dt);
    springStep(this.lz, _lookWanted.z, 9.0, dt);
    this.lookPos.set(this.lx.value, this.ly.value, this.lz.value);

    // A slow roll through the arc. Tiny, but it is what turns an orbit into
    // a shot.
    this.roll = lerp(-0.05, 0.06, ease.inOutCubic(u));
  }

  /**
   * Sample the replay trajectory at time `t` into replayPosition/Orientation.
   * Public so the Game can pose a ghost bike on exactly the same curve.
   */
  sampleReplay(t: number): void {
    const rec = this.replaySource;
    if (!rec || rec.frames.length === 0) return;
    const f = rec.frames;

    if (t <= f[0].t) {
      this.applyFrame(f[0]);
      return;
    }
    const last = f[f.length - 1];
    if (t >= last.t) {
      this.applyFrame(last);
      return;
    }

    // Binary search — replay windows are thousands of frames long and a linear
    // scan here would show up on the results screen.
    let lo = 0;
    let hi = f.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (f[mid].t <= t) lo = mid;
      else hi = mid;
    }
    const a = f[lo];
    const b = f[hi];
    const span = b.t - a.t;
    const k = span > 1e-6 ? (t - a.t) / span : 0;

    this.replayPosition.set(
      a.px + (b.px - a.px) * k,
      a.py + (b.py - a.py) * k,
      a.pz + (b.pz - a.pz) * k,
    );
    _qa.set(a.qx, a.qy, a.qz, a.qw);
    _qb.set(b.qx, b.qy, b.qz, b.qw);
    this.replayOrientation.copy(_qa).slerp(_qb, k);
  }

  private applyFrame(f: ReplayFrame): void {
    this.replayPosition.set(f.px, f.py, f.pz);
    this.replayOrientation.set(f.qx, f.qy, f.qz, f.qw);
  }

  // ── Compose ───────────────────────────────────────────────────────────────

  private compose(dt: number): void {
    this.applyShake(dt);

    _camFinal.copy(this.camPos).add(this.shakeOffset);
    _lookFinal.copy(this.lookPos);

    this.resolveCollision(_camFinal, _lookFinal, dt);

    this.camera.position.copy(_camFinal);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(_lookFinal);

    const roll = this.roll + this.shakeRoll;
    if (Math.abs(roll) > 1e-5) this.camera.rotateZ(roll);
    this.camera.updateMatrixWorld();
  }

  /**
   * Keep the camera out of the hillside.
   *
   * Sampled along the segment from the look point to the camera rather than at
   * the camera alone, because the failure that actually happens is a ridge
   * BETWEEN the two — the camera is over open air, the rider is fine, and the
   * shot is a wall of rock. Lifting the camera end by L raises the segment at
   * parameter s by L*s, so the lift a violation at s demands is its depth over
   * s; taking the max over the samples solves all of them at once.
   */
  private resolveCollision(cam: Vector3, look: Vector3, dt: number): void {
    if (!this.terrain) return;

    let need = 0;
    const N = CAMERA_TUNING.collisionSamples;
    for (let i = 1; i <= N; i++) {
      const s = i / N;
      const px = look.x + (cam.x - look.x) * s;
      const pz = look.z + (cam.z - look.z) * s;
      const py = look.y + (cam.y - look.y) * s;
      const h = this.terrain.heightAt(px, pz);
      const want = h + CAMERA_TUNING.collisionMargin;
      if (py < want) need = Math.max(need, (want - py) / s);
    }

    // Asymmetric. Push out in about two frames — a camera inside a hillside is
    // a hard failure and there is no elegant version of it. Come back over a
    // third of a second, because a camera that drops the instant a ridge clears
    // is a visible pop and the eye catches it every time.
    this.collisionLift =
      need > this.collisionLift
        ? dampHL(this.collisionLift, need, 0.030, dt)
        : dampHL(this.collisionLift, need, 0.30, dt);

    if (this.collisionLift > 1e-4) cam.y += this.collisionLift;
  }

  dispose(): void {
    this.replaySource = null;
    this.subject = null;
    this.onLandingEvent = null;
    this.onCrashEvent = null;
  }
}
