/**
 * RallyCarPhysics — four-wheel arcade rally handling behind the existing
 * vehicle contract.
 *
 * The rest of DESCENT deliberately talks to an abstract BikeState/IBike seam.
 * Keeping that seam means the mountain, race director, camera, effects, replay,
 * ghost and HUD can stay battle-tested while the vehicle underneath becomes a
 * rally car. The old type names are historical implementation details; every
 * physical decision in this class is car-specific.
 *
 * The model is intentionally game-first:
 *  - a stable sprung chassis follows the terrain normal while grounded;
 *  - steering changes heading, while lateral velocity is retained long enough
 *    to produce readable Scandinavian-flick drifts;
 *  - grip, rolling drag and dust still come from the authored surface table;
 *  - crests, jumps, landings, crashes and recovery remain fully physical;
 *  - all hot-path scratch storage is allocated once.
 */

import { Matrix4, Object3D, Quaternion, Vector3 } from 'three';

import {
  BikeMode,
  type BikeInput,
  type BikeState,
  type SurfaceProperties,
  type TerrainSample,
  type WheelState,
} from '../game/Contracts';
import { clamp, clamp01, dampHL, lerp, moveTowards } from '../core/MathX';
import {
  DEFAULT_SURFACE,
  makeTerrainSample,
  type BikeTerrain,
} from '../bike/Wheel';

export const RALLY_TUNE = {
  mass: 1180,
  wheelbase: 2.56,
  trackWidth: 1.64,
  wheelRadius: 0.345,
  rideHeight: 0.315,
  suspensionTravel: 0.245,

  gravity: 20.4,
  engineForce: 9850,
  boostForce: 5200,
  brakeFront: 11800,
  brakeRear: 9200,
  dragK: 0.55,
  rollingScale: 145,
  maxSpeed: 43,

  maxSteer: 0.54,
  steerRate: 3.9,
  highSpeedSteer: 0.42,
  yawResponse: 0.86,
  maxYawRate: 2.65,

  lateralGrip: 10.8,
  handbrakeGrip: 0.28,
  bodyRoll: 0.17,
  bodyPitch: 0.085,

  airPitchRate: 2.2,
  airYawRate: 1.65,
  airRollRate: 1.75,
  airAuthority: 4.6,
  launchClearance: 0.09,
  landingCrashSpeed: 17.5,
  recoveryTime: 1.35,

  boostDrain: 0.34,
  driftCharge: 0.018,
  cleanLandingCharge: 0.12,
} as const;

export interface RallyCarPhysicsOptions {
  terrain?: BikeTerrain;
  mass?: number;
  /** AI cars get a small grip/heading-assist increase, never raw velocity. */
  stabilityBias?: number;
}

export interface RallyCarState extends BikeState {
  /** Monotonic render-safe event counters, matching the previous implementation. */
  landCount: number;
  crashCount: number;
}

const UP = new Vector3(0, 1, 0);
const LOCAL_FORWARD = new Vector3(0, 0, 1);
const LOCAL_LEFT = new Vector3(1, 0, 0);

const _frontPos = new Vector3();
const _rearPos = new Vector3();
const _forward = new Vector3();
const _left = new Vector3();
const _normal = new Vector3();
const _groundForward = new Vector3();
const _groundLeft = new Vector3();
const _gravityTangent = new Vector3();
const _velocity = new Vector3();
const _candidate = new Vector3();
const _axis = new Vector3();
const _basis = new Matrix4();
const _targetQ = new Quaternion();
const _rollQ = new Quaternion();
const _stepQ = new Quaternion();
const _crashDir = new Vector3();

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

function makeWheelState(): WheelState {
  return {
    contactPoint: new Vector3(),
    contactNormal: new Vector3(0, 1, 0),
    grounded: true,
    compression: 0.32,
    compressionVelocity: 0,
    spin: 0,
    spinRate: 0,
    slipRatio: 0,
    lateralSlip: 0,
    surface: DEFAULT_SURFACE,
    load: 0,
  };
}

function horizontalForward(yaw: number, out: Vector3): Vector3 {
  return out.set(Math.sin(yaw), 0, Math.cos(yaw));
}

function signedMove(value: number, amount: number): number {
  if (value > 0) return Math.max(0, value - amount);
  if (value < 0) return Math.min(0, value + amount);
  return 0;
}

export class RallyCarPhysics {
  readonly state: RallyCarState;
  readonly object = new Object3D();
  readonly front = makeWheelState();
  readonly rear = makeWheelState();

  terrain: BikeTerrain;
  mass: number;

  readonly prevPosition = new Vector3();
  readonly prevOrientation = new Quaternion();

  private readonly frontSample: TerrainSample = makeTerrainSample();
  private readonly rearSample: TerrainSample = makeTerrainSample();
  private readonly nextFrontSample: TerrainSample = makeTerrainSample();
  private readonly nextRearSample: TerrainSample = makeTerrainSample();

  private stability: number;
  private yaw = 0;
  private steer = 0;
  private bodyRoll = 0;
  private bodyPitch = 0;
  private prevFrontCompression = 0.32;
  private prevRearCompression = 0.32;
  private prevFrontSpin = 0;
  private prevRearSpin = 0;
  private frontSpin = 0;
  private rearSpin = 0;
  private frontCompression = 0.32;
  private rearCompression = 0.32;
  private launchGrace = 0;
  private crashClock = 0;
  private throttlePrev = 0;
  private lastSupportY = 0;

  constructor(opts: RallyCarPhysicsOptions = {}) {
    this.terrain = opts.terrain ?? {
      heightAt: () => 0,
      normalAt: (_x, _z, out = new Vector3()) => out.set(0, 1, 0),
      sampleAt: (_x, _z, out = makeTerrainSample()) => {
        out.height = 0;
        out.normal.set(0, 1, 0);
        out.slope = 0;
        out.surface = DEFAULT_SURFACE;
        return out;
      },
    };
    this.mass = opts.mass ?? RALLY_TUNE.mass;
    this.stability = clamp01(opts.stabilityBias ?? 0);

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
      boost: 0.62,
      boosting: false,
      landedThisStep: false,
      landingImpact: 0,
      landingQuality: 1,
      crashedThisStep: false,
      crashDirection: new Vector3(0, 0, -1),
      crashSeverity: 0,
      landCount: 0,
      crashCount: 0,
    };
  }

  get surface(): SurfaceProperties {
    return this.rear.surface;
  }

  step(inputRaw: BikeInput, dt: number): void {
    if (!(dt > 0) || !Number.isFinite(dt)) return;

    const s = this.state;
    this.prevPosition.copy(s.position);
    this.prevOrientation.copy(s.orientation);
    this.prevFrontCompression = this.frontCompression;
    this.prevRearCompression = this.rearCompression;
    this.prevFrontSpin = this.frontSpin;
    this.prevRearSpin = this.rearSpin;

    s.landedThisStep = false;
    s.crashedThisStep = false;
    s.modeTime += dt;
    this.launchGrace = Math.max(0, this.launchGrace - dt);

    if (s.mode === BikeMode.Crashing) {
      this.stepCrash(dt);
      this.updateContacts(dt, false);
      this.finishState();
      return;
    }

    const input = s.mode === BikeMode.Finished ? ZERO_INPUT : inputRaw;
    if (s.mode === BikeMode.Airborne) this.stepAir(input, dt);
    else this.stepGround(input, dt);

    this.finishState();
  }

  private stepGround(input: BikeInput, dt: number): void {
    const s = this.state;
    this.sampleAxles(s.position, this.frontSample, this.rearSample);
    this.supportFrame(this.frontSample, this.rearSample, _normal, _groundForward, _groundLeft);

    const currentRideY = this.rideHeightAt(s.position, this.frontSample, this.rearSample);
    if (!Number.isFinite(this.lastSupportY)) this.lastSupportY = currentRideY;

    // Steering angle is speed-sensitive, but never disappears at pace.
    const speed01 = clamp01(Math.abs(s.forwardSpeed) / RALLY_TUNE.maxSpeed);
    const steerLimit = RALLY_TUNE.maxSteer * lerp(1, RALLY_TUNE.highSpeedSteer, speed01);
    this.steer = moveTowards(
      this.steer,
      clamp(input.steer, -1, 1) * steerLimit,
      RALLY_TUNE.steerRate * dt,
    );

    horizontalForward(this.yaw, _forward);
    _left.copy(UP).cross(_forward).normalize();

    let longitudinal = s.velocity.dot(_groundForward);
    let lateral = s.velocity.dot(_groundLeft);

    const surface = this.rearSample.surface;
    const throttle = clamp01(input.pedal);
    const speedAbs = Math.abs(longitudinal);
    const speedFade = clamp01(1 - speedAbs / RALLY_TUNE.maxSpeed);
    let drive = throttle * RALLY_TUNE.engineForce * (0.32 + speedFade * 0.68);

    s.boosting = input.wantBoost && s.boost > 0.015 && s.mode !== BikeMode.Finished;
    if (s.boosting) {
      drive += RALLY_TUNE.boostForce;
      s.boost = Math.max(0, s.boost - RALLY_TUNE.boostDrain * dt);
    }

    const brake = clamp01(input.brakeFront * 0.68 + input.brakeRear * 0.52);
    const brakeForce = input.brakeFront * RALLY_TUNE.brakeFront + input.brakeRear * RALLY_TUNE.brakeRear;
    const driveAccel = drive / this.mass;
    const brakeAccel = brakeForce / this.mass;

    longitudinal += driveAccel * dt;
    longitudinal = signedMove(longitudinal, brakeAccel * dt);

    const rolling = surface.rollingResistance * RALLY_TUNE.rollingScale / this.mass;
    longitudinal = signedMove(longitudinal, rolling * dt);

    // Gravity along the terrain is what makes this still feel like a descent.
    _gravityTangent.set(0, -RALLY_TUNE.gravity, 0)
      .addScaledVector(_normal, RALLY_TUNE.gravity * _normal.y);
    longitudinal += _gravityTangent.dot(_groundForward) * dt;
    lateral += _gravityTangent.dot(_groundLeft) * dt;

    // Aerodynamic drag is quadratic and acts on both components.
    const planarSpeed = Math.hypot(longitudinal, lateral);
    if (planarSpeed > 1e-4) {
      const dragDv = Math.min(planarSpeed, (RALLY_TUNE.dragK * planarSpeed * planarSpeed / this.mass) * dt);
      const scale = (planarSpeed - dragDv) / planarSpeed;
      longitudinal *= scale;
      lateral *= scale;
    }

    // Rear brake deliberately releases the rear axle. The remaining lateral
    // velocity is what draws a visible, controllable rally slide instead of
    // rotating the velocity vector instantly with the car.
    const handbrake = clamp01(input.brakeRear * 1.15);
    const gripScale = lerp(1, RALLY_TUNE.handbrakeGrip, handbrake);
    const stabilityGain = 1 + this.stability * 0.42;
    const lateralRate = RALLY_TUNE.lateralGrip * surface.grip * gripScale * stabilityGain;
    lateral *= Math.exp(-lateralRate * dt);

    const speedForYaw = Math.max(Math.abs(longitudinal), 1.5);
    const rawYawRate = -(Math.tan(this.steer) * speedForYaw / RALLY_TUNE.wheelbase);
    const yawRate = clamp(rawYawRate * RALLY_TUNE.yawResponse, -RALLY_TUNE.maxYawRate, RALLY_TUNE.maxYawRate);
    this.yaw += yawRate * dt * Math.sign(longitudinal || 1);
    s.angularVelocity.set(0, yawRate, 0);

    // Rebuild the terrain frame after heading changed. This preserves momentum
    // through a flick: heading moves first, velocity catches it through grip.
    horizontalForward(this.yaw, _forward);
    _groundForward.copy(_forward).addScaledVector(_normal, -_forward.dot(_normal)).normalize();
    _groundLeft.copy(_normal).cross(_groundForward).normalize();
    s.velocity.copy(_groundForward).multiplyScalar(longitudinal)
      .addScaledVector(_groundLeft, lateral);

    // Spring-like body language without making the chassis dynamically unstable.
    const rollTarget = clamp(
      -yawRate * Math.abs(longitudinal) * 0.0105 + lateral * 0.010,
      -RALLY_TUNE.bodyRoll,
      RALLY_TUNE.bodyRoll,
    );
    const pitchTarget = clamp(
      (brake * 0.070) - (throttle * 0.035),
      -RALLY_TUNE.bodyPitch,
      RALLY_TUNE.bodyPitch,
    );
    this.bodyRoll = dampHL(this.bodyRoll, rollTarget, 0.105, dt);
    this.bodyPitch = dampHL(this.bodyPitch, pitchTarget, 0.13, dt);

    // Crouch is retained as the launch/preload channel so all existing input,
    // AI and capture scripts remain useful. In a car it reads as suspension
    // loading before a crest rather than a rider crouching.
    s.preload = dampHL(s.preload, clamp01(input.crouch), input.crouch > s.preload ? 0.09 : 0.045, dt);
    if (input.crouch > 0.55) s.pumpCharge = Math.min(1, s.pumpCharge + dt * 2.1);
    else s.pumpCharge = Math.max(0, s.pumpCharge - dt * 1.3);

    const released = this.throttlePrev > 0.58 && input.crouch < 0.28;
    this.throttlePrev = input.crouch;
    if ((released && s.pumpCharge > 0.12) || input.wantHop) {
      const impulse = input.wantHop ? 3.4 : 1.8 + s.pumpCharge * 2.15;
      s.velocity.addScaledVector(_normal, impulse);
      s.pumpCharge = 0;
      this.enterAir();
    }

    _candidate.copy(s.position).addScaledVector(s.velocity, dt);
    this.sampleAxles(_candidate, this.nextFrontSample, this.nextRearSample);
    const nextRideY = this.rideHeightAt(_candidate, this.nextFrontSample, this.nextRearSample);

    // On a smooth descent the tangent velocity arrives at the next support
    // height. At a lip it does not: the ground falls away while the chassis
    // continues along the launch tangent, which is the correct takeoff test.
    const clearance = _candidate.y - nextRideY;
    if (s.mode === BikeMode.Airborne || (clearance > RALLY_TUNE.launchClearance && speedForYaw > 6.5)) {
      s.position.copy(_candidate);
      this.enterAir();
      this.updateContacts(dt, false);
      return;
    }

    s.position.copy(_candidate);
    s.position.y = dampHL(s.position.y, nextRideY, 0.032, dt);
    this.lastSupportY = nextRideY;

    this.sampleAxles(s.position, this.frontSample, this.rearSample);
    this.supportFrame(this.frontSample, this.rearSample, _normal, _groundForward, _groundLeft);
    this.composeGroundOrientation(_normal, _groundForward);

    s.mode = s.mode === BikeMode.Recovering ? BikeMode.Recovering : BikeMode.Grounded;
    if (s.mode === BikeMode.Recovering && s.modeTime > 0.42) {
      s.mode = BikeMode.Grounded;
      s.modeTime = 0;
    }
    s.airTime = 0;
    s.airHeight = 0;
    s.peakAirHeight = 0;
    s.lean = this.bodyRoll;
    s.pitch = Math.asin(clamp(-_groundForward.y, -1, 1)) + this.bodyPitch;
    s.steerAngle = this.steer;
    s.manualling = handbrake > 0.45 && Math.abs(lateral) > 2.5;
    s.manualAmount = clamp01(Math.abs(lateral) / 9) * handbrake;

    const drift = Math.abs(lateral);
    if (drift > 2.4 && speedForYaw > 8) {
      s.boost = Math.min(1, s.boost + (drift - 2.4) * RALLY_TUNE.driftCharge * dt);
    }

    this.updateContacts(dt, true);
  }

  private stepAir(input: BikeInput, dt: number): void {
    const s = this.state;
    s.mode = BikeMode.Airborne;
    s.airTime += dt;
    s.modeTime += 0;

    s.velocity.y -= RALLY_TUNE.gravity * dt;
    const drag = Math.exp(-0.055 * dt * Math.max(1, s.velocity.length()));
    s.velocity.multiplyScalar(drag);

    // Existing air axes now mean car attitude correction. The rates are much
    // lower than the BMX values, so a crest can be trimmed without turning the
    // rally car into a stunt toy.
    _axis.set(
      input.airPitch * RALLY_TUNE.airPitchRate,
      -input.airYaw * RALLY_TUNE.airYawRate,
      input.airRoll * RALLY_TUNE.airRollRate,
    );
    s.angularVelocity.x = dampHL(s.angularVelocity.x, _axis.x, 0.20, dt);
    s.angularVelocity.y = dampHL(s.angularVelocity.y, _axis.y, 0.24, dt);
    s.angularVelocity.z = dampHL(s.angularVelocity.z, _axis.z, 0.20, dt);

    const omega = s.angularVelocity.length();
    if (omega > 1e-5) {
      _axis.copy(s.angularVelocity).multiplyScalar(1 / omega);
      _stepQ.setFromAxisAngle(_axis, omega * dt * RALLY_TUNE.airAuthority * 0.22);
      s.orientation.premultiply(_stepQ).normalize();
    }

    s.position.addScaledVector(s.velocity, dt);
    this.sampleAxles(s.position, this.frontSample, this.rearSample);
    const rideY = this.rideHeightAt(s.position, this.frontSample, this.rearSample);
    this.supportFrame(this.frontSample, this.rearSample, _normal, _groundForward, _groundLeft);

    s.airHeight = Math.max(0, s.position.y - rideY);
    s.peakAirHeight = Math.max(s.peakAirHeight, s.airHeight);
    s.forwardSpeed = s.velocity.dot(_forward.copy(LOCAL_FORWARD).applyQuaternion(s.orientation));
    s.speed = s.velocity.length();
    s.lean = this.bodyRoll;
    s.steerAngle = this.steer;

    if (this.launchGrace <= 0 && s.position.y <= rideY && s.velocity.dot(_normal) <= 0.5) {
      const impactSpeed = Math.max(0, -s.velocity.dot(_normal));
      const up = _left.copy(UP).applyQuaternion(s.orientation).normalize();
      const alignment = clamp(up.dot(_normal), -1, 1);
      const impact = clamp01(impactSpeed / RALLY_TUNE.landingCrashSpeed);
      const quality = clamp01((alignment + 0.15) / 1.15);

      s.position.y = rideY;
      s.landedThisStep = true;
      s.landCount++;
      s.landingImpact = impact;
      s.landingQuality = quality;
      s.modeTime = 0;

      if (impactSpeed > RALLY_TUNE.landingCrashSpeed || alignment < 0.42) {
        this.forceCrash(clamp01(Math.max(impact, 1 - quality)));
        return;
      }

      // Remove inward velocity, keep the valuable down-course momentum.
      const vn = s.velocity.dot(_normal);
      if (vn < 0) s.velocity.addScaledVector(_normal, -vn * 0.82);
      this.yaw = Math.atan2(
        _forward.copy(LOCAL_FORWARD).applyQuaternion(s.orientation).x,
        _forward.z,
      );
      this.bodyRoll *= 0.45;
      this.bodyPitch *= 0.4;
      s.boost = Math.min(1, s.boost + RALLY_TUNE.cleanLandingCharge * quality);
      s.mode = BikeMode.Grounded;
      this.lastSupportY = rideY;
      this.composeGroundOrientation(_normal, _groundForward);
      this.updateContacts(dt, true);
      return;
    }

    this.updateContacts(dt, false);
  }

  private stepCrash(dt: number): void {
    const s = this.state;
    this.crashClock += dt;
    s.velocity.y -= RALLY_TUNE.gravity * dt;
    s.velocity.multiplyScalar(Math.exp(-1.8 * dt));
    s.position.addScaledVector(s.velocity, dt);

    const omega = s.angularVelocity.length();
    if (omega > 1e-5) {
      _axis.copy(s.angularVelocity).multiplyScalar(1 / omega);
      _stepQ.setFromAxisAngle(_axis, omega * dt);
      s.orientation.premultiply(_stepQ).normalize();
    }
    s.angularVelocity.multiplyScalar(Math.exp(-1.35 * dt));

    this.sampleAxles(s.position, this.frontSample, this.rearSample);
    const floor = this.rideHeightAt(s.position, this.frontSample, this.rearSample) - 0.20;
    if (s.position.y < floor) {
      s.position.y = floor;
      if (s.velocity.y < 0) s.velocity.y *= -0.16;
      s.velocity.x *= 0.82;
      s.velocity.z *= 0.82;
    }

    s.airHeight = Math.max(0, s.position.y - floor);
    s.speed = s.velocity.length();
    s.forwardSpeed = s.velocity.dot(_forward.copy(LOCAL_FORWARD).applyQuaternion(s.orientation));

    if (this.crashClock >= RALLY_TUNE.recoveryTime && s.speed < 6.5) {
      horizontalForward(this.yaw, _forward);
      const rideY = this.rideHeightAt(s.position, this.frontSample, this.rearSample);
      s.position.y = Math.max(s.position.y, rideY);
      s.orientation.setFromAxisAngle(UP, this.yaw);
      s.angularVelocity.set(0, 0, 0);
      s.mode = BikeMode.Recovering;
      s.modeTime = 0;
      s.crashSeverity = 0;
      this.bodyRoll = 0;
      this.bodyPitch = 0;
      this.lastSupportY = rideY;
    }
  }

  private enterAir(): void {
    const s = this.state;
    if (s.mode !== BikeMode.Airborne) {
      s.mode = BikeMode.Airborne;
      s.modeTime = 0;
      s.airTime = 0;
      s.peakAirHeight = 0;
      this.launchGrace = 0.11;
    }
  }

  forceCrash(severity = 0.75): void {
    const s = this.state;
    if (s.mode === BikeMode.Crashing) return;
    const sev = clamp01(severity);
    s.mode = BikeMode.Crashing;
    s.modeTime = 0;
    s.crashedThisStep = true;
    s.crashSeverity = sev;
    s.crashCount++;
    this.crashClock = 0;

    _forward.copy(LOCAL_FORWARD).applyQuaternion(s.orientation).normalize();
    _crashDir.copy(s.velocity).normalize();
    if (_crashDir.lengthSq() < 1e-5) _crashDir.copy(_forward).multiplyScalar(-1);
    s.crashDirection.copy(_crashDir).multiplyScalar(-1);

    s.angularVelocity.set(
      (0.7 + sev * 1.8) * (Math.random() > 0.5 ? 1 : -1),
      (0.5 + sev * 2.2) * (this.steer >= 0 ? -1 : 1),
      (0.8 + sev * 2.6) * (Math.random() > 0.5 ? 1 : -1),
    );
    s.velocity.addScaledVector(UP, 1.6 + sev * 2.4);
  }

  reset(position: Vector3, forward: Vector3): void {
    const s = this.state;
    this.yaw = Math.atan2(forward.x, forward.z);
    this.steer = 0;
    this.bodyRoll = 0;
    this.bodyPitch = 0;
    this.crashClock = 0;
    this.launchGrace = 0;
    this.throttlePrev = 0;

    s.position.copy(position);
    this.sampleAxles(s.position, this.frontSample, this.rearSample);
    const rideY = this.rideHeightAt(s.position, this.frontSample, this.rearSample);
    if (s.position.y < rideY || Math.abs(s.position.y - rideY) < 1.2) s.position.y = rideY;
    this.lastSupportY = rideY;

    s.velocity.set(0, 0, 0);
    s.orientation.setFromAxisAngle(UP, this.yaw);
    s.angularVelocity.set(0, 0, 0);
    s.forwardSpeed = 0;
    s.speed = 0;
    s.lean = 0;
    s.steerAngle = 0;
    s.pitch = 0;
    s.mode = BikeMode.Grounded;
    s.modeTime = 0;
    s.airHeight = 0;
    s.airTime = 0;
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
    s.landCount = 0;
    s.crashCount = 0;

    this.frontSpin = this.rearSpin = 0;
    this.prevFrontSpin = this.prevRearSpin = 0;
    this.frontCompression = this.rearCompression = 0.32;
    this.prevFrontCompression = this.prevRearCompression = 0.32;
    this.prevPosition.copy(s.position);
    this.prevOrientation.copy(s.orientation);
    this.updateContacts(1 / 120, true);
  }

  interpolatedPosition(alpha: number, out: Vector3): Vector3 {
    return out.copy(this.prevPosition).lerp(this.state.position, clamp01(alpha));
  }

  interpolatedOrientation(alpha: number, out: Quaternion): Quaternion {
    return out.copy(this.prevOrientation).slerp(this.state.orientation, clamp01(alpha));
  }

  interpolatedFrontCompression(alpha: number): number {
    return lerp(this.prevFrontCompression, this.frontCompression, clamp01(alpha));
  }

  interpolatedRearCompression(alpha: number): number {
    return lerp(this.prevRearCompression, this.rearCompression, clamp01(alpha));
  }

  interpolatedFrontSpin(alpha: number): number {
    return lerp(this.prevFrontSpin, this.frontSpin, clamp01(alpha));
  }

  interpolatedRearSpin(alpha: number): number {
    return lerp(this.prevRearSpin, this.rearSpin, clamp01(alpha));
  }

  private finishState(): void {
    const s = this.state;
    _forward.copy(LOCAL_FORWARD).applyQuaternion(s.orientation).normalize();
    s.speed = s.velocity.length();
    s.forwardSpeed = s.velocity.dot(_forward);
    s.steerAngle = this.steer;
    this.object.position.copy(s.position);
    this.object.quaternion.copy(s.orientation);
  }

  private sampleAxles(
    position: Vector3,
    frontOut: TerrainSample,
    rearOut: TerrainSample,
  ): void {
    horizontalForward(this.yaw, _forward);
    _frontPos.copy(position).addScaledVector(_forward, RALLY_TUNE.wheelbase * 0.5);
    _rearPos.copy(position).addScaledVector(_forward, -RALLY_TUNE.wheelbase * 0.5);
    this.terrain.sampleAt(_frontPos.x, _frontPos.z, frontOut);
    this.terrain.sampleAt(_rearPos.x, _rearPos.z, rearOut);
  }

  private supportFrame(
    frontSample: TerrainSample,
    rearSample: TerrainSample,
    normalOut: Vector3,
    forwardOut: Vector3,
    leftOut: Vector3,
  ): void {
    normalOut.copy(frontSample.normal).add(rearSample.normal).normalize();
    if (normalOut.lengthSq() < 1e-8) normalOut.copy(UP);
    horizontalForward(this.yaw, forwardOut);
    forwardOut.addScaledVector(normalOut, -forwardOut.dot(normalOut)).normalize();
    if (forwardOut.lengthSq() < 1e-8) forwardOut.copy(LOCAL_FORWARD);
    leftOut.copy(normalOut).cross(forwardOut).normalize();
  }

  private rideHeightAt(
    _position: Vector3,
    frontSample: TerrainSample,
    rearSample: TerrainSample,
  ): number {
    const ground = (frontSample.height + rearSample.height) * 0.5;
    return ground + RALLY_TUNE.wheelRadius + RALLY_TUNE.rideHeight;
  }

  private composeGroundOrientation(normal: Vector3, groundForward: Vector3): void {
    _groundLeft.copy(normal).cross(groundForward).normalize();
    _basis.makeBasis(_groundLeft, normal, groundForward);
    _targetQ.setFromRotationMatrix(_basis);
    _rollQ.setFromAxisAngle(LOCAL_FORWARD, this.bodyRoll);
    _targetQ.multiply(_rollQ);
    _rollQ.setFromAxisAngle(LOCAL_LEFT, this.bodyPitch);
    _targetQ.multiply(_rollQ);
    this.state.orientation.slerp(_targetQ, 0.42).normalize();
  }

  private updateContacts(dt: number, grounded: boolean): void {
    const s = this.state;
    horizontalForward(this.yaw, _forward);
    _frontPos.copy(s.position).addScaledVector(_forward, RALLY_TUNE.wheelbase * 0.5);
    _rearPos.copy(s.position).addScaledVector(_forward, -RALLY_TUNE.wheelbase * 0.5);

    this.terrain.sampleAt(_frontPos.x, _frontPos.z, this.frontSample);
    this.terrain.sampleAt(_rearPos.x, _rearPos.z, this.rearSample);

    const frontTarget = clamp01(
      (this.frontSample.height + RALLY_TUNE.wheelRadius + RALLY_TUNE.rideHeight - _frontPos.y) /
      RALLY_TUNE.suspensionTravel + 0.36,
    );
    const rearTarget = clamp01(
      (this.rearSample.height + RALLY_TUNE.wheelRadius + RALLY_TUNE.rideHeight - _rearPos.y) /
      RALLY_TUNE.suspensionTravel + 0.36,
    );
    const oldFront = this.frontCompression;
    const oldRear = this.rearCompression;
    this.frontCompression = dampHL(this.frontCompression, grounded ? frontTarget : 0.06, 0.055, dt);
    this.rearCompression = dampHL(this.rearCompression, grounded ? rearTarget : 0.06, 0.065, dt);

    const spinRate = s.forwardSpeed / RALLY_TUNE.wheelRadius;
    this.frontSpin += spinRate * dt;
    this.rearSpin += spinRate * dt * (1 + clamp01(Math.abs(this.rear.lateralSlip) / 12) * 0.08);

    this.front.contactPoint.set(_frontPos.x, this.frontSample.height, _frontPos.z);
    this.rear.contactPoint.set(_rearPos.x, this.rearSample.height, _rearPos.z);
    this.front.contactNormal.copy(this.frontSample.normal);
    this.rear.contactNormal.copy(this.rearSample.normal);
    this.front.grounded = grounded;
    this.rear.grounded = grounded;
    this.front.compression = this.frontCompression;
    this.rear.compression = this.rearCompression;
    this.front.compressionVelocity = (this.frontCompression - oldFront) / Math.max(dt, 1e-4);
    this.rear.compressionVelocity = (this.rearCompression - oldRear) / Math.max(dt, 1e-4);
    this.front.spin = this.frontSpin;
    this.rear.spin = this.rearSpin;
    this.front.spinRate = spinRate;
    this.rear.spinRate = spinRate;
    this.front.surface = this.frontSample.surface;
    this.rear.surface = this.rearSample.surface;
    this.front.load = grounded ? this.mass * RALLY_TUNE.gravity * 0.47 : 0;
    this.rear.load = grounded ? this.mass * RALLY_TUNE.gravity * 0.53 : 0;

    horizontalForward(this.yaw, _groundForward);
    _groundLeft.copy(UP).cross(_groundForward).normalize();
    const lateral = s.velocity.dot(_groundLeft);
    this.front.lateralSlip = lateral * 0.78;
    this.rear.lateralSlip = lateral;
    this.front.slipRatio = clamp(-Math.abs(this.steer) * Math.abs(lateral) * 0.025, -1, 1);
    this.rear.slipRatio = clamp((s.boosting ? 0.22 : 0) + Math.abs(lateral) * 0.018, -1, 1);
  }
}
