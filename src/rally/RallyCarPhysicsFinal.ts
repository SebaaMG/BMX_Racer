/**
 * Final rally physics integration.
 *
 * RallyCarPhysics remains the stable handling model. This class adds the car
 * information that the old two-wheel contract cannot express: four terrain
 * samples, four loads, four surfaces and a signed slip angle. The historical
 * front/rear WheelStates are rebuilt as axle aggregates every step so existing
 * audio, HUD and camera code remains compatible.
 */

import { Vector3 } from 'three';

import type { BikeInput, TerrainSample, WheelState } from '../game/Contracts';
import { clamp, clamp01 } from '../core/MathX';
import {
  DEFAULT_SURFACE,
  makeTerrainSample,
} from '../bike/Wheel';
import {
  RallyCarPhysics as BaseRallyCarPhysics,
  RALLY_TUNE,
  type RallyCarPhysicsOptions,
  type RallyCarState as BaseRallyCarState,
} from './RallyCarPhysics';
import type {
  RallyContactBikeState,
  RallyWheelTuple,
} from './RallyContactPatch';

export { RALLY_TUNE };
export type { RallyCarPhysicsOptions };

export interface RallyCarState extends BaseRallyCarState, RallyContactBikeState {}

const LOCAL_FORWARD = new Vector3(0, 0, 1);
const LOCAL_LEFT = new Vector3(1, 0, 0);
const _forward = new Vector3();
const _left = new Vector3();
const _offset = new Vector3();
const _wheelCenter = new Vector3();

function wheelState(): WheelState {
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

function averageAxle(out: WheelState, a: WheelState, b: WheelState): void {
  out.contactPoint.copy(a.contactPoint).add(b.contactPoint).multiplyScalar(0.5);
  out.contactNormal.copy(a.contactNormal).add(b.contactNormal);
  if (out.contactNormal.lengthSq() < 1e-8) out.contactNormal.set(0, 1, 0);
  else out.contactNormal.normalize();
  out.grounded = a.grounded || b.grounded;
  out.compression = (a.compression + b.compression) * 0.5;
  out.compressionVelocity = (a.compressionVelocity + b.compressionVelocity) * 0.5;
  out.spin = (a.spin + b.spin) * 0.5;
  out.spinRate = (a.spinRate + b.spinRate) * 0.5;
  out.slipRatio = (a.slipRatio + b.slipRatio) * 0.5;
  out.lateralSlip = (a.lateralSlip + b.lateralSlip) * 0.5;
  out.surface = a.load >= b.load ? a.surface : b.surface;
  out.load = a.load + b.load;
}

export class RallyCarPhysics extends BaseRallyCarPhysics {
  declare readonly state: RallyCarState;

  readonly frontLeft = wheelState();
  readonly frontRight = wheelState();
  readonly rearLeft = wheelState();
  readonly rearRight = wheelState();
  readonly rallyWheels: RallyWheelTuple = [
    this.frontLeft,
    this.frontRight,
    this.rearLeft,
    this.rearRight,
  ];

  private readonly patchSamples: readonly [TerrainSample, TerrainSample, TerrainSample, TerrainSample] = [
    makeTerrainSample(),
    makeTerrainSample(),
    makeTerrainSample(),
    makeTerrainSample(),
  ];

  constructor(opts: RallyCarPhysicsOptions = {}) {
    super(opts);
    Object.assign(this.state, {
      rallyWheels: this.rallyWheels,
      rallyLateralVelocity: 0,
      rallySlipAngle: 0,
    });
    this.updateContactPatches(1 / 120);
  }

  override step(input: BikeInput, dt: number): void {
    super.step(input, dt);
    this.updateContactPatches(dt);
  }

  override reset(position: Vector3, forward: Vector3): void {
    super.reset(position, forward);
    this.updateContactPatches(1 / 120);
  }

  /** Used by the car-footprint contact solver for a short, physical yaw nudge. */
  applyRallyContact(normal: Vector3, deltaVelocity: number, yawDelta: number): void {
    this.state.velocity.addScaledVector(normal, deltaVelocity);
    const internals = this as unknown as { yaw: number; bodyRoll: number };
    internals.yaw += yawDelta;
    internals.bodyRoll = clamp(internals.bodyRoll - yawDelta * 0.32, -RALLY_TUNE.bodyRoll, RALLY_TUNE.bodyRoll);
    this.state.angularVelocity.y += yawDelta * 7.5;
  }

  private updateContactPatches(dt: number): void {
    const s = this.state;
    _forward.copy(LOCAL_FORWARD).applyQuaternion(s.orientation);
    _forward.y = 0;
    if (_forward.lengthSq() < 1e-8) _forward.set(0, 0, 1);
    else _forward.normalize();
    _left.copy(LOCAL_LEFT).applyQuaternion(s.orientation);
    _left.y = 0;
    if (_left.lengthSq() < 1e-8) _left.set(1, 0, 0);
    else _left.normalize();

    const lateral = s.velocity.dot(_left);
    const longitudinal = s.velocity.dot(_forward);
    s.rallyLateralVelocity = lateral;
    s.rallySlipAngle = Math.atan2(lateral, Math.max(Math.abs(longitudinal), 0.75));

    // Positive transfer loads the right/outside tyres in a left turn.
    const transfer = clamp(
      s.angularVelocity.y * s.speed * 0.018 + lateral * 0.020,
      -0.34,
      0.34,
    );

    this.updatePatch(this.frontLeft, this.patchSamples[0], true, 1, transfer, dt);
    this.updatePatch(this.frontRight, this.patchSamples[1], true, -1, transfer, dt);
    this.updatePatch(this.rearLeft, this.patchSamples[2], false, 1, transfer, dt);
    this.updatePatch(this.rearRight, this.patchSamples[3], false, -1, transfer, dt);

    averageAxle(this.front, this.frontLeft, this.frontRight);
    averageAxle(this.rear, this.rearLeft, this.rearRight);
  }

  private updatePatch(
    patch: WheelState,
    sample: TerrainSample,
    front: boolean,
    side: 1 | -1,
    transfer: number,
    dt: number,
  ): void {
    const s = this.state;
    const axleZ = (front ? 1 : -1) * RALLY_TUNE.wheelbase * 0.5;
    const sideX = side * RALLY_TUNE.trackWidth * 0.5;
    _offset.set(sideX, -RALLY_TUNE.rideHeight, axleZ).applyQuaternion(s.orientation);
    _wheelCenter.copy(s.position).add(_offset);
    this.terrain.sampleAt(_wheelCenter.x, _wheelCenter.z, sample);

    const axle = front ? this.front : this.rear;
    const previousCompression = patch.compression;
    const compression = clamp01(
      (sample.height + RALLY_TUNE.wheelRadius - _wheelCenter.y) /
      RALLY_TUNE.suspensionTravel + 0.34,
    );

    patch.contactPoint.set(_wheelCenter.x, sample.height, _wheelCenter.z);
    patch.contactNormal.copy(sample.normal);
    if (patch.contactNormal.lengthSq() < 1e-8) patch.contactNormal.set(0, 1, 0);
    else patch.contactNormal.normalize();
    patch.grounded = axle.grounded;
    patch.compression = patch.grounded ? compression : 0.06;
    patch.compressionVelocity = (patch.compression - previousCompression) / Math.max(dt, 1e-4);
    patch.spin = axle.spin;
    patch.spinRate = axle.spinRate;
    patch.slipRatio = axle.slipRatio;
    patch.lateralSlip = s.rallyLateralVelocity + s.angularVelocity.y * axleZ;
    patch.surface = sample.surface;

    const axleLoad = axle.load;
    const sideFactor = clamp(1 - side * transfer, 0.28, 1.72);
    patch.load = patch.grounded ? axleLoad * 0.5 * sideFactor : 0;
  }
}
