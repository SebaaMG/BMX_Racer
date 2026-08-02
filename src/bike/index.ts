/**
 * Vehicle seam.
 *
 * The public class is still named `Bike` because the shared game contracts,
 * replay format and several mature subsystems use that historical name. The
 * implementation is now a procedural four-wheel rally car: RallyCarPhysics +
 * RallyCarVisual. Keeping the seam stable lets the conversion retain the
 * original mountain, race director, AI, camera, NPR pipeline, ghost and HUD
 * without a risky whole-project rename.
 */

import { Color, Object3D, Quaternion, Vector3 } from 'three';

import {
  BikeMode,
  TrickKind,
  type BikeAnchors,
  type BikeInput,
  type IBike,
  type TrickState,
} from '../game/Contracts';
import { clamp01 } from '../core/MathX';
import {
  RallyCarPhysics,
  RALLY_TUNE,
  type RallyCarPhysicsOptions,
  type RallyCarState,
} from '../rally/RallyCarPhysics';
import {
  RallyCarVisual,
  type RallyCarVisualState,
} from '../rally/RallyCarVisual';
import type { BikeTerrain } from './Wheel';

// Keep the legacy low-level exports available for physics harnesses and tools.
// The game factory below no longer instantiates them.
export { BikePhysics, BODY_TUNE, buildOrientation } from './BikePhysics';
export type { BikeStateEx } from './BikePhysics';
export { BikeVisual, crankRate } from './BikeVisual';
export type { BikeVisualState, BikeVisualOptions } from './BikeVisual';
export { Wheel, SURFACE_TABLE, createFallbackTerrain, slipCurve } from './Wheel';
export type { BikeTerrain, WheelConfig, TyreConfig } from './Wheel';
export { BIKE_GEOM, getBikeGeometries } from './BikeModel';
export { TrickSystem, TRICK_TUNE } from './TrickSystem';
export { RallyCarPhysics, RALLY_TUNE } from '../rally/RallyCarPhysics';
export type { RallyCarPhysicsOptions, RallyCarState } from '../rally/RallyCarPhysics';
export { RallyCarVisual } from '../rally/RallyCarVisual';
export type { RallyCarVisualState, RallyCarVisualOptions } from '../rally/RallyCarVisual';

const _position = new Vector3();
const _orientation = new Quaternion();
const _cameraPosition = new Vector3();
const _normal = new Vector3();

const EMPTY_INPUT: BikeInput = {
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

export interface BikeOptions extends RallyCarPhysicsOptions {
  /** Rally body colour. Kept as frameColor for RacerSpec compatibility. */
  frameColor: Color | number;
  accentColor?: Color | number;
  name?: string;
  detail?: 'full' | 'reduced';
  parent?: Object3D;
}

/**
 * Compatibility wrapper used by the rest of the game.
 *
 * Consumers still receive an IBike, but every visible and physical operation is
 * delegated to rally-car systems. This avoids brittle casts and keeps the
 * capture harness, effects and race director functioning without special cases.
 */
export class Bike implements IBike {
  readonly physics: RallyCarPhysics;
  readonly visual: RallyCarVisual;
  readonly object: Object3D;
  readonly anchors: BikeAnchors;
  readonly isRallyCar = true;

  /**
   * RacerBase adopts this state instead of its BMX trick driver. Keeping it at
   * None prevents tailwhip/tabletop/manual labels leaking into the rally HUD;
   * boost is earned directly by controlled drifts and clean landings in physics.
   */
  readonly trick: TrickState = {
    kind: TrickKind.None,
    phase: 0,
    rotations: 0,
    pendingScore: 0,
    committed: true,
  };

  private cameraRef: Object3D | null = null;
  private linkedTrick: TrickState | null = null;
  private readonly lastInput: BikeInput = { ...EMPTY_INPUT };

  private readonly visualState: RallyCarVisualState = {
    steerAngle: 0,
    frontCompression: 0,
    rearCompression: 0,
    frontSpin: 0,
    rearSpin: 0,
    speed: 0,
    lateralSlip: 0,
    boosting: false,
    brake: 0,
    crashed: false,
  };

  constructor(opts: BikeOptions) {
    const bodyColor = opts.frameColor instanceof Color ? opts.frameColor : new Color(opts.frameColor);
    const accentColor = opts.accentColor instanceof Color
      ? opts.accentColor
      : opts.accentColor !== undefined
        ? new Color(opts.accentColor)
        : bodyColor.clone().offsetHSL(0.08, 0.10, 0.15);

    this.physics = new RallyCarPhysics(opts);
    this.visual = new RallyCarVisual({
      frameColor: bodyColor,
      accentColor,
      detail: opts.detail,
      name: opts.name?.replace(/^bike:/, 'rally-car:'),
    });
    this.object = this.visual.root;
    this.anchors = this.visual.anchors;
    syncOutlineTransforms(this.object);
    opts.parent?.add(this.object);
  }

  get state(): RallyCarState {
    return this.physics.state;
  }

  set terrain(terrain: BikeTerrain) {
    this.physics.terrain = terrain;
  }

  setCameraRef(camera: Object3D | null): void {
    this.cameraRef = camera;
  }

  /** Retained for the race layer and future rally-specific style telemetry. */
  linkTrick(trick: TrickState): void {
    this.linkedTrick = trick;
  }

  step(input: BikeInput, dt: number): void {
    copyInput(input, this.lastInput);
    this.physics.step(input, dt);
  }

  noteInput(input: BikeInput): void {
    copyInput(input, this.lastInput);
  }

  updateVisual(alpha: number, dt: number): void {
    const s = this.physics.state;
    this.physics.interpolatedPosition(alpha, _position);
    this.physics.interpolatedOrientation(alpha, _orientation);
    this.object.position.copy(_position);
    this.object.quaternion.copy(_orientation);

    const v = this.visualState;
    v.steerAngle = s.steerAngle;
    v.frontCompression = this.physics.interpolatedFrontCompression(alpha);
    v.rearCompression = this.physics.interpolatedRearCompression(alpha);
    v.frontSpin = this.physics.interpolatedFrontSpin(alpha);
    v.rearSpin = this.physics.interpolatedRearSpin(alpha);
    v.speed = s.speed;
    v.lateralSlip = Math.abs(s.rear.lateralSlip);
    v.boosting = s.boosting;
    v.brake = clamp01(this.lastInput.brakeFront * 0.72 + this.lastInput.brakeRear * 0.62);
    v.crashed = s.mode === BikeMode.Crashing;

    const cameraDistance = this.cameraRef
      ? this.cameraRef.getWorldPosition(_cameraPosition).distanceTo(_position)
      : 7;
    this.visual.update(v, dt, cameraDistance);
    this.visual.setContacts(s.front.contactPoint, s.rear.contactPoint);

    void this.linkedTrick;
  }

  reset(position: Vector3, forward: Vector3): void {
    this.physics.reset(position, forward);
    copyInput(EMPTY_INPUT, this.lastInput);
    this.updateVisual(1, 1 / 60);
  }

  dispose(): void {
    this.visual.dispose();
  }
}

export function createBike(opts: BikeOptions): Bike {
  return new Bike(opts);
}

/** Place a rally car on the terrain at axle height, aligned down-course. */
export function seatBikeOnTerrain(
  car: Bike,
  terrain: BikeTerrain,
  x: number,
  z: number,
  forward: Vector3,
): void {
  const height = terrain.heightAt(x, z);
  terrain.normalAt(x, z, _normal);
  if (_normal.lengthSq() < 1e-8) _normal.set(0, 1, 0);
  else _normal.normalize();
  _position.set(x, height, z)
    .addScaledVector(_normal, RALLY_TUNE.wheelRadius + RALLY_TUNE.rideHeight);
  car.reset(_position, forward);
}

function copyInput(src: BikeInput, dst: BikeInput): void {
  dst.steer = src.steer;
  dst.pedal = src.pedal;
  dst.brakeRear = src.brakeRear;
  dst.brakeFront = src.brakeFront;
  dst.crouch = src.crouch;
  dst.pitchLean = src.pitchLean;
  dst.airPitch = src.airPitch;
  dst.airYaw = src.airYaw;
  dst.airRoll = src.airRoll;
  dst.wantBoost = src.wantBoost;
  dst.wantHop = src.wantHop;
}

/** attachOutline creates a sibling; copy authored transforms onto that sibling. */
function syncOutlineTransforms(root: Object3D): void {
  root.traverse((node) => {
    const hull = node.userData.hull as Object3D | undefined;
    if (!hull) return;
    hull.position.copy(node.position);
    hull.quaternion.copy(node.quaternion);
    hull.scale.copy(node.scale);
  });
}
