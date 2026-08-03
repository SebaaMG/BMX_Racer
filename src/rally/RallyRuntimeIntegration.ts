/**
 * Runtime integration for systems whose public contracts still use bike-era
 * shapes. Kept in one explicit compatibility module so the rally path is easy
 * to remove once the shared contracts are renamed.
 */

import { Vector3 } from 'three';

import { RaceDirector } from '../ai/RaceDirector';
import { Effects } from '../fx';
import {
  BikeMode,
  type BikeState,
  type IAudio,
  type IBike,
  type IEffects,
  type IRacer,
  type WheelState,
} from '../game/Contracts';
import { BIKE } from '../game/WorldConstants';
import { clamp, clamp01 } from '../core/MathX';
import { getRallyWheels } from './RallyContactPatch';

const CONTACT_RESTITUTION = 0.24;
const CONTACT_COOLDOWN = 0.24;
const _af = new Vector3();
const _al = new Vector3();
const _bf = new Vector3();
const _bl = new Vector3();
const _normal = new Vector3();
const _mid = new Vector3();
const _impactVelocity = new Vector3();

interface RallyCollisionBike extends IBike {
  collisionHalfLength?: number;
  collisionHalfWidth?: number;
  collisionHalfHeight?: number;
  collisionMass?: number;
  applyRallyContact?: (normal: Vector3, deltaVelocity: number, yawDelta: number) => void;
}

interface DirectorInternals {
  racers: IRacer[];
  contactCooldown: number[];
  audio?: IAudio;
  effects?: IEffects;
}

interface EffectsInternals {
  emitFromState(state: BikeState, dt: number): void;
  emitWheel(wheel: WheelState, state: BikeState, dt: number, weight: number): void;
}

function horizontalAxes(state: BikeState, forward: Vector3, left: Vector3): void {
  forward.set(0, 0, 1).applyQuaternion(state.orientation);
  forward.y = 0;
  if (forward.lengthSq() < 1e-8) forward.set(0, 0, 1);
  else forward.normalize();
  left.set(1, 0, 0).applyQuaternion(state.orientation);
  left.y = 0;
  if (left.lengthSq() < 1e-8) left.set(1, 0, 0);
  else left.normalize();
}

function patchEffects(): void {
  const proto = Effects.prototype as unknown as EffectsInternals & { __rallyFourWheelFx?: boolean };
  if (proto.__rallyFourWheelFx) return;
  proto.__rallyFourWheelFx = true;
  const original = proto.emitFromState;

  proto.emitFromState = function emitRallyState(this: EffectsInternals, state: BikeState, dt: number): void {
    const wheels = getRallyWheels(state);
    if (!wheels) {
      original.call(this, state, dt);
      return;
    }

    // Preserve roughly the old total density while separating the plume into
    // four readable origins. Rear tyres carry more drive/handbrake material.
    this.emitWheel(wheels[0], state, dt, 0.34);
    this.emitWheel(wheels[1], state, dt, 0.34);
    this.emitWheel(wheels[2], state, dt, 0.50);
    this.emitWheel(wheels[3], state, dt, 0.50);
  };
}

function patchRaceContacts(): void {
  const proto = RaceDirector.prototype as unknown as {
    resolveContacts(dt: number): void;
    __rallyObbContacts?: boolean;
  };
  if (proto.__rallyObbContacts) return;
  proto.__rallyObbContacts = true;
  const original = proto.resolveContacts;

  proto.resolveContacts = function resolveRallyContacts(this: DirectorInternals, dt: number): void {
    const rallyReady = this.racers.every((r) => {
      const bike = r.bike as RallyCollisionBike;
      return typeof bike.collisionHalfLength === 'number' && typeof bike.collisionHalfWidth === 'number';
    });
    if (!rallyReady) {
      original.call(this, dt);
      return;
    }

    let pair = 0;
    for (let i = 0; i < this.racers.length; i++) {
      for (let j = i + 1; j < this.racers.length; j++, pair++) {
        if (this.contactCooldown[pair] > 0) this.contactCooldown[pair] -= dt;

        const aBike = this.racers[i].bike as RallyCollisionBike;
        const bBike = this.racers[j].bike as RallyCollisionBike;
        const a = aBike.state;
        const b = bBike.state;
        if (a.mode === BikeMode.Finished || b.mode === BikeMode.Finished) continue;

        const halfHeight = (aBike.collisionHalfHeight ?? 0.72) + (bBike.collisionHalfHeight ?? 0.72);
        if (Math.abs(b.position.y - a.position.y) > halfHeight) continue;

        horizontalAxes(a, _af, _al);
        horizontalAxes(b, _bf, _bl);
        const dx = b.position.x - a.position.x;
        const dz = b.position.z - a.position.z;
        const aL = aBike.collisionHalfLength ?? 2.0;
        const aW = aBike.collisionHalfWidth ?? 0.91;
        const bL = bBike.collisionHalfLength ?? 2.0;
        const bW = bBike.collisionHalfWidth ?? 0.91;

        let bestOverlap = Infinity;
        let bestNx = 0;
        let bestNz = 0;
        let separated = false;
        const testAxis = (ax: number, az: number): void => {
          if (separated) return;
          const signed = dx * ax + dz * az;
          const distance = Math.abs(signed);
          const ra = aL * Math.abs(_af.x * ax + _af.z * az) + aW * Math.abs(_al.x * ax + _al.z * az);
          const rb = bL * Math.abs(_bf.x * ax + _bf.z * az) + bW * Math.abs(_bl.x * ax + _bl.z * az);
          const overlap = ra + rb - distance;
          if (overlap <= 0) {
            separated = true;
            return;
          }
          if (overlap < bestOverlap) {
            bestOverlap = overlap;
            const sign = signed >= 0 ? 1 : -1;
            bestNx = ax * sign;
            bestNz = az * sign;
          }
        };

        testAxis(_af.x, _af.z);
        testAxis(_al.x, _al.z);
        testAxis(_bf.x, _bf.z);
        testAxis(_bl.x, _bl.z);
        if (separated || !Number.isFinite(bestOverlap)) continue;

        _normal.set(bestNx, 0, bestNz);
        const correction = Math.max(0, bestOverlap - 0.025) * 0.36;
        a.position.addScaledVector(_normal, -correction * 0.5);
        b.position.addScaledVector(_normal, correction * 0.5);

        const rvx = b.velocity.x - a.velocity.x;
        const rvz = b.velocity.z - a.velocity.z;
        const approach = rvx * bestNx + rvz * bestNz;
        if (approach >= 0) continue;

        const ma = aBike.collisionMass ?? BIKE.mass;
        const mb = bBike.collisionMass ?? BIKE.mass;
        const impulse = -(1 + CONTACT_RESTITUTION) * approach / (1 / ma + 1 / mb);
        const aDv = -impulse / ma;
        const bDv = impulse / mb;

        const sideA = Math.sign(_af.z * bestNx - _af.x * bestNz) || 1;
        const sideB = Math.sign(_bf.z * -bestNx - _bf.x * -bestNz) || -1;
        const yawMagnitude = clamp(-approach * 0.0105, 0, 0.105);
        if (aBike.applyRallyContact) aBike.applyRallyContact(_normal, aDv, yawMagnitude * sideA);
        else a.velocity.addScaledVector(_normal, aDv);
        if (bBike.applyRallyContact) bBike.applyRallyContact(_normal, bDv, yawMagnitude * sideB);
        else b.velocity.addScaledVector(_normal, bDv);

        const severity = clamp01(-approach / 8.5);
        if (severity > 0.10 && this.contactCooldown[pair] <= 0) {
          this.contactCooldown[pair] = CONTACT_COOLDOWN;
          const surface = a.rear.surface;
          this.audio?.playImpact(severity * 0.62, surface);
          if (severity > 0.34) {
            this.effects?.impactFrame(severity * 0.48);
            _mid.copy(a.position).lerp(b.position, 0.5);
            _impactVelocity.copy(_normal).multiplyScalar(-approach);
            this.effects?.dustBurst(_mid, _normal, _impactVelocity, severity * 0.52, surface);
          }
        }
      }
    }
  };
}

patchEffects();
patchRaceContacts();
