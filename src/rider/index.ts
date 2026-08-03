/**
 * Driver rig assembly.
 *
 * The race factory now creates RallyDriverRig instances seated inside each car.
 * The original full BMX RiderRig remains exported for the standalone model
 * viewer and historical capture tools, but it is no longer part of gameplay.
 */

import type { BikeAnchors, IBike, IRiderRig } from '../game/Contracts';
import type { RacerSpec } from '../ai/RaceDirector';
import { RiderRig, type RiderRigOptions } from './RiderRig';
import { RallyDriverRig } from '../rally/RallyDriverRigPolished';

export { RiderRig, type RiderRigOptions } from './RiderRig';
export { RallyDriverRig, type RallyDriverRigOptions } from '../rally/RallyDriverRigPolished';
export { RiderSkeleton, REST, LIMB, RIDER_DIMS, BONE_NAMES, BONE_INDEX, type BoneName } from './Skeleton';
export {
  buildRiderMeshes,
  getRiderGeometries,
  applyRiderColors,
  type RiderMeshSet,
  type RiderMeshOptions,
  type RiderPart,
} from './RiderMesh';
export * from './Poses';
export * from './IK';

/** Deterministic per-driver animation phase so the grid never moves in unison. */
function phaseFor(id: string): number {
  let h = 2166136261;
  for (let i = 0; i < id.length; i++) {
    h ^= id.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) / 0xffffffff) * Math.PI * 2;
}

/** Gameplay factory: a helmeted driver seated behind the rally windshield. */
export function createRiderRig(spec: RacerSpec): IRiderRig {
  return new RallyDriverRig({
    jersey: spec.jersey,
    accent: spec.accent,
    name: `rally-driver:${spec.id}`,
    phase: phaseFor(spec.id),
  });
}

/** Build the legacy exposed rider outside a race for old tooling/model views. */
export function makeRiderRig(opts: RiderRigOptions = {}): RiderRig {
  return new RiderRig(opts);
}

/**
 * Attach whichever rig the active vehicle uses. Both the legacy RiderRig and
 * RallyDriverRig expose the same small attach hook over BikeAnchors.
 */
export function attachRigToBike(rig: IRiderRig, bike: IBike): void {
  const attachable = rig as IRiderRig & { attach?: (anchors: BikeAnchors) => void };
  attachable.attach?.(bike.anchors);
}
