/**
 * Four-contact-patch state shared by rally physics, FX and audio.
 *
 * The historical BikeState contract exposes one front and one rear WheelState.
 * Rally gameplay keeps those two aggregate axle states for compatibility while
 * publishing the real four tyre patches here. Consumers that understand cars
 * use the tuple; legacy consumers continue to read front/rear unchanged.
 */

import type { BikeState, WheelState } from '../game/Contracts';

export type RallyWheelTuple = readonly [
  frontLeft: WheelState,
  frontRight: WheelState,
  rearLeft: WheelState,
  rearRight: WheelState,
];

export interface RallyContactBikeState extends BikeState {
  rallyWheels: RallyWheelTuple;
  /** Signed local lateral velocity: +left, -right, metres per second. */
  rallyLateralVelocity: number;
  /** Signed velocity-heading angle, radians. */
  rallySlipAngle: number;
}

export function getRallyWheels(state: BikeState): RallyWheelTuple | null {
  const candidate = (state as Partial<RallyContactBikeState>).rallyWheels;
  return candidate && candidate.length === 4 ? candidate : null;
}

export function isRallyContactState(state: BikeState): state is RallyContactBikeState {
  return getRallyWheels(state) !== null;
}
