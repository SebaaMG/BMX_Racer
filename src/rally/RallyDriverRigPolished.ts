/** Cockpit placement correction for the proportion-led rally body. */

import type { BikeAnchors } from '../game/Contracts';
import {
  RallyDriverRig as BaseRallyDriverRig,
  type RallyDriverRigOptions,
} from './RallyDriverRig';

export type { RallyDriverRigOptions };

export class RallyDriverRig extends BaseRallyDriverRig {
  override attach(anchors: BikeAnchors): void {
    super.attach(anchors);
    // The first loft capture put the helmet above the roof. The base rig was
    // authored for the taller rejected cabin; lower it into the bucket seat and
    // move it slightly rearward so the visor sits behind the windshield header.
    this.object.position.y -= 0.23;
    this.object.position.z -= 0.04;
  }
}
