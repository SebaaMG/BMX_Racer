/**
 * Final construction corrections over the proportion-led loft.
 *
 * Kept separate while the new body is under critic review so corrections are
 * small, readable and easy to compare against captures. Once approved this
 * layer can be folded into RallyCarVisual.ts and the rejected V2 deleted.
 */

import { Object3D } from 'three';
import {
  RallyCarVisual as LoftedRallyCarVisual,
  RALLY_BODY_DIMENSIONS,
  type RallyCarVisualOptions,
  type RallyCarVisualState,
} from './RallyCarVisual';

export { RALLY_BODY_DIMENSIONS };
export type { RallyCarVisualOptions, RallyCarVisualState };

function syncHull(node: Object3D): void {
  const hull = node.userData.hull as Object3D | undefined;
  if (!hull) return;
  hull.position.copy(node.position);
  hull.quaternion.copy(node.quaternion);
  hull.scale.copy(node.scale);
}

export class RallyCarVisual extends LoftedRallyCarVisual {
  constructor(opts: RallyCarVisualOptions) {
    super(opts);

    this.root.traverse((node) => {
      const name = node.name;

      // These geometries were authored along Z already. A second placement
      // rotation turned them sideways in the first loft capture.
      if (
        name.startsWith('rally:spot-lamp:') ||
        name.startsWith('rally:exhaust:') ||
        name.startsWith('rally:flame:')
      ) {
        node.rotation.x = 0;
        syncHull(node);
      }

      // Rotate positions around the axle as well as the spoke geometry itself.
      // The rejected capture stacked all five spokes at one local offset.
      if (name.startsWith('rally:spoke:')) {
        const index = Number(name.slice(name.lastIndexOf(':') + 1));
        const angle = (index / 5) * Math.PI * 2;
        const radius = 0.095;
        node.position.set(0, -Math.sin(angle) * radius, Math.cos(angle) * radius);
        node.rotation.x = angle;
        syncHull(node);
      }

      // The loft fixed the utility-vehicle silhouette; this keeps the aero from
      // replacing it with a dominant horizontal bar in the chase camera.
      if (name === 'rally:rear-wing') {
        node.position.y -= 0.055;
        node.scale.x *= 0.82;
        node.scale.z *= 0.78;
        syncHull(node);
      } else if (name.startsWith('rally:wing-post-')) {
        node.position.y -= 0.045;
        node.scale.y *= 0.72;
        syncHull(node);
      } else if (name.startsWith('rally:wing-plate-')) {
        node.position.y -= 0.055;
        node.position.x *= 0.84;
        node.scale.y *= 0.78;
        node.scale.z *= 0.78;
        syncHull(node);
      }

      // Keep the roof vent as a competition detail, not a second cabin block.
      if (name === 'rally:roof-vent') {
        node.scale.set(0.82, 0.68, 0.80);
        syncHull(node);
      }
    });
  }
}
