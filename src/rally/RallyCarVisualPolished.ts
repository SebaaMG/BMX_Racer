/**
 * Final construction corrections over the proportion-led loft.
 *
 * Kept separate while the new body is under critic review so corrections are
 * small, readable and easy to compare against captures. Once approved this
 * layer can be folded into RallyCarVisual.ts and the rejected V2 deleted.
 */

import { BufferGeometry, Mesh, Object3D, ShaderMaterial } from 'three';
import { prepareOutlineGeometry } from '../npr/OutlineGeometry';
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

function geometryOf(node: Object3D): BufferGeometry | null {
  const geometry = (node as Mesh).geometry;
  return geometry?.isBufferGeometry ? geometry : null;
}

function translateGeometryZ(node: Object3D, dz: number, key: string): void {
  const geometry = geometryOf(node);
  if (!geometry || geometry.userData[key]) return;
  const position = geometry.getAttribute('position');
  if (!position) return;
  for (let i = 0; i < position.count; i++) position.setZ(i, position.getZ(i) + dz);
  position.needsUpdate = true;
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  geometry.userData[key] = true;
}

function makeGlassOpaque(node: Object3D): void {
  const mesh = node as Mesh;
  const material = mesh.material;
  if (!material || Array.isArray(material)) return;
  material.transparent = false;
  material.depthWrite = true;
  material.needsUpdate = true;
  mesh.renderOrder = 1;

  const shader = material as ShaderMaterial;
  if (shader.uniforms?.uOpacity) shader.uniforms.uOpacity.value = 1;
  if (shader.uniforms?.uTintStrength) shader.uniforms.uTintStrength.value = 0.70;
  if (shader.uniforms?.uSpecStrength) shader.uniforms.uSpecStrength.value = 0.24;
}

/**
 * Move only the upper cabin forward. Low sill/shoulder vertices keep the real
 * wheelbase and overhangs, while belt/roof vertices advance up to 24 cm. This
 * converts the first loft's sports-coupe hood into a compact rally greenhouse.
 */
function advanceGreenhouse(node: Object3D): void {
  const geometry = geometryOf(node);
  if (!geometry || geometry.userData.rallyGreenhouseAdvanced) return;
  const position = geometry.getAttribute('position');
  if (!position) return;

  for (let i = 0; i < position.count; i++) {
    const y = position.getY(i);
    const z = position.getZ(i);
    if (y <= 0.40 || z <= -1.18 || z >= 0.58) continue;
    const height = Math.min(1, Math.max(0, (y - 0.40) / 0.36));
    const longitudinal = Math.min(1, Math.max(0, (z + 1.18) / 0.66));
    position.setZ(i, z + 0.24 * height * longitudinal);
  }

  position.needsUpdate = true;
  geometry.computeVertexNormals();
  // Rebuild the outline attributes after changing positions. The hull shares
  // this exact buffer, so main, prepass and ink remain in lockstep.
  prepareOutlineGeometry(geometry, { maxWeldAngle: 66, curvatureGain: 1.26 });
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  geometry.userData.rallyGreenhouseAdvanced = true;
}

export class RallyCarVisual extends LoftedRallyCarVisual {
  constructor(opts: RallyCarVisualOptions) {
    super(opts);

    this.root.traverse((node) => {
      // Hulls are siblings that receive the final authored transform through
      // syncHull(). Mutating them a second time double-scales aero and makes a
      // spoke name end in ":hull", which parsed as NaN in the first pass.
      if (node.userData.isHull) return;

      const name = node.name;

      if (name === 'rally:lofted-shell') advanceGreenhouse(node);
      if (name === 'rally:windshield') {
        translateGeometryZ(node, 0.24, 'rallyWindshieldAdvanced');
        node.position.z += 0.012;
        makeGlassOpaque(node);
      } else if (name === 'rally:rear-glass') {
        node.position.z -= 0.012;
        makeGlassOpaque(node);
      } else if (name.startsWith('rally:front-side-glass:')) {
        translateGeometryZ(node, 0.18, 'rallyFrontSideGlassAdvanced');
        node.position.x += Math.sign(node.scale.x || 1) * 0.012;
        makeGlassOpaque(node);
      } else if (name.startsWith('rally:rear-side-glass:')) {
        node.position.x += Math.sign(node.scale.x || 1) * 0.012;
        makeGlassOpaque(node);
      }

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

      // Follow the forward greenhouse, and keep the vent as a detail rather
      // than a second cabin block.
      if (name === 'rally:roof-vent') {
        node.position.z += 0.15;
        node.scale.set(0.82, 0.68, 0.80);
        syncHull(node);
      } else if (name.startsWith('rally:mirror:')) {
        node.position.z += 0.13;
        syncHull(node);
      }
    });
  }
}
