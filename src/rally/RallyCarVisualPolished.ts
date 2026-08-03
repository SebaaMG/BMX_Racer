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

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function syncHull(node: Object3D): void {
  const hull = node.userData.hull as Object3D | undefined;
  if (!hull) return;
  hull.position.copy(node.position);
  hull.quaternion.copy(node.quaternion);
  hull.scale.copy(node.scale);
  hull.visible = node.visible;
}

function geometryOf(node: Object3D): BufferGeometry | null {
  const geometry = (node as Mesh).geometry;
  return geometry?.isBufferGeometry ? geometry : null;
}

function finishGeometry(geometry: BufferGeometry, outline = false): void {
  geometry.computeVertexNormals();
  if (outline) prepareOutlineGeometry(geometry, { maxWeldAngle: 66, curvatureGain: 1.26 });
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
}

function reshapeGeometry(
  node: Object3D,
  key: string,
  edit: (x: number, y: number, z: number) => [number, number, number],
  outline = false,
): void {
  const geometry = geometryOf(node);
  if (!geometry || geometry.userData[key]) return;
  const position = geometry.getAttribute('position');
  if (!position) return;
  for (let i = 0; i < position.count; i++) {
    const [x, y, z] = edit(position.getX(i), position.getY(i), position.getZ(i));
    position.setXYZ(i, x, y, z);
  }
  position.needsUpdate = true;
  finishGeometry(geometry, outline);
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
  if (shader.uniforms?.uTintStrength) shader.uniforms.uTintStrength.value = 0.74;
  if (shader.uniforms?.uSpecStrength) shader.uniforms.uSpecStrength.value = 0.22;
}

/**
 * Convert the loft into a compact five-door-style rally hatch.
 *
 * The audited 4.10 m body and 2.56 m wheelbase remain unchanged. The cowl moves
 * toward the front axle, while the rear roof and glass move much closer to the
 * rear axle and the rear shoulder rises into a near-vertical hatch. This is a
 * silhouette correction, not a global scale or camera trick.
 */
function reshapeHatchGreenhouse(node: Object3D): void {
  reshapeGeometry(
    node,
    'rallyHatchGreenhouseV3',
    (x, y, z) => {
      if (y <= 0.34) return [x, y, z];

      const height = clamp01((y - 0.34) / 0.46);
      let dz = 0;
      if (z > -0.20) {
        const front = clamp01((z + 0.20) / 0.80);
        dz += (0.20 + front * 0.16) * (0.62 + height * 0.38);
      }
      if (z < -0.50) {
        const rear = clamp01((-z - 0.50) / 0.85);
        dz -= 0.36 * rear * (0.62 + height * 0.38);
        y = Math.min(0.82, y + 0.13 * rear * (1 - height * 0.42));
      }
      return [x, y, z + dz];
    },
    true,
  );
}

function reshapeFrontGlass(node: Object3D): void {
  reshapeGeometry(node, 'rallyFrontGlassV3', (x, y, z) => {
    const front = clamp01((z + 0.20) / 0.80);
    return [x, y, z + (0.20 + front * 0.16)];
  });
}

function reshapeRearGlass(node: Object3D): void {
  reshapeGeometry(node, 'rallyRearGlassV3', (x, y, z) => {
    const rear = clamp01((-z - 0.50) / 0.85);
    return [x, Math.min(0.82, y + 0.10 * rear), z - 0.36 * rear];
  });
}

function hidePart(node: Object3D): void {
  node.visible = false;
  const hull = node.userData.hull as Object3D | undefined;
  if (hull) hull.visible = false;
}

export class RallyCarVisual extends LoftedRallyCarVisual {
  constructor(opts: RallyCarVisualOptions) {
    super(opts);

    this.root.traverse((node) => {
      if (node.userData.isHull) return;

      const name = node.name;

      if (name === 'rally:lofted-shell') reshapeHatchGreenhouse(node);
      if (name === 'rally:windshield') {
        reshapeFrontGlass(node);
        node.position.z += 0.012;
        makeGlassOpaque(node);
      } else if (name === 'rally:rear-glass') {
        reshapeRearGlass(node);
        node.position.z -= 0.012;
        makeGlassOpaque(node);
      } else if (name.startsWith('rally:front-side-glass:')) {
        reshapeFrontGlass(node);
        node.position.x += Math.sign(node.scale.x || 1) * 0.012;
        makeGlassOpaque(node);
      } else if (name.startsWith('rally:rear-side-glass:')) {
        reshapeRearGlass(node);
        node.position.x += Math.sign(node.scale.x || 1) * 0.012;
        makeGlassOpaque(node);
      }

      if (name.startsWith('rally:spot-lamp:')) hidePart(node);

      if (name.startsWith('rally:exhaust:') || name.startsWith('rally:flame:')) {
        node.rotation.x = 0;
        syncHull(node);
      }

      if (name.startsWith('rally:spoke:')) {
        const index = Number(name.slice(name.lastIndexOf(':') + 1));
        const angle = (index / 5) * Math.PI * 2;
        const radius = 0.095;
        node.position.set(0, -Math.sin(angle) * radius, Math.cos(angle) * radius);
        node.rotation.x = angle;
        syncHull(node);
      }

      if (name === 'rally:front-bumper' || name === 'rally:rear-bumper') {
        node.scale.x *= 0.94;
        node.scale.y *= 0.52;
        node.scale.z *= 0.78;
        node.position.y -= 0.035;
        syncHull(node);
      } else if (name.startsWith('rally:sill:')) {
        node.scale.y *= 0.58;
        node.position.y -= 0.025;
        syncHull(node);
      }

      // One low spoiler blade is enough. End plates, visible posts and the roof
      // vent all read as toy roof-rack hardware in the cel silhouette.
      if (name === 'rally:rear-wing') {
        node.position.y -= 0.15;
        node.position.z -= 0.20;
        node.scale.x *= 0.56;
        node.scale.z *= 0.46;
        syncHull(node);
      } else if (
        name.startsWith('rally:wing-post-') ||
        name.startsWith('rally:wing-plate-') ||
        name === 'rally:roof-vent'
      ) {
        hidePart(node);
      } else if (name.startsWith('rally:mirror:')) {
        node.position.z += 0.24;
        node.scale.multiplyScalar(0.88);
        syncHull(node);
      }
    });
  }
}
