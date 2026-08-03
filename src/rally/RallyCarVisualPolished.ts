/**
 * Final construction corrections over the proportion-led loft.
 *
 * The base loft provides deterministic topology and audited dimensions. This
 * layer performs the critic-led body refinement that turns that neutral cage
 * into a planted gravel rally hatch: lower sill, broader greenhouse, compact
 * hatch, smooth panel normals, open-spoke wheels and restrained aero.
 */

import {
  BufferGeometry,
  Mesh,
  Object3D,
  ShaderMaterial,
  TorusGeometry,
  Vector3,
} from 'three';
import { prepareOutlineGeometry } from '../npr/OutlineGeometry';
import {
  RallyCarVisual as LoftedRallyCarVisual,
  RALLY_BODY_DIMENSIONS,
  type RallyCarVisualOptions,
  type RallyCarVisualState,
} from './RallyCarVisual';

export { RALLY_BODY_DIMENSIONS };
export type { RallyCarVisualOptions, RallyCarVisualState };

const _normal = new Vector3();

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
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

/**
 * Average face normals at coincident positions. The loft remains non-indexed so
 * material/outline code stays simple, but broad body panels no longer fracture
 * into unrelated triangular light bands.
 */
function smoothCoincidentNormals(geometry: BufferGeometry): void {
  geometry.computeVertexNormals();
  const position = geometry.getAttribute('position');
  const normal = geometry.getAttribute('normal');
  if (!position || !normal) return;

  const groups = new Map<string, { x: number; y: number; z: number; indices: number[] }>();
  for (let i = 0; i < position.count; i++) {
    const key = `${Math.round(position.getX(i) * 10000)},${Math.round(position.getY(i) * 10000)},${Math.round(position.getZ(i) * 10000)}`;
    let group = groups.get(key);
    if (!group) {
      group = { x: 0, y: 0, z: 0, indices: [] };
      groups.set(key, group);
    }
    group.x += normal.getX(i);
    group.y += normal.getY(i);
    group.z += normal.getZ(i);
    group.indices.push(i);
  }

  for (const group of groups.values()) {
    _normal.set(group.x, group.y, group.z).normalize();
    for (const index of group.indices) normal.setXYZ(index, _normal.x, _normal.y, _normal.z);
  }
  normal.needsUpdate = true;
}

function finishGeometry(geometry: BufferGeometry, outline = false, smooth = false): void {
  if (smooth) smoothCoincidentNormals(geometry);
  else geometry.computeVertexNormals();
  if (outline) prepareOutlineGeometry(geometry, { maxWeldAngle: 72, curvatureGain: 1.18 });
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
}

function reshapeGeometry(
  node: Object3D,
  key: string,
  edit: (x: number, y: number, z: number) => [number, number, number],
  outline = false,
  smooth = false,
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
  finishGeometry(geometry, outline, smooth);
  geometry.userData[key] = true;
}

/** Shared body-space deformation for the shell and conformed surface panels. */
function mapBodyPoint(x: number, y: number, z: number): [number, number, number] {
  const originalY = y;
  const upper = clamp01((originalY - 0.30) / 0.50);

  // Lower the floor/sill from crossover clearance to gravel-rally clearance,
  // while keeping the audited roof height almost unchanged.
  y = originalY * 1.13 - 0.124;

  // A rally greenhouse is broad, not pinched like a sports coupe.
  x *= 1 + upper * 0.105;

  // Compact hood: move the cowl toward the front axle.
  if (z > -0.20 && upper > 0) {
    const front = clamp01((z + 0.20) / 0.86);
    z += (0.20 + front * 0.15) * (0.56 + upper * 0.44);
  }

  // Compact hatch: compress the upper rear longitudinally. The lower rear body
  // and bumper remain at the audited 4.10 m envelope, producing a steep hatch
  // instead of a long fastback.
  if (z < -0.55 && upper > 0) {
    const compressed = -0.55 + (z + 0.55) * 0.69;
    z = lerp(z, compressed, 0.58 + upper * 0.42);
    const rear = clamp01((-z - 0.55) / 1.10);
    y += 0.055 * rear * (1 - upper * 0.34);
  }

  return [x, y, z];
}

function mapFrontGlass(x: number, y: number, z: number): [number, number, number] {
  const point = mapBodyPoint(x, y, z);
  return [point[0], point[1], point[2] + 0.018];
}

function mapRearGlass(x: number, y: number, z: number): [number, number, number] {
  // Rear glazing gets a steeper dedicated mapping than the surrounding shell.
  const point = mapBodyPoint(x, y, z);
  const t = clamp01((-z - 0.78) / 0.62);
  point[2] -= 0.12 + t * 0.12;
  point[1] += t * 0.035;
  return point;
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
  if (shader.uniforms?.uTintStrength) shader.uniforms.uTintStrength.value = 0.78;
  if (shader.uniforms?.uSpecStrength) shader.uniforms.uSpecStrength.value = 0.18;
}

function hidePart(node: Object3D): void {
  node.visible = false;
  const hull = node.userData.hull as Object3D | undefined;
  if (hull) hull.visible = false;
}

export class RallyCarVisual extends LoftedRallyCarVisual {
  private readonly refinedRimGeometry: BufferGeometry;

  constructor(opts: RallyCarVisualOptions) {
    super(opts);

    this.refinedRimGeometry = new TorusGeometry(0.174, 0.026, 8, 20);
    this.refinedRimGeometry.rotateY(Math.PI * 0.5);
    prepareOutlineGeometry(this.refinedRimGeometry, { maxWeldAngle: 120, curvatureGain: 0.82 });

    const shell = this.root.getObjectByName('rally:lofted-shell') as Mesh | undefined;
    const bodyMaterial = shell?.material;
    const bodyPrepass = shell?.userData.prepassMaterial;
    const bodyShadow = shell?.userData.shadowMaterial;

    this.root.traverse((node) => {
      if (node.userData.isHull) return;
      const name = node.name;

      if (name === 'rally:lofted-shell') {
        reshapeGeometry(node, 'rallyBodyV4', mapBodyPoint, true, true);
      } else if (
        name === 'rally:hood-livery' ||
        name.startsWith('rally:door-number-panel:')
      ) {
        reshapeGeometry(node, 'rallyBodyPanelV4', mapBodyPoint, false, true);
      } else if (name === 'rally:windshield' || name.startsWith('rally:front-side-glass:')) {
        reshapeGeometry(node, 'rallyFrontGlassV4', mapFrontGlass, false, true);
        makeGlassOpaque(node);
      } else if (name === 'rally:rear-glass' || name.startsWith('rally:rear-side-glass:')) {
        reshapeGeometry(node, 'rallyRearGlassV4', mapRearGlass, false, true);
        makeGlassOpaque(node);
      }

      if (name.startsWith('rally:spot-lamp:')) hidePart(node);

      if (name.startsWith('rally:rim:')) {
        const mesh = node as Mesh;
        mesh.geometry = this.refinedRimGeometry;
        const hull = node.userData.hull as Mesh | undefined;
        if (hull) hull.geometry = this.refinedRimGeometry;
      }

      if (name.startsWith('rally:spoke:')) {
        const index = Number(name.slice(name.lastIndexOf(':') + 1));
        const angle = (index / 5) * Math.PI * 2;
        const radius = 0.103;
        node.position.set(0, -Math.sin(angle) * radius, Math.cos(angle) * radius);
        node.rotation.x = angle;
        node.scale.set(0.86, 0.86, 0.86);
        syncHull(node);
      }

      // Body-colour bumpers and mirrors remove the toy gold-bar read.
      if (
        bodyMaterial &&
        (name === 'rally:front-bumper' || name === 'rally:rear-bumper' || name.startsWith('rally:mirror:'))
      ) {
        (node as Mesh).material = bodyMaterial;
        if (bodyPrepass) node.userData.prepassMaterial = bodyPrepass;
        if (bodyShadow) node.userData.shadowMaterial = bodyShadow;
      }

      // Primitive accessories must follow the vertical body remap explicitly.
      if (name === 'rally:front-bumper' || name === 'rally:rear-bumper') {
        node.position.y = node.position.y * 1.13 - 0.124;
        node.scale.y *= 0.68;
        node.scale.z *= 0.82;
        syncHull(node);
      } else if (name === 'rally:front-splitter') {
        node.position.y = -0.425;
        node.scale.z *= 0.78;
        syncHull(node);
      } else if (name.startsWith('rally:sill:')) {
        node.position.y = -0.285;
        node.scale.y *= 0.48;
        syncHull(node);
      } else if (name === 'rally:grille') {
        node.position.y = -0.055;
        node.scale.y *= 0.76;
        syncHull(node);
      } else if (name.startsWith('rally:mirror:')) {
        node.position.y = node.position.y * 1.13 - 0.124;
        node.position.z += 0.23;
        node.scale.multiplyScalar(0.82);
        syncHull(node);
      } else if (name.startsWith('rally:mudflap:') || name.includes('-flap:')) {
        node.position.y = -0.405;
        node.scale.y *= 0.82;
        syncHull(node);
      } else if (name.startsWith('rally:exhaust:') || name.startsWith('rally:flame:')) {
        node.rotation.x = 0;
        node.position.y = -0.345;
        syncHull(node);
      }

      if (name.startsWith('rally:headlight:')) {
        reshapeGeometry(node, 'rallyHeadlightV4', mapBodyPoint, false, true);
        node.position.z += 0.025;
      } else if (name.startsWith('rally:taillight:')) {
        reshapeGeometry(node, 'rallyTaillightV4', mapBodyPoint, false, true);
        node.position.z -= 0.025;
      }

      // One low integrated spoiler blade. Everything that looked like roof-rack
      // hardware in the critic sheet is removed.
      if (name === 'rally:rear-wing') {
        node.position.set(0, 0.47, -1.57);
        node.scale.set(0.48, 0.55, 0.38);
        syncHull(node);
      } else if (
        name.startsWith('rally:wing-post-') ||
        name.startsWith('rally:wing-plate-') ||
        name === 'rally:roof-vent'
      ) {
        hidePart(node);
      }
    });
  }

  override dispose(): void {
    super.dispose();
    this.refinedRimGeometry.dispose();
  }
}
