/**
 * Production rally hatch body.
 *
 * The physics, wheel animation, anchors and material integration from
 * RallyCarVisual are retained. Its neutral/rejected body is hidden and replaced
 * with this explicit 13-station hatch shell. The design is built around the
 * proportions a rally silhouette actually needs: long usable roof, broad
 * greenhouse, short hood, steep C-pillar, low sill and a vertical rear volume.
 *
 * No imported mesh or texture is used. All visible body geometry is generated
 * from the station table below and rendered through the existing NPR pipeline.
 */

import {
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  Color,
  Group,
  Mesh,
  Object3D,
  ShaderMaterial,
  TorusGeometry,
  Vector3,
} from 'three';

import {
  CelMaterial,
  attachOutline,
  registerNprMesh,
  type CelOptions,
} from '../npr/CelMaterial';
import { prepareOutlineGeometry } from '../npr/OutlineGeometry';
import { RAMPS, type RampPreset } from '../npr/Palette';
import {
  RallyCarVisual as VehicleFoundation,
  RALLY_BODY_DIMENSIONS,
  type RallyCarVisualOptions,
  type RallyCarVisualState,
} from './RallyCarVisual';

export { RALLY_BODY_DIMENSIONS };
export type { RallyCarVisualOptions, RallyCarVisualState };

interface HatchStation {
  z: number;
  floorY: number;
  sillHalf: number;
  lowerY: number;
  lowerHalf: number;
  shoulderY: number;
  shoulderHalf: number;
  beltY: number;
  beltHalf: number;
  roofY: number;
  roofHalf: number;
  crownY: number;
}

/**
 * +Z is the nose. Ground is approximately -0.66 in this vehicle space.
 * Maximum shoulder width is 1.86 m and the roof crown reaches 0.80, giving the
 * audited 1.46 m contact-patch-to-roof height.
 */
const HATCH: readonly HatchStation[] = [
  { z:  2.05, floorY: -0.35, sillHalf: 0.64, lowerY: -0.13, lowerHalf: 0.70, shoulderY: 0.02, shoulderHalf: 0.74, beltY: 0.13, beltHalf: 0.68, roofY: 0.17, roofHalf: 0.48, crownY: 0.19 },
  { z:  1.82, floorY: -0.38, sillHalf: 0.82, lowerY: -0.11, lowerHalf: 0.88, shoulderY: 0.08, shoulderHalf: 0.92, beltY: 0.22, beltHalf: 0.84, roofY: 0.27, roofHalf: 0.59, crownY: 0.29 },
  { z:  1.45, floorY: -0.38, sillHalf: 0.91, lowerY: -0.09, lowerHalf: 0.95, shoulderY: 0.14, shoulderHalf: 0.93, beltY: 0.28, beltHalf: 0.87, roofY: 0.34, roofHalf: 0.65, crownY: 0.36 },
  { z:  1.13, floorY: -0.38, sillHalf: 0.92, lowerY: -0.07, lowerHalf: 0.96, shoulderY: 0.17, shoulderHalf: 0.93, beltY: 0.32, beltHalf: 0.86, roofY: 0.38, roofHalf: 0.66, crownY: 0.40 },
  { z:  0.78, floorY: -0.38, sillHalf: 0.91, lowerY: -0.06, lowerHalf: 0.94, shoulderY: 0.18, shoulderHalf: 0.92, beltY: 0.37, beltHalf: 0.83, roofY: 0.43, roofHalf: 0.68, crownY: 0.45 },
  { z:  0.48, floorY: -0.38, sillHalf: 0.90, lowerY: -0.06, lowerHalf: 0.92, shoulderY: 0.18, shoulderHalf: 0.90, beltY: 0.41, beltHalf: 0.80, roofY: 0.59, roofHalf: 0.69, crownY: 0.61 },
  { z:  0.27, floorY: -0.38, sillHalf: 0.90, lowerY: -0.06, lowerHalf: 0.91, shoulderY: 0.18, shoulderHalf: 0.89, beltY: 0.44, beltHalf: 0.78, roofY: 0.77, roofHalf: 0.69, crownY: 0.80 },
  { z: -0.36, floorY: -0.38, sillHalf: 0.90, lowerY: -0.06, lowerHalf: 0.91, shoulderY: 0.18, shoulderHalf: 0.89, beltY: 0.45, beltHalf: 0.79, roofY: 0.78, roofHalf: 0.70, crownY: 0.80 },
  { z: -0.88, floorY: -0.38, sillHalf: 0.91, lowerY: -0.06, lowerHalf: 0.93, shoulderY: 0.18, shoulderHalf: 0.91, beltY: 0.44, beltHalf: 0.81, roofY: 0.75, roofHalf: 0.68, crownY: 0.78 },
  { z: -1.16, floorY: -0.38, sillHalf: 0.92, lowerY: -0.07, lowerHalf: 0.96, shoulderY: 0.17, shoulderHalf: 0.93, beltY: 0.41, beltHalf: 0.84, roofY: 0.66, roofHalf: 0.66, crownY: 0.69 },
  { z: -1.42, floorY: -0.38, sillHalf: 0.92, lowerY: -0.08, lowerHalf: 0.97, shoulderY: 0.15, shoulderHalf: 0.93, beltY: 0.35, beltHalf: 0.86, roofY: 0.49, roofHalf: 0.68, crownY: 0.52 },
  { z: -1.64, floorY: -0.38, sillHalf: 0.90, lowerY: -0.10, lowerHalf: 0.94, shoulderY: 0.10, shoulderHalf: 0.91, beltY: 0.27, beltHalf: 0.84, roofY: 0.31, roofHalf: 0.63, crownY: 0.34 },
  { z: -1.86, floorY: -0.38, sillHalf: 0.82, lowerY: -0.13, lowerHalf: 0.88, shoulderY: 0.04, shoulderHalf: 0.87, beltY: 0.18, beltHalf: 0.78, roofY: 0.22, roofHalf: 0.54, crownY: 0.24 },
  { z: -2.05, floorY: -0.35, sillHalf: 0.67, lowerY: -0.15, lowerHalf: 0.74, shoulderY: 0.00, shoulderHalf: 0.76, beltY: 0.12, beltHalf: 0.69, roofY: 0.16, roofHalf: 0.47, crownY: 0.18 },
];

const _normal = new Vector3();

function ring(s: HatchStation): readonly [number, number, number][] {
  const underside = s.floorY - 0.07;
  return [
    [0, underside, s.z],
    [s.sillHalf * 0.70, underside, s.z],
    [s.sillHalf, s.floorY, s.z],
    [s.lowerHalf, s.lowerY, s.z],
    [s.shoulderHalf, s.shoulderY, s.z],
    [s.beltHalf, s.beltY, s.z],
    [s.roofHalf, s.roofY, s.z],
    [0, s.crownY, s.z],
    [-s.roofHalf, s.roofY, s.z],
    [-s.beltHalf, s.beltY, s.z],
    [-s.shoulderHalf, s.shoulderY, s.z],
    [-s.lowerHalf, s.lowerY, s.z],
    [-s.sillHalf, s.floorY, s.z],
    [-s.sillHalf * 0.70, underside, s.z],
  ];
}

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
    for (const i of group.indices) normal.setXYZ(i, _normal.x, _normal.y, _normal.z);
  }
  normal.needsUpdate = true;
}

function loft(stations: readonly HatchStation[]): BufferGeometry {
  const ringSize = ring(stations[0]).length;
  const positions: number[] = [];
  for (const station of stations) for (const p of ring(station)) positions.push(...p);
  const indices: number[] = [];
  for (let i = 0; i < stations.length - 1; i++) {
    const a0 = i * ringSize;
    const b0 = (i + 1) * ringSize;
    for (let j = 0; j < ringSize; j++) {
      const k = (j + 1) % ringSize;
      const a = a0 + j;
      const b = a0 + k;
      const c = b0 + k;
      const d = b0 + j;
      indices.push(a, c, b, a, d, c);
    }
  }
  for (let i = 1; i < ringSize - 1; i++) indices.push(0, i, i + 1);
  const rear = (stations.length - 1) * ringSize;
  for (let i = 1; i < ringSize - 1; i++) indices.push(rear, rear + i + 1, rear + i);

  const indexed = new BufferGeometry();
  indexed.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
  indexed.setIndex(indices);
  indexed.computeVertexNormals();
  const geometry = indexed.toNonIndexed();
  indexed.dispose();
  smoothCoincidentNormals(geometry);
  prepareOutlineGeometry(geometry, { maxWeldAngle: 74, curvatureGain: 1.12 });
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function panel(points: readonly [number, number, number][], outline = false): BufferGeometry {
  const positions = new Float32Array(points.flat());
  const indices: number[] = [];
  for (let i = 1; i < points.length - 1; i++) indices.push(0, i, i + 1);
  const indexed = new BufferGeometry();
  indexed.setAttribute('position', new BufferAttribute(positions, 3));
  indexed.setIndex(indices);
  indexed.computeVertexNormals();
  const geometry = indexed.toNonIndexed();
  indexed.dispose();
  smoothCoincidentNormals(geometry);
  if (outline) prepareOutlineGeometry(geometry, { maxWeldAngle: 82, curvatureGain: 0.92 });
  return geometry;
}

function prep<T extends BufferGeometry>(geometry: T, angle = 78, gain = 1.0): T {
  prepareOutlineGeometry(geometry, { maxWeldAngle: angle, curvatureGain: gain });
  return geometry;
}

function hide(node: Object3D): void {
  node.visible = false;
  const hull = node.userData.hull as Object3D | undefined;
  if (hull) hull.visible = false;
}

function syncHull(node: Object3D): void {
  const hull = node.userData.hull as Object3D | undefined;
  if (!hull) return;
  hull.position.copy(node.position);
  hull.quaternion.copy(node.quaternion);
  hull.scale.copy(node.scale);
}

export class RallyCarVisual extends VehicleFoundation {
  private readonly addedGeometries: BufferGeometry[] = [];
  private readonly addedHullMaterials: ShaderMaterial[] = [];
  private readonly refinedRimGeometry: BufferGeometry;

  constructor(opts: RallyCarVisualOptions) {
    super(opts);

    const rejectedShell = this.root.getObjectByName('rally:lofted-shell') as Mesh | undefined;
    const bodyParent = rejectedShell?.parent ?? this.root;
    const bodyMaterial = rejectedShell?.material as CelMaterial | undefined;
    const glassMaterial = (this.root.getObjectByName('rally:windshield') as Mesh | undefined)?.material as CelMaterial | undefined;
    const rubberMaterial = (this.root.getObjectByName('rally:grille') as Mesh | undefined)?.material as CelMaterial | undefined;
    const lampMaterial = (this.root.getObjectByName('rally:headlight:1') as Mesh | undefined)?.material as CelMaterial | undefined;
    const brakeMaterial = (this.root.getObjectByName('rally:taillight:1') as Mesh | undefined)?.material as CelMaterial | undefined;
    const accentMaterial = (this.root.getObjectByName('rally:sill:1') as Mesh | undefined)?.material as CelMaterial | undefined;

    if (!bodyMaterial || !glassMaterial || !rubberMaterial || !lampMaterial || !brakeMaterial || !accentMaterial) {
      throw new Error('rally production body: foundation materials unavailable');
    }

    // Hide every rejected body component. Wheel assemblies, flares, mudflaps,
    // exhausts, boost flames and the cockpit anchors remain active.
    this.root.traverse((node) => {
      if (node.userData.isHull) return;
      const n = node.name;
      if (
        n === 'rally:lofted-shell' ||
        n === 'rally:windshield' ||
        n === 'rally:rear-glass' ||
        n.startsWith('rally:front-side-glass:') ||
        n.startsWith('rally:rear-side-glass:') ||
        n.startsWith('rally:door-number-panel:') ||
        n === 'rally:hood-livery' ||
        n === 'rally:front-bumper' ||
        n === 'rally:rear-bumper' ||
        n === 'rally:front-splitter' ||
        n.startsWith('rally:sill:') ||
        n === 'rally:grille' ||
        n.startsWith('rally:headlight:') ||
        n.startsWith('rally:taillight:') ||
        n.startsWith('rally:spot-lamp:') ||
        n === 'rally:roof-vent' ||
        n.startsWith('rally:wing-') ||
        n === 'rally:rear-wing' ||
        n.startsWith('rally:mirror:')
      ) hide(node);
    });

    const body = new Group();
    body.name = 'rally:production-hatch-body';
    bodyParent.add(body);

    const shellGeometry = loft(HATCH);
    this.addedGeometries.push(shellGeometry);
    this.addPart(body, shellGeometry, bodyMaterial, RAMPS.frame, 'rally:production-shell', [0, 0, 0], true);

    // Dark generated glass masks the closed shell while preserving clean cel
    // values. Side windows are split by a real B-pillar gap.
    this.addGlass(body, panel([
      [-0.75, 0.42, 0.735], [0.75, 0.42, 0.735],
      [0.66, 0.755, 0.305], [-0.66, 0.755, 0.305],
    ]), glassMaterial, 'rally:production-windshield');
    this.addGlass(body, panel([
      [-0.66, 0.735, -0.90], [0.66, 0.735, -0.90],
      [0.75, 0.42, -1.49], [-0.75, 0.42, -1.49],
    ]), glassMaterial, 'rally:production-rear-glass');

    const frontSide = panel([
      [0.807, 0.42, 0.68], [0.695, 0.755, 0.30],
      [0.695, 0.775, -0.30], [0.807, 0.43, -0.33],
    ]);
    const rearSide = panel([
      [0.807, 0.43, -0.37], [0.695, 0.775, -0.34],
      [0.675, 0.735, -0.91], [0.792, 0.49, -1.17],
      [0.815, 0.42, -1.20],
    ]);
    this.addedGeometries.push(frontSide, rearSide);
    for (const side of [-1, 1]) {
      this.addGlass(body, frontSide, glassMaterial, `rally:production-front-side-glass:${side}`, side);
      this.addGlass(body, rearSide, glassMaterial, `rally:production-rear-side-glass:${side}`, side);
    }

    // Body-colour bumpers and a dark lower opening. No decorative gold bars.
    this.addBox(body, bodyMaterial, RAMPS.frame, 'rally:production-front-bumper', [0, -0.26, 1.99], [1.78, 0.18, 0.18]);
    this.addBox(body, bodyMaterial, RAMPS.frame, 'rally:production-rear-bumper', [0, -0.26, -1.99], [1.76, 0.18, 0.18]);
    this.addBox(body, rubberMaterial, RAMPS.rubber, 'rally:production-grille', [0, -0.035, 2.062], [0.78, 0.16, 0.045], false);
    this.addBox(body, rubberMaterial, RAMPS.rubber, 'rally:production-splitter', [0, -0.44, 1.91], [1.64, 0.045, 0.30]);

    // Thin rally sill protection, not a full-length toy stripe.
    for (const side of [-1, 1]) {
      this.addBox(body, accentMaterial, RAMPS.marker, `rally:production-sill:${side}`, [side * 0.91, -0.315, -0.03], [0.07, 0.055, 2.20]);
    }

    // Large, simple lamp shapes are more legible in an anime render than tiny
    // photoreal lamp internals.
    const headlight = panel([
      [-0.28, 0.11, 2.070], [-0.71, 0.12, 2.070],
      [-0.65, 0.29, 2.070], [-0.24, 0.27, 2.070],
    ]);
    const taillight = panel([
      [-0.30, 0.10, -2.070], [-0.72, 0.11, -2.070],
      [-0.68, 0.27, -2.070], [-0.27, 0.27, -2.070],
    ]);
    this.addedGeometries.push(headlight, taillight);
    for (const side of [-1, 1]) {
      this.addPart(body, headlight, lampMaterial, RAMPS.marker, `rally:production-headlight:${side}`, [0, 0, 0], false, side);
      this.addPart(body, taillight, brakeMaterial, RAMPS.marker, `rally:production-taillight:${side}`, [0, 0, 0], false, side);
    }

    // Small body-colour mirrors and an integrated roof-edge spoiler.
    for (const side of [-1, 1]) {
      this.addBox(body, bodyMaterial, RAMPS.frame, `rally:production-mirror:${side}`, [side * 0.86, 0.45, 0.44], [0.15, 0.10, 0.22]);
    }
    this.addBox(body, bodyMaterial, RAMPS.frame, 'rally:production-spoiler', [0, 0.53, -1.48], [1.18, 0.045, 0.16]);

    // Replace solid wheel discs with a rim ring so the generated spokes and
    // brake hardware are visible.
    this.refinedRimGeometry = prep(new TorusGeometry(0.174, 0.026, 8, 20), 120, 0.82);
    this.refinedRimGeometry.rotateY(Math.PI * 0.5);
    this.addedGeometries.push(this.refinedRimGeometry);
    this.root.traverse((node) => {
      if (node.userData.isHull) return;
      if (node.name.startsWith('rally:rim:')) {
        (node as Mesh).geometry = this.refinedRimGeometry;
        const hull = node.userData.hull as Mesh | undefined;
        if (hull) hull.geometry = this.refinedRimGeometry;
      }
      if (node.name.startsWith('rally:spoke:')) {
        const index = Number(node.name.slice(node.name.lastIndexOf(':') + 1));
        const angle = (index / 5) * Math.PI * 2;
        node.position.set(0, -Math.sin(angle) * 0.103, Math.cos(angle) * 0.103);
        node.rotation.x = angle;
        node.scale.setScalar(0.86);
        syncHull(node);
      }
    });
  }

  private addPart(
    parent: Object3D,
    geometry: BufferGeometry,
    material: CelMaterial,
    ramp: RampPreset,
    name: string,
    position: [number, number, number],
    outline: boolean,
    mirrorX = 1,
  ): Mesh {
    const mesh = new Mesh(geometry, material);
    mesh.name = name;
    mesh.position.set(...position);
    mesh.scale.x = mirrorX;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    registerNprMesh(mesh, material);
    if (outline) {
      const hull = attachOutline(mesh, ramp, material.celOptions);
      if (hull) {
        hull.position.copy(mesh.position);
        hull.quaternion.copy(mesh.quaternion);
        hull.scale.copy(mesh.scale);
        this.addedHullMaterials.push(hull.material as ShaderMaterial);
        parent.add(hull);
      }
    }
    parent.add(mesh);
    return mesh;
  }

  private addGlass(
    parent: Object3D,
    geometry: BufferGeometry,
    material: CelMaterial,
    name: string,
    mirrorX = 1,
  ): Mesh {
    if (!this.addedGeometries.includes(geometry)) this.addedGeometries.push(geometry);
    const mesh = this.addPart(parent, geometry, material, RAMPS.lens, name, [0, 0, 0], false, mirrorX);
    mesh.castShadow = false;
    mesh.renderOrder = 2;
    return mesh;
  }

  private addBox(
    parent: Object3D,
    material: CelMaterial,
    ramp: RampPreset,
    name: string,
    position: [number, number, number],
    size: [number, number, number],
    outline = true,
  ): Mesh {
    const geometry = prep(new BoxGeometry(...size), 82, 0.92);
    this.addedGeometries.push(geometry);
    return this.addPart(parent, geometry, material, ramp, name, position, outline);
  }

  override dispose(): void {
    super.dispose();
    for (const material of this.addedHullMaterials) material.dispose();
    for (const geometry of this.addedGeometries) geometry.dispose();
    this.addedHullMaterials.length = 0;
    this.addedGeometries.length = 0;
  }
}
