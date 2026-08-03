/**
 * RallyCarVisual — proportion-led procedural gravel rally car.
 *
 * The previous body was a constant-width side extrusion. That construction
 * compressed a 2.56 m wheelbase into a 3.24 m body and produced the slab-sided,
 * upright utility-vehicle read visible in review captures. This model instead
 * uses a full three-dimensional longitudinal loft: every station controls sill,
 * shoulder, belt and roof widths independently, so the nose tapers, the cabin
 * tucks inward and the wheel arches carry the silhouette.
 *
 * Target proportions are Rally2-inspired rather than copied from one car:
 * 4.10 m long, 1.86 m wide, 2.56 m wheelbase, 0.70 m gravel tyre and a
 * 1.46 m roof height from the contact patch. Everything remains generated in
 * TypeScript and participates in the existing cel, G-buffer and ink pipeline.
 */

import {
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  Color,
  CylinderGeometry,
  DoubleSide,
  Group,
  Mesh,
  Object3D,
  ShaderMaterial,
  SphereGeometry,
  TorusGeometry,
  Vector3,
} from 'three';

import type { BikeAnchors } from '../game/Contracts';
import { clamp01, dampHL, lerp } from '../core/MathX';
import {
  CelMaterial,
  attachOutline,
  disposeCelMaterial,
  registerNprMesh,
  type CelOptions,
} from '../npr/CelMaterial';
import { prepareOutlineGeometry } from '../npr/OutlineGeometry';
import { RAMPS, type RampPreset } from '../npr/Palette';
import { RALLY_TUNE } from './RallyCarPhysics';

export const RALLY_BODY_DIMENSIONS = {
  length: 4.10,
  width: 1.86,
  wheelbase: 2.56,
  gravelTyreDiameter: 0.70,
  roofHeightFromGround: 1.46,
} as const;

interface BodyStation {
  z: number;
  floorY: number;
  sillHalf: number;
  shoulderY: number;
  shoulderHalf: number;
  beltY: number;
  beltHalf: number;
  roofY: number;
  roofHalf: number;
  crownY: number;
}

/**
 * A Rally2-like compact hatch/sedan cage. The origin is the sprung chassis
 * centre; ground is approximately -0.66 m because wheel centres sit at
 * -RALLY_TUNE.rideHeight and tyre radius is 0.345 m.
 */
const BODY_STATIONS: readonly BodyStation[] = [
  { z:  2.05, floorY: -0.21, sillHalf: 0.66, shoulderY: 0.08, shoulderHalf: 0.76, beltY: 0.21, beltHalf: 0.68, roofY: 0.25, roofHalf: 0.48, crownY: 0.27 },
  { z:  1.76, floorY: -0.21, sillHalf: 0.83, shoulderY: 0.15, shoulderHalf: 0.91, beltY: 0.30, beltHalf: 0.82, roofY: 0.34, roofHalf: 0.61, crownY: 0.36 },
  { z:  1.28, floorY: -0.20, sillHalf: 0.91, shoulderY: 0.24, shoulderHalf: 0.95, beltY: 0.38, beltHalf: 0.84, roofY: 0.41, roofHalf: 0.64, crownY: 0.43 },
  { z:  0.76, floorY: -0.19, sillHalf: 0.90, shoulderY: 0.26, shoulderHalf: 0.92, beltY: 0.41, beltHalf: 0.80, roofY: 0.44, roofHalf: 0.65, crownY: 0.46 },
  { z:  0.38, floorY: -0.19, sillHalf: 0.88, shoulderY: 0.27, shoulderHalf: 0.88, beltY: 0.45, beltHalf: 0.76, roofY: 0.50, roofHalf: 0.63, crownY: 0.52 },
  { z:  0.04, floorY: -0.19, sillHalf: 0.87, shoulderY: 0.28, shoulderHalf: 0.86, beltY: 0.49, beltHalf: 0.72, roofY: 0.73, roofHalf: 0.57, crownY: 0.76 },
  { z: -0.55, floorY: -0.19, sillHalf: 0.87, shoulderY: 0.28, shoulderHalf: 0.86, beltY: 0.49, beltHalf: 0.72, roofY: 0.77, roofHalf: 0.58, crownY: 0.80 },
  { z: -0.96, floorY: -0.19, sillHalf: 0.88, shoulderY: 0.27, shoulderHalf: 0.88, beltY: 0.47, beltHalf: 0.74, roofY: 0.70, roofHalf: 0.57, crownY: 0.73 },
  { z: -1.28, floorY: -0.20, sillHalf: 0.92, shoulderY: 0.24, shoulderHalf: 0.96, beltY: 0.40, beltHalf: 0.85, roofY: 0.53, roofHalf: 0.66, crownY: 0.55 },
  { z: -1.74, floorY: -0.21, sillHalf: 0.85, shoulderY: 0.16, shoulderHalf: 0.91, beltY: 0.31, beltHalf: 0.82, roofY: 0.38, roofHalf: 0.61, crownY: 0.40 },
  { z: -2.05, floorY: -0.21, sillHalf: 0.69, shoulderY: 0.09, shoulderHalf: 0.79, beltY: 0.23, beltHalf: 0.71, roofY: 0.28, roofHalf: 0.49, crownY: 0.30 },
];

const _local = new Vector3();

function ringForStation(s: BodyStation): readonly [number, number, number][] {
  const underside = s.floorY - 0.075;
  return [
    [0, underside, s.z],
    [s.sillHalf * 0.72, underside, s.z],
    [s.sillHalf, s.floorY, s.z],
    [s.shoulderHalf, s.shoulderY, s.z],
    [s.beltHalf, s.beltY, s.z],
    [s.roofHalf, s.roofY, s.z],
    [0, s.crownY, s.z],
    [-s.roofHalf, s.roofY, s.z],
    [-s.beltHalf, s.beltY, s.z],
    [-s.shoulderHalf, s.shoulderY, s.z],
    [-s.sillHalf, s.floorY, s.z],
    [-s.sillHalf * 0.72, underside, s.z],
  ];
}

function loftGeometry(stations: readonly BodyStation[]): BufferGeometry {
  const ringSize = ringForStation(stations[0]).length;
  const positions: number[] = [];
  for (const station of stations) {
    for (const [x, y, z] of ringForStation(station)) positions.push(x, y, z);
  }

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
      // Stations run +Z to -Z, so this winding points outwards.
      indices.push(a, c, b, a, d, c);
    }
  }

  // Front cap (+Z).
  for (let i = 1; i < ringSize - 1; i++) indices.push(0, i, i + 1);
  // Rear cap (-Z).
  const rear = (stations.length - 1) * ringSize;
  for (let i = 1; i < ringSize - 1; i++) indices.push(rear, rear + i + 1, rear + i);

  const indexed = new BufferGeometry();
  indexed.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
  indexed.setIndex(indices);
  indexed.computeVertexNormals();
  const flat = indexed.toNonIndexed();
  flat.computeVertexNormals();
  indexed.dispose();
  return flat;
}

function panelGeometry(points: readonly [number, number, number][]): BufferGeometry {
  const positions: number[] = [];
  for (const p of points) positions.push(...p);
  const indices: number[] = [];
  for (let i = 1; i < points.length - 1; i++) indices.push(0, i, i + 1);
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry.toNonIndexed();
}

/** Top-half wheel-arch ribbon, extruded along X. */
function archGeometry(innerR: number, outerR: number, width: number, segments = 15): BufferGeometry {
  const positions: number[] = [];
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI;
    const zIn = Math.cos(a) * innerR;
    const yIn = Math.sin(a) * innerR;
    const zOut = Math.cos(a) * outerR;
    const yOut = Math.sin(a) * outerR;
    positions.push(
      width * 0.5, yIn, zIn,
      width * 0.5, yOut, zOut,
      -width * 0.5, yIn, zIn,
      -width * 0.5, yOut, zOut,
    );
  }

  const indices: number[] = [];
  for (let i = 0; i < segments; i++) {
    const a = i * 4;
    const b = (i + 1) * 4;
    // +X and -X faces.
    indices.push(a, b + 1, a + 1, a, b, b + 1);
    indices.push(a + 2, a + 3, b + 3, a + 2, b + 3, b + 2);
    // Inner and outer radial surfaces.
    indices.push(a, a + 2, b + 2, a, b + 2, b);
    indices.push(a + 1, b + 3, a + 3, a + 1, b + 1, b + 3);
  }
  // End caps.
  const last = segments * 4;
  indices.push(0, 1, 3, 0, 3, 2);
  indices.push(last, last + 3, last + 1, last, last + 2, last + 3);

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  const flat = geometry.toNonIndexed();
  flat.computeVertexNormals();
  geometry.dispose();
  return flat;
}

function prep<T extends BufferGeometry>(geometry: T, angle = 72, gain = 1.15): T {
  prepareOutlineGeometry(geometry, { maxWeldAngle: angle, curvatureGain: gain });
  return geometry;
}

interface GeometrySet {
  shell: BufferGeometry;
  frontArch: BufferGeometry;
  rearArch: BufferGeometry;
  frontGlass: BufferGeometry;
  rearGlass: BufferGeometry;
  sideGlass: BufferGeometry;
  sideGlassRear: BufferGeometry;
  sidePanel: BufferGeometry;
  hoodPanel: BufferGeometry;
  bumper: BufferGeometry;
  splitter: BufferGeometry;
  sill: BufferGeometry;
  grille: BufferGeometry;
  headlight: BufferGeometry;
  taillight: BufferGeometry;
  roundLamp: BufferGeometry;
  roofVent: BufferGeometry;
  wing: BufferGeometry;
  wingPost: BufferGeometry;
  wingPlate: BufferGeometry;
  mirror: BufferGeometry;
  mudflap: BufferGeometry;
  tyre: BufferGeometry;
  rim: BufferGeometry;
  hub: BufferGeometry;
  disc: BufferGeometry;
  spoke: BufferGeometry;
  caliper: BufferGeometry;
  exhaust: BufferGeometry;
  flame: BufferGeometry;
}

let geometryCache: GeometrySet | null = null;

function geometrySet(): GeometrySet {
  if (geometryCache) return geometryCache;

  const tyre = new TorusGeometry(0.272, 0.078, 10, 24);
  tyre.rotateY(Math.PI * 0.5);
  const rim = new CylinderGeometry(0.205, 0.205, 0.17, 16, 1, false);
  rim.rotateZ(Math.PI * 0.5);
  const hub = new CylinderGeometry(0.055, 0.055, 0.19, 12, 1, false);
  hub.rotateZ(Math.PI * 0.5);
  const disc = new CylinderGeometry(0.145, 0.145, 0.026, 16, 1, false);
  disc.rotateZ(Math.PI * 0.5);
  const roundLamp = new CylinderGeometry(0.095, 0.095, 0.065, 12, 1, false);
  roundLamp.rotateX(Math.PI * 0.5);
  const exhaust = new CylinderGeometry(0.045, 0.058, 0.30, 10, 1, false);
  exhaust.rotateX(Math.PI * 0.5);
  const flame = new CylinderGeometry(0.012, 0.075, 0.36, 8, 1, false);
  flame.rotateX(Math.PI * 0.5);

  geometryCache = {
    shell: prep(loftGeometry(BODY_STATIONS), 66, 1.26),
    frontArch: prep(archGeometry(0.36, 0.48, 0.18), 70, 1.30),
    rearArch: prep(archGeometry(0.36, 0.49, 0.18), 70, 1.30),
    frontGlass: prep(panelGeometry([
      [-0.69, 0.49, 0.392],
      [0.69, 0.49, 0.392],
      [0.56, 0.735, 0.032],
      [-0.56, 0.735, 0.032],
    ]), 86, 0.85),
    rearGlass: prep(panelGeometry([
      [-0.56, 0.715, -0.962],
      [0.56, 0.715, -0.962],
      [0.68, 0.515, -1.292],
      [-0.68, 0.515, -1.292],
    ]), 86, 0.85),
    sideGlass: prep(panelGeometry([
      [0.735, 0.475, 0.34],
      [0.585, 0.725, 0.02],
      [0.585, 0.745, -0.36],
      [0.715, 0.485, -0.37],
    ]), 86, 0.85),
    sideGlassRear: prep(panelGeometry([
      [0.715, 0.485, -0.39],
      [0.585, 0.745, -0.38],
      [0.575, 0.705, -0.83],
      [0.72, 0.505, -1.10],
    ]), 86, 0.85),
    sidePanel: prep(panelGeometry([
      [0.903, 0.13, 0.62],
      [0.885, 0.40, 0.38],
      [0.875, 0.40, -0.75],
      [0.915, 0.12, -0.93],
    ]), 80, 0.96),
    hoodPanel: prep(panelGeometry([
      [-0.44, 0.438, 0.72],
      [0.44, 0.438, 0.72],
      [0.34, 0.344, 1.67],
      [-0.34, 0.344, 1.67],
    ]), 82, 0.90),
    bumper: prep(new BoxGeometry(1.72, 0.17, 0.18), 78, 1.08),
    splitter: prep(new BoxGeometry(1.68, 0.055, 0.34), 80, 1.12),
    sill: prep(new BoxGeometry(0.115, 0.15, 2.38), 78, 1.12),
    grille: prep(new BoxGeometry(0.86, 0.18, 0.055), 82, 0.94),
    headlight: prep(panelGeometry([
      [-0.37, 0.15, 2.057],
      [-0.69, 0.17, 2.057],
      [-0.63, 0.30, 2.057],
      [-0.31, 0.28, 2.057],
    ]), 86, 0.86),
    taillight: prep(panelGeometry([
      [-0.34, 0.18, -2.057],
      [-0.70, 0.17, -2.057],
      [-0.64, 0.30, -2.057],
      [-0.31, 0.29, -2.057],
    ]), 86, 0.86),
    roundLamp: prep(roundLamp, 78, 0.92),
    roofVent: prep(new BoxGeometry(0.36, 0.075, 0.30), 78, 1.08),
    wing: prep(new BoxGeometry(1.26, 0.055, 0.22), 80, 1.08),
    wingPost: prep(new BoxGeometry(0.055, 0.18, 0.075), 78, 1.0),
    wingPlate: prep(new BoxGeometry(0.035, 0.18, 0.24), 80, 1.0),
    mirror: prep(new SphereGeometry(0.12, 8, 5), 74, 1.15),
    mudflap: prep(new BoxGeometry(0.035, 0.25, 0.22), 82, 0.86),
    tyre: prep(tyre, 180, 0.66),
    rim: prep(rim, 76, 1.18),
    hub: prep(hub, 76, 1.12),
    disc: prep(disc, 76, 1.0),
    spoke: prep(new BoxGeometry(0.175, 0.035, 0.19), 76, 1.04),
    caliper: prep(new BoxGeometry(0.07, 0.11, 0.055), 76, 1.02),
    exhaust: prep(exhaust, 76, 1.05),
    flame: prep(flame, 76, 0.92),
  };
  return geometryCache;
}

function rehue(base: RampPreset, target: Color, name: string, amount = 0.58): RampPreset {
  return {
    ...base,
    name,
    colors: base.colors.map((band, i) => band.clone().lerp(
      target,
      amount * lerp(0.74, 1, i / Math.max(1, base.colors.length - 1)),
    )),
    thresholds: [...base.thresholds],
  };
}

export interface RallyCarVisualOptions {
  frameColor: Color;
  accentColor?: Color;
  detail?: 'full' | 'reduced';
  name?: string;
}

export interface RallyCarVisualState {
  steerAngle: number;
  frontCompression: number;
  rearCompression: number;
  frontSpin: number;
  rearSpin: number;
  speed: number;
  lateralSlip: number;
  boosting: boolean;
  brake: number;
  crashed: boolean;
}

interface WheelAssembly {
  steer: Object3D;
  spin: Object3D;
  front: boolean;
}

export class RallyCarVisual {
  readonly root = new Group();
  readonly anchors: BikeAnchors;

  private readonly g = geometrySet();
  private readonly chassis = new Group();
  private readonly body = new Group();
  private readonly frontAxle = new Group();
  private readonly rearAxle = new Group();
  private readonly wheels: WheelAssembly[] = [];
  private readonly materials: CelMaterial[] = [];
  private readonly hullMaterials = new Set<ShaderMaterial>();
  private readonly boostFlames = new Group();

  private readonly bodyMaterial: CelMaterial;
  private readonly accentMaterial: CelMaterial;
  private readonly glassMaterial: CelMaterial;
  private readonly brakeMaterial: CelMaterial;
  private suspensionBob = 0;
  private wheelBlur = 0;

  constructor(opts: RallyCarVisualOptions) {
    const bodyColor = opts.frameColor;
    const accentColor = opts.accentColor ?? bodyColor.clone().offsetHSL(0.08, 0.12, 0.16);
    const detail = opts.detail ?? 'full';
    this.root.name = opts.name ?? 'rally-car';
    this.root.add(this.chassis);
    this.chassis.add(this.body, this.frontAxle, this.rearAxle);

    const bodyRamp = rehue(RAMPS.frame, bodyColor, `${this.root.name}:body-ramp`);
    const accentRamp = rehue(RAMPS.marker, accentColor, `${this.root.name}:accent-ramp`, 0.46);
    this.bodyMaterial = this.makeMaterial(bodyRamp, {
      name: `${this.root.name}:body`,
      matcapMix: 0.20,
      specPower: 12,
      specStrength: 0.27,
      specInk: 0.74,
      identity: {
        color: bodyColor,
        height: 1.46,
        fullPx: 58,
        floorPx: 18,
        chroma: 0.58,
        rim: 0.42,
        inkFloor: 0.53,
      },
    });
    this.accentMaterial = this.makeMaterial(accentRamp, {
      name: `${this.root.name}:accent`,
      matcapMix: 0.13,
      specPower: 10,
      specStrength: 0.20,
      identity: {
        color: accentColor,
        height: 1.40,
        fullPx: 50,
        floorPx: 16,
        chroma: 0.50,
        rim: 0.32,
        inkFloor: 0.56,
      },
    });
    const metal = this.makeMaterial(RAMPS.metal, {
      name: `${this.root.name}:metal`,
      matcapMix: 0.38,
      specPower: 11,
      specStrength: 0.31,
      specInk: 0.66,
    });
    const tyre = this.makeMaterial(RAMPS.tyre, {
      name: `${this.root.name}:tyre`,
      rimStrength: 0.48,
    });
    const rubber = this.makeMaterial(RAMPS.rubber, {
      name: `${this.root.name}:rubber`,
      rimStrength: 0.28,
    });
    this.glassMaterial = this.makeMaterial(RAMPS.lens, {
      name: `${this.root.name}:glass`,
      transparent: true,
      depthWrite: false,
      side: DoubleSide,
      tint: new Color(0x3c7e95),
      tintStrength: 0.30,
      matcapMix: 0.42,
      specPower: 6,
      specStrength: 0.32,
      noShadow: true,
    });
    this.glassMaterial.uniforms.uOpacity.value = 0.68;
    const lamp = this.makeMaterial(RAMPS.marker, {
      name: `${this.root.name}:lamp`,
      tint: new Color(0xffedbd),
      tintStrength: 0.84,
      rimStrength: 0.74,
      noShadow: true,
    });
    this.brakeMaterial = this.makeMaterial(RAMPS.marker, {
      name: `${this.root.name}:brake`,
      tint: new Color(0xe84655),
      tintStrength: 0.68,
      noShadow: true,
    });

    this.buildBody(bodyRamp, accentRamp, metal, rubber, lamp, detail);
    this.buildWheel(this.frontAxle, 1, true, tyre, metal, detail);
    this.buildWheel(this.frontAxle, -1, true, tyre, metal, detail);
    this.buildWheel(this.rearAxle, 1, false, tyre, metal, detail);
    this.buildWheel(this.rearAxle, -1, false, tyre, metal, detail);

    const barLeft = new Object3D();
    const barRight = new Object3D();
    const pedalLeft = new Object3D();
    const pedalRight = new Object3D();
    const seat = new Object3D();
    const frontContact = new Object3D();
    const rearContact = new Object3D();
    barLeft.position.set(0.25, 0.39, 0.12);
    barRight.position.set(0.48, 0.39, 0.12);
    pedalLeft.position.set(0.27, -0.02, 0.49);
    pedalRight.position.set(0.48, -0.02, 0.49);
    seat.position.set(0.37, 0.21, -0.31);
    frontContact.position.set(0, -RALLY_TUNE.rideHeight - RALLY_TUNE.wheelRadius, RALLY_TUNE.wheelbase * 0.5);
    rearContact.position.set(0, -RALLY_TUNE.rideHeight - RALLY_TUNE.wheelRadius, -RALLY_TUNE.wheelbase * 0.5);
    this.chassis.add(barLeft, barRight, pedalLeft, pedalRight, seat, frontContact, rearContact);
    this.anchors = { barLeft, barRight, pedalLeft, pedalRight, seat, frame: this.chassis, frontContact, rearContact };
  }

  private makeMaterial(preset: RampPreset, opts: CelOptions): CelMaterial {
    const material = new CelMaterial(preset, opts);
    this.materials.push(material);
    return material;
  }

  private part(
    parent: Object3D,
    geometry: BufferGeometry,
    material: CelMaterial,
    preset: RampPreset,
    name: string,
    position: [number, number, number],
    rotation: [number, number, number] = [0, 0, 0],
    scale: [number, number, number] = [1, 1, 1],
    outline = true,
    prepass = true,
  ): Mesh {
    const mesh = new Mesh(geometry, material);
    mesh.name = name;
    mesh.position.set(...position);
    mesh.rotation.set(...rotation);
    mesh.scale.set(...scale);
    mesh.castShadow = !material.celOptions.noShadow;
    mesh.receiveShadow = true;
    if (prepass) registerNprMesh(mesh, material);
    mesh.renderOrder = material.transparent ? 4 : 0;

    if (outline) {
      const hull = attachOutline(mesh, preset, material.celOptions);
      if (hull) {
        hull.position.copy(mesh.position);
        hull.rotation.copy(mesh.rotation);
        hull.scale.copy(mesh.scale);
        this.hullMaterials.add(hull.material as ShaderMaterial);
        parent.add(hull);
      }
    }
    parent.add(mesh);
    return mesh;
  }

  private buildBody(
    bodyRamp: RampPreset,
    accentRamp: RampPreset,
    metal: CelMaterial,
    rubber: CelMaterial,
    lamp: CelMaterial,
    detail: 'full' | 'reduced',
  ): void {
    const b = this.body;
    this.part(b, this.g.shell, this.bodyMaterial, bodyRamp, 'rally:lofted-shell', [0, 0, 0]);

    // The arch ribbons sit proud of the loft and create the wheel-first stance
    // missing from the previous constant-width body.
    for (const side of [-1, 1]) {
      const x = side * 0.91;
      this.part(b, this.g.frontArch, this.bodyMaterial, bodyRamp, `rally:front-flare:${side}`, [x, -RALLY_TUNE.rideHeight, 1.28]);
      this.part(b, this.g.rearArch, this.bodyMaterial, bodyRamp, `rally:rear-flare:${side}`, [x, -RALLY_TUNE.rideHeight, -1.28]);
      this.part(b, this.g.sill, this.accentMaterial, accentRamp, `rally:sill:${side}`, [side * 0.895, -0.10, -0.04]);
      this.part(b, this.g.sidePanel, this.accentMaterial, accentRamp, `rally:door-number-panel:${side}`, [0, 0, 0], [0, 0, 0], [side, 1, 1], false);
      this.part(b, this.g.sideGlass, this.glassMaterial, RAMPS.lens, `rally:front-side-glass:${side}`, [0, 0, 0], [0, 0, 0], [side, 1, 1], false, false);
      this.part(b, this.g.sideGlassRear, this.glassMaterial, RAMPS.lens, `rally:rear-side-glass:${side}`, [0, 0, 0], [0, 0, 0], [side, 1, 1], false, false);
      this.part(b, this.g.mirror, this.accentMaterial, accentRamp, `rally:mirror:${side}`, [side * 0.82, 0.50, 0.20], [0, 0, 0], [1.0, 0.55, 1.35]);
      this.part(b, this.g.mudflap, rubber, RAMPS.rubber, `rally:front-flap:${side}`, [side * 0.99, -0.24, 0.87]);
      this.part(b, this.g.mudflap, rubber, RAMPS.rubber, `rally:rear-flap:${side}`, [side * 0.99, -0.24, -1.60]);
    }

    this.part(b, this.g.frontGlass, this.glassMaterial, RAMPS.lens, 'rally:windshield', [0, 0, 0], [0, 0, 0], [1, 1, 1], false, false);
    this.part(b, this.g.rearGlass, this.glassMaterial, RAMPS.lens, 'rally:rear-glass', [0, 0, 0], [0, 0, 0], [1, 1, 1], false, false);

    // Motorsport surfaces: integrated bumper/splitter, skid-like lower edge and
    // a restrained body-mounted aero package.
    this.part(b, this.g.bumper, this.accentMaterial, accentRamp, 'rally:front-bumper', [0, -0.12, 2.01]);
    this.part(b, this.g.bumper, this.accentMaterial, accentRamp, 'rally:rear-bumper', [0, -0.12, -2.01]);
    this.part(b, this.g.splitter, rubber, RAMPS.rubber, 'rally:front-splitter', [0, -0.245, 1.91]);
    this.part(b, this.g.grille, rubber, RAMPS.rubber, 'rally:grille', [0, 0.035, 2.063]);
    this.part(b, this.g.hoodPanel, this.accentMaterial, accentRamp, 'rally:hood-livery', [0, 0, 0], [0, 0, 0], [0.62, 1, 1], false);
    this.part(b, this.g.roofVent, this.accentMaterial, accentRamp, 'rally:roof-vent', [0, 0.825, -0.16], [-0.05, 0, 0], [1, 1, 1]);

    for (const side of [-1, 1]) {
      this.part(b, this.g.headlight, lamp, RAMPS.marker, `rally:headlight:${side}`, [0, 0, 0], [0, 0, 0], [side, 1, 1], false);
      this.part(b, this.g.taillight, this.brakeMaterial, RAMPS.marker, `rally:taillight:${side}`, [0, 0, 0], [0, 0, 0], [side, 1, 1], false);
    }

    // Compact roof-edge wing. End plates make it read as engineered aero from
    // rear three-quarter without becoming the dominant horizontal bar.
    this.part(b, this.g.wingPost, metal, RAMPS.metal, 'rally:wing-post-left', [0.43, 0.63, -1.39]);
    this.part(b, this.g.wingPost, metal, RAMPS.metal, 'rally:wing-post-right', [-0.43, 0.63, -1.39]);
    this.part(b, this.g.wing, this.accentMaterial, accentRamp, 'rally:rear-wing', [0, 0.72, -1.43], [-0.055, 0, 0]);
    this.part(b, this.g.wingPlate, this.accentMaterial, accentRamp, 'rally:wing-plate-left', [0.64, 0.72, -1.43]);
    this.part(b, this.g.wingPlate, this.accentMaterial, accentRamp, 'rally:wing-plate-right', [-0.64, 0.72, -1.43]);

    if (detail === 'full') {
      // Four round auxiliary lamps are an unambiguous gravel-rally signifier,
      // but stay below the hood line so they do not turn the nose into a wall.
      for (const x of [-0.31, -0.105, 0.105, 0.31]) {
        this.part(b, this.g.roundLamp, lamp, RAMPS.marker, `rally:spot-lamp:${x}`, [x, 0.25, 1.99], [Math.PI * 0.5, 0, 0], [0.78, 0.78, 0.55]);
      }
    }

    for (const side of [-1, 1]) {
      this.part(b, this.g.exhaust, metal, RAMPS.metal, `rally:exhaust:${side}`, [side * 0.34, -0.17, -2.05], [Math.PI * 0.5, 0, 0]);
      this.part(this.boostFlames, this.g.flame, this.accentMaterial, accentRamp, `rally:flame:${side}`, [side * 0.34, -0.17, -2.27], [Math.PI * 0.5, 0, 0], [1, 1, 1], false, false);
    }
    this.boostFlames.visible = false;
    b.add(this.boostFlames);
  }

  private buildWheel(
    axle: Object3D,
    side: number,
    front: boolean,
    tyre: CelMaterial,
    metal: CelMaterial,
    detail: 'full' | 'reduced',
  ): void {
    const z = front ? 1.28 : -1.28;
    const steer = new Group();
    steer.position.set(side * 0.95, -RALLY_TUNE.rideHeight, z);
    axle.add(steer);
    const spin = new Group();
    steer.add(spin);

    this.part(spin, this.g.tyre, tyre, RAMPS.tyre, `rally:tyre:${front}:${side}`, [0, 0, 0]);
    this.part(spin, this.g.rim, metal, RAMPS.metal, `rally:rim:${front}:${side}`, [0, 0, 0]);
    this.part(spin, this.g.hub, this.accentMaterial, this.accentMaterial.preset, `rally:hub:${front}:${side}`, [0, 0, 0]);
    if (detail === 'full') {
      this.part(spin, this.g.disc, metal, RAMPS.metal, `rally:disc:${front}:${side}`, [side * -0.018, 0, 0], [0, 0, 0], [1, 1, 1], false);
      for (let i = 0; i < 5; i++) {
        const angle = (i / 5) * Math.PI * 2;
        this.part(spin, this.g.spoke, metal, RAMPS.metal, `rally:spoke:${front}:${side}:${i}`, [0, 0, 0.095], [angle, 0, 0], [1, 1, 1], false);
      }
      this.part(steer, this.g.caliper, this.accentMaterial, this.accentMaterial.preset, `rally:caliper:${front}:${side}`, [side * -0.10, 0.02, -0.14], [0, 0, 0], [1, 1, 1], false);
    }
    this.wheels.push({ steer, spin, front });
  }

  update(state: RallyCarVisualState, dt: number, cameraDistance = 7): void {
    const frontY = (state.frontCompression - 0.32) * RALLY_TUNE.suspensionTravel * 0.78;
    const rearY = (state.rearCompression - 0.32) * RALLY_TUNE.suspensionTravel * 0.78;
    this.frontAxle.position.y = frontY;
    this.rearAxle.position.y = rearY;

    const average = (state.frontCompression + state.rearCompression) * 0.5;
    this.suspensionBob = dampHL(this.suspensionBob, (average - 0.32) * -0.052, 0.055, dt);
    this.body.position.y = this.suspensionBob;
    this.body.rotation.x = dampHL(
      this.body.rotation.x,
      (state.rearCompression - state.frontCompression) * 0.065,
      0.075,
      dt,
    );
    this.body.rotation.z = dampHL(
      this.body.rotation.z,
      clamp01(state.lateralSlip / 8) * Math.sign(state.steerAngle) * -0.018,
      0.09,
      dt,
    );

    this.wheelBlur = dampHL(this.wheelBlur, clamp01((state.speed - 13) / 22), 0.12, dt);
    for (const wheel of this.wheels) {
      wheel.steer.rotation.y = wheel.front ? state.steerAngle : 0;
      wheel.spin.rotation.x = wheel.front ? state.frontSpin : state.rearSpin;
      const smear = this.wheelBlur * (cameraDistance < 18 ? 0.028 : 0.015);
      wheel.spin.scale.set(1, 1 + smear, 1 - smear * 0.36);
    }

    const braking = clamp01(state.brake);
    this.brakeMaterial.uniforms.uTintStrength.value = lerp(0.44, 1, braking);
    this.brakeMaterial.uniforms.uRimStrength.value = lerp(0.22, 1.02, braking);

    this.boostFlames.visible = state.boosting;
    if (state.boosting) {
      const pulse = 0.92 + Math.sin(performance.now() * 0.04) * 0.08;
      this.boostFlames.scale.set(pulse, pulse, lerp(0.92, 1.30, clamp01(state.speed / 36)));
    }

    this.chassis.scale.y = dampHL(this.chassis.scale.y, state.crashed ? 0.96 : 1, 0.08, dt);
  }

  setContacts(frontWorld: Vector3, rearWorld: Vector3): void {
    this.root.updateMatrixWorld(true);
    _local.copy(frontWorld);
    this.root.worldToLocal(_local);
    this.anchors.frontContact.position.copy(_local);
    _local.copy(rearWorld);
    this.root.worldToLocal(_local);
    this.anchors.rearContact.position.copy(_local);
  }

  setBodyColor(color: Color): void {
    this.bodyMaterial.setTint(color, 0.72);
  }

  dispose(): void {
    this.root.removeFromParent();
    for (const material of this.hullMaterials) material.dispose();
    for (const material of this.materials) disposeCelMaterial(material);
    this.hullMaterials.clear();
    this.materials.length = 0;
  }
}
