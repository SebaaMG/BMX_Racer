/**
 * RallyCarVisualV2 — procedural cel-shaded rally hatch.
 *
 * The first conversion proved the rendering/vehicle seam, but its stacked box
 * silhouette read more like a utility truck. This version authors the hero
 * volumes as side-profile prisms: low wedge nose, raked windshield, compact
 * roof, sloped rear hatch and exposed wheels. It remains entirely generated in
 * code and uses the same cel, prepass and ink systems as the mountain.
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

interface ProfilePoint {
  z: number;
  y: number;
}

function profilePrism(profile: readonly ProfilePoint[], width: number): BufferGeometry {
  const n = profile.length;
  const positions: number[] = [];
  for (const x of [width * 0.5, -width * 0.5]) {
    for (const p of profile) positions.push(x, p.y, p.z);
  }

  const indices: number[] = [];
  // +X face. Profile is counter-clockwise when viewed from +X, so reverse it.
  for (let i = 1; i < n - 1; i++) indices.push(0, i + 1, i);
  // -X face.
  for (let i = 1; i < n - 1; i++) indices.push(n, n + i, n + i + 1);
  // Perimeter faces.
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const li = i;
    const lj = j;
    const ri = n + i;
    const rj = n + j;
    indices.push(li, lj, rj, li, rj, ri);
  }

  const indexed = new BufferGeometry();
  indexed.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
  indexed.setIndex(indices);
  indexed.computeVertexNormals();
  const flat = indexed.toNonIndexed();
  flat.computeVertexNormals();
  indexed.dispose();
  return flat;
}

function prep<T extends BufferGeometry>(geometry: T, angle = 78, gain = 1.1): T {
  prepareOutlineGeometry(geometry, { maxWeldAngle: angle, curvatureGain: gain });
  return geometry;
}

interface GeometrySet {
  lower: BufferGeometry;
  cabin: BufferGeometry;
  bumper: BufferGeometry;
  sill: BufferGeometry;
  arch: BufferGeometry;
  windshield: BufferGeometry;
  rearGlass: BufferGeometry;
  sideGlass: BufferGeometry;
  pillar: BufferGeometry;
  light: BufferGeometry;
  grille: BufferGeometry;
  wing: BufferGeometry;
  wingPost: BufferGeometry;
  scoop: BufferGeometry;
  mirror: BufferGeometry;
  numberPanel: BufferGeometry;
  stripe: BufferGeometry;
  mudflap: BufferGeometry;
  tyre: BufferGeometry;
  rim: BufferGeometry;
  hub: BufferGeometry;
  disc: BufferGeometry;
  exhaust: BufferGeometry;
  flame: BufferGeometry;
}

let geometryCache: GeometrySet | null = null;

function geometrySet(): GeometrySet {
  if (geometryCache) return geometryCache;

  const lower = profilePrism([
    { z: 1.62, y: -0.18 },
    { z: 1.56, y: 0.27 },
    { z: 0.55, y: 0.49 },
    { z: -1.20, y: 0.43 },
    { z: -1.58, y: 0.20 },
    { z: -1.62, y: -0.18 },
  ], 1.76);
  const cabin = profilePrism([
    { z: 0.48, y: 0.43 },
    { z: 0.00, y: 1.10 },
    { z: -0.67, y: 1.10 },
    { z: -1.26, y: 0.45 },
  ], 1.46);

  const tyre = new TorusGeometry(0.285, 0.082, 9, 20);
  tyre.rotateY(Math.PI * 0.5);
  const rim = new CylinderGeometry(0.205, 0.205, 0.155, 14, 1, false);
  rim.rotateZ(Math.PI * 0.5);
  const hub = new CylinderGeometry(0.062, 0.062, 0.18, 10, 1, false);
  hub.rotateZ(Math.PI * 0.5);
  const disc = new CylinderGeometry(0.145, 0.145, 0.025, 14, 1, false);
  disc.rotateZ(Math.PI * 0.5);
  const exhaust = new CylinderGeometry(0.044, 0.058, 0.29, 9, 1, false);
  exhaust.rotateX(Math.PI * 0.5);
  const flame = new CylinderGeometry(0.012, 0.075, 0.34, 8, 1, false);
  flame.rotateX(Math.PI * 0.5);

  geometryCache = {
    lower: prep(lower, 76, 1.18),
    cabin: prep(cabin, 72, 1.18),
    bumper: prep(new BoxGeometry(1.74, 0.16, 0.20), 78, 1.1),
    sill: prep(new BoxGeometry(0.12, 0.18, 2.28), 78, 1.12),
    arch: prep(new BoxGeometry(0.16, 0.22, 0.78), 78, 1.18),
    windshield: prep(new BoxGeometry(1.34, 0.54, 0.038), 84, 0.9),
    rearGlass: prep(new BoxGeometry(1.31, 0.50, 0.038), 84, 0.9),
    sideGlass: prep(new BoxGeometry(0.035, 0.39, 0.91), 84, 0.9),
    pillar: prep(new BoxGeometry(0.045, 0.48, 0.08), 78, 1.0),
    light: prep(new BoxGeometry(0.39, 0.13, 0.062), 80, 0.95),
    grille: prep(new BoxGeometry(0.82, 0.18, 0.055), 80, 0.95),
    wing: prep(new BoxGeometry(1.28, 0.055, 0.24), 80, 1.12),
    wingPost: prep(new BoxGeometry(0.055, 0.22, 0.07), 78, 1.0),
    scoop: prep(new BoxGeometry(0.38, 0.095, 0.34), 78, 1.1),
    mirror: prep(new BoxGeometry(0.16, 0.10, 0.22), 80, 1.1),
    numberPanel: prep(new BoxGeometry(0.028, 0.35, 0.48), 82, 0.9),
    stripe: prep(new BoxGeometry(0.28, 0.035, 1.18), 82, 0.92),
    mudflap: prep(new BoxGeometry(0.04, 0.25, 0.24), 82, 0.85),
    tyre: prep(tyre, 180, 0.68),
    rim: prep(rim, 76, 1.2),
    hub: prep(hub, 76, 1.16),
    disc: prep(disc, 76, 1.0),
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
      matcapMix: 0.24,
      specPower: 15,
      specStrength: 0.31,
      specInk: 0.76,
      identity: {
        color: bodyColor,
        height: 1.35,
        fullPx: 55,
        floorPx: 17,
        chroma: 0.58,
        rim: 0.44,
        inkFloor: 0.56,
      },
    });
    this.accentMaterial = this.makeMaterial(accentRamp, {
      name: `${this.root.name}:accent`,
      matcapMix: 0.16,
      specPower: 12,
      specStrength: 0.23,
      identity: {
        color: accentColor,
        height: 1.25,
        fullPx: 48,
        floorPx: 16,
        chroma: 0.52,
        rim: 0.34,
        inkFloor: 0.58,
      },
    });
    const metal = this.makeMaterial(RAMPS.metal, {
      name: `${this.root.name}:metal`,
      matcapMix: 0.40,
      specPower: 12,
      specStrength: 0.34,
      specInk: 0.68,
    });
    const tyre = this.makeMaterial(RAMPS.tyre, {
      name: `${this.root.name}:tyre`,
      rimStrength: 0.46,
    });
    const rubber = this.makeMaterial(RAMPS.rubber, {
      name: `${this.root.name}:rubber`,
      rimStrength: 0.30,
    });
    this.glassMaterial = this.makeMaterial(RAMPS.lens, {
      name: `${this.root.name}:glass`,
      transparent: true,
      depthWrite: false,
      side: DoubleSide,
      tint: new Color(0x4f9aad),
      tintStrength: 0.34,
      matcapMix: 0.45,
      specPower: 7,
      specStrength: 0.35,
      noShadow: true,
    });
    this.glassMaterial.uniforms.uOpacity.value = 0.72;
    const lamp = this.makeMaterial(RAMPS.marker, {
      name: `${this.root.name}:lamp`,
      tint: new Color(0xffedbd),
      tintStrength: 0.82,
      rimStrength: 0.72,
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
    barLeft.position.set(0.26, 0.47, 0.16);
    barRight.position.set(0.49, 0.47, 0.16);
    pedalLeft.position.set(0.27, 0.02, 0.47);
    pedalRight.position.set(0.48, 0.02, 0.47);
    seat.position.set(0.38, 0.28, -0.32);
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
    this.part(b, this.g.lower, this.bodyMaterial, bodyRamp, 'rally:hatch-shell', [0, 0, 0]);
    this.part(b, this.g.cabin, this.bodyMaterial, bodyRamp, 'rally:hatch-cabin', [0, 0, 0]);

    this.part(b, this.g.bumper, this.accentMaterial, accentRamp, 'rally:front-bumper', [0, -0.11, 1.63]);
    this.part(b, this.g.bumper, this.accentMaterial, accentRamp, 'rally:rear-bumper', [0, -0.10, -1.63]);
    this.part(b, this.g.grille, rubber, RAMPS.rubber, 'rally:grille', [0, 0.10, 1.655]);

    for (const side of [-1, 1]) {
      this.part(b, this.g.sill, this.accentMaterial, accentRamp, `rally:sill:${side}`, [side * 0.865, -0.07, -0.03]);
      this.part(b, this.g.arch, this.bodyMaterial, bodyRamp, `rally:front-arch:${side}`, [side * 0.855, 0.20, 1.18]);
      this.part(b, this.g.arch, this.bodyMaterial, bodyRamp, `rally:rear-arch:${side}`, [side * 0.855, 0.20, -1.18]);
      this.part(b, this.g.mirror, this.accentMaterial, accentRamp, `rally:mirror:${side}`, [side * 0.83, 0.78, 0.17]);
      this.part(b, this.g.sideGlass, this.glassMaterial, RAMPS.lens, `rally:side-glass:${side}`, [side * 0.738, 0.81, -0.33], [0, 0, 0], [1, 1, 1], true, false);
      this.part(b, this.g.pillar, this.bodyMaterial, bodyRamp, `rally:b-pillar:${side}`, [side * 0.758, 0.81, -0.31], [0, 0, 0], [1, 1, 1], false);
      this.part(b, this.g.mudflap, rubber, RAMPS.rubber, `rally:front-flap:${side}`, [side * 0.91, -0.20, 0.83]);
      this.part(b, this.g.mudflap, rubber, RAMPS.rubber, `rally:rear-flap:${side}`, [side * 0.91, -0.20, -1.53]);
    }

    this.part(b, this.g.windshield, this.glassMaterial, RAMPS.lens, 'rally:windshield', [0, 0.80, 0.27], [-0.62, 0, 0], [1, 1, 1], true, false);
    this.part(b, this.g.rearGlass, this.glassMaterial, RAMPS.lens, 'rally:rear-glass', [0, 0.79, -0.96], [0.69, 0, 0], [1, 1, 1], true, false);

    this.part(b, this.g.stripe, this.accentMaterial, accentRamp, 'rally:hood-stripe', [0, 0.505, 0.95], [-0.035, 0, 0], [0.78, 1, 0.92]);
    this.part(b, this.g.stripe, this.accentMaterial, accentRamp, 'rally:roof-stripe', [0, 1.122, -0.35], [0, 0, 0], [0.72, 0.9, 0.58]);
    this.part(b, this.g.scoop, this.accentMaterial, accentRamp, 'rally:roof-scoop', [0, 1.19, -0.05], [0.035, 0, 0]);

    for (const side of [-1, 1]) {
      this.part(b, this.g.light, lamp, RAMPS.marker, `rally:headlight:${side}`, [side * 0.48, 0.24, 1.648]);
      this.part(b, this.g.light, this.brakeMaterial, RAMPS.marker, `rally:taillight:${side}`, [side * 0.49, 0.30, -1.616], [0, Math.PI, 0], [0.90, 0.92, 1]);
    }

    // Compact WRC-style wing: enough to identify the rear, not enough to turn
    // the silhouette into two horizontal bars in chase view.
    this.part(b, this.g.wingPost, metal, RAMPS.metal, 'rally:wing-post-left', [0.43, 0.72, -1.27]);
    this.part(b, this.g.wingPost, metal, RAMPS.metal, 'rally:wing-post-right', [-0.43, 0.72, -1.27]);
    this.part(b, this.g.wing, this.accentMaterial, accentRamp, 'rally:rear-wing', [0, 0.84, -1.29], [-0.05, 0, 0]);

    if (detail === 'full') {
      for (const side of [-1, 1]) {
        this.part(b, this.g.numberPanel, this.accentMaterial, accentRamp, `rally:number-panel:${side}`, [side * 0.885, 0.48, -0.28], [0, 0, 0], [1, 1, 1], false);
      }
      this.part(b, this.g.grille, lamp, RAMPS.marker, 'rally:lamp-pod', [0, 0.39, 1.47], [-0.09, 0, 0], [0.72, 0.58, 1]);
    }

    for (const side of [-1, 1]) {
      this.part(b, this.g.exhaust, metal, RAMPS.metal, `rally:exhaust:${side}`, [side * 0.37, -0.15, -1.67], [Math.PI * 0.5, 0, 0]);
      this.part(this.boostFlames, this.g.flame, this.accentMaterial, accentRamp, `rally:flame:${side}`, [side * 0.37, -0.15, -1.89], [Math.PI * 0.5, 0, 0], [1, 1, 1], false, false);
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
    const z = front ? 1.19 : -1.19;
    const steer = new Group();
    steer.position.set(side * 0.92, -RALLY_TUNE.rideHeight, z);
    axle.add(steer);
    const spin = new Group();
    steer.add(spin);

    this.part(spin, this.g.tyre, tyre, RAMPS.tyre, `rally:tyre:${front}:${side}`, [0, 0, 0]);
    this.part(spin, this.g.rim, metal, RAMPS.metal, `rally:rim:${front}:${side}`, [0, 0, 0]);
    this.part(spin, this.g.hub, this.accentMaterial, this.accentMaterial.preset, `rally:hub:${front}:${side}`, [0, 0, 0]);
    if (detail === 'full') this.part(spin, this.g.disc, metal, RAMPS.metal, `rally:disc:${front}:${side}`, [side * -0.018, 0, 0], [0, 0, 0], [1, 1, 1], false);
    this.wheels.push({ steer, spin, front });
  }

  update(state: RallyCarVisualState, dt: number, cameraDistance = 7): void {
    const frontY = (state.frontCompression - 0.32) * RALLY_TUNE.suspensionTravel * 0.72;
    const rearY = (state.rearCompression - 0.32) * RALLY_TUNE.suspensionTravel * 0.72;
    this.frontAxle.position.y = frontY;
    this.rearAxle.position.y = rearY;

    const average = (state.frontCompression + state.rearCompression) * 0.5;
    this.suspensionBob = dampHL(this.suspensionBob, (average - 0.32) * -0.060, 0.06, dt);
    this.body.position.y = this.suspensionBob;
    this.body.rotation.x = dampHL(
      this.body.rotation.x,
      (state.rearCompression - state.frontCompression) * 0.055,
      0.08,
      dt,
    );

    this.wheelBlur = dampHL(this.wheelBlur, clamp01((state.speed - 13) / 22), 0.12, dt);
    for (const wheel of this.wheels) {
      wheel.steer.rotation.y = wheel.front ? state.steerAngle : 0;
      wheel.spin.rotation.x = wheel.front ? state.frontSpin : state.rearSpin;
      const smear = this.wheelBlur * (cameraDistance < 18 ? 0.030 : 0.016);
      wheel.spin.scale.set(1, 1 + smear, 1 - smear * 0.38);
    }

    const braking = clamp01(state.brake);
    this.brakeMaterial.uniforms.uTintStrength.value = lerp(0.45, 1, braking);
    this.brakeMaterial.uniforms.uRimStrength.value = lerp(0.24, 1.02, braking);

    this.boostFlames.visible = state.boosting;
    if (state.boosting) {
      const pulse = 0.92 + Math.sin(performance.now() * 0.04) * 0.08;
      this.boostFlames.scale.set(pulse, pulse, lerp(0.92, 1.28, clamp01(state.speed / 36)));
    }

    this.chassis.scale.y = dampHL(this.chassis.scale.y, state.crashed ? 0.96 : 1, 0.08, dt);
  }

  setContacts(frontWorld: Vector3, rearWorld: Vector3): void {
    this.root.updateMatrixWorld(true);
    const local = new Vector3();
    local.copy(frontWorld);
    this.root.worldToLocal(local);
    this.anchors.frontContact.position.copy(local);
    local.copy(rearWorld);
    this.root.worldToLocal(local);
    this.anchors.rearContact.position.copy(local);
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
