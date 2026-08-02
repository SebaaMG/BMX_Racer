/**
 * RallyCarVisual — a complete procedural rally car authored from primitives.
 *
 * No model, texture, font or decal is loaded. The silhouette is assembled from
 * deliberately chunky, animation-friendly hard-surface forms, then every part
 * is passed through the same cel ramps, G-buffer registration and variable
 * inverted-hull line system as the mountain and the original riders.
 *
 * Car space: +Y up, +Z forward, +X left. The origin is the sprung chassis
 * centre; wheel centres sit below it by RALLY_TUNE.rideHeight.
 */

import {
  BoxGeometry,
  BufferGeometry,
  Color,
  CylinderGeometry,
  Group,
  Mesh,
  Object3D,
  Quaternion,
  SphereGeometry,
  TorusGeometry,
  Vector3,
} from 'three';

import {
  CelMaterial,
  attachOutline,
  disposeCelMaterial,
  registerNprMesh,
  type CelOptions,
} from '../npr/CelMaterial';
import { prepareOutlineGeometry } from '../npr/OutlineGeometry';
import { RAMPS, type RampPreset } from '../npr/Palette';
import type { BikeAnchors } from '../game/Contracts';
import { clamp01, dampHL, lerp } from '../core/MathX';
import { RALLY_TUNE } from './RallyCarPhysics';

const _local = new Vector3();
const _world = new Vector3();
const _q = new Quaternion();

interface RallyGeometrySet {
  lowerBody: BufferGeometry;
  upperBody: BufferGeometry;
  hood: BufferGeometry;
  roof: BufferGeometry;
  bumper: BufferGeometry;
  sill: BufferGeometry;
  fender: BufferGeometry;
  glassFront: BufferGeometry;
  glassRear: BufferGeometry;
  glassSide: BufferGeometry;
  lamp: BufferGeometry;
  lampBar: BufferGeometry;
  wing: BufferGeometry;
  wingPost: BufferGeometry;
  scoop: BufferGeometry;
  mirror: BufferGeometry;
  skid: BufferGeometry;
  tyre: BufferGeometry;
  rim: BufferGeometry;
  hub: BufferGeometry;
  brakeDisc: BufferGeometry;
  exhaust: BufferGeometry;
}

let geometryCache: RallyGeometrySet | null = null;

function outlined<T extends BufferGeometry>(geo: T, angle = 72, gain = 1.12): T {
  prepareOutlineGeometry(geo, {
    maxWeldAngle: angle,
    curvatureGain: gain,
  });
  return geo;
}

function getGeometry(): RallyGeometrySet {
  if (geometryCache) return geometryCache;

  const tyre = new TorusGeometry(RALLY_TUNE.wheelRadius - 0.082, 0.082, 8, 18);
  tyre.rotateY(Math.PI * 0.5);
  const rim = new CylinderGeometry(0.205, 0.205, 0.15, 12, 1, false);
  rim.rotateZ(Math.PI * 0.5);
  const hub = new CylinderGeometry(0.067, 0.067, 0.172, 10, 1, false);
  hub.rotateZ(Math.PI * 0.5);
  const brakeDisc = new CylinderGeometry(0.15, 0.15, 0.022, 14, 1, false);
  brakeDisc.rotateZ(Math.PI * 0.5);
  const exhaust = new CylinderGeometry(0.055, 0.072, 0.34, 10, 1, false);
  exhaust.rotateX(Math.PI * 0.5);

  geometryCache = {
    lowerBody: outlined(new BoxGeometry(1.90, 0.42, 3.58), 84, 1.0),
    upperBody: outlined(new BoxGeometry(1.56, 0.68, 1.54), 76, 1.15),
    hood: outlined(new BoxGeometry(1.72, 0.25, 1.20), 78, 1.08),
    roof: outlined(new BoxGeometry(1.48, 0.13, 1.36), 76, 1.15),
    bumper: outlined(new BoxGeometry(1.86, 0.22, 0.24), 72, 1.18),
    sill: outlined(new BoxGeometry(0.13, 0.24, 2.46), 76, 1.12),
    fender: outlined(new BoxGeometry(0.22, 0.34, 0.82), 74, 1.22),
    glassFront: outlined(new BoxGeometry(1.40, 0.52, 0.055), 82, 0.9),
    glassRear: outlined(new BoxGeometry(1.36, 0.44, 0.052), 82, 0.9),
    glassSide: outlined(new BoxGeometry(0.052, 0.43, 1.04), 82, 0.9),
    lamp: outlined(new BoxGeometry(0.42, 0.16, 0.075), 78, 0.95),
    lampBar: outlined(new BoxGeometry(0.88, 0.10, 0.09), 78, 0.95),
    wing: outlined(new BoxGeometry(1.52, 0.10, 0.34), 76, 1.2),
    wingPost: outlined(new BoxGeometry(0.085, 0.37, 0.10), 76, 1.1),
    scoop: outlined(new BoxGeometry(0.54, 0.17, 0.42), 76, 1.18),
    mirror: outlined(new BoxGeometry(0.19, 0.13, 0.28), 78, 1.2),
    skid: outlined(new BoxGeometry(1.38, 0.065, 0.72), 76, 1.05),
    tyre: outlined(tyre, 180, 0.72),
    rim: outlined(rim, 74, 1.24),
    hub: outlined(hub, 74, 1.18),
    brakeDisc: outlined(brakeDisc, 74, 1.0),
    exhaust: outlined(exhaust, 72, 1.15),
  };
  return geometryCache;
}

function rehueRamp(base: RampPreset, target: Color, name: string, amount = 0.56): RampPreset {
  return {
    ...base,
    name,
    colors: base.colors.map((band, index) => {
      const t = amount * lerp(0.76, 1, index / Math.max(1, base.colors.length - 1));
      return band.clone().lerp(target, t);
    }),
    thresholds: [...base.thresholds],
  };
}

interface PartResult {
  mesh: Mesh;
  hull: Mesh | null;
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
  side: number;
  front: boolean;
}

export class RallyCarVisual {
  readonly root = new Group();
  readonly anchors: BikeAnchors;

  private readonly geometry = getGeometry();
  private readonly chassis = new Group();
  private readonly bodyFloat = new Group();
  private readonly frontAxle = new Group();
  private readonly rearAxle = new Group();
  private readonly wheels: WheelAssembly[] = [];
  private readonly ownedMaterials: CelMaterial[] = [];
  private readonly hullMaterials = new Set<THREE.ShaderMaterial>();
  private readonly boostFlames = new Group();

  private readonly bodyMaterial: CelMaterial;
  private readonly accentMaterial: CelMaterial;
  private readonly glassMaterial: CelMaterial;
  private readonly brakeMaterial: CelMaterial;
  private suspensionBob = 0;
  private wheelBlur = 0;

  constructor(opts: RallyCarVisualOptions) {
    const detail = opts.detail ?? 'full';
    const accent = opts.accentColor ?? opts.frameColor.clone().offsetHSL(0.08, 0.12, 0.16);
    this.root.name = opts.name ?? 'rally-car';
    this.chassis.name = 'rally:chassis';
    this.bodyFloat.name = 'rally:sprung-body';
    this.root.add(this.chassis);
    this.chassis.add(this.bodyFloat, this.frontAxle, this.rearAxle);

    const bodyRamp = rehueRamp(RAMPS.frame, opts.frameColor, `rally-body:${this.root.name}`);
    const accentRamp = rehueRamp(RAMPS.marker, accent, `rally-accent:${this.root.name}`, 0.44);

    this.bodyMaterial = this.material(bodyRamp, {
      name: `rally:body:${this.root.name}`,
      matcapMix: 0.28,
      specPower: 18,
      specStrength: 0.34,
      specInk: 0.72,
      identity: {
        color: opts.frameColor,
        height: 1.55,
        fullPx: 56,
        floorPx: 18,
        chroma: 0.56,
        rim: 0.42,
        inkFloor: 0.55,
      },
    });
    this.accentMaterial = this.material(accentRamp, {
      name: `rally:accent:${this.root.name}`,
      matcapMix: 0.18,
      specPower: 14,
      specStrength: 0.25,
      identity: {
        color: accent,
        height: 1.4,
        fullPx: 52,
        floorPx: 17,
        chroma: 0.48,
        rim: 0.34,
        inkFloor: 0.58,
      },
    });
    const metal = this.material(RAMPS.metal, {
      name: `rally:metal:${this.root.name}`,
      matcapMix: 0.42,
      specPower: 14,
      specStrength: 0.36,
      specInk: 0.65,
    });
    const tyre = this.material(RAMPS.tyre, {
      name: `rally:tyre:${this.root.name}`,
      rimStrength: 0.42,
    });
    this.glassMaterial = this.material(RAMPS.lens, {
      name: `rally:glass:${this.root.name}`,
      matcapMix: 0.58,
      specPower: 7,
      specStrength: 0.46,
      specInk: 0.86,
      valueSteps: 4,
      tint: new Color(0x315b73),
      tintStrength: 0.34,
    });
    const headlamp = this.material(RAMPS.marker, {
      name: `rally:headlamp:${this.root.name}`,
      tint: new Color(0xffe7b0),
      tintStrength: 0.78,
      rimStrength: 0.78,
      noShadow: true,
    });
    this.brakeMaterial = this.material(RAMPS.marker, {
      name: `rally:brake:${this.root.name}`,
      tint: new Color(0xe43f4f),
      tintStrength: 0.72,
      rimStrength: 0.4,
      noShadow: true,
    });

    this.buildBody(metal, headlamp, detail);
    this.buildWheel(this.frontAxle, +1, true, tyre, metal, detail);
    this.buildWheel(this.frontAxle, -1, true, tyre, metal, detail);
    this.buildWheel(this.rearAxle, +1, false, tyre, metal, detail);
    this.buildWheel(this.rearAxle, -1, false, tyre, metal, detail);

    const barLeft = new Object3D();
    const barRight = new Object3D();
    const pedalLeft = new Object3D();
    const pedalRight = new Object3D();
    const seat = new Object3D();
    const frontContact = new Object3D();
    const rearContact = new Object3D();
    barLeft.name = 'anchor:steeringLeft';
    barRight.name = 'anchor:steeringRight';
    pedalLeft.name = 'anchor:pedalClutch';
    pedalRight.name = 'anchor:pedalThrottle';
    seat.name = 'anchor:driverSeat';
    frontContact.name = 'anchor:frontAxleContact';
    rearContact.name = 'anchor:rearAxleContact';
    barLeft.position.set(0.30, 0.57, 0.15);
    barRight.position.set(0.57, 0.57, 0.15);
    pedalLeft.position.set(0.31, 0.08, 0.54);
    pedalRight.position.set(0.53, 0.08, 0.54);
    seat.position.set(0.43, 0.42, -0.35);
    frontContact.position.set(0, -RALLY_TUNE.rideHeight - RALLY_TUNE.wheelRadius, RALLY_TUNE.wheelbase * 0.5);
    rearContact.position.set(0, -RALLY_TUNE.rideHeight - RALLY_TUNE.wheelRadius, -RALLY_TUNE.wheelbase * 0.5);
    this.chassis.add(barLeft, barRight, pedalLeft, pedalRight, seat, frontContact, rearContact);

    this.anchors = {
      barLeft,
      barRight,
      pedalLeft,
      pedalRight,
      seat,
      frame: this.chassis,
      frontContact,
      rearContact,
    };
  }

  private material(preset: RampPreset, opts: CelOptions): CelMaterial {
    const material = new CelMaterial(preset, opts);
    this.ownedMaterials.push(material);
    return material;
  }

  private addPart(
    parent: Object3D,
    geometry: BufferGeometry,
    material: CelMaterial,
    preset: RampPreset,
    name: string,
    position: [number, number, number],
    rotation: [number, number, number] = [0, 0, 0],
    scale: [number, number, number] = [1, 1, 1],
    outline = true,
  ): PartResult {
    const mesh = new Mesh(geometry, material);
    mesh.name = name;
    mesh.position.set(...position);
    mesh.rotation.set(...rotation);
    mesh.scale.set(...scale);
    mesh.castShadow = !material.celOptions.noShadow;
    mesh.receiveShadow = true;
    registerNprMesh(mesh, material);

    let hull: Mesh | null = null;
    if (outline) {
      hull = attachOutline(mesh, preset, material.celOptions);
      if (hull) {
        this.hullMaterials.add(hull.material as THREE.ShaderMaterial);
        parent.add(hull);
      }
    }
    parent.add(mesh);
    return { mesh, hull };
  }

  private buildBody(metal: CelMaterial, headlamp: CelMaterial, detail: 'full' | 'reduced'): void {
    const b = this.bodyFloat;
    const bodyPreset = this.bodyMaterial.preset;
    const accentPreset = this.accentMaterial.preset;

    this.addPart(b, this.geometry.lowerBody, this.bodyMaterial, bodyPreset, 'rally:lower-body', [0, 0.05, 0]);
    this.addPart(b, this.geometry.hood, this.bodyMaterial, bodyPreset, 'rally:hood', [0, 0.42, 0.92], [-0.035, 0, 0]);
    this.addPart(b, this.geometry.upperBody, this.bodyMaterial, bodyPreset, 'rally:cabin', [0, 0.77, -0.27], [-0.015, 0, 0], [1, 1, 0.98]);
    this.addPart(b, this.geometry.roof, this.bodyMaterial, bodyPreset, 'rally:roof', [0, 1.18, -0.29]);

    this.addPart(b, this.geometry.bumper, this.accentMaterial, accentPreset, 'rally:front-bumper', [0, -0.04, 1.82]);
    this.addPart(b, this.geometry.bumper, this.accentMaterial, accentPreset, 'rally:rear-bumper', [0, -0.02, -1.82]);
    this.addPart(b, this.geometry.skid, metal, RAMPS.metal, 'rally:front-skid', [0, -0.235, 1.34], [0.08, 0, 0]);

    for (const side of [-1, 1]) {
      this.addPart(b, this.geometry.sill, this.accentMaterial, accentPreset, `rally:sill:${side}`, [side * 0.925, -0.08, -0.08]);
      this.addPart(b, this.geometry.fender, this.bodyMaterial, bodyPreset, `rally:fender-front:${side}`, [side * 0.89, 0.17, 1.27]);
      this.addPart(b, this.geometry.fender, this.bodyMaterial, bodyPreset, `rally:fender-rear:${side}`, [side * 0.89, 0.17, -1.27]);
      this.addPart(b, this.geometry.mirror, this.accentMaterial, accentPreset, `rally:mirror:${side}`, [side * 0.90, 0.88, 0.19]);
      this.addPart(b, this.geometry.glassSide, this.glassMaterial, RAMPS.lens, `rally:side-glass:${side}`, [side * 0.792, 0.89, -0.23]);
    }

    this.addPart(b, this.geometry.glassFront, this.glassMaterial, RAMPS.lens, 'rally:windshield', [0, 0.91, 0.515], [-0.48, 0, 0]);
    this.addPart(b, this.geometry.glassRear, this.glassMaterial, RAMPS.lens, 'rally:rear-window', [0, 0.91, -1.02], [0.44, 0, 0]);

    // A single bold livery stroke carries identity at race distance better than
    // tiny sponsor decals, and remains entirely procedural.
    this.addPart(b, this.geometry.sill, this.accentMaterial, accentPreset, 'rally:hood-stripe', [0, 0.558, 0.94], [-0.035, 0, Math.PI * 0.5], [0.72, 0.86, 0.55]);
    this.addPart(b, this.geometry.sill, this.accentMaterial, accentPreset, 'rally:roof-stripe', [0, 1.258, -0.29], [0, 0, Math.PI * 0.5], [0.78, 0.76, 0.48]);

    for (const side of [-1, 1]) {
      this.addPart(b, this.geometry.lamp, headlamp, RAMPS.marker, `rally:headlight:${side}`, [side * 0.51, 0.30, 1.805], [0, 0, 0], [1, 1, 1], true);
      this.addPart(b, this.geometry.lamp, this.brakeMaterial, RAMPS.marker, `rally:taillight:${side}`, [side * 0.51, 0.31, -1.805], [0, Math.PI, 0], [0.88, 0.9, 1], true);
    }

    this.addPart(b, this.geometry.scoop, this.accentMaterial, accentPreset, 'rally:roof-scoop', [0, 1.34, -0.05], [0.04, 0, 0]);
    this.addPart(b, this.geometry.wingPost, metal, RAMPS.metal, 'rally:wing-post-left', [0.48, 0.56, -1.58]);
    this.addPart(b, this.geometry.wingPost, metal, RAMPS.metal, 'rally:wing-post-right', [-0.48, 0.56, -1.58]);
    this.addPart(b, this.geometry.wing, this.accentMaterial, accentPreset, 'rally:rear-wing', [0, 0.78, -1.60], [-0.08, 0, 0]);

    if (detail === 'full') {
      this.addPart(b, this.geometry.lampBar, headlamp, RAMPS.marker, 'rally:lamp-pod', [0, 0.51, 1.52], [-0.06, 0, 0]);
      this.addPart(b, this.geometry.skid, this.accentMaterial, accentPreset, 'rally:door-number-left', [0.966, 0.47, -0.26], [0, 0, Math.PI * 0.5], [0.55, 0.52, 0.55]);
      this.addPart(b, this.geometry.skid, this.accentMaterial, accentPreset, 'rally:door-number-right', [-0.966, 0.47, -0.26], [0, 0, Math.PI * 0.5], [0.55, 0.52, 0.55]);
    }

    // Twin exhausts and stylised hard-edged boost flames.
    for (const side of [-1, 1]) {
      this.addPart(b, this.geometry.exhaust, metal, RAMPS.metal, `rally:exhaust:${side}`, [side * 0.44, -0.11, -1.83], [Math.PI * 0.5, 0, 0], [1, 1, 1], true);
      const flame = this.addPart(
        this.boostFlames,
        this.geometry.exhaust,
        this.accentMaterial,
        accentPreset,
        `rally:boost-flame:${side}`,
        [side * 0.44, -0.11, -2.10],
        [Math.PI * 0.5, 0, 0],
        [0.72, 0.72, 1.15],
        false,
      ).mesh;
      flame.castShadow = false;
      flame.receiveShadow = false;
    }
    this.boostFlames.visible = false;
    b.add(this.boostFlames);
  }

  private buildWheel(
    axle: Object3D,
    side: number,
    front: boolean,
    tyreMaterial: CelMaterial,
    metal: CelMaterial,
    detail: 'full' | 'reduced',
  ): void {
    const z = front ? RALLY_TUNE.wheelbase * 0.5 : -RALLY_TUNE.wheelbase * 0.5;
    axle.position.z = 0;

    const steer = new Group();
    steer.name = `rally:${front ? 'front' : 'rear'}:${side > 0 ? 'left' : 'right'}:steer`;
    steer.position.set(side * RALLY_TUNE.trackWidth * 0.5, -RALLY_TUNE.rideHeight, z);
    axle.add(steer);

    const spin = new Group();
    spin.name = `${steer.name}:spin`;
    steer.add(spin);

    this.addPart(spin, this.geometry.tyre, tyreMaterial, RAMPS.tyre, `${steer.name}:tyre`, [0, 0, 0]);
    this.addPart(spin, this.geometry.rim, metal, RAMPS.metal, `${steer.name}:rim`, [0, 0, 0]);
    this.addPart(spin, this.geometry.hub, this.accentMaterial, this.accentMaterial.preset, `${steer.name}:hub`, [0, 0, 0]);
    if (detail === 'full') {
      this.addPart(spin, this.geometry.brakeDisc, metal, RAMPS.metal, `${steer.name}:disc`, [side * -0.018, 0, 0], [0, 0, 0], [1, 1, 1], false);
    }

    this.wheels.push({ steer, spin, side, front });
  }

  update(state: RallyCarVisualState, dt: number, cameraDistance = 7): void {
    const frontY = (state.frontCompression - 0.32) * RALLY_TUNE.suspensionTravel * 0.72;
    const rearY = (state.rearCompression - 0.32) * RALLY_TUNE.suspensionTravel * 0.72;
    this.frontAxle.position.y = frontY;
    this.rearAxle.position.y = rearY;

    const averageCompression = (state.frontCompression + state.rearCompression) * 0.5;
    this.suspensionBob = dampHL(this.suspensionBob, (averageCompression - 0.32) * -0.075, 0.06, dt);
    this.bodyFloat.position.y = this.suspensionBob;
    this.bodyFloat.rotation.x = dampHL(
      this.bodyFloat.rotation.x,
      (state.rearCompression - state.frontCompression) * 0.065,
      0.08,
      dt,
    );

    this.wheelBlur = dampHL(this.wheelBlur, clamp01((state.speed - 12) / 23), 0.12, dt);
    for (const wheel of this.wheels) {
      wheel.steer.rotation.y = wheel.front ? state.steerAngle : 0;
      wheel.spin.rotation.x = wheel.front ? state.frontSpin : state.rearSpin;
      // A tiny squash at speed reads as drawn rotational smear without a
      // transparent post-process and remains silhouette-safe.
      const smear = this.wheelBlur * (cameraDistance < 18 ? 0.035 : 0.018);
      wheel.spin.scale.set(1, 1 + smear, 1 - smear * 0.42);
    }

    const braking = clamp01(state.brake);
    this.brakeMaterial.uniforms.uTintStrength.value = lerp(0.45, 1.0, braking);
    this.brakeMaterial.uniforms.uRimStrength.value = lerp(0.25, 1.05, braking);

    this.boostFlames.visible = state.boosting;
    if (state.boosting) {
      const pulse = 0.88 + Math.sin(performance.now() * 0.045) * 0.12;
      this.boostFlames.scale.set(pulse, pulse, lerp(1.0, 1.42, clamp01(state.speed / 35)));
    }

    const crashSquash = state.crashed ? 0.94 : 1;
    this.chassis.scale.y = dampHL(this.chassis.scale.y, crashSquash, 0.08, dt);
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
    this.hullMaterials.clear();
    for (const material of this.ownedMaterials) disposeCelMaterial(material);
    this.ownedMaterials.length = 0;
  }
}
