/**
 * RallyDriverRig — a compact procedural cockpit character.
 *
 * The driver is intentionally read through the glass as a few strong anime
 * shapes: helmet, visor, shoulders, gloves and steering wheel. That keeps the
 * silhouette clean at speed, removes the old exposed bicycle mannequin, and
 * costs a fraction of a full-body skinning rig while still reacting to steer,
 * suspension travel, air time and crashes.
 */

import {
  BoxGeometry,
  BufferGeometry,
  Color,
  CylinderGeometry,
  Group,
  Mesh,
  Object3D,
  ShaderMaterial,
  SphereGeometry,
  TorusGeometry,
} from 'three';

import {
  BikeMode,
  type BikeAnchors,
  type BikeState,
  type IRiderRig,
  type TrickState,
} from '../game/Contracts';
import {
  CelMaterial,
  attachOutline,
  disposeCelMaterial,
  registerNprMesh,
  type CelOptions,
} from '../npr/CelMaterial';
import { prepareOutlineGeometry } from '../npr/OutlineGeometry';
import { RAMPS, type RampPreset } from '../npr/Palette';
import { clamp01, dampHL } from '../core/MathX';

interface DriverGeometry {
  torso: BufferGeometry;
  shoulder: BufferGeometry;
  helmet: BufferGeometry;
  visor: BufferGeometry;
  arm: BufferGeometry;
  glove: BufferGeometry;
  steeringWheel: BufferGeometry;
  wheelHub: BufferGeometry;
}

let geometryCache: DriverGeometry | null = null;

function prep<T extends BufferGeometry>(geo: T, angle = 120, gain = 1): T {
  prepareOutlineGeometry(geo, { maxWeldAngle: angle, curvatureGain: gain });
  return geo;
}

function geometries(): DriverGeometry {
  if (geometryCache) return geometryCache;
  const arm = new CylinderGeometry(0.055, 0.068, 0.42, 9, 1, false);
  const steeringWheel = new TorusGeometry(0.18, 0.025, 7, 16);
  geometryCache = {
    torso: prep(new BoxGeometry(0.46, 0.48, 0.26), 78, 1.12),
    shoulder: prep(new BoxGeometry(0.60, 0.16, 0.25), 78, 1.15),
    helmet: prep(new SphereGeometry(0.205, 14, 10), 180, 0.84),
    visor: prep(new BoxGeometry(0.31, 0.105, 0.055), 86, 0.92),
    arm: prep(arm, 110, 0.9),
    glove: prep(new SphereGeometry(0.068, 9, 7), 180, 0.92),
    steeringWheel: prep(steeringWheel, 180, 0.72),
    wheelHub: prep(new CylinderGeometry(0.045, 0.045, 0.08, 9), 82, 1),
  };
  return geometryCache;
}

export interface RallyDriverRigOptions {
  jersey?: number;
  accent?: number;
  name?: string;
  phase?: number;
}

export class RallyDriverRig implements IRiderRig {
  readonly object = new Group();

  private readonly geo = geometries();
  private readonly materials: CelMaterial[] = [];
  private readonly hullMaterials = new Set<ShaderMaterial>();
  private readonly torsoRoot = new Group();
  private readonly head = new Group();
  private readonly leftArm = new Group();
  private readonly rightArm = new Group();
  private readonly steering = new Group();
  private readonly leftHand = new Group();
  private readonly rightHand = new Group();
  private readonly phase: number;

  private jerseyMaterial: CelMaterial;
  private helmetMaterial: CelMaterial;
  private steerVisual = 0;
  private compression = 0;
  private attached = false;

  constructor(opts: RallyDriverRigOptions = {}) {
    const jersey = new Color(opts.jersey ?? 0xe0574c);
    const accent = new Color(opts.accent ?? 0xff9a5c);
    this.phase = opts.phase ?? 0;
    this.object.name = opts.name ?? 'rally-driver';

    this.jerseyMaterial = this.material(RAMPS.jerseyPlayer, {
      name: `${this.object.name}:suit`,
      tint: jersey,
      tintStrength: 0.72,
      valueSteps: 4,
      identity: {
        color: jersey,
        height: 1.1,
        fullPx: 48,
        floorPx: 16,
        chroma: 0.58,
        rim: 0.46,
        inkFloor: 0.58,
      },
    });
    this.helmetMaterial = this.material(RAMPS.helmet, {
      name: `${this.object.name}:helmet`,
      tint: accent,
      tintStrength: 0.67,
      matcapMix: 0.32,
      specPower: 12,
      specStrength: 0.28,
      specInk: 0.72,
    });
    const visorMaterial = this.material(RAMPS.lens, {
      name: `${this.object.name}:visor`,
      tint: new Color(0x4e8fa1),
      tintStrength: 0.45,
      matcapMix: 0.58,
      valueSteps: 3,
      noShadow: true,
    });
    const gloveMaterial = this.material(RAMPS.rubber, {
      name: `${this.object.name}:gloves`,
      tint: accent,
      tintStrength: 0.22,
    });
    const wheelMaterial = this.material(RAMPS.metal, {
      name: `${this.object.name}:wheel`,
      matcapMix: 0.32,
      specPower: 13,
      specStrength: 0.25,
    });

    this.object.add(this.torsoRoot);
    this.part(this.torsoRoot, this.geo.torso, this.jerseyMaterial, RAMPS.jerseyPlayer, 'driver:torso', [0, 0.22, 0]);
    this.part(this.torsoRoot, this.geo.shoulder, this.jerseyMaterial, RAMPS.jerseyPlayer, 'driver:shoulders', [0, 0.43, 0.035]);

    this.head.position.set(0, 0.68, 0.03);
    this.torsoRoot.add(this.head);
    this.part(this.head, this.geo.helmet, this.helmetMaterial, RAMPS.helmet, 'driver:helmet', [0, 0, 0]);
    this.part(this.head, this.geo.visor, visorMaterial, RAMPS.lens, 'driver:visor', [0, 0.025, 0.186], [-0.05, 0, 0]);

    this.leftArm.position.set(0.245, 0.46, 0.05);
    this.rightArm.position.set(-0.245, 0.46, 0.05);
    this.torsoRoot.add(this.leftArm, this.rightArm);
    this.part(this.leftArm, this.geo.arm, this.jerseyMaterial, RAMPS.jerseyPlayer, 'driver:left-arm', [0, -0.15, 0.12], [1.02, 0, -0.12]);
    this.part(this.rightArm, this.geo.arm, this.jerseyMaterial, RAMPS.jerseyPlayer, 'driver:right-arm', [0, -0.15, 0.12], [1.02, 0, 0.12]);

    this.leftHand.position.set(0.16, 0.24, 0.39);
    this.rightHand.position.set(-0.16, 0.24, 0.39);
    this.torsoRoot.add(this.leftHand, this.rightHand);
    this.part(this.leftHand, this.geo.glove, gloveMaterial, RAMPS.rubber, 'driver:left-glove', [0, 0, 0]);
    this.part(this.rightHand, this.geo.glove, gloveMaterial, RAMPS.rubber, 'driver:right-glove', [0, 0, 0]);

    this.steering.position.set(0, 0.25, 0.42);
    this.steering.rotation.x = -0.22;
    this.torsoRoot.add(this.steering);
    this.part(this.steering, this.geo.steeringWheel, wheelMaterial, RAMPS.metal, 'driver:steering-wheel', [0, 0, 0]);
    this.part(this.steering, this.geo.wheelHub, wheelMaterial, RAMPS.metal, 'driver:steering-hub', [0, 0, 0], [Math.PI * 0.5, 0, 0]);
  }

  private material(preset: RampPreset, opts: CelOptions): CelMaterial {
    const m = new CelMaterial(preset, opts);
    this.materials.push(m);
    return m;
  }

  private part(
    parent: Object3D,
    geometry: BufferGeometry,
    material: CelMaterial,
    preset: RampPreset,
    name: string,
    position: [number, number, number],
    rotation: [number, number, number] = [0, 0, 0],
  ): Mesh {
    const mesh = new Mesh(geometry, material);
    mesh.name = name;
    mesh.position.set(...position);
    mesh.rotation.set(...rotation);
    mesh.castShadow = !material.celOptions.noShadow;
    mesh.receiveShadow = true;
    registerNprMesh(mesh, material);
    const hull = attachOutline(mesh, preset, material.celOptions);
    if (hull) {
      hull.position.copy(mesh.position);
      hull.rotation.copy(mesh.rotation);
      hull.scale.copy(mesh.scale);
      this.hullMaterials.add(hull.material as ShaderMaterial);
      parent.add(hull);
    }
    parent.add(mesh);
    return mesh;
  }

  attach(anchors: BikeAnchors): void {
    if (this.object.parent !== anchors.frame) anchors.frame.add(this.object);
    this.object.position.copy(anchors.seat.position);
    // The seat anchor is authored at cushion height. Lift the rig so its pelvis
    // sits there and keep the helmet behind the windshield header.
    this.object.position.y -= 0.05;
    this.object.rotation.set(0, 0, 0);
    this.attached = true;
  }

  update(state: BikeState, _trick: TrickState, dt: number, time: number): void {
    this.steerVisual = dampHL(this.steerVisual, state.steerAngle, 0.055, dt);
    const averageCompression = (state.front.compression + state.rear.compression) * 0.5;
    this.compression = dampHL(this.compression, averageCompression, 0.075, dt);

    const wheelTurn = -this.steerVisual * 2.2;
    this.steering.rotation.z = wheelTurn;
    this.leftHand.rotation.z = wheelTurn;
    this.rightHand.rotation.z = wheelTurn;

    // Hands remain visually welded to the wheel rim while shoulders counter
    // rotate. The small asymmetry makes steering readable through the glass.
    this.leftArm.rotation.z = -this.steerVisual * 0.34;
    this.rightArm.rotation.z = -this.steerVisual * 0.34;
    this.leftArm.rotation.y = this.steerVisual * 0.20;
    this.rightArm.rotation.y = this.steerVisual * 0.20;

    const suspensionPunch = clamp01(this.compression) * 0.055;
    const airFloat = state.mode === BikeMode.Airborne ? 0.035 : 0;
    const idle = Math.sin(time * 2.2 + this.phase) * 0.004;
    this.torsoRoot.position.y = -suspensionPunch + airFloat + idle;
    this.torsoRoot.rotation.x = dampHL(
      this.torsoRoot.rotation.x,
      state.mode === BikeMode.Airborne ? -0.055 : state.pitch * -0.16,
      0.12,
      dt,
    );
    this.torsoRoot.rotation.z = dampHL(
      this.torsoRoot.rotation.z,
      state.mode === BikeMode.Crashing ? -state.crashDirection.x * 0.45 : state.lean * -0.32,
      0.10,
      dt,
    );

    // Look through the corner and stabilise against chassis roll.
    this.head.rotation.y = dampHL(this.head.rotation.y, -this.steerVisual * 0.72, 0.11, dt);
    this.head.rotation.z = dampHL(this.head.rotation.z, state.lean * -0.38, 0.13, dt);
    this.head.rotation.x = dampHL(
      this.head.rotation.x,
      state.mode === BikeMode.Crashing ? 0.24 : clamp01(state.landingImpact) * -0.12,
      0.09,
      dt,
    );

    if (!this.attached) {
      // Ghost/standalone fallback: follow the synthetic vehicle pose directly.
      this.object.position.copy(state.position);
      this.object.quaternion.copy(state.orientation);
      this.object.translateX(0.43);
      this.object.translateY(0.37);
      this.object.translateZ(-0.35);
    }
  }

  setJerseyColor(jersey: number, accent: number): void {
    this.jerseyMaterial.setTint(new Color(jersey), 0.76);
    this.helmetMaterial.setTint(new Color(accent), 0.70);
  }

  dispose(): void {
    this.object.removeFromParent();
    for (const material of this.hullMaterials) material.dispose();
    this.hullMaterials.clear();
    for (const material of this.materials) disposeCelMaterial(material);
    this.materials.length = 0;
  }
}
