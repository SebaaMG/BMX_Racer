/**
 * Game — the orchestrator.
 *
 * MILESTONE 0. Right now this stands up the engine, the sky, and a small
 * calibration scene used to tune the cel ramps by eye. Subsystems land here as
 * they come online; the wiring order is fixed:
 *
 *   input → race director → per-racer input → physics (fixed step)
 *   → rider rigs → camera → fx → hud → post pipeline
 */

import {
  Color,
  IcosahedronGeometry,
  Mesh,
  Object3D,
  PerspectiveCamera,
  PlaneGeometry,
  SphereGeometry,
  TorusKnotGeometry,
  Vector3,
  CylinderGeometry,
  BoxGeometry,
} from 'three';

import { Engine } from '../core/Engine';
import { Input } from '../core/Input';
import { Sky } from '../npr/Sky';
import { NPR, updateNprGlobals } from '../npr/NprGlobals';
import { initGeneratedTextures } from '../npr/GeneratedTextures';
import { makeCelMesh } from '../npr/CelMaterial';
import { finalizeGeometry } from '../npr/OutlineGeometry';
import { RAMPS } from '../npr/Palette';
import { dampHL } from '../core/MathX';

export interface GameOptions {
  params: URLSearchParams;
}

/**
 * The surface the Playwright harness drives. Kept deliberately small: take the
 * clock, set a named situation, step a fixed dt, shoot. Anything the harness
 * needs that isn't expressible as "a named pose" belongs in the game, not here.
 */
export interface CaptureApi {
  takeControl(): void;
  releaseControl(): void;
  step(dt: number): void;
  setPose(name: string): boolean;
  setSequence(name: string): boolean;
  listPoses(): string[];
  listSequences(): string[];
}

export class Game {
  private engine: Engine;
  private input: Input;
  private sky!: Sky;
  private calibration = new Object3D();

  private orbit = { yaw: 0.6, pitch: 0.28, dist: 14, target: new Vector3(0, 1.4, 0) };
  private smoothYaw = 0.6;
  private smoothPitch = 0.28;

  /** Camera framings the capture harness can request by name. */
  private poses: Record<string, { yaw: number; pitch: number; dist: number; target: [number, number, number] }> = {
    'summit-wide': { yaw: 0.0, pitch: 0.16, dist: 46, target: [0, 6, -40] },
    'summit-rider': { yaw: 0.9, pitch: 0.2, dist: 9, target: [-10, 1.6, 0] },
    'rider-closeup': { yaw: 2.3, pitch: 0.12, dist: 3.6, target: [-3.4, 1.5, 0] },
    'rider-threequarter': { yaw: 0.95, pitch: 0.22, dist: 5.2, target: [0, 1.5, 0] },
    'bike-detail': { yaw: 1.9, pitch: 0.06, dist: 3.0, target: [3.4, 1.4, 0] },
    'treeline-silhouette': { yaw: 0.35, pitch: 0.05, dist: 18, target: [6.8, 1.6, 0] },
    'valley-vista': { yaw: 0.0, pitch: 0.10, dist: 120, target: [0, 30, -400] },
    'scree-speed': { yaw: 1.4, pitch: 0.18, dist: 12, target: [-6.8, 1.5, 0] },
    'switchback-lean': { yaw: 2.9, pitch: 0.25, dist: 8, target: [0, 1.5, 3.6] },
    'rockgarden-low': { yaw: 0.6, pitch: -0.06, dist: 7, target: [3.4, 1.0, 3.6] },
    'tabletop-air': { yaw: 1.1, pitch: 0.35, dist: 14, target: [0, 4, 0] },
    'ravine-gap': { yaw: 2.1, pitch: 0.30, dist: 20, target: [0, 3, 0] },
    'ridge-exposure': { yaw: 0.2, pitch: 0.34, dist: 30, target: [0, 8, -60] },
    streambed: { yaw: 1.7, pitch: 0.09, dist: 9, target: [-3.4, 1.2, 3.6] },
    'finish-sprint': { yaw: 3.4, pitch: 0.14, dist: 11, target: [0, 1.6, 0] },
    crash: { yaw: 1.25, pitch: 0.2, dist: 6, target: [0, 1.2, 0] },
  };

  /** Motion setups. In milestone 0 these are slow camera moves over the rig. */
  private sequences: Record<string, () => void> = {};

  private captureControlled = false;

  readonly capture: CaptureApi;

  constructor(engine: Engine, _options: GameOptions) {
    this.engine = engine;
    this.input = new Input(window);

    this.sequences = {
      launch: () => this.applyPose('summit-rider'),
      switchback: () => this.applyPose('switchback-lean'),
      'tabletop-air': () => this.applyPose('tabletop-air'),
      landing: () => this.applyPose('rockgarden-low'),
      crash: () => this.applyPose('crash'),
      'scree-speed': () => this.applyPose('scree-speed'),
      'trick-360': () => this.applyPose('rider-threequarter'),
      'pack-race': () => this.applyPose('summit-wide'),
    };

    this.capture = {
      takeControl: () => {
        this.captureControlled = true;
        this.input.setScripted(true);
        this.engine.stop();
        this.engine.setFixedPixelRatio(this.engine.stats.pixelRatio || 2);
      },
      releaseControl: () => {
        this.captureControlled = false;
        this.input.setScripted(false);
        this.engine.setFixedPixelRatio(null);
        this.engine.start();
      },
      step: (dt: number) => this.engine.stepManual(dt),
      setPose: (name: string) => this.applyPose(name),
      setSequence: (name: string) => {
        const s = this.sequences[name];
        if (!s) return false;
        s();
        return true;
      },
      listPoses: () => Object.keys(this.poses),
      listSequences: () => Object.keys(this.sequences),
    };
  }

  private applyPose(name: string): boolean {
    const p = this.poses[name];
    if (!p) return false;
    this.orbit.yaw = this.smoothYaw = p.yaw;
    this.orbit.pitch = this.smoothPitch = p.pitch;
    this.orbit.dist = p.dist;
    this.orbit.target.set(p.target[0], p.target[1], p.target[2]);
    return true;
  }

  async load(progress: (p: number, label?: string) => void): Promise<void> {
    progress(0.05, 'Generating textures');
    initGeneratedTextures(NPR);
    await frame();

    progress(0.25, 'Building sky');
    this.sky = new Sky();
    this.sky.buildShafts();
    this.engine.scene.add(this.sky.group);
    await frame();

    progress(0.55, 'Calibration scene');
    this.buildCalibrationScene();
    await frame();

    progress(0.85, 'Wiring loop');
    this.engine.onFixedUpdate((dt) => this.fixedUpdate(dt));
    this.engine.onRender((dt, alpha, elapsed) => this.render(dt, alpha, elapsed));
    await frame();
  }

  /**
   * A row of primitives, one per ramp preset, on a ground plane. This is what
   * the ramp thresholds and band colours are tuned against — you cannot tune a
   * cel ramp against a screenshot of a mountain, because you can't tell whether
   * a bad read is the ramp or the geometry. Isolate it first.
   */
  private buildCalibrationScene(): void {
    this.calibration.name = 'calibration';

    // Ground.
    const ground = finalizeGeometry(new PlaneGeometry(400, 400, 1, 1).rotateX(-Math.PI / 2), {
      tolerance: 1e-3,
    });
    const g = makeCelMesh(ground, 'grass', { name: 'ground', outlineWidth: 0 });
    g.mesh.receiveShadow = true;
    this.calibration.add(g.group);

    const presets: (keyof typeof RAMPS)[] = [
      'rock', 'dirt', 'scree', 'snow', 'foliage', 'bark',
      'skin', 'jerseyPlayer', 'cloth', 'helmet', 'frame', 'metal', 'tyre',
    ];

    presets.forEach((name, i) => {
      const col = i % 7;
      const row = Math.floor(i / 7);
      const x = (col - 3) * 3.4;
      const z = row * 3.6 - 1.8;

      // Alternate shapes so the ramp is judged on both a smooth sphere (where
      // the terminator shows) and a hard-surface form (where the outline and
      // curvature weighting show).
      let geo;
      if (i % 3 === 0) geo = new SphereGeometry(1.1, 48, 32);
      else if (i % 3 === 1) geo = new TorusKnotGeometry(0.72, 0.26, 128, 24);
      else geo = new IcosahedronGeometry(1.2, 1);

      finalizeGeometry(geo, { tolerance: 1e-4, maxWeldAngle: 180, ao: true });

      const opts =
        name === 'metal' || name === 'frame' || name === 'helmet'
          ? { matcapMix: 0.42 }
          : {};

      const m = makeCelMesh(geo, name, { ...opts, name: `cal:${name}` });
      m.group.position.set(x, 1.4, z);
      this.calibration.add(m.group);

      // A plinth under each, so there's a hard-surface object with a real
      // cast shadow relationship in every test frame.
      const plinth = finalizeGeometry(new CylinderGeometry(1.35, 1.5, 0.36, 12), {
        tolerance: 1e-4,
        maxWeldAngle: 70,
      });
      const p = makeCelMesh(plinth, 'wood', { name: `plinth:${name}` });
      p.group.position.set(x, 0.18, z);
      this.calibration.add(p.group);
    });

    // A tall slab to catch the sun rake and show the fog banding at distance.
    for (let i = 0; i < 7; i++) {
      const d = 90 + i * 260;
      const geo = finalizeGeometry(new BoxGeometry(320, 60 + i * 40, 30), {
        tolerance: 1e-3,
        maxWeldAngle: 70,
      });
      const r = makeCelMesh(geo, 'rock', { name: `ridge${i}` });
      r.group.position.set((i % 2 === 0 ? -1 : 1) * 40, (60 + i * 40) / 2 - 8, -d);
      this.calibration.add(r.group);
    }

    this.engine.scene.add(this.calibration);
  }

  private fixedUpdate(_dt: number): void {
    // Physics lands here in milestone 2.
  }

  private render(dt: number, _alpha: number, elapsed: number): void {
    this.input.update(dt);

    // Temporary orbit camera for calibration.
    if (!this.captureControlled) {
      const i = this.input.intent;
      this.orbit.yaw += i.steer * dt * 1.4;
      this.orbit.pitch += i.pitchLean * -0.5 * dt;
      this.orbit.pitch = Math.max(-0.35, Math.min(1.25, this.orbit.pitch));
      this.orbit.dist *= 1 + (i.brakeRear - i.pedal) * dt * 0.9;
      this.orbit.dist = Math.max(3, Math.min(600, this.orbit.dist));
    } else {
      // Sequences in milestone 0 are a slow arc, so motion frames have motion.
      this.orbit.yaw += dt * 0.35;
    }

    this.smoothYaw = dampHL(this.smoothYaw, this.orbit.yaw, 0.06, dt);
    this.smoothPitch = dampHL(this.smoothPitch, this.orbit.pitch, 0.06, dt);

    const cam = this.engine.camera as PerspectiveCamera;
    const cy = Math.cos(this.smoothPitch);
    cam.position.set(
      this.orbit.target.x + Math.sin(this.smoothYaw) * cy * this.orbit.dist,
      this.orbit.target.y + Math.sin(this.smoothPitch) * this.orbit.dist,
      this.orbit.target.z + Math.cos(this.smoothYaw) * cy * this.orbit.dist,
    );
    cam.lookAt(this.orbit.target);

    updateNprGlobals(elapsed, cam, this.engine.renderSize.x, this.engine.renderSize.y);
    this.sky.update(cam, 0.55);

    const r = this.engine.renderer;
    r.setRenderTarget(null);
    r.clear(true, true, false);
    r.render(this.engine.scene, cam);

    this.input.clearEdges();
  }

  dispose(): void {
    this.input.dispose();
    this.sky.dispose();
  }
}

function frame(): Promise<void> {
  return new Promise((res) => requestAnimationFrame(() => res()));
}
