/**
 * RallyStageFurniture — car-scale roadside language layered over the original
 * mountain course.
 *
 * The ribbon is already 14–23 m wide through most of the playable stage. Its
 * problem is not literal width; it still reads as a downhill trail because the
 * objects beside it are sparse and bike-scaled. These instanced delineators,
 * outside-corner boards and safety ribbons establish a closed rally stage while
 * preserving the hand-built dawn-gold art direction and zero-asset rule.
 */

import {
  BoxGeometry,
  BufferGeometry,
  CylinderGeometry,
  Group,
  InstancedBufferAttribute,
  InstancedMesh,
  Matrix4,
  Vector3,
} from 'three';

import type { ITerrain, TrackSampleResult } from '../game/Contracts';
import {
  CelMaterial,
  attachOutline,
  disposeCelMaterial,
  registerNprMesh,
  type CelOptions,
} from '../npr/CelMaterial';
import { finalizeGeometry } from '../npr/OutlineGeometry';
import { RAMPS } from '../npr/Palette';
import { clamp } from '../core/MathX';
import { TrackSpline, createTrackSample } from './TrackSpline';

const _left = new Vector3();
const _up = new Vector3();
const _forward = new Vector3();
const _right = new Vector3();
const _position = new Vector3();
const _positionB = new Vector3();
const _mid = new Vector3();
const _scale = new Vector3();
const _matrix = new Matrix4();

interface InstanceGroup {
  geometry: BufferGeometry;
  material: CelMaterial;
  mesh: InstancedMesh;
}

export class RallyStageFurniture {
  readonly object = new Group();

  private readonly groups: InstanceGroup[] = [];
  private readonly sample: TrackSampleResult = createTrackSample();
  private readonly sampleB: TrackSampleResult = createTrackSample();

  constructor(private spline: TrackSpline, private terrain: ITerrain) {
    this.object.name = 'rally-stage-furniture';
    this.buildDelineators();
    this.buildCornerBoards();
    this.buildSafetyRibbon();
  }

  private frameAt(distance: number, out: Matrix4, position: Vector3, scale: Vector3): Matrix4 {
    const s = this.spline.sampleAtDistance(distance, this.sample);
    this.spline.surfaceLeftAt(distance, _left);
    _up.copy(s.up).normalize();
    _forward.copy(s.tangent).addScaledVector(_up, -s.tangent.dot(_up)).normalize();
    _right.copy(_left).normalize();
    out.makeBasis(_right, _up, _forward);
    out.scale(scale);
    out.setPosition(position);
    return out;
  }

  private roadsidePoint(distance: number, side: number, margin: number, height: number, out: Vector3): Vector3 {
    const s = this.spline.sampleAtDistance(distance, this.sampleB);
    this.spline.surfaceLeftAt(distance, _left);
    const lateral = side * (s.halfWidth + margin);
    out.copy(s.position).addScaledVector(_left, lateral);
    const terrainY = this.terrain.heightAt(out.x, out.z);
    const groundY = Number.isFinite(terrainY)
      ? clamp(terrainY, s.position.y - 1.2, s.position.y + 0.45)
      : s.position.y;
    out.y = groundY + height;
    return out;
  }

  private addInstances(
    geometry: BufferGeometry,
    transforms: Matrix4[],
    name: string,
    options: CelOptions = {},
  ): void {
    if (transforms.length === 0) {
      geometry.dispose();
      return;
    }

    finalizeGeometry(geometry, {
      tolerance: 5e-4,
      maxWeldAngle: 76,
      ao: true,
      aoStrength: 0.42,
    });

    const count = transforms.length;
    const tint = new Float32Array(count * 3);
    const fade = new Float32Array(count);
    const phase = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const value = 0.92 + ((i * 17) % 11) * 0.012;
      tint[i * 3] = value;
      tint[i * 3 + 1] = value;
      tint[i * 3 + 2] = value;
      fade[i] = 1;
      phase[i] = (i * 2.399963) % (Math.PI * 2);
    }
    geometry.setAttribute('aInstanceTint', new InstancedBufferAttribute(tint, 3));
    geometry.setAttribute('aInstanceFade', new InstancedBufferAttribute(fade, 1));
    geometry.setAttribute('aInstancePhase', new InstancedBufferAttribute(phase, 1));

    const materialOptions: CelOptions = {
      name,
      instanced: true,
      vertexAo: true,
      matcapMix: 0.08,
      ...options,
    };
    const material = new CelMaterial(RAMPS.marker, materialOptions);
    const mesh = new InstancedMesh(geometry, material, count);
    mesh.name = name;
    for (let i = 0; i < count; i++) mesh.setMatrixAt(i, transforms[i]);
    mesh.instanceMatrix.needsUpdate = true;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.computeBoundingSphere();
    registerNprMesh(mesh, material);

    const group = new Group();
    group.name = `${name}:group`;
    const hull = attachOutline(mesh, RAMPS.marker, materialOptions);
    if (hull) group.add(hull);
    group.add(mesh);
    this.object.add(group);
    this.groups.push({ geometry, material, mesh });
  }

  /** White/red stage stakes every 24 m, outside the usable road shoulder. */
  private buildDelineators(): void {
    const transforms: Matrix4[] = [];
    const step = 24;
    for (let d = 14; d < this.spline.length - 12; d += step) {
      for (const side of [-1, 1]) {
        this.roadsidePoint(d, side, 0.72, 0, _position);
        transforms.push(
          this.frameAt(d, _matrix, _position, _scale.set(1, 1, 1)).clone(),
        );
      }
    }
    const geometry = new CylinderGeometry(0.055, 0.072, 0.92, 6, 1, false);
    geometry.translate(0, 0.46, 0);
    this.addInstances(geometry, transforms, 'rally-stage-delineators', {
      outlineWidth: 0.009,
    });
  }

  /**
   * Three-board clusters on the outside of meaningful bends. They start before
   * the apex, so a driver reads the corner's direction and duration at speed.
   */
  private buildCornerBoards(): void {
    const transforms: Matrix4[] = [];
    let lastBoard = -100;
    for (let d = 35; d < this.spline.length - 25; d += 4) {
      const s = this.spline.sampleAtDistance(d, this.sample);
      if (Math.abs(s.curvature) < 0.0042 || d - lastBoard < 42) continue;
      const outside = s.curvature > 0 ? -1 : 1;
      for (let i = 0; i < 3; i++) {
        const boardD = Math.max(4, d - 12 + i * 7);
        this.roadsidePoint(boardD, outside, 1.05, 0.18, _position);
        const matrix = this.frameAt(boardD, _matrix, _position, _scale.set(1, 1, 1)).clone();
        // Cant each panel toward the racing line; alternating height makes the
        // cluster hand-set rather than mechanically copied.
        matrix.multiply(new Matrix4().makeRotationY(outside * -0.18));
        transforms.push(matrix);
      }
      lastBoard = d;
    }
    const geometry = new BoxGeometry(0.42, 0.58, 0.075);
    geometry.translate(0, 0.29, 0);
    this.addInstances(geometry, transforms, 'rally-stage-corner-boards', {
      outlineWidth: 0.012,
      rimStrength: 0.42,
    });
  }

  /**
   * Low safety ribbon through broad/open sections. It is intentionally absent
   * from exposed ridge segments where tape would visually deny the drop.
   */
  private buildSafetyRibbon(): void {
    const transforms: Matrix4[] = [];
    const step = 18;
    for (let d = 8; d < this.spline.length - step; d += step) {
      const a = this.spline.sampleAtDistance(d, this.sample);
      const b = this.spline.sampleAtDistance(d + step, this.sampleB);
      if (Math.min(a.halfWidth, b.halfWidth) < 7.25) continue;

      for (const side of [-1, 1]) {
        this.roadsidePoint(d, side, 1.18, 1.05, _position);
        this.roadsidePoint(d + step, side, 1.18, 1.05, _positionB);
        _mid.copy(_position).lerp(_positionB, 0.5);
        const length = _position.distanceTo(_positionB);
        const matrix = this.frameAt(d + step * 0.5, _matrix, _mid, _scale.set(1, 1, length)).clone();
        transforms.push(matrix);
      }
    }
    const geometry = new BoxGeometry(0.065, 0.095, 1);
    this.addInstances(geometry, transforms, 'rally-stage-safety-ribbon', {
      outlineWidth: 0.006,
      noShadow: true,
    });
  }

  dispose(): void {
    this.object.removeFromParent();
    for (const group of this.groups) {
      group.geometry.dispose();
      disposeCelMaterial(group.material);
    }
    this.groups.length = 0;
    this.object.clear();
  }
}
