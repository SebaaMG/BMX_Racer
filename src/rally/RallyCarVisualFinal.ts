/**
 * Final visual critic layer for the production hatch.
 *
 * The production shell already owns the correct materials, suspension, wheels,
 * driver anchors and lifecycle. This pass performs the two remaining authored
 * corrections found by orthographic and switchback review:
 *
 *  - move the upper greenhouse toward the front axle and hold a long, flatter
 *    roof before a steeper hatch, removing the residual sports-coupe read;
 *  - drive the visible front wheels from signed slip as well as rack input, so
 *    a sustained slide visibly countersteers instead of looking like neutral yaw.
 */

import { BufferGeometry, Mesh, Object3D } from 'three';

import { clamp, clamp01, dampHL, lerp } from '../core/MathX';
import { prepareOutlineGeometry } from '../npr/OutlineGeometry';
import {
  RallyCarVisual as ProductionRallyCarVisual,
  RALLY_BODY_DIMENSIONS,
  type RallyCarVisualOptions,
  type RallyCarVisualState as ProductionVisualState,
} from './RallyCarVisualProduction';

export { RALLY_BODY_DIMENSIONS };
export type { RallyCarVisualOptions };

export interface RallyCarVisualState extends ProductionVisualState {
  /** Signed local lateral velocity: +left, -right, metres per second. */
  signedLateralSlip?: number;
}

function geometryOf(node: Object3D): BufferGeometry | null {
  const geometry = (node as Mesh).geometry;
  return geometry?.isBufferGeometry ? geometry : null;
}

function finishGeometry(geometry: BufferGeometry, outline = false): void {
  geometry.computeVertexNormals();
  if (outline) prepareOutlineGeometry(geometry, { maxWeldAngle: 76, curvatureGain: 1.10 });
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
}

function reshape(
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

function syncHull(node: Object3D): void {
  const hull = node.userData.hull as Object3D | undefined;
  if (!hull) return;
  hull.position.copy(node.position);
  hull.quaternion.copy(node.quaternion);
  hull.scale.copy(node.scale);
}

function reshapeShell(node: Object3D): void {
  reshape(
    node,
    'rallyFinalHatchShell',
    (x, y, z) => {
      const upper = clamp01((y - 0.27) / 0.48);
      if (upper <= 0) return [x, y, z];

      // The cowl was roughly half a metre behind the front axle. Advance only
      // the upper body, preserving wheelbase, bumper and crumple volume.
      const cowlZone = 1 - clamp01(Math.abs(z - 0.62) / 0.72);
      z += 0.24 * upper * cowlZone;

      // Hold the roof as a usable hatch roof instead of a continuous coupe arc.
      if (y > 0.55 && z < 0.72 && z > -1.24) {
        const roofTarget = 0.785 - Math.abs(z + 0.28) * 0.018;
        y = Math.max(y, roofTarget);
      }

      // Walk the upper rear toward the rear axle. Low shoulder vertices move
      // less, which makes the C-pillar steep rather than stretching the whole car.
      if (z < -0.62) {
        const rear = clamp01((-z - 0.62) / 0.92);
        z -= 0.23 * upper * rear;
        if (y > 0.38) y += 0.055 * rear * (1 - upper * 0.25);
      }
      return [x, y, z];
    },
    true,
  );
}

function reshapeWindshield(node: Object3D): void {
  reshape(node, 'rallyFinalWindshield', (x, y, z) => {
    const top = clamp01((y - 0.40) / 0.38);
    return [x, y, z + lerp(0.29, 0.18, top)];
  });
}

function reshapeFrontSideGlass(node: Object3D): void {
  reshape(node, 'rallyFinalFrontSideGlass', (x, y, z) => {
    const front = clamp01((z + 0.34) / 1.05);
    return [x, y, z + 0.20 + front * 0.08];
  });
}

function reshapeRearGlass(node: Object3D): void {
  reshape(node, 'rallyFinalRearGlass', (x, y, z) => {
    const top = clamp01((y - 0.40) / 0.38);
    return [x, y, z - lerp(0.035, 0.235, top)];
  });
}

function reshapeRearSideGlass(node: Object3D): void {
  reshape(node, 'rallyFinalRearSideGlass', (x, y, z) => {
    const rear = clamp01((-z - 0.34) / 0.92);
    const upper = clamp01((y - 0.40) / 0.38);
    return [x, y, z - rear * lerp(0.05, 0.20, upper)];
  });
}

export class RallyCarVisual extends ProductionRallyCarVisual {
  private readonly frontSteerRoots: Object3D[] = [];

  constructor(opts: RallyCarVisualOptions) {
    super(opts);

    this.root.traverse((node) => {
      if (node.userData.isHull) return;
      const name = node.name;
      if (name === 'rally:production-shell') reshapeShell(node);
      else if (name === 'rally:production-windshield') reshapeWindshield(node);
      else if (name === 'rally:production-rear-glass') reshapeRearGlass(node);
      else if (name.startsWith('rally:production-front-side-glass:')) reshapeFrontSideGlass(node);
      else if (name.startsWith('rally:production-rear-side-glass:')) reshapeRearSideGlass(node);
      else if (name.startsWith('rally:production-mirror:')) {
        node.position.z += 0.16;
        syncHull(node);
      } else if (name === 'rally:production-spoiler') {
        node.position.z -= 0.15;
        node.position.y += 0.035;
        node.scale.z *= 0.82;
        syncHull(node);
      }

      if (name.startsWith('rally:tyre:true:')) {
        const steer = node.parent?.parent;
        if (steer && !this.frontSteerRoots.includes(steer)) this.frontSteerRoots.push(steer);
      }
    });
  }

  override update(state: RallyCarVisualState, dt: number, cameraDistance = 7): void {
    super.update(state, dt, cameraDistance);

    const signedSlip = state.signedLateralSlip ?? 0;
    const driftGate =
      clamp01((Math.abs(signedSlip) - 0.75) / 4.5) *
      clamp01((state.speed - 5.5) / 11.0);
    const counter = clamp(-signedSlip * 0.060, -0.38, 0.38) * driftGate;
    const visibleSteer = clamp(state.steerAngle + counter, -0.62, 0.62);

    for (const root of this.frontSteerRoots) {
      root.rotation.y = dampHL(root.rotation.y, visibleSteer, driftGate > 0.1 ? 0.035 : 0.055, dt);
    }
  }
}
