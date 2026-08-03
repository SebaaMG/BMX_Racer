/**
 * Generated rally number panels.
 *
 * No font, image or texture is loaded. The pale panel and seven-segment number
 * are tiny meshes using the existing cel ramps, so they remain crisp under the
 * NPR pipeline and readable on reduced-detail opponents.
 */

import {
  BoxGeometry,
  BufferGeometry,
  Group,
  Mesh,
  Object3D,
  ShaderMaterial,
} from 'three';

import {
  CelMaterial,
  attachOutline,
  disposeCelMaterial,
  registerNprMesh,
  type CelOptions,
} from '../npr/CelMaterial';
import { prepareOutlineGeometry } from '../npr/OutlineGeometry';
import { RAMPS } from '../npr/Palette';

const SEGMENTS: Record<number, readonly string[]> = {
  1: ['b', 'c'],
  2: ['a', 'b', 'g', 'e', 'd'],
  3: ['a', 'b', 'c', 'd', 'g'],
  4: ['f', 'g', 'b', 'c'],
};

const SEGMENT_LAYOUT: Record<string, { y: number; z: number; vertical: boolean }> = {
  a: { y: 0.18, z: 0, vertical: false },
  b: { y: 0.09, z: -0.115, vertical: true },
  c: { y: -0.09, z: -0.115, vertical: true },
  d: { y: -0.18, z: 0, vertical: false },
  e: { y: -0.09, z: 0.115, vertical: true },
  f: { y: 0.09, z: 0.115, vertical: true },
  g: { y: 0, z: 0, vertical: false },
};

export interface RallyNumberPlateHandle {
  object: Object3D;
  dispose(): void;
}

export function rallyNumberFromName(name: string | undefined): number {
  const n = (name ?? '').toLowerCase();
  if (n.includes('player') || n.includes('you')) return 1;
  if (n.includes('kestrel') || n.endsWith(':ai-0') || n.endsWith(':1')) return 2;
  if (n.includes('mags') || n.endsWith(':ai-1') || n.endsWith(':2')) return 3;
  if (n.includes('okoye') || n.endsWith(':ai-2') || n.endsWith(':3')) return 4;
  let hash = 2166136261;
  for (let i = 0; i < n.length; i++) hash = Math.imul(hash ^ n.charCodeAt(i), 16777619);
  return 1 + ((hash >>> 0) % 4);
}

export function addRallyNumberPlates(
  parent: Object3D,
  number: number,
  name = 'rally-number',
): RallyNumberPlateHandle {
  const object = new Group();
  object.name = `${name}:group`;
  parent.add(object);

  const materials: CelMaterial[] = [];
  const hullMaterials: ShaderMaterial[] = [];
  const geometries: BufferGeometry[] = [];

  const panelOpts: CelOptions = {
    name: `${name}:panel-material`,
    matcapMix: 0.04,
    noShadow: true,
  };
  const panelMaterial = new CelMaterial(RAMPS.snow, panelOpts);
  const digitMaterial = new CelMaterial(RAMPS.rubber, {
    name: `${name}:digit-material`,
    matcapMix: 0,
    noShadow: true,
    rimStrength: 0.12,
  });
  materials.push(panelMaterial, digitMaterial);

  const panelGeometry = new BoxGeometry(0.026, 0.49, 0.68);
  prepareOutlineGeometry(panelGeometry, { maxWeldAngle: 78, curvatureGain: 1.0 });
  geometries.push(panelGeometry);

  const horizontal = new BoxGeometry(0.032, 0.055, 0.23);
  const vertical = new BoxGeometry(0.032, 0.165, 0.055);
  geometries.push(horizontal, vertical);

  const active = SEGMENTS[Math.max(1, Math.min(4, Math.round(number)))] ?? SEGMENTS[1];

  for (const side of [-1, 1]) {
    const x = side * 0.934;
    const panel = new Mesh(panelGeometry, panelMaterial);
    panel.name = `${name}:panel:${side}`;
    panel.position.set(x, 0.27, -0.08);
    panel.castShadow = false;
    panel.receiveShadow = true;
    registerNprMesh(panel, panelMaterial);
    const hull = attachOutline(panel, RAMPS.snow, panelOpts);
    if (hull) {
      hull.position.copy(panel.position);
      hull.quaternion.copy(panel.quaternion);
      hull.scale.copy(panel.scale);
      hullMaterials.push(hull.material as ShaderMaterial);
      object.add(hull);
    }
    object.add(panel);

    for (const key of active) {
      const layout = SEGMENT_LAYOUT[key];
      const segment = new Mesh(layout.vertical ? vertical : horizontal, digitMaterial);
      segment.name = `${name}:digit:${number}:${key}:${side}`;
      segment.position.set(side * 0.952, 0.27 + layout.y, -0.08 + layout.z);
      segment.castShadow = false;
      segment.receiveShadow = false;
      registerNprMesh(segment, digitMaterial);
      object.add(segment);
    }
  }

  return {
    object,
    dispose() {
      object.removeFromParent();
      for (const material of hullMaterials) material.dispose();
      for (const material of materials) disposeCelMaterial(material);
      for (const geometry of geometries) geometry.dispose();
      object.clear();
    },
  };
}
