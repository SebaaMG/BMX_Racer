#!/usr/bin/env node
/**
 * Export the exact BODY_STATIONS loft as an OBJ for optional offline inspection.
 *
 * The OBJ is a generated review artefact, never a game dependency. Runtime
 * geometry still comes from RallyCarVisual.ts. This gives Blender or any mesh
 * inspector access to the same proportions without introducing a downloaded
 * model or a second source of truth.
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const sourceUrl = new URL('../../src/rally/RallyCarVisual.ts', import.meta.url);
const source = await readFile(sourceUrl, 'utf8');
const block = source.match(/const BODY_STATIONS:[\s\S]*?=\s*\[([\s\S]*?)\n\];/)?.[1];
if (!block) throw new Error('BODY_STATIONS not found');

const fields = [
  'z', 'floorY', 'sillHalf', 'shoulderY', 'shoulderHalf',
  'beltY', 'beltHalf', 'roofY', 'roofHalf', 'crownY',
];
const stations = [...block.matchAll(/\{([^{}]+)\}/g)].map((match) => {
  const text = match[1];
  const station = {};
  for (const field of fields) {
    const value = text.match(new RegExp(`\\b${field}:\\s*(-?[0-9.]+)`))?.[1];
    if (value === undefined) throw new Error(`station missing ${field}: ${text}`);
    station[field] = Number(value);
  }
  return station;
});
if (stations.length < 3) throw new Error('not enough loft stations');

function ring(s) {
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

const rings = stations.map(ring);
const ringSize = rings[0].length;
const lines = [
  '# DESCENT RALLY generated body review',
  '# Source: src/rally/RallyCarVisual.ts BODY_STATIONS',
  '# +Y up, +Z forward, +X left',
  'o rally_body_loft',
];

for (const points of rings) {
  for (const [x, y, z] of points) lines.push(`v ${x.toFixed(6)} ${y.toFixed(6)} ${z.toFixed(6)}`);
}

const at = (station, point) => station * ringSize + point + 1;
for (let i = 0; i < rings.length - 1; i++) {
  for (let j = 0; j < ringSize; j++) {
    const k = (j + 1) % ringSize;
    lines.push(`f ${at(i, j)} ${at(i + 1, j)} ${at(i + 1, k)} ${at(i, k)}`);
  }
}
// Caps.
lines.push(`f ${[...Array(ringSize)].map((_, i) => at(0, i)).join(' ')}`);
lines.push(`f ${[...Array(ringSize)].map((_, i) => at(rings.length - 1, ringSize - 1 - i)).join(' ')}`);

const arg = process.argv.find((value) => value.startsWith('--out='));
const output = path.resolve(arg ? arg.slice('--out='.length) : 'captures/rally-body-review.obj');
await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, `${lines.join('\n')}\n`, 'utf8');
console.log(`rally body OBJ → ${output}`);
