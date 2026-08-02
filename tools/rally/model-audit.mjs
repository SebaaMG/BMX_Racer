#!/usr/bin/env node
/**
 * Static quality gate for the generated rally model.
 *
 * This intentionally reads source rather than importing TypeScript: it runs in
 * plain Node before Vite and catches proportion regressions even when the model
 * still compiles. Visual review remains mandatory; this only prevents known
 * numerical failure modes from returning.
 */

import { readFile } from 'node:fs/promises';

const visualPath = new URL('../../src/rally/RallyCarVisual.ts', import.meta.url);
const seamPath = new URL('../../src/bike/index.ts', import.meta.url);
const visual = await readFile(visualPath, 'utf8');
const seam = await readFile(seamPath, 'utf8');

function numberField(name) {
  const match = visual.match(new RegExp(`\\b${name}:\\s*([0-9.]+)`));
  if (!match) throw new Error(`rally model audit: missing ${name}`);
  return Number(match[1]);
}

function within(name, value, min, max) {
  if (!(value >= min && value <= max)) {
    throw new Error(`rally model audit: ${name}=${value} outside ${min}..${max}`);
  }
}

const length = numberField('length');
const width = numberField('width');
const wheelbase = numberField('wheelbase');
const tyre = numberField('gravelTyreDiameter');
const roof = numberField('roofHeightFromGround');
const ratio = length / wheelbase;

// A broad fictional Rally1/Rally2-inspired envelope. These are guardrails, not
// homologation claims and not a copy of any one manufacturer's dimensions.
within('length', length, 4.00, 4.30);
within('width', width, 1.80, 1.90);
within('wheelbase', wheelbase, 2.50, 2.70);
within('gravelTyreDiameter', tyre, 0.66, 0.74);
within('roofHeightFromGround', roof, 1.36, 1.54);
within('length/wheelbase', ratio, 1.54, 1.66);

const stationZ = [...visual.matchAll(/\{\s*z:\s*(-?[0-9.]+)/g)].map((m) => Number(m[1]));
if (stationZ.length < 9) throw new Error(`rally model audit: only ${stationZ.length} body stations`);
const stationSpan = Math.max(...stationZ) - Math.min(...stationZ);
if (Math.abs(stationSpan - length) > 0.03) {
  throw new Error(`rally model audit: station span ${stationSpan.toFixed(3)} does not match length ${length}`);
}

const shoulderHalf = [...visual.matchAll(/shoulderHalf:\s*([0-9.]+)/g)].map((m) => Number(m[1]));
const maxShoulderWidth = Math.max(...shoulderHalf) * 2;
if (Math.abs(maxShoulderWidth - width) > 0.08) {
  throw new Error(
    `rally model audit: shoulder width ${maxShoulderWidth.toFixed(3)} does not match declared width ${width}`,
  );
}

for (const forbidden of ['GLTFLoader', '.glb', '.gltf', '.fbx', 'MeshStandardMaterial']) {
  if (visual.includes(forbidden) || seam.includes(forbidden)) {
    throw new Error(`rally model audit: forbidden external/PBR dependency ${forbidden}`);
  }
}

if (!seam.includes("RallyCarVisualPolished")) {
  throw new Error('rally model audit: gameplay is not wired to the reviewed rally model');
}

console.log([
  'rally model audit: PASS',
  `  envelope ${length.toFixed(2)}m × ${width.toFixed(2)}m`,
  `  wheelbase ${wheelbase.toFixed(2)}m  ratio ${ratio.toFixed(3)}`,
  `  tyre ${tyre.toFixed(2)}m  roof ${roof.toFixed(2)}m`,
  `  ${stationZ.length} loft stations`,
].join('\n'));
