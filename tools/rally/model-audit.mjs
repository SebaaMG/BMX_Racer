#!/usr/bin/env node
/**
 * Static quality gate for the generated rally model and its car-specific seams.
 */

import { readFile } from 'node:fs/promises';

const visualPath = new URL('../../src/rally/RallyCarVisual.ts', import.meta.url);
const productionPath = new URL('../../src/rally/RallyCarVisualProduction.ts', import.meta.url);
const finalVisualPath = new URL('../../src/rally/RallyCarVisualFinal.ts', import.meta.url);
const finalPhysicsPath = new URL('../../src/rally/RallyCarPhysicsFinal.ts', import.meta.url);
const runtimePath = new URL('../../src/rally/RallyRuntimeIntegration.ts', import.meta.url);
const seamPath = new URL('../../src/bike/index.ts', import.meta.url);
const reviewPath = new URL('./model-review.ts', import.meta.url);

const visual = await readFile(visualPath, 'utf8');
const production = await readFile(productionPath, 'utf8');
const finalVisual = await readFile(finalVisualPath, 'utf8');
const finalPhysics = await readFile(finalPhysicsPath, 'utf8');
const runtime = await readFile(runtimePath, 'utf8');
const seam = await readFile(seamPath, 'utf8');
const review = await readFile(reviewPath, 'utf8');

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

within('length', length, 4.00, 4.30);
within('width', width, 1.80, 1.90);
within('wheelbase', wheelbase, 2.50, 2.70);
within('gravelTyreDiameter', tyre, 0.66, 0.74);
within('roofHeightFromGround', roof, 1.36, 1.54);
within('length/wheelbase', ratio, 1.54, 1.66);

const stationMatch = production.match(
  /const HATCH:[\s\S]*?=\s*\[([\s\S]*?)\n\];/,
);
if (!stationMatch) throw new Error('rally model audit: production HATCH block not found');
const stations = stationMatch[1];
const stationZ = [...stations.matchAll(/\{\s*z:\s*(-?[0-9.]+)/g)].map((m) => Number(m[1]));
if (stationZ.length < 12) throw new Error(`rally model audit: only ${stationZ.length} production stations`);
const stationSpan = Math.max(...stationZ) - Math.min(...stationZ);
if (Math.abs(stationSpan - length) > 0.03) {
  throw new Error(`rally model audit: station span ${stationSpan.toFixed(3)} does not match length ${length}`);
}

const shoulderHalf = [...stations.matchAll(/shoulderHalf:\s*([0-9.]+)/g)].map((m) => Number(m[1]));
const maxShoulderWidth = Math.max(...shoulderHalf) * 2;
if (Math.abs(maxShoulderWidth - width) > 0.08) {
  throw new Error(
    `rally model audit: shoulder width ${maxShoulderWidth.toFixed(3)} does not match declared width ${width}`,
  );
}

const floorY = [...stations.matchAll(/floorY:\s*(-?[0-9.]+)/g)].map((m) => Number(m[1]));
within('production floorY', Math.min(...floorY), -0.43, -0.34);

for (const token of [
  'rallyFinalHatchShell',
  'signedLateralSlip',
  'frontSteerRoots',
]) {
  if (!finalVisual.includes(token)) throw new Error(`rally model audit: final visual missing ${token}`);
}
for (const token of ['rallyWheels', 'rallyLateralVelocity', 'updateContactPatches']) {
  if (!finalPhysics.includes(token)) throw new Error(`rally model audit: four-wheel physics missing ${token}`);
}
for (const token of ['collisionHalfLength', 'testAxis', 'emitWheel(wheels[3]']) {
  if (!runtime.includes(token) && !seam.includes(token)) {
    throw new Error(`rally model audit: runtime rally integration missing ${token}`);
  }
}

for (const forbidden of ['GLTFLoader', '.glb', '.gltf', '.fbx', 'MeshStandardMaterial']) {
  if ([visual, production, finalVisual, finalPhysics, runtime, seam].some((source) => source.includes(forbidden))) {
    throw new Error(`rally model audit: forbidden external/PBR dependency ${forbidden}`);
  }
}

if (!seam.includes('RallyCarVisualFinal') || !seam.includes('RallyCarPhysicsFinal')) {
  throw new Error('rally model audit: gameplay is not wired to final rally systems');
}
if (!review.includes('RallyCarVisualFinal')) {
  throw new Error('rally model audit: orthographic review is not rendering the shipping visual');
}

console.log([
  'rally model audit: PASS',
  `  envelope ${length.toFixed(2)}m × ${width.toFixed(2)}m`,
  `  wheelbase ${wheelbase.toFixed(2)}m  ratio ${ratio.toFixed(3)}`,
  `  tyre ${tyre.toFixed(2)}m  roof ${roof.toFixed(2)}m`,
  `  ${stationZ.length} production hatch stations`,
  '  final greenhouse + countersteer + four patches + OBB contacts',
].join('\n'));
