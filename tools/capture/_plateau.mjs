/**
 * Plateau probe — the critic's own measurement, run as a number.
 *
 *   node tools/capture/_plateau.mjs <png> col <x> <y0> <y1> [minStep]
 *   node tools/capture/_plateau.mjs <png> row <y> <x0> <x1> [minStep]
 *
 * Walks EVERY pixel along the line (no stride, no box blur — a box blur turns
 * a hard step into a ramp, which is the thing under test), then reports:
 *   - the number of distinct quantised values seen
 *   - every adjacent-sample jump of at least minStep levels, with its position
 *   - the plateau run lengths between those jumps
 *
 * A 5px median is applied ACROSS the line (perpendicular) only, so a single
 * ink stroke or a hatch pixel does not register as a plateau boundary while a
 * genuine horizontal terrace still steps in one row.
 */
import { readPng } from './_pngread.mjs';

const [, , file, mode, aS, b0S, b1S, minStepS] = process.argv;
const a = Number(aS), b0 = Number(b0S), b1 = Number(b1S);
const minStep = Number(minStepS ?? 4);

const img = readPng(file);
const { width: W, height: H, channels: CH, data } = img;
const px = (x, y) => {
  const i = (Math.min(H - 1, Math.max(0, y)) * W + Math.min(W - 1, Math.max(0, x))) * CH;
  return [data[i], data[i + 1], data[i + 2]];
};

// Median of 5 samples taken PERPENDICULAR to the scan direction.
function sampleAt(t) {
  const rs = [], gs = [], bs = [];
  for (let k = -2; k <= 2; k++) {
    const [r, g, bl] = mode === 'col' ? px(a + k, t) : px(t, a + k);
    rs.push(r); gs.push(g); bs.push(bl);
  }
  const med = (v) => v.sort((p, q) => p - q)[2];
  return [med(rs), med(gs), med(bs)];
}

const vals = [];
for (let t = b0; t <= b1; t++) vals.push([t, ...sampleAt(t)]);

const lum = (v) => 0.299 * v[1] + 0.587 * v[2] + 0.114 * v[3];

if (process.env.DUMP) {
  for (const v of vals) console.log(`  ${String(v[0]).padStart(5)}  ${String(v[1]).padStart(4)} ${String(v[2]).padStart(4)} ${String(v[3]).padStart(4)}   luma ${(0.299 * v[1] + 0.587 * v[2] + 0.114 * v[3]).toFixed(1)}`);
}

const distinct = new Set(vals.map((v) => `${v[1]},${v[2]},${v[3]}`));
console.log(`${file}  ${mode} ${a}  ${b0}..${b1}   n=${vals.length}  distinct RGB triples=${distinct.size}`);

// Hard steps: an adjacent-sample jump. Report position, size, and the flat run
// that preceded it.
const steps = [];
let runStart = b0;
for (let i = 1; i < vals.length; i++) {
  const d = Math.abs(lum(vals[i]) - lum(vals[i - 1]));
  if (d >= minStep) {
    steps.push({ at: vals[i][0], jump: d.toFixed(1), runBefore: vals[i][0] - runStart,
                 from: vals[i - 1].slice(1).join(','), to: vals[i].slice(1).join(',') });
    runStart = vals[i][0];
  }
}
console.log(`hard steps (>= ${minStep} luma levels between adjacent rows): ${steps.length}`);
for (const s of steps) {
  console.log(`   at ${String(s.at).padStart(5)}  jump ${String(s.jump).padStart(6)}  run before ${String(s.runBefore).padStart(4)}  (${s.from}) -> (${s.to})`);
}
console.log(`   final run ${b1 - runStart}`);

// Total drift and the largest single jump, so a "smooth drift" can be told
// apart from a terrace with the same endpoints.
let maxJump = 0;
for (let i = 1; i < vals.length; i++) maxJump = Math.max(maxJump, Math.abs(lum(vals[i]) - lum(vals[i - 1])));
console.log(`luma ${lum(vals[0]).toFixed(1)} -> ${lum(vals[vals.length - 1]).toFixed(1)}   drift ${(lum(vals[vals.length - 1]) - lum(vals[0])).toFixed(1)}   largest single-sample jump ${maxJump.toFixed(1)}`);

// Alternation test for the scanline moire: how often the sign of the delta
// flips between consecutive samples. A clean surface flips rarely; a one-pixel
// checkerboard flips on nearly every row.
let flips = 0, nz = 0;
let prevSign = 0;
for (let i = 1; i < vals.length; i++) {
  const d = lum(vals[i]) - lum(vals[i - 1]);
  if (Math.abs(d) < 0.4) continue;
  nz++;
  const s = Math.sign(d);
  if (prevSign !== 0 && s !== prevSign) flips++;
  prevSign = s;
}
console.log(`alternation: ${flips} sign flips over ${nz} non-flat deltas (${((flips / Math.max(nz, 1)) * 100).toFixed(0)}%)`);
