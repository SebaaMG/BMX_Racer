/**
 * _ride.mjs — DOES IT RIDE? A whole-course regression test.
 *
 * The named capture poses are a bad regression test for handling: most of them
 * hold a constant input with no steering while the trail turns underneath, so
 * after two seconds the rider is off-piste on untracked mountainside and where
 * it dies next is chaos, not a measurement. Measured: `rockgarden-low` leaves a
 * 5.2 m half-width ribbon at 2.3 s and is 14.6 m outside it by 3.6 s.
 *
 * So this runs the actual race with the actual AI riders — which use the same
 * BikePhysics — down the real course, and reports what a rider would notice:
 * distance covered, how close to the ribbon they stayed, how much speed they
 * carried, and how often they fell over.
 *
 * Usage: node tools/capture/_ride.mjs [seconds] [--tag before]
 */

import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const argv = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const SECONDS = Number(argv[0] ?? 45);
const TAG = (process.argv.includes('--tag') ? process.argv[process.argv.indexOf('--tag') + 1] : null) ?? 'run';
const OUT = path.resolve('tools/capture/_out');
const GAIN = Number(argv[1] ?? 2.2);

const b = await chromium.launch({
  headless: true,
  args: ['--use-angle=default', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-unsafe-swiftshader'],
});
const p = await b.newPage();
p.on('pageerror', (e) => console.log('ERR', e.message.slice(0, 200)));
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.capture, null, { timeout: 300000 });

const out = await p.evaluate(([seconds, GAIN]) => {
  const g = window.__DESCENT__.game;
  g.capture.takeControl();
  g.capture.setPose('summit-rider');
  const bike = g.race.player.bike;
  const st = bike.state;
  const V = st.position.constructor;
  const fwd = new V(), tgt = new V(), up = new V(0, 1, 0);

  // Put the rider on the ribbon at the top with a little speed.
  const smp = g.track.sampleAtDistance(60);
  const pos = smp.position.clone(); pos.y += 0.267;
  bike.reset(pos, smp.tangent.clone());
  st.velocity.copy(smp.tangent).multiplyScalar(8);
  for (const r of g.race.racers) {
    if (r === g.race.player) continue;
    r.bike.state.position.y -= 900; r.bike.state.velocity.set(0, 0, 0);
  }

  const input = { steer: 0, pedal: 1, brakeRear: 0, brakeFront: 0, crouch: 0,
    pitchLean: 0, airPitch: 0, airYaw: 0, airRoll: 0, wantBoost: false, wantHop: false };
  g.race.player.scripted = input;

  const phys = bike.physics;
  if (!phys.__origStep) phys.__origStep = phys.step.bind(phys);
  const a = { n: 0, sumSpeed: 0, sumLat: 0, maxLat: 0, off: 0, air: 0, maxLean: 0,
              crash0: st.crashCount, d0: 0, dLast: 0, sumSteer: 0, trace: [] };
  let started = false;

  phys.step = (inp, dt) => {
    // ── Autopilot: follow the ribbon. Pure P on heading and lateral offset,
    // no state, so it is deterministic and identical between runs.
    const pr = g.track.spline.project(st.position);
    if (!started) { a.d0 = pr.distance; started = true; }
    const look = g.track.sampleAtDistance(Math.min(g.track.length - 2, pr.distance + 9));
    tgt.copy(look.position).sub(st.position); tgt.y = 0;
    fwd.set(0, 0, 1).applyQuaternion(st.orientation); fwd.y = 0;
    if (tgt.lengthSq() > 1e-6) tgt.normalize();
    if (fwd.lengthSq() > 1e-6) fwd.normalize();
    // + when the target is to the LEFT of the nose.
    const cross = fwd.x * tgt.z - fwd.z * tgt.x;
    const dot = fwd.x * tgt.x + fwd.z * tgt.z;
    const headErr = Math.atan2(-cross, dot);
    // pr.lateral > 0 means the rider is LEFT of centre, so steer right.
    const cmd = -headErr * GAIN + pr.lateral * 0.16;
    inp.steer = Math.max(-1, Math.min(1, cmd));
    inp.pedal = 1;
    phys.__origStep(inp, dt);

    a.n++;
    a.sumSpeed += st.speed;
    a.sumSteer += Math.abs(inp.steer);
    const lat = Math.abs(pr.lateral);
    a.sumLat += lat;
    if (lat > a.maxLat) a.maxLat = lat;
    if (lat > pr.sample.halfWidth) a.off++;
    if (!st.front.grounded && !st.rear.grounded) a.air++;
    if (Math.abs(st.lean) > a.maxLean) a.maxLean = Math.abs(st.lean);
    a.dLast = pr.distance;
    if (a.n % 30 === 0) a.trace.push(+pr.lateral.toFixed(2));
  };
  const frames = Math.round(seconds * 60);
  for (let i = 0; i < frames; i++) g.capture.step(1 / 60);
  phys.step = phys.__origStep;

  return [{
    id: 'autopilot',
    metres: +(a.dLast - a.d0).toFixed(1),
    meanSpeed: +(a.sumSpeed / a.n).toFixed(2),
    meanAbsLat: +(a.sumLat / a.n).toFixed(2),
    maxAbsLat: +a.maxLat.toFixed(2),
    offTrackPct: +((100 * a.off) / a.n).toFixed(1),
    airPct: +((100 * a.air) / a.n).toFixed(1),
    crashes: st.crashCount - a.crash0,
    landings: st.landCount,
    maxLean: +a.maxLean.toFixed(2),
    meanSteer: +(a.sumSteer / a.n).toFixed(3),
    trace: a.trace,
  }];
}, [SECONDS, GAIN]);

await b.close();

console.log(`RIDE — ${SECONDS}s of the real course, AI riders on the shipped physics\n`);
console.log('rider       metres  mean m/s  mean|lat|  max|lat|  off%   air%  crashes  lands  maxLean');
for (const r of out) {
  console.log(
    `${String(r.id).padEnd(10)} ${String(r.metres).padStart(7)} ${String(r.meanSpeed).padStart(9)} ` +
    `${String(r.meanAbsLat).padStart(10)} ${String(r.maxAbsLat).padStart(9)} ${String(r.offTrackPct).padStart(5)} ` +
    `${String(r.airPct).padStart(6)} ${String(r.crashes).padStart(8)} ${String(r.landings).padStart(6)} ${String(r.maxLean).padStart(8)}`,
  );
}
const mean = (f) => +(out.reduce((a, r) => a + f(r), 0) / out.length).toFixed(2);
console.log(`\nPACK MEAN  metres ${mean((r) => r.metres)}  speed ${mean((r) => r.meanSpeed)}  ` +
  `|lat| ${mean((r) => r.meanAbsLat)}  off% ${mean((r) => r.offTrackPct)}  crashes ${mean((r) => r.crashes)}`);

await mkdir(OUT, { recursive: true });
await writeFile(path.join(OUT, `ride-${TAG}.json`), JSON.stringify(out, null, 1));
console.log(`\n→ tools/capture/_out/ride-${TAG}.json`);
