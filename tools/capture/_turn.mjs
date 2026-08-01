/**
 * _turn.mjs — STEERING / YAW probe.
 *
 * Defect 1 is "the bike leans but does not turn". That cannot be settled on
 * real terrain, where slope, surface and the track's own curvature all move the
 * numbers. So this runs a SKIDPAD: the bike's terrain is swapped for a flat
 * plane at runtime, the bike is placed on it pointing +Z at a chosen speed, and
 * full steer is held.
 *
 * Reports, per speed: heading change, mean/steady yaw rate, achieved turn
 * radius, lean vs target lean, steer angle, per-wheel lateral force and the
 * yaw-moment budget (front lever vs rear lever vs the yaw damper).
 *
 * Usage: node tools/capture/_turn.mjs [--tag before]
 */

import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const args = {};
for (let i = 0; i < process.argv.length; i++) {
  const a = process.argv[i];
  if (a.startsWith('--')) {
    const n = process.argv[i + 1];
    args[a.slice(2)] = n && !n.startsWith('--') ? n : true;
  }
}
const TAG = args.tag ?? 'run';
const OUT = path.resolve('tools/capture/_out');

const b = await chromium.launch({
  headless: true,
  args: ['--use-angle=default', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-unsafe-swiftshader'],
});
let p = await b.newPage();
p.on('pageerror', (e) => console.log('ERR', e.message.slice(0, 300)));

const INSTALL = () => {
  const g = window.__DESCENT__.game;
  g.capture.takeControl();
  window.__TURN__ = {
    /** Swap the bike's terrain for a flat plane at height y0. */
    flat(y0) {
      const bike = g.race.player.bike;
      const phys = bike.physics;
      if (!phys.__realTerrain) phys.__realTerrain = phys.terrain;
      // Borrow a real SurfaceProperties so grip/drag are the shipped values.
      const surf = bike.state.rear.surface ?? bike.state.front.surface;
      phys.terrain = {
        heightAt: () => y0,
        normalAt: (x, z, out) => out.set(0, 1, 0),
        sampleAt: (x, z, out) => {
          out.height = y0;
          out.normal.set(0, 1, 0);
          out.slope = 0;
          out.kind = surf.kind;
          out.surface = surf;
          return out;
        },
      };
      return surf.kind + ' grip ' + surf.grip;
    },
    unflat() {
      const phys = g.race.player.bike.physics;
      if (phys.__realTerrain) phys.terrain = phys.__realTerrain;
    },

    /**
     * Hold `steer` for `seconds` at `speed` on the flat plane. Returns a
     * per-step record sampled INSIDE the 120 Hz step.
     */
    skidpad(speed, steer, seconds, extra) {
      const bike = g.race.player.bike;
      const phys = bike.physics;
      const st = bike.state;
      // Park the rivals so nothing collides.
      for (const r of g.race.racers) {
        if (r === g.race.player) continue;
        r.bike.state.position.y -= 900;
        r.bike.state.velocity.set(0, 0, 0);
      }
      const y0 = 100;
      window.__TURN__.flat(y0);
      const pos = st.position.clone();
      pos.set(0, y0 + 0.267, 0);
      bike.reset(pos, new (pos.constructor)(0, 0, 1));
      st.velocity.set(0, 0, speed);

      const base = {
        steer: 0, pedal: 0, brakeRear: 0, brakeFront: 0, crouch: 0,
        pitchLean: 0, airPitch: 0, airYaw: 0, airRoll: 0,
        wantBoost: false, wantHop: false,
      };
      g.race.player.scripted = Object.assign(base, { steer }, extra ?? {});

      if (!phys.__origStep) phys.__origStep = phys.step.bind(phys);
      const rec = [];
      let n = 0;
      const D = phys.diag;
      D.enabled = true;
      phys.step = (input, dt) => {
        phys.__origStep(input, dt);
        const fwd = new (st.position.constructor)(0, 0, 1).applyQuaternion(st.orientation);
        const up = new (st.position.constructor)(0, 1, 0).applyQuaternion(st.orientation);
        rec.push({
          t: n / 120,
          x: st.position.x, y: st.position.y, z: st.position.z,
          heading: Math.atan2(fwd.x, fwd.z),
          yawRate: st.angularVelocity.dot(up),
          speed: st.speed, fwdSpeed: st.forwardSpeed,
          lean: st.lean, targetLean: D.targetLean, steerA: st.steerAngle,
          fg: st.front.grounded ? 1 : 0, rg: st.rear.grounded ? 1 : 0,
          fl: st.front.load, rl: st.rear.load,
          fSlip: st.front.slipAngle, rSlip: st.rear.slipAngle,
          fLat: st.front.vLat, rLat: st.rear.vLat,
          fFy: st.front.tyreForce.dot(st.front.left),
          rFy: st.rear.tyreForce.dot(st.rear.left),
          fCap: st.front.load * st.front.surface.grip * 1.03,
          rCap: st.rear.load * st.rear.surface.grip * 0.96,
          yawDampT: D.yawDampT, rollServo: D.rollServo, tipRoll: D.tipRoll,
          mode: st.mode,
        });
        n++;
      };
      const frames = Math.round(seconds * 60);
      for (let i = 0; i < frames; i++) g.capture.step(1 / 60);
      phys.step = phys.__origStep;
      D.enabled = false;
      window.__TURN__.unflat();
      return rec;
    },
  };
};

async function boot() {
  await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
  await p.waitForFunction(() => !!window.__DESCENT__?.game?.capture, null, { timeout: 300000 });
  await p.evaluate(INSTALL);
}
await boot();

async function evalRetry(fn, arg, tries = 4) {
  for (let i = 0; ; i++) {
    try {
      const has = await p.evaluate(() => !!window.__TURN__).catch(() => false);
      if (!has) await p.evaluate(INSTALL);
      return await p.evaluate(fn, arg);
    } catch (e) {
      if (i >= tries) throw e;
      console.log(`  (retry ${i + 1}: ${String(e.message).split('\n')[0].slice(0, 80)})`);
      try { await p.close(); } catch {}
      p = await b.newPage();
      p.on('pageerror', (ev) => console.log('ERR', ev.message.slice(0, 200)));
      await boot();
    }
  }
}

function unwrap(rec) {
  let acc = 0;
  let prev = rec[0].heading;
  const out = [];
  for (const r of rec) {
    let d = r.heading - prev;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    acc += d;
    prev = r.heading;
    out.push(acc);
  }
  return out;
}

const report = { tag: TAG, at: new Date().toISOString(), runs: [] };
const SPEEDS = [8, 14, 20];
const SECS = 2.5;

console.log(`SKIDPAD — flat plane, full steer (1.0) held for ${SECS}s\n`);
console.log('  v      heading  pathRate  chassisYaw  radius   lean(p50/max)  target  steerA   |v|end');
for (const v of SPEEDS) {
  const rec = await evalRetry(
    ([v, secs]) => window.__TURN__.skidpad(v, 1.0, secs, { pedal: 0 }),
    [v, SECS],
  );
  const hdg = unwrap(rec);
  const N = rec.length;
  // PATH heading, from successive positions. The chassis heading is not the
  // path: with sideslip the two differ, and "does the bike turn" is a question
  // about where it GOES.
  for (let k = 1; k < N; k++) {
    rec[k].pathHeading = Math.atan2(rec[k].x - rec[k - 1].x, rec[k].z - rec[k - 1].z);
  }
  rec[0].pathHeading = rec[1] ? rec[1].pathHeading : 0;
  let pacc = 0, pprev = rec[0].pathHeading;
  for (const r of rec) {
    let d = r.pathHeading - pprev;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    pacc += d; pprev = r.pathHeading; r.pathAcc = pacc;
  }
  const totalHeading = hdg[N - 1];
  const meanYaw = totalHeading / rec[N - 1].t;
  // Steady state: last 40% of the run.
  const i0 = Math.floor(N * 0.6);
  const steadyYaw = (hdg[N - 1] - hdg[i0]) / (rec[N - 1].t - rec[i0].t);
  const steadySpeed = rec.slice(i0).reduce((a, r) => a + r.speed, 0) / (N - i0);
  const steadyPath = (rec[N - 1].pathAcc - rec[i0].pathAcc) / (rec[N - 1].t - rec[i0].t);
  // Radius of the PATH, which is what a rider means by "turn radius".
  const radius = Math.abs(steadyPath) > 1e-4 ? steadySpeed / Math.abs(steadyPath) : Infinity;
  const leans = rec.map((r) => r.lean).sort((a, c) => a - c);
  const run = {
    v, totalHeadingDeg: +(totalHeading * 180 / Math.PI).toFixed(1),
    meanYaw: +meanYaw.toFixed(4), steadyYaw: +steadyYaw.toFixed(4),
    steadyPath: +steadyPath.toFixed(4),
    radius: +radius.toFixed(2),
    leanP50: +leans[N >> 1].toFixed(3), leanMax: +leans[N - 1].toFixed(3),
    leanMin: +leans[0].toFixed(3),
    target: +rec[N - 1].targetLean.toFixed(3), steerA: +rec[N - 1].steerA.toFixed(4),
    endSpeed: +rec[N - 1].speed.toFixed(2),
    groundedPct: +(100 * rec.filter((r) => r.fg || r.rg).length / N).toFixed(1),
    yawDampT: +rec[N - 1].yawDampT.toFixed(1),
    fSlip: +rec[N - 1].fSlip.toFixed(4), rSlip: +rec[N - 1].rSlip.toFixed(4),
    fLat: +rec[N - 1].fLat.toFixed(3), rLat: +rec[N - 1].rLat.toFixed(3),
    fl: +rec[N - 1].fl.toFixed(0), rl: +rec[N - 1].rl.toFixed(0),
    modes: [...new Set(rec.map((r) => r.mode))],
  };
  report.runs.push(run);
  console.log(
    `${String(v).padStart(3)} m/s  ${String(run.totalHeadingDeg).padStart(7)}°  ` +
    `${String(run.steadyPath).padStart(8)}  ${String(run.steadyYaw).padStart(9)}  ` +
    `${String(run.radius).padStart(7)}  ${String(run.leanP50).padStart(6)}/${String(run.leanMax).padStart(6)}  ` +
    `${String(run.target).padStart(6)}  ${String(run.steerA).padStart(7)}  ${String(run.endSpeed).padStart(6)}`,
  );
  console.log(
    `        grounded ${run.groundedPct}%  slip f/r ${run.fSlip}/${run.rSlip}  ` +
    `vLat f/r ${run.fLat}/${run.rLat}  load f/r ${run.fl}/${run.rl}  yawDampT ${run.yawDampT}  modes ${run.modes.join(',')}`,
  );
  // Steady-state lean/turn agreement: does 40 degrees of lean produce the
  // lateral acceleration 40 degrees of lean is a promise of?
  const sl = rec.slice(i0);
  const pathRate = (rec[N - 1].pathAcc - rec[i0].pathAcc) / (rec[N - 1].t - rec[i0].t);
  const aAct = -pathRate * steadySpeed;
  const aDem = sl.reduce((a, r) => a + Math.tan(r.lean) * 20.4, 0) / sl.length;
  const fyF = sl.reduce((a, r) => a + Math.abs(r.fFy), 0) / sl.length;
  const fyR = sl.reduce((a, r) => a + Math.abs(r.rFy), 0) / sl.length;
  const capF = sl.reduce((a, r) => a + r.fCap, 0) / sl.length;
  const capR = sl.reduce((a, r) => a + r.rCap, 0) / sl.length;
  run.aActual = +aAct.toFixed(2); run.aDemand = +aDem.toFixed(2);
  run.turnRatio = +(aAct / aDem).toFixed(3);
  run.fyUse = +(fyF / capF).toFixed(3); run.ryUse = +(fyR / capR).toFixed(3);
  console.log(
    `        aLat actual ${aAct.toFixed(2)} m/s2 vs lean promise ${aDem.toFixed(2)} = ${(100 * aAct / aDem).toFixed(0)}%   ` +
    `tyre use f ${(100 * fyF / capF).toFixed(0)}% r ${(100 * fyR / capR).toFixed(0)}%  (Fy ${fyF.toFixed(0)}/${fyR.toFixed(0)} N)`,
  );
}

await mkdir(OUT, { recursive: true });
await writeFile(path.join(OUT, `turn-${TAG}.json`), JSON.stringify(report, null, 1));
console.log(`\n→ tools/capture/_out/turn-${TAG}.json`);
await b.close();
