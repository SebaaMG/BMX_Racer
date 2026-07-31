/**
 * _diag.mjs — per-step forensic dump for one section.
 * node tools/capture/_diag.mjs [pose] [seconds] [--roll]
 */
import { chromium } from 'playwright';

const POSE = process.argv[2] ?? 'scree-speed';
const SECS = Number(process.argv[3] ?? 1.5);

const b = await chromium.launch({ headless: true, args: ['--use-angle=default', '--enable-unsafe-swiftshader'] });
const p = await b.newPage();
p.on('pageerror', (e) => console.log('ERR', e.message.slice(0, 300)));
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.capture, null, { timeout: 300000 });

const out = await p.evaluate(([POSE, SECS]) => {
  const g = window.__DESCENT__.game;
  g.capture.takeControl();
  g.capture.setPose(POSE);
  const bike = g.race.player.bike;
  const phys = bike.physics;
  const st = bike.state;
  const orig = phys.step.bind(phys);
  const rows = [];
  let n = 0;
  const THREE = { };
  phys.step = (input, dt) => {
    orig(input, dt);
    // terrain under each mount
    const fm = phys.front.mountWorld(st.position, st.orientation, new (st.position.constructor)());
    const rm = phys.rear.mountWorld(st.position, st.orientation, new (st.position.constructor)());
    rows.push({
      n,
      y: +st.position.y.toFixed(3),
      vy: +st.velocity.y.toFixed(2),
      sp: +st.speed.toFixed(2),
      lean: +st.lean.toFixed(3),
      wR: +st.angularVelocity.dot(new (st.position.constructor)(0, 0, 1).applyQuaternion(st.orientation)).toFixed(2),
      pitch: +st.pitch.toFixed(3),
      fg: st.front.grounded ? 1 : 0,
      rg: st.rear.grounded ? 1 : 0,
      fL: Math.round(st.front.load),
      rL: Math.round(st.rear.load),
      fC: +st.front.compression.toFixed(2),
      rC: +st.rear.compression.toFixed(2),
      fSusp: +phys.front.suspensionLength.toFixed(3),
      rSusp: +phys.rear.suspensionLength.toFixed(3),
      fGap: +(fm.y - g.terrain.heightAt(fm.x, fm.z)).toFixed(3),
      rGap: +(rm.y - g.terrain.heightAt(rm.x, rm.z)).toFixed(3),
      mode: st.mode.slice(0, 3),
    });
    n++;
  };
  const frames = Math.round(SECS * 60);
  for (let i = 0; i < frames; i++) g.capture.step(1 / 60);
  phys.step = orig;

  // terrain roughness along the path: sample height every 0.25 m of travel
  const pts = rows.map((r) => r.y);
  return { rows };
}, [POSE, SECS]);

const R = out.rows;
console.log('n     y      vy    sp    lean   wRoll  pitch  fg rg  fLoad rLoad  fC   rC   fGap   rGap  mode');
for (let i = 0; i < R.length; i += 2) {
  const r = R[i];
  console.log(
    `${String(r.n).padStart(4)} ${String(r.y).padStart(7)} ${String(r.vy).padStart(6)} ${String(r.sp).padStart(5)} ` +
    `${String(r.lean).padStart(6)} ${String(r.wR).padStart(6)} ${String(r.pitch).padStart(6)}  ` +
    `${r.fg}  ${r.rg}  ${String(r.fL).padStart(5)} ${String(r.rL).padStart(5)} ` +
    `${String(r.fC).padStart(4)} ${String(r.rC).padStart(4)} ${String(r.fGap).padStart(6)} ${String(r.rGap).padStart(6)}  ${r.mode}`,
  );
}
await b.close();
