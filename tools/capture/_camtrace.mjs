/**
 * _camtrace.mjs — per-frame camera trace + the critic's pixel-delta metric.
 *
 *   node tools/capture/_camtrace.mjs --seq scree-speed --frames 120
 *   node tools/capture/_camtrace.mjs --all --tag before
 *
 * Dumps, per frame: fov, shake offset (m and px), timeScale, boom length,
 * camera yaw/roll, swing, crashFocus, subject px, and the mean absolute
 * frame-to-frame pixel delta measured off the WebGL canvas in-page.
 */
import { chromium } from 'playwright';
import { writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

const args = {};
for (let i = 0; i < process.argv.length; i++) {
  const a = process.argv[i];
  if (!a.startsWith('--')) continue;
  const n = process.argv[i + 1];
  args[a.slice(2)] = n && !n.startsWith('--') ? (i++, n) : true;
}

const URL_BASE = args.url ?? 'http://127.0.0.1:5173';
const FPS = 60;
const FRAMES = Number(args.frames ?? 120);
const ALL = ['scree-speed', 'switchback', 'tabletop-air', 'crash', 'landing', 'trick-360', 'pack-race', 'launch'];
const SEQS = args.all ? ALL : String(args.seq ?? 'scree-speed').split(',');
const TAG = args.tag ?? 'run';
const QUIET = !!args.quiet;

const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-swiftshader', '--use-angle=default', '--enable-gpu', '--ignore-gpu-blocklist'],
});
const ctx = await browser.newContext({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1 });
const page = await ctx.newPage();
page.on('pageerror', (e) => console.log('  [exception]', e.message));

async function ensureReady() {
  await page.goto(`${URL_BASE}/?capture=1&pr=1`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForFunction(() => !!window.__DESCENT__?.game?.capture, null, { timeout: 240000 });
  await page.evaluate(() => window.__DESCENT__.game.capture.takeControl());

  await page.evaluate(() => {
    const g = window.__DESCENT__.game;
    const V = g.effects.cameraDirector.camera.position.constructor;
    const cv = document.createElement('canvas');
    cv.width = 400; cv.height = 225;
    const c2 = cv.getContext('2d', { willReadFrequently: true });
    let prev = null;
    window.__RESETDIFF__ = () => { prev = null; };
    window.__PROBE__ = () => {
      const dir = g.effects.cameraDirector;
      const cam = dir.camera;
      const st = g.race.player.bike.state;
      const p = st.position;
      const c = cam.position;
      const H = window.innerHeight;

      // Subject screen extent (wheel bottom to helmet top).
      const a = new V(p.x, p.y - 0.35, p.z).project(cam);
      const b = new V(p.x, p.y + 1.60, p.z).project(cam);
      const pxA = (1 - a.y) * 0.5 * H;
      const pxB = (1 - b.y) * 0.5 * H;
      const riderPx = Math.abs(pxA - pxB);
      const cy = (1 - (a.y + b.y) * 0.5) * 0.5;
      const cxn = ((a.x + b.x) * 0.5 + 1) * 0.5;
      const onScreen = a.z > -1 && a.z < 1 && cxn > -0.05 && cxn < 1.05;

      // Camera basis.
      const e = cam.matrixWorld.elements;
      const fwdx = -e[8], fwdy = -e[9], fwdz = -e[10];
      const yaw = Math.atan2(fwdx, fwdz);
      const pitch = Math.asin(Math.max(-1, Math.min(1, fwdy)));

      const dx = c.x - p.x, dy = c.y - p.y, dz = c.z - p.z;
      const dist = Math.hypot(dx, dy, dz);

      const so = dir.shakeOffset;
      const shakeM = Math.hypot(so.x, so.y, so.z);
      const tanHalf = Math.tan((cam.fov * Math.PI) / 360);
      const shakePx = dist > 0.01 ? (shakeM / dist) / (2 * tanHalf) * H : 0;

      // Pixel delta off the render canvas, downsampled.
      let delta = 0;
      try {
        const src = document.querySelector('canvas');
        c2.drawImage(src, 0, 0, 400, 225);
        const img = c2.getImageData(0, 0, 400, 225).data;
        if (prev) {
          let s = 0;
          for (let i = 0; i < img.length; i += 4) {
            s += Math.abs(img[i] - prev[i]) + Math.abs(img[i + 1] - prev[i + 1]) + Math.abs(img[i + 2] - prev[i + 2]);
          }
          delta = s / (3 * (img.length / 4));
        }
        prev = img.slice();
      } catch (err) { delta = -1; }

      return {
        fov: cam.fov,
        ts: dir.timeScale,
        shakeM, shakePx, shakeAmp: dir.shakeAmp, shakeT: dir.shakeT, shakeDur: dir.shakeDur,
        bLen: dir.boomLength, bWant: dir.boomDesired, bRet: dir.boomRetract,
        lift: dir.collisionLift, rise: dir.framedRise, az: dir.framedAz,
        bias: dir.frameBias, cf: dir.crashFocus,
        swing: dir.swingAmount, swingOn: !!dir.swingActive, swingCd: dir.swingCooldown,
        slowOn: !!dir.slowActive, slowCd: dir.slowCooldown,
        yaw, pitch, roll: dir.roll, aimYaw: dir.aimYaw, yawRate: dir.yawRate,
        dist, riderPx, cy, cxn, onScreen,
        speed: st.speed, mode: st.mode, air: st.airHeight, peak: st.peakAirHeight, vy: st.velocity.y,
        airTime: st.airTime, kick: dir.kick,
        delta,
        cxw: c.x, cyw: c.y, czw: c.z, px: p.x, py: p.y, pz: p.z,
      };
    };
  });
}
await ensureReady();

async function runSeq(seq) {
  const ok = await page.evaluate((n) => window.__DESCENT__.game.capture.setSequence(n), seq);
  if (!ok) { console.log('  unknown seq', seq); return null; }
  return page.evaluate(
    ([frames, dt]) => {
      const g = window.__DESCENT__.game;
      window.__RESETDIFF__();
      for (let i = 0; i < 4; i++) g.capture.step(dt);
      const r = [];
      for (let f = 0; f < frames; f++) {
        g.capture.step(dt);
        r.push(window.__PROBE__());
      }
      return r;
    },
    [FRAMES, 1 / FPS],
  );
}

const f2 = (x) => (x === undefined || x === null ? '   -  ' : x.toFixed(2).padStart(6));
const f3 = (x) => (x === undefined || x === null ? '    -  ' : x.toFixed(3).padStart(7));

const out = {};
for (const seq of SEQS) {
  let rows = null;
  for (let attempt = 0; attempt < 3 && !rows; attempt++) {
    try { rows = await runSeq(seq); }
    catch (e) { console.log(`  ! ${seq} attempt ${attempt + 1}: ${String(e.message).split('\n')[0]}`); await ensureReady(); }
  }
  if (!rows) { console.log('  xx gave up on', seq); continue; }
  out[seq] = rows;

  const q = (arr, p) => { const s = [...arr].sort((a, b) => a - b); return s[Math.floor(p * (s.length - 1))]; };
  const col = (k) => rows.map((r) => r[k]);
  console.log(`\n════ ${seq}  ${rows.length} frames ════`);
  console.log(`  speed  ${(Math.min(...col('speed')) * 3.6).toFixed(0)}→${(Math.max(...col('speed')) * 3.6).toFixed(0)} km/h`);
  console.log(`  fov    ${Math.min(...col('fov')).toFixed(1)} → ${Math.max(...col('fov')).toFixed(1)}   (span ${(Math.max(...col('fov')) - Math.min(...col('fov'))).toFixed(2)})`);
  console.log(`  boom   min ${Math.min(...col('bLen')).toFixed(2)} med ${q(col('bLen'), 0.5).toFixed(2)} max ${Math.max(...col('bLen')).toFixed(2)}`);
  console.log(`  riderPx min ${Math.min(...col('riderPx')).toFixed(0)} med ${q(col('riderPx'), 0.5).toFixed(0)} max ${Math.max(...col('riderPx')).toFixed(0)}   offscreen ${rows.filter((r) => !r.onScreen).length}`);
  console.log(`  shake  max ${Math.max(...col('shakeM')).toFixed(3)} m  = ${Math.max(...col('shakePx')).toFixed(1)} px   frames>0.5px ${rows.filter((r) => r.shakePx > 0.5).length}`);
  console.log(`  timeScale min ${Math.min(...col('ts')).toFixed(3)}   slow frames ${rows.filter((r) => r.ts < 0.995).length}`);
  console.log(`  swing  max ${Math.max(...col('swing')).toFixed(3)}   active frames ${rows.filter((r) => r.swingOn).length}`);
  console.log(`  roll   |max| ${Math.max(...col('roll').map(Math.abs)).toFixed(4)} rad   yawRate |max| ${Math.max(...col('yawRate').map(Math.abs)).toFixed(3)}`);
  console.log(`  crashFocus max ${Math.max(...col('cf')).toFixed(2)}   lift max ${Math.max(...col('lift')).toFixed(2)}   rise max ${Math.max(...col('rise')).toFixed(3)}`);
  const dl = col('delta').slice(1);
  console.log(`  pixel delta  min ${Math.min(...dl).toFixed(2)} med ${q(dl, 0.5).toFixed(2)} max ${Math.max(...dl).toFixed(2)}`);
  // Correlate delta with speed (first vs last third).
  const n = rows.length, a = rows.slice(2, Math.floor(n / 3)), b = rows.slice(Math.floor((2 * n) / 3));
  const mean = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;
  console.log(`  delta  first-third ${mean(a.map((r) => r.delta)).toFixed(2)} @ ${(mean(a.map((r) => r.speed)) * 3.6).toFixed(0)} km/h   last-third ${mean(b.map((r) => r.delta)).toFixed(2)} @ ${(mean(b.map((r) => r.speed)) * 3.6).toFixed(0)} km/h`);

  if (!QUIET) {
    console.log('   f | speed  fov   boom  dist  ridPx  shkPx   ts   swing  roll   yaw    cf   delta  mode air');
    for (let i = 0; i < rows.length; i += Number(args.every ?? 4)) {
      const r = rows[i];
      console.log(
        `  ${String(i).padStart(3)}|${f2(r.speed * 3.6)} ${f2(r.fov)} ${f2(r.bLen)} ${f2(r.dist)} ${f2(r.riderPx)} ${f2(r.shakePx)} ${f2(r.ts)} ${f2(r.swing)} ${f3(r.roll)} ${f3(r.yaw)} ${f2(r.cf)} ${f2(r.delta)}  ${r.mode} ${r.air.toFixed(1)}`,
      );
    }
  }
}

await mkdir(path.resolve('tools/capture/_out'), { recursive: true });
await writeFile(path.resolve(`tools/capture/_out/camtrace-${TAG}.json`), JSON.stringify(out));
console.log(`\n→ tools/capture/_out/camtrace-${TAG}.json`);
await browser.close();
