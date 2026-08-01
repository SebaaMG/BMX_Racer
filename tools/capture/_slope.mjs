import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
await p.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
const r = await p.evaluate(() => {
  const g = window.__DESCENT__.game, t = g.track, terr = g.terrain;
  const out = [];
  for (const d of [255, 265, 275, 285, 295]) {
    const s = t.sampleAtDistance(d);
    let maxSlope = 0, prof = [];
    let prevH = null;
    for (let o = -22; o <= 22; o += 2) {
      const h = terr.heightAt(s.position.x + s.left.x*o, s.position.z + s.left.z*o);
      if (prevH !== null) maxSlope = Math.max(maxSlope, Math.abs(Math.atan2(h - prevH, 2.5) * 180/Math.PI));
      prevH = h;
      if (o % 4 === 0) prof.push(+(h - s.position.y).toFixed(1));
    }
    out.push({ d, maxSideSlope: +maxSlope.toFixed(0), prof });
  }
  return out;
});
console.log('Cross-sections through the cut. prof = terrain height rel. to trail, -22..+22 m in 4 m steps (inside the carved corridor only).');
console.log('maxSideSlope is the steepest wall anywhere in the section (angle of repose for scree is ~34 deg).\n');
for (const x of r) console.log(` d=${x.d}  maxSideSlope=${String(x.maxSideSlope).padStart(2)} deg   ${x.prof.map(v=>String(v).padStart(6)).join('')}`);
await b.close();
