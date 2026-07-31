import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('ERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.track, null, { timeout: 300000 });
console.log(JSON.stringify(await p.evaluate(() => {
  const g = window.__DESCENT__.game, tr = g.track, te = g.terrain;
  const secs = tr.sectionRanges.map(r => ({ kind: r.kind, d0: r.start, d1: r.end }));
  const out = [];
  for (const sec of secs) {
    let mn = 1e9, mx = -1e9, sum = 0, n = 0;
    const prev = [];
    let maxSlopeDeg = 0;
    for (let d = sec.d0; d <= sec.d1; d += 1.5) {
      const s = tr.sampleAtDistance(d);
      const e = te.heightAt(s.position.x, s.position.z) - s.position.y;
      mn = Math.min(mn, e); mx = Math.max(mx, e); sum += e; n++;
      prev.push([s.position.x, s.position.y, s.position.z]);
    }
    // steepest centreline segment, degrees from horizontal
    for (let i = 1; i < prev.length; i++) {
      const dx = prev[i][0]-prev[i-1][0], dy = prev[i][1]-prev[i-1][1], dz = prev[i][2]-prev[i-1][2];
      const horiz = Math.hypot(dx, dz);
      const deg = Math.abs(Math.atan2(dy, Math.max(horiz, 1e-4))) * 180/Math.PI;
      if (deg > maxSlopeDeg) maxSlopeDeg = deg;
    }
    out.push({ sec: sec.kind, min: +mn.toFixed(2), max: +mx.toFixed(2), mean: +(sum/n).toFixed(2), steepestDeg: +maxSlopeDeg.toFixed(1) });
  }
  return out;
}, null), null, 1));
await b.close();
