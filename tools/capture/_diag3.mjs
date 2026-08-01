import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
await p.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
const r = await p.evaluate(() => {
  const g = window.__DESCENT__.game;
  const t = g.track;
  const L = t.length;
  const step = 2;
  const n = Math.floor(L / step) + 1;
  const ys = new Float64Array(n);
  const xs = new Float64Array(n);
  const zs = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const s = t.sampleAtDistance(i * step);
    ys[i] = s.position.y; xs[i] = s.position.x; zs[i] = s.position.z;
  }
  const climbs = [];
  let run = null;
  for (let i = 1; i < n; i++) {
    const grade = Math.atan2(ys[i] - ys[i-1], step) * 180 / Math.PI;
    if (grade > 3) {
      if (!run) run = { from: (i-1)*step, rise: 0, peak: 0 };
      run.to = i*step; run.rise += ys[i]-ys[i-1]; run.peak = Math.max(run.peak, grade);
    } else if (run) {
      if (run.rise > 1.0) climbs.push({ from: run.from, to: run.to, rise: +run.rise.toFixed(1), peak: +run.peak.toFixed(1) });
      run = null;
    }
  }
  if (run && run.rise > 1.0) climbs.push({ from: run.from, to: run.to, rise: +run.rise.toFixed(1), peak: +run.peak.toFixed(1) });
  const feats = [];
  for (const d of [250,260,270,284,316,330,338]) {
    const s = t.sampleAtDistance(d);
    feats.push({ d, section: s.sectionId, hw: +s.halfWidth.toFixed(2), y: +s.position.y.toFixed(2), lip: s.lipMask, gap: s.gapMask });
  }
  return { L: +L.toFixed(1), drop: +(ys[0]-ys[n-1]).toFixed(1), climbs, feats,
           totalClimb: +climbs.reduce((a,c)=>a+c.rise,0).toFixed(1) };
});
console.log(`track ${r.L} m, net drop ${r.drop} m, total CLIMBING ${r.totalClimb} m`);
console.log('\nEvery sustained climb (>1 m rise, >3 deg) on the centreline:');
for (const c of r.climbs) console.log(`  ${c.from}-${c.to} m   +${c.rise} m   peak ${c.peak} deg`);
console.log('\nsamples around the first two:');
for (const f of r.feats) console.log(' ', JSON.stringify(f));
await b.close();
