import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
await p.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
const r = await p.evaluate(() => {
  const g = window.__DESCENT__.game, t = g.track, terr = g.terrain;
  const rows = [];
  for (const d of [246, 262, 270, 278, 284, 2604, 3258]) {
    const s = t.sampleAtDistance(d);
    const lat = [];
    for (let o = -60; o <= 60; o += 6) {
      const x = s.position.x + s.left.x * o, z = s.position.z + s.left.z * o;
      lat.push(+(terr.heightAt(x, z) - s.position.y).toFixed(1));
    }
    rows.push({ d, y: +s.position.y.toFixed(1), lat });
  }
  return rows;
});
console.log('Terrain height RELATIVE to the centreline, -60 m (left) to +60 m (right), 6 m steps:');
for (const row of rows_(r)) console.log(row);
function rows_(r){ return r.map(x => ` d=${String(x.d).padStart(4)}  ` + x.lat.map(v=>String(v).padStart(6)).join('')); }
await b.close();
