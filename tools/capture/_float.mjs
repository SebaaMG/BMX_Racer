import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
await p.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
const r = await p.evaluate(() => {
  const g = window.__DESCENT__.game, t = g.track, terr = g.terrain;
  const gaps = [];
  for (let d=0; d<=t.length; d+=2) {
    const s = t.sampleAtDistance(d);
    const gap = s.position.y - terr.heightAt(s.position.x, s.position.z);
    if (Math.abs(gap) < 3) gaps.push({ d, gap });   // exclude the ravine span
  }
  const v = gaps.map(x=>x.gap).sort((a,b)=>a-b);
  const q = f => +v[Math.floor(f*(v.length-1))].toFixed(3);
  const worst = gaps.slice().sort((a,b)=>Math.abs(b.gap)-Math.abs(a.gap)).slice(0,6);
  return { n: v.length, mean: +(v.reduce((a,x)=>a+x,0)/v.length).toFixed(3),
           p05:q(0.05), p50:q(0.50), p95:q(0.95), min:q(0), max:q(1),
           worst: worst.map(w=>({d:w.d, gap:+w.gap.toFixed(2)})),
           cell: g.terrain.cellSize ?? null };
});
console.log(JSON.stringify(r, null, 1));
await b.close();
