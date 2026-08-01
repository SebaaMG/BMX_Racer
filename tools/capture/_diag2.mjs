import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('PAGEERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });

// ---- THE BUMP: profile the collision surface far enough to see it whole ----
const prof = await p.evaluate(() => {
  const g = window.__DESCENT__.game, t = g.track, terr = g.terrain;
  const rows = [];
  for (let d = 0; d <= 600; d += 2) {
    const s = t.sampleAtDistance(d);
    rows.push({ d, ry: s.position.y, ty: terr.heightAt(s.position.x, s.position.z) });
  }
  return rows;
});
console.log('=== THE BUMP: collision-surface grade along the course opening ===');
console.log('(+ = climbing. A downhill course should be negative almost everywhere.)');
let run = null; const walls = [];
for (let i=1;i<prof.length;i++){
  const g = Math.atan2(prof[i].ty - prof[i-1].ty, 2) * 180/Math.PI;
  if (g > 5) { if (!run) run = { from: prof[i-1].d, peak: g, rise: 0 }; run.to = prof[i].d; run.peak = Math.max(run.peak, g); run.rise += prof[i].ty - prof[i-1].ty; }
  else if (run) { walls.push(run); run = null; }
}
if (run) walls.push(run);
for (const w of walls) console.log(`  climb ${w.from}->${w.to} m : ${(w.to-w.from)} m long, +${w.rise.toFixed(2)} m rise, peak ${w.peak.toFixed(1)} deg`);
console.log('ribbon-vs-terrain: ribbon floats above collision by',
  (prof.reduce((a,r)=>a+(r.ry-r.ty),0)/prof.length).toFixed(3), 'm mean;',
  'min', Math.min(...prof.map(r=>r.ry-r.ty)).toFixed(3),
  'max', Math.max(...prof.map(r=>r.ry-r.ty)).toFixed(3));

// ---- PLAYER + AI over 10 s of real racing ---------------------------------
await p.evaluate(() => { window.__DESCENT__.game.race.forceRacing?.(); });
await p.evaluate(() => new Promise(r => setTimeout(r, 10000)));
const out = await p.evaluate(() => {
  const g = window.__DESCENT__.game;
  return g.race.racers.map(r => {
    const st = r.bike.state;
    const fwd = new st.position.constructor(0,0,-1).applyQuaternion(st.orientation);
    const s = g.track.sampleAtDistance(r.progress?.distance ?? 0);
    return { id: r.id, kmh: +(st.speed*3.6).toFixed(0),
             dist: +(r.progress?.distance ?? -1).toFixed(1),
             headErr: +(Math.acos(Math.max(-1,Math.min(1,fwd.dot(s.tangent))))*180/Math.PI).toFixed(0),
             lateral: +(r.progress?.lateral ?? 0).toFixed(1),
             mode: st.mode };
  });
});
console.log('\n=== 10 s of racing (headErr 0 = facing down-course, 180 = backwards) ===');
for (const r of out) console.log(' ', JSON.stringify(r));
await b.close();
