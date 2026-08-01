import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('PAGEERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });

// ---- A. TERRAIN vs RIBBON along the opening of the course -------------------
const profile = await p.evaluate(() => {
  const g = window.__DESCENT__.game, t = g.track, terr = g.terrain;
  const rows = [];
  for (let d = 0; d <= 260; d += 2) {
    const s = t.sampleAtDistance(d);
    const h = terr.heightAt(s.position.x, s.position.z);
    rows.push({ d, ribbonY: +s.position.y.toFixed(3), terrainY: +h.toFixed(3),
                delta: +(h - s.position.y).toFixed(3) });
  }
  return rows;
});
console.log('=== A. ribbon vs terrain, first 260 m ===');
const bad = profile.filter(r => Math.abs(r.delta) > 0.10);
console.log(`samples ${profile.length}, |delta|>0.10 m on ${bad.length}`);
let maxD = profile.reduce((a,r)=>Math.abs(r.delta)>Math.abs(a.delta)?r:a);
console.log('worst:', JSON.stringify(maxD));
// slope of the COLLISION surface, which is what the bike actually rides
console.log('--- collision-surface grade, first 260 m (deg, + = uphill) ---');
const grades = [];
for (let i=1;i<profile.length;i++){
  const dy = profile[i].terrainY - profile[i-1].terrainY;
  grades.push({ d: profile[i].d, g: +(Math.atan2(dy, 2) * 180/Math.PI).toFixed(1) });
}
console.log(grades.filter(x=>x.g > 6).map(x=>`${x.d}m:+${x.g}`).join(' ') || 'no uphill >6 deg');
console.log('steepest uphill:', JSON.stringify(grades.reduce((a,x)=>x.g>a.g?x:a)));

// ---- B. AI DIRECTION at the start -----------------------------------------
const ai0 = await p.evaluate(() => {
  const g = window.__DESCENT__.game;
  g.race.forceRacing?.();
  return g.race.racers.map(r => {
    const st = r.bike.state;
    const f = new (window.__DESCENT__.THREE?.Vector3 ?? Object)();
    return { id: r.id, name: r.name,
             pos: [+st.position.x.toFixed(1), +st.position.y.toFixed(1), +st.position.z.toFixed(1)],
             trackDist: +(r.progress?.distance ?? -1).toFixed(1) };
  });
});
console.log('\n=== B. racers at t=0 ==='); console.log(JSON.stringify(ai0, null, 1));

// run 6 s of real physics
await p.evaluate(() => new Promise(r => setTimeout(r, 6000)));
const ai1 = await p.evaluate(() => {
  const g = window.__DESCENT__.game;
  return g.race.racers.map(r => {
    const st = r.bike.state;
    const fwd = new st.position.constructor(0,0,-1).applyQuaternion(st.orientation);
    const s = g.track.sampleAtDistance(r.progress?.distance ?? 0);
    const dot = fwd.dot(s.tangent);
    return { id: r.id,
             pos: [+st.position.x.toFixed(1), +st.position.y.toFixed(1), +st.position.z.toFixed(1)],
             kmh: +(st.speed*3.6).toFixed(0),
             fwdSpeed: +st.forwardSpeed.toFixed(1),
             trackDist: +(r.progress?.distance ?? -1).toFixed(1),
             headingVsTrack: +(Math.acos(Math.max(-1,Math.min(1,dot)))*180/Math.PI).toFixed(0),
             mode: st.mode,
             groundY: +g.terrain.heightAt(st.position.x, st.position.z).toFixed(2),
             sink: +(g.terrain.heightAt(st.position.x, st.position.z) - st.position.y).toFixed(2) };
  });
});
console.log('\n=== C. racers after 6 s (headingVsTrack: 0=correct, 180=backwards) ===');
console.log(JSON.stringify(ai1, null, 1));
await b.close();
