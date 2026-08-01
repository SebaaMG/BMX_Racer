import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('PAGEERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });

// 1. RIBBON vs COLLISION SURFACE, whole course
const float = await p.evaluate(() => {
  const g = window.__DESCENT__.game, t = g.track, terr = g.terrain;
  let sum=0, n=0, mx=-9, mn=9, off=0;
  for (let d=0; d<=t.length; d+=4) {
    const s = t.sampleAtDistance(d);
    const gap = s.position.y - terr.heightAt(s.position.x, s.position.z);
    sum+=gap; n++; mx=Math.max(mx,gap); mn=Math.min(mn,gap);
    if (Math.abs(gap) > 0.06) off++;
  }
  return { mean:+(sum/n).toFixed(3), min:+mn.toFixed(3), max:+mx.toFixed(3), n, offBy6cm:off };
});
console.log('1. RIBBON FLOAT over the whole course:', JSON.stringify(float));

// 2. AI DIRECTION — 20 s of real racing
await p.evaluate(() => { window.__DESCENT__.game.race.forceRacing?.(); });
await p.evaluate(() => new Promise(r => setTimeout(r, 20000)));
const ai = await p.evaluate(() => {
  const g = window.__DESCENT__.game;
  const FWD = new g.race.player.bike.state.position.constructor(0,0,1);
  return g.race.racers.map(r => {
    const st = r.bike.state;
    const fwd = FWD.clone().applyQuaternion(st.orientation); fwd.y = 0; fwd.normalize();
    const s = g.track.sampleAtDistance(r.progress?.distance ?? 0);
    const tan = s.tangent.clone(); tan.y = 0; tan.normalize();
    return { id: r.id, kmh: +(st.speed*3.6).toFixed(0),
             dist: +(r.progress?.distance ?? -1).toFixed(0),
             headErr: +(Math.acos(Math.max(-1,Math.min(1,fwd.dot(tan))))*180/Math.PI).toFixed(0),
             lat: +(r.trackLateral ?? 0).toFixed(1), mode: st.mode };
  });
});
console.log('\n2. AFTER 20 s OF RACING (headErr 0 = facing down-course):');
for (const r of ai) console.log('  ', JSON.stringify(r));
await b.close();
