import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('PAGEERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
await p.evaluate(() => {
  const g = window.__DESCENT__.game;
  g.race.forceRacing?.();
  window.__L = [];
  const r = g.race.racers.find(x => x.id === 'ai1');
  window.__I = setInterval(() => {
    const st = r.bike.state, pl = r.plan, s = g.track.sampleAtDistance(r.progress.distance);
    window.__L.push({ d:+r.progress.distance.toFixed(0), v:+(st.speed*3.6).toFixed(0),
      tgt:+((pl?.targetSpeed ?? -1)*3.6).toFixed(0), bR:+(r.input.brakeRear).toFixed(2),
      bF:+(r.input.brakeFront).toFixed(2), ped:+(r.input.pedal).toFixed(2),
      slope:+(Math.asin(-s.tangent.y)*180/Math.PI).toFixed(0), lat:+(r.trackLateral).toFixed(1) });
  }, 500);
});
await p.evaluate(() => new Promise(r => setTimeout(r, 75000)));
const L = await p.evaluate(() => { clearInterval(window.__I); return window.__L; });
const fast = L.filter(x => x.v > x.tgt + 12);
console.log(`${L.length} samples, ${fast.length} where speed exceeds target by >12 km/h`);
console.log('d      v   tgt   bR    bF   ped  slope  lat');
for (const x of L.filter((_,i)=>i%4===0)) console.log(
  String(x.d).padStart(5), String(x.v).padStart(4), String(x.tgt).padStart(5),
  String(x.bR).padStart(5), String(x.bF).padStart(5), String(x.ped).padStart(5),
  String(x.slope).padStart(5), String(x.lat).padStart(6));
const top = L.reduce((a,x)=>x.v>a.v?x:a);
console.log('\nfastest sample:', JSON.stringify(top));
await b.close();
