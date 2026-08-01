import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e=>console.log('PAGEERR', e.message.slice(0,160)));
await p.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
await p.evaluate(() => {
  const g = window.__DESCENT__.game;
  g.race.forceRacing?.();
  window.__L = [];
  window.__I = setInterval(() => {
    for (const r of g.race.aiRiders) {
      const s = g.track.sampleAtDistance(r.progress.distance);
      window.__L.push({ id:r.id, t:+g.race.raceTime.toFixed(1), d:+r.progress.distance.toFixed(0),
        v:+(r.bike.state.speed*3.6).toFixed(0), lat:+(r.trackLateral??0).toFixed(1), hw:+s.halfWidth.toFixed(1),
        fin:!!r.progress.finished, ft:r.progress.finishTime, cr:r.progress.crashCount });
    }
  }, 400);
});
await p.evaluate(() => new Promise(r => setTimeout(r, 75000)));
const out = await p.evaluate(() => { clearInterval(window.__I);
  const g = window.__DESCENT__.game;
  return { log: window.__L, rescues: g.race.rescues,
    fin: g.race.aiRiders.map(r=>({id:r.id, fin:!!r.progress.finished,
      ft: r.progress.finishTime? +r.progress.finishTime.toFixed(1):null, cr:r.progress.crashCount })) };
});
const L = out.log;
const racing = L.filter(x=>!x.fin); const off = racing.filter(x=>Math.abs(x.lat) > x.hw);
console.log(`course 240 m | rescues ${out.rescues} | off-track ${(off.length/racing.length*100).toFixed(0)}% of samples`);
console.log('finish:', JSON.stringify(out.fin));
for (const id of ['ai0','ai1','ai2']) {
  const r = L.filter(x=>x.id===id);
  console.log(' '+id+' lat: ' + r.filter((_,i)=>i%3===0).slice(0,16).map(x=>x.lat).join(' ') + '   (halfWidth '+r[0].hw+')');
}
await b.close();
