import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
await p.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
await p.evaluate(() => { window.__DESCENT__.game.race.forceRacing?.();
  window.__W = []; const g = window.__DESCENT__.game;
  window.__I = setInterval(() => {
    for (const r of g.race.racers) { if (r.id==='player') continue;
      const s = g.track.sampleAtDistance(r.progress.distance);
      window.__W.push({ id:r.id, t:+g.race.raceTime.toFixed(1), lat:+(r.trackLateral??0).toFixed(1), hw:+s.halfWidth.toFixed(1), crash:r.progress.crashCount }); }
  }, 500); });
await p.evaluate(() => new Promise(r => setTimeout(r, 20000)));
const W = await p.evaluate(() => { clearInterval(window.__I); return window.__W; });
const t0 = W.filter(x=>x.t<=6);
console.log('lateral vs time, first 20 s (hw ~3.0-3.6 m):');for (const id of ['ai0','ai1','ai2']) { console.log(' '+id+': '+W.filter(x=>x.id===id).map(x=>x.lat).join(' ')); }
const byId = {};
for (const x of W) { byId[x.id] = byId[x.id] || {max:0,crash:0}; byId[x.id].max = Math.max(byId[x.id].max, Math.abs(x.lat)); byId[x.id].crash = x.crash; }
for (const k in byId) console.log(`  ${k}: max |lateral| ${byId[k].max.toFixed(1)} m, crashes ${byId[k].crash}`);
await b.close();
