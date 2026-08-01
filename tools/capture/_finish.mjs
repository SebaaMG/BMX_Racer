import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('PAGEERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
await p.evaluate(() => window.__DESCENT__.game.race.forceRacing?.());
const L = await p.evaluate(() => window.__DESCENT__.game.track.length);
for (let t = 30; t <= 330; t += 30) {
  await p.evaluate(() => new Promise(r => setTimeout(r, 30000)));
  const row = await p.evaluate(() => {
    const g = window.__DESCENT__.game;
    return { resc: g.race.rescues, r: g.race.racers.filter(x=>x.id!=='player').map(r => ({
      id:r.id, d:+(r.progress.distance).toFixed(0), v:+(r.bike.state.speed*3.6).toFixed(0),
      fin:!!r.progress.finished, ft: r.progress.finishTime ? +r.progress.finishTime.toFixed(1) : null,
      cr:r.progress.crashCount }))};
  });
  console.log(`t=${String(t).padStart(3)}s rescues=${row.resc}  ` +
    row.r.map(x=>`${x.id}:${String(x.d).padStart(4)}m/${String(x.v).padStart(3)}${x.fin?` FIN@${x.ft}s`:''} x${x.cr}`).join('  '));
  if (row.r.every(x=>x.fin)) { console.log('\nALL RIDERS FINISHED'); break; }
}
console.log(`course ${L.toFixed(0)} m`);
await b.close();
