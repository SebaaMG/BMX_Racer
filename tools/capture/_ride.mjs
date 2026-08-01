import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('PAGEERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
await p.evaluate(() => {
  const g = window.__DESCENT__.game;
  g.race.forceRacing?.();
  window.__T = { samples: [], stalls: [] };
  const seen = new Map();
  const id = setInterval(() => {
    for (const r of g.race.racers) {
      if (r.id === 'player') continue;
      const d = r.progress?.distance ?? 0, v = r.bike.state.speed * 3.6;
      const prev = seen.get(r.id) ?? { d: -1, t: 0 };
      if (d - prev.d < 0.5) { prev.t++; if (prev.t === 12) window.__T.stalls.push({ id: r.id, at: +d.toFixed(0), kmh: +v.toFixed(0) }); }
      else prev.t = 0;
      prev.d = d; seen.set(r.id, prev);
    }
  }, 250);
  window.__STOP = () => clearInterval(id);
});
const L = await p.evaluate(() => window.__DESCENT__.game.track.length);
for (const s of [30, 60, 90, 120, 150]) {
  await p.evaluate(() => new Promise(r => setTimeout(r, 30000)));
  const row = await p.evaluate(() => window.__DESCENT__.game.race.racers.filter(r=>r.id!=='player').map(r => ({
    id: r.id, d: +(r.progress?.distance ?? 0).toFixed(0), kmh: +(r.bike.state.speed*3.6).toFixed(0),
    fin: !!r.progress?.finished, crashes: r.progress?.crashCount ?? 0, mode: r.bike.state.mode })));
  console.log(`t=${s}s  ` + row.map(r=>`${r.id}:${r.d}m/${r.kmh}kmh${r.fin?' FINISHED':''}${r.mode!=='grounded'?'/'+r.mode:''} x${r.crashes}`).join('   '));
}
const st = await p.evaluate(() => { window.__STOP(); return window.__T.stalls; });
console.log(`\ncourse ${L.toFixed(0)} m. Stalls (no progress for 3 s):`);
console.log(st.length ? st.map(s=>`  ${s.id} stuck at ${s.at} m (${s.kmh} km/h)`).join('\n') : '  none');
await b.close();
