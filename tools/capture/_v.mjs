import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
const errs = [];
p.on('pageerror', e => errs.push('PAGEERR ' + e.message.slice(0,300)));
p.on('console', m => { if (m.type()==='error') errs.push('CONSOLE ' + m.text().slice(0,300)); });
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
try {
  await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
  const r = await p.evaluate(() => {
    const g = window.__DESCENT__.game; g.capture.takeControl();
    const out = {};
    for (const s of ['bike-detail','scree-speed','streambed','ridge-exposure','crash']) {
      g.capture.setPose(s);
      for (let i=0;i<12;i++) g.capture.step(1/60);
      const st = g.race.player.bike.state;
      out[s] = { kmh: +(st.speed*3.6).toFixed(0), mode: st.mode };
    }
    out._programs = g.engine.stats.programs;
    return out;
  });
  console.log('RENDERS OK', JSON.stringify(r));
} catch (e) { console.log('FAILED:', e.message.slice(0,200)); }
console.log(errs.length ? 'ERRORS:\n' + errs.slice(0,5).join('\n') : 'no page/gl errors');
await b.close();
