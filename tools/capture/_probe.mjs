import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('PAGEERR', e.message.slice(0,400)));
p.on('console', m => { if (m.type()==='error') console.log('CONSOLE', m.text().slice(0,500)); });
p.on('crash', () => console.log('*** PAGE CRASHED ***'));
p.on('close', () => console.log('*** PAGE CLOSED ***'));
try {
  await p.goto('http://127.0.0.1:5173/?capture=1&pr=2', { waitUntil: 'domcontentloaded' });
  await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 240000 });
  console.log('booted ok');
  const r = await p.evaluate(() => {
    const g = window.__DESCENT__.game;
    g.capture.takeControl();
    const ok = g.capture.setSequence('trick-360');
    let n = 0;
    try { for (; n < 60; n++) g.capture.step(1/60); }
    catch (e) { return { ok, crashedAtFrame: n, err: String(e).slice(0,300) }; }
    return { ok, framesRun: n, programs: g.engine.stats.programs };
  });
  console.log('RESULT', JSON.stringify(r));
} catch (e) { console.log('OUTER FAIL:', e.message.slice(0,300)); }
await b.close().catch(()=>{});
