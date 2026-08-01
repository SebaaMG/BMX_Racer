import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('ERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
console.log(JSON.stringify(await p.evaluate(() => {
  const g = window.__DESCENT__.game; g.capture.takeControl();
  g.capture.setSequence('launch');
  const bike = g.race.player.bike;
  const rows = [];
  let prev = bike.visual ? null : null;
  for (let f = 0; f < 16; f++) {
    g.capture.step(1/60);
    if (f % 3 === 0) rows.push({ f, crankRad: +(bike.vis?.crankAngle ?? 0).toFixed(2),
      rpm: +((bike.physics.rear.spinRate * 0.0222/0.089) * 60/(2*Math.PI)).toFixed(0),
      kmh: +(bike.state.speed*3.6).toFixed(0) });
  }
  return rows;
}, null), null, 0));
await b.close();
