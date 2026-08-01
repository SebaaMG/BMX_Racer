import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('ERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
console.log(await p.evaluate(() => {
  const g = window.__DESCENT__.game; g.capture.takeControl();
  const out = [];
  // (a) Does frame 0 of a sequence have wheel contact now?
  for (const seq of ['scree-speed','launch','switchback']) {
    g.capture.setSequence(seq);
    for (let i = 0; i < 4; i++) g.capture.step(1/60);   // harness settle
    const st = g.race.player.bike.state;
    out.push(`${seq.padEnd(13)} f0000  rear=${st.rear.grounded} front=${st.front.grounded} airH=${st.airHeight.toFixed(2)}`);
  }
  // (b) Does the crash now land inside the captured window?
  g.capture.setSequence('crash');
  const st = g.race.player.bike.state;
  const c0 = st.crashCount;
  let hitFrame = -1;
  for (let f = 0; f < 60; f++) {
    g.capture.step(1/60);
    if (hitFrame < 0 && st.crashCount > c0) hitFrame = f;
  }
  out.push(`crash        crashCount at spawn=${c0}, fires at captured frame ${hitFrame}`);
  return out.join('\n');
}));
await b.close();
