import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('ERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
console.log(JSON.stringify(await p.evaluate(() => {
  const g = window.__DESCENT__.game;
  g.capture.takeControl();
  const S = g.constructor; // not used; we drive via setPose then step manually
  const out = {};
  // Re-implement the pose spawn so we can sweep preroll without editing Game.
  const run = (poseName, tFrac, speed, prerollSteps) => {
    g.capture.setPose(poseName);          // positions everything, runs its own preroll
    const st = g.race.player.bike.state;
    return null;
  };
  // Simpler: for each pose, step in 10-step increments from the spawn and log air state.
  for (const pose of ['tabletop-air', 'ravine-gap']) {
    g.capture.setPose(pose);
    const st = g.race.player.bike.state;
    const trace = [];
    for (let k = 0; k < 40; k++) {
      for (let i = 0; i < 10; i++) g.capture.step(1/120);
      trace.push({ step: (k+1)*10, mode: st.mode, air: +st.airHeight.toFixed(2), kmh: +(st.speed*3.6).toFixed(0) });
    }
    out[pose] = trace.filter(r => r.mode === 'airborne' || r.step % 50 === 0);
  }
  return out;
}, null), null, 0));
await b.close();
