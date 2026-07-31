import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('ERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
console.log(await p.evaluate(() => {
  const g = window.__DESCENT__.game;
  g.capture.takeControl();
  const out = {};
  for (const seq of ['tabletop-air','landing','trick-360','pack-race','scree-speed']) {
    g.capture.setSequence(seq);
    const st = g.race.player.bike.state;
    let air = 0, firstAir = -1, maxYaw = 0, oppInFront = 0;
    const cam = g.engine.camera;
    for (let f = 0; f < 144; f++) {
      g.capture.step(1/60);
      if (st.mode === 'airborne') { air++; if (firstAir < 0) firstAir = f; }
      maxYaw = Math.max(maxYaw, Math.abs(st.angularVelocity.y));
      // Is any rival in front of the camera and within 40 m?
      for (const r of g.race.racers) {
        if (r === g.race.player) continue;
        const d = cam.position.distanceTo(r.bike.state.position);
        if (d < 40) { oppInFront++; break; }
      }
    }
    out[seq] = { airFrames: air, firstAirFrame: firstAir, maxYawRate: +maxYaw.toFixed(2), framesWithRivalNear: oppInFront };
  }
  return JSON.stringify(out, null, 1);
}));
await b.close();
