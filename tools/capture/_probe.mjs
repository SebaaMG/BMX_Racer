import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('ERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.effects, null, { timeout: 240000 });
console.log(JSON.stringify(await p.evaluate(() => {
  const g = window.__DESCENT__.game;
  g.capture.takeControl(); g.capture.setPose('scree-speed');
  const st = g.race.player.bike.state, dust = g.effects.dust;
  let everGnd = 0, everTrail = 0, maxAlive = 0, cpZero = 0, samples = 0;
  const origTrail = dust.trail.bind(dust);
  dust.trail = (...a) => { everTrail++; return origTrail(...a); };
  const pool = dust.pool;
  for (let i = 0; i < 180; i++) {
    g.capture.step(1/60);
    samples++;
    if (st.rear.grounded) everGnd++;
    const cp = st.rear.contactPoint;
    if (st.rear.grounded && Math.abs(cp.x) < 1e-6 && Math.abs(cp.z) < 1e-6) cpZero++;
    const alive = pool?.used ?? 0;
    if (alive > maxAlive) maxAlive = alive;
  }
  return {
    samples, framesRearGrounded: everGnd, trailCallsMade: everTrail,
    contactPointAtOrigin: cpZero, maxAliveParticles: maxAlive,
    poolKeys: pool ? Object.keys(pool) : null,
    finalSpeed: +st.speed.toFixed(2),
    frontGrounded: st.front.grounded, rearGrounded: st.rear.grounded,
    pitch: +st.pitch.toFixed(3), lean: +st.lean.toFixed(3),
    rearContact: [+st.rear.contactPoint.x.toFixed(1), +st.rear.contactPoint.y.toFixed(1), +st.rear.contactPoint.z.toFixed(1)],
  };
}, null), null, 1));
await b.close();
