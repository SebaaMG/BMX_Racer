import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist','--enable-unsafe-swiftshader'] });
const p = await b.newPage({ viewport: { width: 900, height: 520 } });
p.on('pageerror', e => console.log('PAGEERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/?capture=1&pr=2', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.effects, null, { timeout: 240000 });
const r = await p.evaluate(() => {
  const g = window.__DESCENT__.game;
  g.capture.takeControl();
  g.capture.setPose('bike-detail');          // side-on, dist 2.3
  const st = g.race.player.bike.state;
  // Pin a real racing speed so the wheels are genuinely spinning.
  for (let i = 0; i < 26; i++) {
    const v = 19.5, sp = st.velocity.length();
    if (sp > 0.01) st.velocity.multiplyScalar(v / sp);
    st.speed = v;
    if (st.front) st.front.spinRate = v / 0.33;
    if (st.rear) st.rear.spinRate = v / 0.33;
    g.capture.step(1/60);
  }
  const sp = g.effects.speed;
  return { kmh: +(st.speed*3.6).toFixed(1), front: +(st.front.spinRate??0).toFixed(1), rear: +(st.rear.spinRate??0).toFixed(1),
           spin01: sp.spins.map(s => +s.material.uniforms.uSpin.value.toFixed(2)),
           vis: sp.spins.map(s => s.mesh.visible) };
});
console.log(JSON.stringify(r));
await p.evaluate(() => new Promise(r => requestAnimationFrame(() => r())));
await p.screenshot({ path: 'captures/_v_wheel.png' });
await b.close();
