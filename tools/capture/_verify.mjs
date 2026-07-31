import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist','--enable-unsafe-swiftshader'] });
const p = await b.newPage({ viewport: { width: 900, height: 520 } });
p.on('pageerror', e => console.log('PAGEERR', e.message.slice(0,300)));
p.on('console', m => { if (m.type()==='error') console.log('[err]', m.text().slice(0,400)); });
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.effects, null, { timeout: 240000 });
await p.evaluate(() => window.__DESCENT__.game.capture.takeControl());

// Impact-frame timeline: how many consecutive frames carry a flash?
const timeline = await p.evaluate(() => {
  const g = window.__DESCENT__.game, fx = g.effects;
  g.capture.setSequence('landing');
  const rows = [];
  for (let i = 0; i < 90; i++) {
    g.capture.step(1/60);
    const u = g.post.composite.uniforms;
    rows.push({ f: i, flash: +u.uImpactFlash.value.toFixed(3), ink: +u.uInkFlood.value.toFixed(3),
                desat: +u.uDesaturate.value.toFixed(3), ts: +fx.timeScale.toFixed(2),
                dust: fx.dust.countAlive() });
  }
  const hot = rows.filter(r => r.flash > 0.01);
  return { flashFrames: hot.length, peakFlash: Math.max(...rows.map(r=>r.flash)),
           peakInk: Math.max(...rows.map(r=>r.ink)), peakDesat: Math.max(...rows.map(r=>r.desat)),
           frozenFrames: rows.filter(r=>r.ts===0).length,
           maxDust: Math.max(...rows.map(r=>r.dust)),
           window: hot.slice(0, 8) };
});
console.log('IMPACT', JSON.stringify(timeline));

async function shot(seq, n, tag) {
  const r = await p.evaluate(([s, k]) => {
    const g = window.__DESCENT__.game, fx = g.effects;
    g.capture.setSequence(s);
    let maxD = 0, maxDeb = 0;
    for (let i = 0; i < k; i++) {
      g.capture.step(1/60);
      maxD = Math.max(maxD, fx.dust.countAlive());
      maxDeb = Math.max(maxDeb, fx.debris.liveCount ?? 0);
    }
    const st = g.race.player.bike.state;
    return { dust: fx.dust.countAlive(), maxDust: maxD, maxDebris: maxDeb,
             surf: st.rear.surface?.kind, dustAmt: st.rear.surface?.dustAmount,
             kmh: +(st.speed*3.6).toFixed(1), mode: st.mode };
  }, [seq, n]);
  console.log(tag, JSON.stringify(r));
  await p.evaluate(() => new Promise(r => requestAnimationFrame(() => r())));
  await p.screenshot({ path: `captures/_v_${tag}.png` });
}
await shot('landing', 34, 'landing');
await shot('crash', 20, 'crash');
await p.evaluate(() => { const g = window.__DESCENT__.game; g.capture.setPose('streambed'); for (let i=0;i<40;i++) g.capture.step(1/60); });
const water = await p.evaluate(() => {
  const g = window.__DESCENT__.game, st = g.race.player.bike.state, fx = g.effects;
  return { frontSurf: st.front.surface?.kind, rearSurf: st.rear.surface?.kind,
           dustAmt: st.rear.surface?.dustAmount, dust: fx.dust.countAlive(),
           debris: fx.debris.liveCount, kmh: +(st.speed*3.6).toFixed(1) };
});
console.log('STREAM', JSON.stringify(water));
await p.evaluate(() => new Promise(r => requestAnimationFrame(() => r())));
await p.screenshot({ path: 'captures/_v_stream.png' });
await b.close();
