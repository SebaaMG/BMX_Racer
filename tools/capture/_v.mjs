import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('ERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
console.log(JSON.stringify(await p.evaluate(() => {
  const g = window.__DESCENT__.game; g.capture.takeControl();
  const out = [];
  // The trim is a module const, so scale the personalities' gains instead —
  // same closed-loop effect, and it is what a real fix would tune.
  const ai0 = g.race.racers.find(r => r !== g.race.player);
  const base = {};
  for (const r of g.race.racers) {
    if (r === g.race.player) continue;
    const p = r.personality ?? r._personality;
    if (p && !base[r.id]) base[r.id] = { kp: p.steerKp, kd: p.steerKd, lg: p.lateralGain };
  }
  for (const k of [1.0, 0.7, 0.5, 0.35, 0.25]) {
    for (const r of g.race.racers) {
      if (r === g.race.player) continue;
      const p = r.personality ?? r._personality;
      const b0 = base[r.id];
      if (p && b0) { p.steerKp = b0.kp * k; p.steerKd = b0.kd * k; p.lateralGain = b0.lg * k; }
    }
    g.capture.setSequence('pack-race');
    let sat = 0, n = 0, maxLat = 0;
    const d0 = ai0.progress.distance;
    for (let f = 0; f < 180; f++) {
      g.capture.step(1/60);
      n++; if (Math.abs(ai0.input.steer) > 0.97) sat++;
      maxLat = Math.max(maxLat, Math.abs(ai0.trackLateral ?? 0));
    }
    out.push({ trim: k, satPct: +(100*sat/n).toFixed(0), maxLat: +maxLat.toFixed(1),
      moved: +(ai0.progress.distance - d0).toFixed(0), endKmh: +(ai0.bike.state.speed*3.6).toFixed(0) });
  }
  return out;
}, null), null, 0));
await b.close();
