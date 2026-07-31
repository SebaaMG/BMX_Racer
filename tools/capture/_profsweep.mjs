/**
 * Throwaway probe: sweep the route-profile marker across the whole route and
 * report the closest approach between the player chevron's ink and the fixed
 * furniture below the baseline rule (the checkpoint digits, SUMMIT, FINISH).
 *
 * The critic tested stills; the marker moves, so a still cannot prove this.
 */
import { chromium } from 'playwright';

const b = await chromium.launch({ args: ['--use-angle=default', '--enable-gpu', '--ignore-gpu-blocklist'] });
const p = await b.newPage({ viewport: { width: 1600, height: 900 } });
p.on('pageerror', (e) => console.log('PAGEERR', e.message.slice(0, 300)));
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.capture, null, { timeout: 240000 });
await p.evaluate(() => window.__DESCENT__.game.capture.takeControl());
await p.evaluate(() => { window.__DESCENT__.game.capture.setPose('summit-wide'); for (let i = 0; i < 8; i++) window.__DESCENT__.game.capture.step(1 / 60); });

const r = await p.evaluate(async () => {
  const g = window.__DESCENT__.game;
  const w = g.hud.profile;
  const mod = await import('/src/game/WorldConstants.ts');
  const CP = mod.CHECKPOINT_TS;

  // Chevron geometry as drawn: chevron(px+2, py-2, 15, 5, dir) with ink 2.6.
  const SIZE = 15, THICK = 5, INK = 2.6;
  const pts = [
    [SIZE, 0], [-SIZE * 0.35, SIZE * 0.92], [-SIZE * 0.35 + THICK * 1.5, SIZE * 0.92],
    [SIZE - THICK * 1.7, 0], [-SIZE * 0.35 + THICK * 1.5, -SIZE * 0.92], [-SIZE * 0.35, -SIZE * 0.92],
  ];

  const out = { chevBottomMax: -1e9, chevAtT: 0, ruleY: w.ruleY, cpBaseY: w.cpBaseY, endBaseY: w.endBaseY,
                digitCapTop: 0, worstGap: 1e9, worstT: 0, worstWhich: '' };
  // digit cap top (PROFILE_CP_ST size 10, weight .20, ink null -> half stroke only)
  const digitTop = w.cpBaseY - 10 - 10 * 0.20 * 0.5;
  out.digitCapTop = digitTop;

  for (let t = 0; t <= 1.0001; t += 0.002) {
    const px = w.plotX + t * w.plotW;
    const py = w.yAt(t);
    const y2 = w.yAt(Math.min(1, t + 0.02));
    const dir = Math.atan2(y2 - py, w.plotW * 0.02);
    const c = Math.cos(dir), s = Math.sin(dir);
    let bot = -1e9, left = 1e9, right = -1e9;
    for (const q of pts) {
      const X = px + 2 + q[0] * c - q[1] * s;
      const Y = py - 2 + q[0] * s + q[1] * c;
      if (Y > bot) bot = Y;
      if (X < left) left = X;
      if (X > right) right = X;
    }
    bot += INK * 0.5; left -= INK * 0.5; right += INK * 0.5;
    if (bot > out.chevBottomMax) { out.chevBottomMax = bot; out.chevAtT = +t.toFixed(3); }

    // Against every checkpoint digit (10 wide-ish; advance .70*10 = 7, centred)
    for (let i = 1; i < CP.length - 1; i++) {
      const dx = w.plotX + CP[i] * w.plotW;
      const dl = dx - 4, dr = dx + 4;
      if (right < dl || left > dr) continue;
      const gap = digitTop - bot;
      if (gap < out.worstGap) { out.worstGap = gap; out.worstT = +t.toFixed(3); out.worstWhich = 'cp' + i; }
    }
  }
  return { out, cp: Array.from(CP).map((x) => +x.toFixed(3)),
           geo: { plotX: w.plotX, plotY: w.plotY, plotW: w.plotW, plotH: w.plotH } };
});
console.log(JSON.stringify(r, null, 1));
await b.close();
