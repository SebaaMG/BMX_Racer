import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('ERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
console.log(JSON.stringify(await p.evaluate(() => {
  const g = window.__DESCENT__.game, tr = g.track;
  g.capture.takeControl();
  // Walk the whole course and record what the preview says at each point.
  const rows = [];
  let nulls = 0, tooClose = 0, n = 0;
  for (let d = 0; d < tr.length - 20; d += 25) {
    const cp = g.race.updateCornerPreview ? null : null;
    // exercise through the public model instead
    g.race.player.progress.distance = d;
    const m = g.race.getHudModel();
    const c = m.cornerPreview;
    n++;
    if (!c) { nulls++; continue; }
    if (c.distance <= 10) tooClose++;
    if (rows.length < 8) rows.push({ d, dist: c.distance, k: +c.curvature.toFixed(2) });
  }
  return { samples: n, nullFrac: +(nulls/n).toFixed(2), atMinRangeFrac: +(tooClose/n).toFixed(2), sample: rows };
}, null), null, 0));
await b.close();
