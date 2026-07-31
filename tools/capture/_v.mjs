import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('ERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
console.log(await p.evaluate(() => {
  const g = window.__DESCENT__.game; g.capture.takeControl();
  const rows = [];
  for (const pose of ['summit-wide','scree-speed','switchback-lean','ravine-gap','finish-sprint']) {
    g.capture.setPose(pose);
    for (let i=0;i<12;i++) g.capture.step(1/60);
    const m = g.race.getHudModel();
    rows.push(`${pose.padEnd(20)} elapsed ${g.race.elapsed.toFixed(1).padStart(6)}s   progress ${(g.race.player.progress.distance/g.track.length*100).toFixed(0).padStart(3)}%`);
  }
  return rows.join('\n');
}));
await b.close();
