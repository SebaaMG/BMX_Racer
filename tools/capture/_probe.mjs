import { chromium } from 'playwright';
import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('ERR', e.message.slice(0,200)));
await p.setViewportSize({ width: 1600, height: 900 });
await p.goto('http://127.0.0.1:5173/?capture=1&pr=2', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
await p.evaluate(() => { const g = window.__DESCENT__.game; g.capture.takeControl(); g.capture.setPose('valley-vista'); });
await p.evaluate(() => { for (let i=0;i<14;i++) window.__DESCENT__.game.capture.step(1/60); });
const buf = await p.screenshot();
await b.close();
const png = PNG.sync.read(buf);
// Single-pixel island fraction over a snow patch (the critic's region).
const x0=2400,x1=2900,y0=700,y1=1100; let islands=0,n=0;
const at=(x,y)=>{const i=(y*png.width+x)*4;return png.data[i]*0.299+png.data[i+1]*0.587+png.data[i+2]*0.114;};
for(let y=y0+1;y<y1-1;y++)for(let x=x0+1;x<x1-1;x++){
  const c=at(x,y); const nb=[at(x-1,y),at(x+1,y),at(x,y-1),at(x,y+1)];
  const far=nb.filter(v=>Math.abs(v-c)>10).length; n++; if(far>=3)islands++;
}
console.log(`single-pixel islands: ${(100*islands/n).toFixed(1)}%  (was 43.1%)`);
