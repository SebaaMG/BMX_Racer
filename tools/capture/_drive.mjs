import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('PAGEERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
await p.evaluate(() => window.__DESCENT__.game.race.forceRacing?.());
// Hold W, as a player would.
await p.evaluate(() => {
  const fire = (t) => window.dispatchEvent(new KeyboardEvent(t, { code: 'KeyW', key: 'w', bubbles: true }));
  fire('keydown');
  window.__H = setInterval(() => fire('keydown'), 200);
  window.__S = [];
  const g = window.__DESCENT__.game;
  window.__I = setInterval(() => {
    const st = g.race.player.bike.state;
    const s = g.track.sampleAtDistance(g.race.player.progress.distance);
    const gh = g.terrain.heightAt(st.rear.contactPoint.x, st.rear.contactPoint.z);
    window.__S.push({ d:+g.race.player.progress.distance.toFixed(0), v:+(st.speed*3.6).toFixed(0),
      ped:+g.race.player.bike.state.mode, slope:+(Math.asin(-s.tangent.y)*180/Math.PI).toFixed(0),
      rearSink:+(gh - st.rear.contactPoint.y).toFixed(3),
      bodyAbove:+(st.position.y - gh).toFixed(3), grounded: st.rear.grounded });
  }, 400);
});
await p.evaluate(() => new Promise(r => setTimeout(r, 45000)));
const S = await p.evaluate(() => { clearInterval(window.__I); clearInterval(window.__H); return window.__S; });
console.log('PLAYER holding W from the line:');
console.log(' t(s)   d(m)  km/h  slope  rearContactBelowGround  bodyAboveGround');
S.filter((_,i)=>i%3===0).forEach((x,i)=>console.log(
  String((i*1.2).toFixed(1)).padStart(5), String(x.d).padStart(6), String(x.v).padStart(5),
  String(x.slope).padStart(6), String(x.rearSink).padStart(22), String(x.bodyAbove).padStart(16)));
const mx = S.reduce((a,x)=>x.v>a.v?x:a);
console.log('\ntop speed reached:', mx.v, 'km/h at', mx.d, 'm');
console.log('distance covered in 45 s:', S[S.length-1].d, 'm');
const sink = S.filter(x=>x.grounded).map(x=>x.rearSink);
if (sink.length) console.log('rear contact vs ground while grounded: mean',
  (sink.reduce((a,x)=>a+x,0)/sink.length).toFixed(3), 'max', Math.max(...sink).toFixed(3));
await b.close();
