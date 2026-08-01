import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('PAGEERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
await p.evaluate(() => window.__DESCENT__.game.race.forceRacing?.());
await p.evaluate(() => new Promise(r => setTimeout(r, 100000)));
const r = await p.evaluate(() => {
  const g = window.__DESCENT__.game, t = g.track, terr = g.terrain;
  const FWDC = g.race.player.bike.state.position.constructor;
  return g.race.racers.filter(x=>x.id!=='player').map(r => {
    const st = r.bike.state;
    const d = r.progress?.distance ?? 0;
    const s = t.sampleAtDistance(d);
    const fwd = new FWDC(0,0,1).applyQuaternion(st.orientation);
    const flat = new FWDC(fwd.x,0,fwd.z).normalize();
    const tan = new FWDC(s.tangent.x,0,s.tangent.z).normalize();
    // what is in front of the bike?
    const ahead = [];
    for (const m of [1,2,4,8]) {
      const px = st.position.x + flat.x*m, pz = st.position.z + flat.z*m;
      ahead.push(+(terr.heightAt(px,pz) - st.position.y).toFixed(2));
    }
    return { id: r.id, d: +d.toFixed(0), kmh:+(st.speed*3.6).toFixed(1),
      lateral: +(r.trackLateral ?? 0).toFixed(2), halfWidth: +s.halfWidth.toFixed(2),
      headErr: +(Math.acos(Math.max(-1,Math.min(1,flat.dot(tan))))*180/Math.PI).toFixed(0),
      steerCmd: +(r.input?.steer ?? 0).toFixed(2), pedal: +(r.input?.pedal ?? 0).toFixed(2),
      mode: st.mode, lean:+(st.lean*180/Math.PI).toFixed(0),
      groundAhead_1_2_4_8m: ahead,
      sink: +(terr.heightAt(st.position.x,st.position.z) - st.position.y).toFixed(2),
      curvature: +(s.curvature ?? 0).toFixed(4) };
  });
});
for (const x of r) console.log(JSON.stringify(x));
await b.close();
