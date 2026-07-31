import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, args: ['--use-angle=default','--enable-gpu','--ignore-gpu-blocklist'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('ERR', e.message.slice(0,200)));
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'domcontentloaded' });
await p.waitForFunction(() => !!window.__DESCENT__?.game?.race, null, { timeout: 300000 });
console.log(JSON.stringify(await p.evaluate(() => {
  const g = window.__DESCENT__.game;
  g.capture.takeControl();
  const V = window.__THREE_V ?? null;
  const findBone = (root, re) => { let f = null; root.traverse(o => { if (!f && re.test(o.name||'')) f = o; }); return f; };

  // ---- (A) Is the rider pose actually moving frame to frame? ----
  g.capture.setPose('switchback-lean');
  for (let i = 0; i < 20; i++) g.capture.step(1/60);
  const rig = g.race.player.rig.object;
  const head = findBone(rig, /head/i);
  const hand = findBone(rig, /hand.*L|L.*hand/i);
  const samples = [];
  let prev = null, moved = 0, still = 0, maxMove = 0;
  for (let f = 0; f < 90; f++) {
    g.capture.step(1/60);
    if (!head) break;
    head.updateWorldMatrix(true, false);
    const m = head.matrixWorld.elements;
    const cur = [m[12], m[13], m[14]];
    if (prev) {
      const d = Math.hypot(cur[0]-prev[0], cur[1]-prev[1], cur[2]-prev[2]);
      if (d < 1e-4) still++; else moved++;
      maxMove = Math.max(maxMove, d);
      samples.push(+(d*1000).toFixed(2));
    }
    prev = cur;
  }
  const headMotion = { boneFound: !!head, movedFrames: moved, stillFrames: still,
                       maxMoveMM: +(maxMove*1000).toFixed(2),
                       medianMM: samples.length ? samples.slice().sort((a,b)=>a-b)[samples.length>>1] : null };

  // ---- (B) Does the crash decelerate monotonically? ----
  g.capture.setPose('crash');
  const st = g.race.player.bike.state;
  const trace = []; let increases = 0; let prevSpd = null;
  for (let f = 0; f < 150; f++) {
    g.capture.step(1/60);
    const kmh = st.speed * 3.6;
    if (prevSpd !== null && kmh > prevSpd + 0.05 && (st.mode === 'crashing' || st.mode === 'recovering')) increases++;
    prevSpd = kmh;
    if (f % 15 === 0) trace.push({ f, kmh: +kmh.toFixed(1), mode: st.mode });
  }
  return { headMotion, crash: { trace, increasesWhileDown: increases } };
}, null), null, 1));
await b.close();
