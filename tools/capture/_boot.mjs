import { chromium } from 'playwright';
const b = await chromium.launch();
const p = await (await b.newContext({ viewport: { width: 800, height: 450 } })).newPage();
p.on('console', (m) => { const t = m.text(); if (t.includes('[terrain]')) console.log(t); });
await p.goto('http://127.0.0.1:5173/?capture=1&pr=1', { waitUntil: 'load' });
await p.waitForFunction(() => window.__DESCENT__ && window.__DESCENT__.game, { timeout: 90000 });
await b.close();
