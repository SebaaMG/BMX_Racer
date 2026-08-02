#!/usr/bin/env node

import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).map((arg, i, all) => {
    if (!arg.startsWith('--')) return [arg, true];
    const key = arg.slice(2);
    const next = all[i + 1];
    return [key, next && !next.startsWith('--') ? next : true];
  }),
);

const base = String(args.url ?? 'http://127.0.0.1:5173');
const outdir = path.resolve(String(args.outdir ?? 'captures/rally-review/orthographic'));
const width = Number(args.width ?? 1200);
const height = Number(args.height ?? 900);
const pr = Number(args.pr ?? 1);
const views = ['front', 'front-threequarter', 'side', 'rear-threequarter', 'rear'];

await mkdir(outdir, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  args: [
    '--enable-unsafe-swiftshader',
    '--enable-webgl',
    '--ignore-gpu-blocklist',
    '--force-color-profile=srgb',
  ],
});
const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: pr });

try {
  for (const view of views) {
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });

    await page.goto(
      `${base}/tools/rally/model-review.html?view=${encodeURIComponent(view)}&pr=${pr}`,
      { waitUntil: 'domcontentloaded', timeout: 60_000 },
    );
    await page.waitForFunction(() => window.__RALLY_MODEL_REVIEW__?.ready === true, null, {
      timeout: 30_000,
    });
    await page.evaluate(() => window.__RALLY_MODEL_REVIEW__.render());
    await page.screenshot({ path: path.join(outdir, `${view}.png`) });
    if (errors.length) throw new Error(`${view}: ${errors.join(' | ')}`);
    await page.close();
  }
} finally {
  await context.close();
  await browser.close();
}

console.log(`rally model review → ${outdir}`);
