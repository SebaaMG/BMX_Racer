#!/usr/bin/env node
/**
 * Backticks inside a GLSL template literal terminate the template and break the
 * TypeScript parse with a cryptic "',' expected" hundreds of lines away. It has
 * cost this repo three debugging cycles. This makes it a build error instead.
 *
 * Walks from each `/* glsl *\/ \`` opener to its matching close, honouring
 * ${...} interpolation (which may legitimately contain nested templates), and
 * reports any bare backtick in the GLSL body.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const files = [];
(function walk(d) {
  for (const e of readdirSync(d)) {
    const p = join(d, e);
    if (statSync(p).isDirectory()) walk(p);
    else if (p.endsWith('.ts')) files.push(p);
  }
})('src');

let bad = 0;
for (const f of files) {
  const src = readFileSync(f, 'utf8');
  const re = /\/\* glsl \*\/\s*`/g;
  let m;
  while ((m = re.exec(src))) {
    let i = m.index + m[0].length;
    let depth = 0;
    while (i < src.length) {
      const c = src[i];
      if (c === '\\') { i += 2; continue; }
      if (c === '$' && src[i + 1] === '{') { depth++; i += 2; continue; }
      if (c === '}' && depth > 0) { depth--; i++; continue; }
      if (c === '`') {
        if (depth === 0) break;          // proper close
        i++; continue;                    // inside ${...}, fine
      }
      i++;
    }
    const body = src.slice(m.index, i);
    const line = src.slice(0, m.index).split('\n').length;
    // A backtick inside a // comment within the GLSL body is the failure mode.
    for (const [n, l] of body.split('\n').entries()) {
      const cmt = l.indexOf('//');
      if (cmt >= 0 && l.slice(cmt).includes('`')) {
        console.error(`${f}:${line + n}  backtick inside a GLSL comment`);
        bad++;
      }
    }
  }
}
if (bad) { console.error(`\n${bad} backtick(s) inside GLSL comments. These break the TS parse.`); process.exit(1); }
console.log('GLSL comments clean.');
