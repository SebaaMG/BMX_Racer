/**
 * TrackRibbon — the dirt the race is actually run on.
 *
 * The ribbon is a swept surface, not a decal projected onto the terrain. It has
 * its own geometry, its own edges, and its own material, for three reasons:
 *
 *  1. THE EDGE. A projected trail texture has a soft, mathematically perfect
 *     boundary. A real trail edge is a break — worked ground stops, hillside
 *     starts, and the transition is a crease you could catch a pedal on. Here
 *     the edge is a genuine geometric fold with a buried skirt behind it, so
 *     the G-buffer sees a normal discontinuity and the Sobel pass inks it. The
 *     material ID changes across the same line, which inks it a second time.
 *     Both are irregular in plan, because a trail cut by feet and tyres wanders
 *     and a constant offset from a spline does not.
 *
 *  2. BERMS. The switchbacks ask for up to half a radian of bank. You cannot
 *     bank a decal. The outer edge of each bermed corner is built up into a
 *     real wall with a real outer flank, which is what the rider rails against
 *     and what the camera sees in silhouette on the exit.
 *
 *  3. THE LIPS. The tabletop and the ravine need a takeoff the rider can READ
 *     from thirty metres out. That means a hard silhouette (a genuine vertical
 *     end cap at the ravine, a kicker crest at the tabletop) and a distinct
 *     value (a bleached, packed strip driven by the spline's lip mask).
 *
 * The whole ribbon is one CelMaterial on RAMPS.trail, split into chunk meshes
 * purely so frustum culling has something to work with. There is no inverted
 * hull: a hull on a ground plane produces a black halo where the ribbon meets
 * the terrain, and the edge we want is already coming from the geometry break.
 */

import {
  BufferAttribute,
  BufferGeometry,
  Group,
  Mesh,
  Vector3,
} from 'three';

import { ITerrain, TrackSectionKind } from '../game/Contracts';
import { CelMaterial, disposeCelMaterial, registerNprMesh } from '../npr/CelMaterial';
import { finalizeGeometry } from '../npr/OutlineGeometry';
import { RAMPS } from '../npr/Palette';
import { trailSurface } from '../npr/GeneratedTextures';
import { Noise2D } from '../core/Noise';
import { clamp, clamp01, lerp, smoothstep } from '../core/MathX';
import { SECTION_ORDER, TrackSpline } from './TrackSpline';

// ─────────────────────────────────────────────────────────────────────────────
// Cross-section
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Nominal columns across the ribbon, in half-width units. +u is LEFT.
 *
 * The pair at +-1.00 is the visible trail edge; the pair at +-1.09 is the skirt
 * that dives under the terrain to seal the seam. The 0.09 gap between them is
 * what makes the edge a near-vertical face rather than a ramp — about 50 degrees
 * on a 3m half-width, which is steep enough for the prepass normals to break
 * hard and the Sobel pass to draw a line.
 */
const COL_U = [-1.09, -1.0, -0.93, -0.66, -0.33, 0.0, 0.33, 0.66, 0.93, 1.0, 1.09];
const NC = COL_U.length;
const SKIRT_COLS = [0, NC - 1];

/** How far the skirt drops below whichever is lower: ribbon edge or terrain. */
const SKIRT_DROP = 0.42;
/** Depth of the vertical end cap at a jump lip. Pure silhouette. */
const CAP_DEPTH = 1.25;
/** Metres of track per V unit of the trail texture. */
const UV_ALONG = 24;
/** Target chunk length. Small enough to cull, large enough not to thrash. */
const CHUNK_LENGTH = 150;

/**
 * Per-section albedo tint. Every value is within 16% of white — this is not
 * recolouring the trail, it is the difference between packed warm dirt and
 * broken cold stone, which is the amount a background painter would shift it.
 */
const SECTION_TINT: Record<TrackSectionKind, [number, number, number]> = {
  [TrackSectionKind.TechnicalStart]: [0.95, 0.95, 1.01],
  [TrackSectionKind.ScreeRun]: [1.02, 1.01, 1.0],
  [TrackSectionKind.Switchbacks]: [1.01, 0.99, 0.95],
  [TrackSectionKind.RockGarden]: [0.9, 0.92, 1.0],
  [TrackSectionKind.Tabletop]: [1.04, 1.02, 0.96],
  [TrackSectionKind.RavineGap]: [1.02, 1.0, 0.96],
  [TrackSectionKind.RidgeSprint]: [0.97, 0.97, 1.01],
  [TrackSectionKind.StreamBed]: [0.84, 0.89, 0.99],
  [TrackSectionKind.FinalSprint]: [1.03, 1.0, 0.95],
};

interface Row {
  d: number;
  pos: Float32Array; // NC * 3
  nrm: Float32Array; // NC * 3
  uv: Float32Array; // NC * 2
  col: Float32Array; // NC * 3
  ao: Float32Array; // NC
  /** Cap rows take a fixed normal and are excluded from surface smoothing. */
  fixedNormal: boolean;
}

interface Link {
  a: Row;
  b: Row;
}

const _c = new Vector3();
const _sl = new Vector3();
const _up = new Vector3();
const _tan = new Vector3();
const _along = new Vector3();
const _across = new Vector3();
const _n = new Vector3();

export class TrackRibbon {
  readonly object = new Group();
  readonly material: CelMaterial;
  private meshes: Mesh[] = [];

  constructor(private spline: TrackSpline, terrain: ITerrain) {
    this.object.name = 'track-ribbon';

    const rows: Row[] = [];
    const links: Link[] = [];
    this.buildRows(terrain, rows, links);

    this.material = buildRibbonMaterial(spline);
    this.emitChunks(links);
  }

  // ───────────────────────────────────────────────────────────────────────────

  private buildRows(terrain: ITerrain, rows: Row[], links: Link[]): void {
    const spline = this.spline;
    const edgeNoise = new Noise2D('trail-edge');
    const pinchNoise = new Noise2D('trail-pinch');

    // Walk the centreline collecting runs of contiguous ground. A run ends at
    // the ravine and a new one starts on the far side; the two get end caps,
    // which is what gives the gap a silhouette instead of a torn hole.
    const runs: Row[][] = [];
    let current: Row[] = [];
    let d = 0;
    for (;;) {
      const dd = Math.min(d, spline.length);
      if (spline.isGap(dd)) {
        if (current.length) {
          runs.push(current);
          current = [];
        }
        d += 1.5;
      } else {
        current.push(this.buildRow(dd, terrain, edgeNoise, pinchNoise));
        d += this.stepAt(dd);
      }
      if (dd >= spline.length) break;
    }
    if (current.length) runs.push(current);

    for (const run of runs) {
      computeRunNormals(run);
      rows.push(...run);
      for (let i = 0; i < run.length - 1; i++) links.push({ a: run[i], b: run[i + 1] });
    }

    // End caps. The takeoff cap faces forward across the gap so it is the
    // silhouette the rider sees on approach; the landing cap faces back so the
    // far lip reads as a wall to clear rather than as a slot to fall into.
    for (let r = 0; r < runs.length; r++) {
      const run = runs[r];
      if (r < runs.length - 1) {
        const top = duplicateRow(run[run.length - 1], true);
        const bot = dropRow(top, CAP_DEPTH);
        this.faceRow(top, run[run.length - 1].d, 1);
        this.faceRow(bot, run[run.length - 1].d, 1);
        rows.push(top, bot);
        links.push({ a: top, b: bot });
      }
      if (r > 0) {
        const top = duplicateRow(run[0], true);
        const bot = dropRow(top, CAP_DEPTH);
        this.faceRow(top, run[0].d, -1);
        this.faceRow(bot, run[0].d, -1);
        rows.push(top, bot);
        links.push({ a: bot, b: top });
      }
    }
  }

  /** Row spacing: dense where the shape matters, sparse where it does not. */
  private stepAt(d: number): number {
    const s = this.spline;
    const i = clamp(Math.round(d / s.spacing), 0, s.count - 1);
    let step = 1.7;
    const k = Math.abs(s.curvature[i]);
    if (k > 0.008) step = Math.min(step, lerp(1.25, 0.55, clamp01((k - 0.008) / 0.05)));
    if (s.bermStrength[i] > 0.05) step = Math.min(step, 0.85);
    if (s.lipMask[i] > 0.02) step = Math.min(step, 0.4);
    return step;
  }

  private buildRow(d: number, terrain: ITerrain, edgeNoise: Noise2D, pinchNoise: Noise2D): Row {
    const s = this.spline;
    const i = clamp(Math.round(d / s.spacing), 0, s.count - 1);
    const sample = s.sampleAtDistance(d);
    _c.copy(sample.position);
    _up.copy(sample.up);
    s.surfaceLeftAt(d, _sl);

    const hw = sample.halfWidth;
    const berm = s.bermStrength[i];
    const bermSide = s.bermSide[i];
    const bermHeight = berm * hw * 0.34;

    // ── Plan-view edge irregularity ──────────────────────────────────────────
    // Two scales per side plus an occasional pinch. Biased outward so the
    // drawn ribbon is never NARROWER than the half-width the physics and the
    // AI believe in — the irregularity is allowed to give the rider more room,
    // never less.
    const eL = edgeIrregularity(edgeNoise, pinchNoise, d, 0);
    const eR = edgeIrregularity(edgeNoise, pinchNoise, d, 71.3);

    const secTint = SECTION_TINT[SECTION_ORDER[s.sectionId[i]]] ?? [1, 1, 1];
    const lip = s.lipMask[i];

    const pos = new Float32Array(NC * 3);
    const nrm = new Float32Array(NC * 3);
    const uv = new Float32Array(NC * 2);
    const col = new Float32Array(NC * 3);
    const ao = new Float32Array(NC);

    for (let c = 0; c < NC; c++) {
      const u = COL_U[c];
      const au = Math.abs(u);
      const side = Math.sign(u);
      const edge = u > 0 ? eL : eR;
      const isSkirt = c === SKIRT_COLS[0] || c === SKIRT_COLS[1];

      // Blend the edge wander in over the outer third so the centre of the
      // trail stays where the spline says it is.
      let uEff = u * lerp(1, edge, smoothstep(0.3, 0.93, au));
      // A berm needs its outer skirt pushed well clear so the flank is a slope
      // rather than a cliff.
      if (isSkirt && side === bermSide && berm > 0) uEff = side * (au + berm * 0.75);

      const lateral = uEff * hw;
      let x = _c.x + _sl.x * lateral;
      let y = _c.y + _sl.y * lateral;
      let z = _c.z + _sl.z * lateral;

      // Berm rise on the outer side only. Quartic-ish so the transition out of
      // the flat is smooth and the top of the wall is steep.
      if (berm > 0 && side === bermSide && !isSkirt) {
        y += bermHeight * Math.pow(clamp01((au - 0.35) / 0.68), 1.85);
      }
      // A whisper of crown on the flat sections. Not drainage realism — it
      // keeps the ribbon from reading as a perfect plane in a wide shot.
      if (!isSkirt) y -= 0.028 * hw * au * au * (1 - berm);
      // The lip crest is deliberately flat across its full width.
      if (lip > 0.01 && !isSkirt) y += 0;

      if (isSkirt) {
        const tY = terrain.heightAt(x, z);
        const ground = Number.isFinite(tY) ? Math.min(y, tY) : y;
        y = ground - SKIRT_DROP;
      }

      pos[c * 3] = x;
      pos[c * 3 + 1] = y;
      pos[c * 3 + 2] = z;

      uv[c * 2] = (u + 1) * 0.5;
      uv[c * 2 + 1] = d / UV_ALONG;

      col[c * 3] = secTint[0];
      col[c * 3 + 1] = secTint[1];
      col[c * 3 + 2] = secTint[2];

      // Vertex AO is doing double duty as the ink: celShade drives occluded
      // vertices toward the darkest band colour rather than toward black, so a
      // dark crest reads as the shadowed cut edge of the trail, in palette.
      let a = 1 - 0.14 * smoothstep(0.35, 0.93, au);
      if (c === 2 || c === NC - 3) a = 0.52;
      if (c === 1 || c === NC - 2) a = 0.28;
      if (isSkirt) a = 0.08;
      // The built-up outer face of a berm catches light; it must not be inked
      // like a cut edge or the corner loses its shape.
      if (berm > 0 && side === bermSide) a = lerp(a, Math.min(1, a + 0.5), berm);
      ao[c] = a;
    }

    return { d, pos, nrm, uv, col, ao, fixedNormal: false };
  }

  /** Force a whole row's normals to +-tangent — used for the jump end caps. */
  private faceRow(row: Row, d: number, sign: number): void {
    const s = this.spline;
    const i = clamp(Math.round(d / s.spacing), 0, s.count - 1);
    _tan.set(s.tx[i], 0, s.tz[i]).normalize().multiplyScalar(sign);
    for (let c = 0; c < NC; c++) {
      row.nrm[c * 3] = _tan.x;
      row.nrm[c * 3 + 1] = _tan.y;
      row.nrm[c * 3 + 2] = _tan.z;
      // The cap face is a cut bank: dark, and outside the rideable surface.
      row.ao[c] = 0.22;
    }
    row.fixedNormal = true;
  }

  // ───────────────────────────────────────────────────────────────────────────

  private emitChunks(links: Link[]): void {
    let start = 0;
    while (start < links.length) {
      let end = start;
      const d0 = links[start].a.d;
      while (end < links.length && Math.abs(links[end].b.d - d0) < CHUNK_LENGTH) end++;
      if (end === start) end = start + 1;
      this.emitChunk(links.slice(start, end), this.meshes.length);
      // Chunks share their boundary link's rows by simply not overlapping —
      // the boundary row is duplicated into both chunks with identical values,
      // so the seam is invisible and no triangle is emitted twice.
      start = end;
    }
  }

  private emitChunk(links: Link[], index: number): void {
    const rowIndex = new Map<Row, number>();
    const order: Row[] = [];
    for (const l of links) {
      if (!rowIndex.has(l.a)) {
        rowIndex.set(l.a, order.length);
        order.push(l.a);
      }
      if (!rowIndex.has(l.b)) {
        rowIndex.set(l.b, order.length);
        order.push(l.b);
      }
    }

    const vcount = order.length * NC;
    const position = new Float32Array(vcount * 3);
    const normal = new Float32Array(vcount * 3);
    const uv = new Float32Array(vcount * 2);
    const color = new Float32Array(vcount * 3);
    const aAo = new Float32Array(vcount);

    for (let r = 0; r < order.length; r++) {
      const row = order[r];
      position.set(row.pos, r * NC * 3);
      normal.set(row.nrm, r * NC * 3);
      uv.set(row.uv, r * NC * 2);
      color.set(row.col, r * NC * 3);
      aAo.set(row.ao, r * NC);
    }

    const indices: number[] = [];
    for (const l of links) {
      const ra = rowIndex.get(l.a)! * NC;
      const rb = rowIndex.get(l.b)! * NC;
      for (let c = 0; c < NC - 1; c++) {
        // Winding chosen so cross(b - a, across) is the face normal; see the
        // note in buildRows about which way each cap has to face.
        indices.push(ra + c, rb + c, ra + c + 1);
        indices.push(ra + c + 1, rb + c, rb + c + 1);
      }
    }

    const geo = new BufferGeometry();
    geo.setAttribute('position', new BufferAttribute(position, 3));
    geo.setAttribute('normal', new BufferAttribute(normal, 3));
    geo.setAttribute('uv', new BufferAttribute(uv, 2));
    geo.setAttribute('color', new BufferAttribute(color, 3));
    geo.setAttribute('aAo', new BufferAttribute(aAo, 1));
    geo.setIndex(indices);
    // No hull on the ribbon, but the prepass still reads aCurvature to taper
    // interior lines, so the geometry goes through finalizeGeometry anyway.
    finalizeGeometry(geo, { tolerance: 1e-3, maxWeldAngle: 100, curvatureGain: 0.8 });

    const mesh = new Mesh(geo, this.material);
    mesh.name = `ribbon:${index}`;
    mesh.receiveShadow = true;
    // Berms and lips are the two places the ribbon casts a shadow that means
    // something. The shadow material culls front faces, which keeps the huge
    // flat expanse from acneing onto itself.
    mesh.castShadow = true;
    mesh.frustumCulled = true;
    registerNprMesh(mesh, this.material);
    this.meshes.push(mesh);
    this.object.add(mesh);
  }

  dispose(): void {
    for (const m of this.meshes) m.geometry.dispose();
    this.meshes.length = 0;
    disposeCelMaterial(this.material);
    this.object.clear();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Edge scale for one side of the ribbon at a distance. Returns a multiplier
 * >= 1: two octaves of wander plus a rare outward scallop where the trail has
 * been widened by a braid around a rock.
 */
function edgeIrregularity(n: Noise2D, pinch: Noise2D, d: number, phase: number): number {
  const slow = n.noise(d * 0.055 + phase, 11.3);
  const fast = n.noise(d * 0.24 + phase, 4.7);
  const scallop = Math.max(0, pinch.noise(d * 0.017 + phase, 21.1) - 0.42) * 0.62;
  return 1.07 + slow * 0.07 + fast * 0.028 + scallop;
}

function duplicateRow(src: Row, fixedNormal: boolean): Row {
  return {
    d: src.d,
    pos: src.pos.slice(),
    nrm: src.nrm.slice(),
    uv: src.uv.slice(),
    col: src.col.slice(),
    ao: src.ao.slice(),
    fixedNormal,
  };
}

function dropRow(src: Row, depth: number): Row {
  const r = duplicateRow(src, src.fixedNormal);
  for (let c = 0; c < NC; c++) r.pos[c * 3 + 1] -= depth;
  return r;
}

function computeRunNormals(run: Row[]): void {
  for (let r = 0; r < run.length; r++) {
    const row = run[r];
    if (row.fixedNormal) continue;
    const prev = run[Math.max(0, r - 1)];
    const next = run[Math.min(run.length - 1, r + 1)];
    for (let c = 0; c < NC; c++) {
      const c0 = Math.max(0, c - 1);
      const c1 = Math.min(NC - 1, c + 1);
      _along.set(
        next.pos[c * 3] - prev.pos[c * 3],
        next.pos[c * 3 + 1] - prev.pos[c * 3 + 1],
        next.pos[c * 3 + 2] - prev.pos[c * 3 + 2],
      );
      _across.set(
        row.pos[c1 * 3] - row.pos[c0 * 3],
        row.pos[c1 * 3 + 1] - row.pos[c0 * 3 + 1],
        row.pos[c1 * 3 + 2] - row.pos[c0 * 3 + 2],
      );
      _n.crossVectors(_along, _across);
      if (_n.lengthSq() < 1e-12) _n.set(0, 1, 0);
      else _n.normalize();
      if (_n.y < 0) _n.negate();
      row.nrm[c * 3] = _n.x;
      row.nrm[c * 3 + 1] = _n.y;
      row.nrm[c * 3 + 2] = _n.z;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Material
// ─────────────────────────────────────────────────────────────────────────────

function buildRibbonMaterial(spline: TrackSpline): CelMaterial {
  const mat = new CelMaterial(RAMPS.trail, {
    name: 'track-ribbon',
    idName: 'trail',
    vertexColors: true,
    vertexAo: true,
    outlineWidth: 0,
    uniforms: {
      uTrailTex: { value: trailSurface() },
      uTrackProfile: { value: spline.profileTexture() },
      uTrackLength: { value: spline.length },
      uUvAlong: { value: UV_ALONG },
    },
    fragmentPreamble: /* glsl */ `
      uniform sampler2D uTrailTex;
      uniform sampler2D uTrackProfile;
      uniform float uTrackLength;
      uniform float uUvAlong;
    `,
    fragmentBody: /* glsl */ `
      // ── Trail surface detail ─────────────────────────────────────────────
      // Everything below is quantised. Nothing here is allowed to introduce a
      // smooth gradient: this is the largest surface in the frame, and one
      // continuous ramp across it would give the whole cel treatment away.
      float dAlong = vUv.y * uUvAlong;
      float across = vUv.x * 2.0 - 1.0;

      // The track profile lookup. One texel per 0.9m of course, carrying what
      // KIND of ground this is: curvature, wetness, rockiness, lip.
      vec4 prof = texture(uTrackProfile, vec2(clamp(dAlong / uTrackLength, 0.0, 1.0), 0.5));
      float curv = (prof.r - 0.5) * 2.0;
      float wet  = prof.g;
      float rock = prof.b;
      float lipM = prof.a;

      vec4 tr = texture(uTrailTex, vUv);
      float wear = tr.r;
      float gravel = tr.g;
      float moisture = tr.b;

      // Two braided tyre lines. They polish where the corner is, because
      // everyone brakes and rails in the same place; a uniform pair of lines
      // down the whole mountain reads as wallpaper.
      float polish = mix(0.30, 1.0, min(abs(curv) * 1.7, 1.0));
      float line = bandStep(wear * polish, 0.26, 0.015);
      celCol = mix(celCol, celCol * 1.15 + vec3(0.022, 0.016, 0.010), line * 0.55);

      // Loose stone. The rock garden is not a texture swap: it is the same
      // ground with the speckle turned up until it stops reading as dirt.
      float speck = 1.0 - bandStep(gravel, 0.21 + rock * 0.10, 0.02);
      celCol = mix(celCol, celCol * vec3(0.84, 0.855, 0.93), speck * (0.16 + rock * 0.52));

      // Wet rock in the stream bed. Darker, bluer, and the only ground on the
      // course with a real highlight — which is the entire warning.
      if (wet > 0.002) {
        vec3 wetCol = celCol * vec3(0.60, 0.68, 0.88);
        float sheen = bandedSpecular(N, V, uSunDir, 110.0);
        wetCol += vec3(0.42, 0.52, 0.60) * sheen * saturate1(dot(N, uSunDir) * 2.0);
        celCol = mix(celCol, wetCol, wet * mix(0.65, 1.0, moisture));
      }

      // The takeoff lip. Deliberately not subtle: a bleached, packed strip a
      // full band brighter than the trail either side of it, so the rider can
      // see exactly where to leave the ground from the top of the approach.
      if (lipM > 0.004) {
        celCol = mix(celCol, uBandColor[3] * 1.04, bandStep(lipM, 0.30, 0.05) * 0.72);
      }

      // The inked trail edge. The plan-view irregularity lives in the geometry;
      // this is the value break that turns a boundary into a drawn line, with a
      // little wander in the threshold so it never runs perfectly parallel to
      // the spline.
      float wob = fbm2(vec2(dAlong * 0.85, 3.1), 3);
      float edgeT = bandStep(abs(across), 0.895 + (wob - 0.5) * 0.055, 0.008);
      celCol = mix(celCol, mix(celCol, uBandColor[0], 0.62), edgeT);
    `,
  });
  // CelMaterial sets the NPR_VERTEX_COLOR define but three only declares the
  // `color` attribute when the Material flag itself is set.
  mat.vertexColors = true;
  return mat;
}
