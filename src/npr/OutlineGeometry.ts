/**
 * OutlineGeometry — the preprocessing that makes inverted-hull outlines work
 * on hard-surface geometry.
 *
 * The naive inverted hull pushes each vertex along its own normal. On a smooth
 * sphere that is fine. On anything with a hard edge — a box, a rock facet, a
 * bike frame lug, a helmet vent — the vertices at that edge are SPLIT: the same
 * position appears several times with different normals. Push each copy along
 * its own normal and the hull tears open at every corner, leaving the gaps you
 * see on almost every browser toon shader.
 *
 * The fix, applied to every outlined mesh in this game:
 *
 *   1. Weld positions on a spatial hash (quantised to a tolerance).
 *   2. Average the normals of every vertex sharing a welded position, weighted
 *      by triangle area so a fan of tiny triangles doesn't drag the average.
 *   3. Write that average back to EVERY split copy as `aSmoothNormal`.
 *
 * The shading normals are left completely untouched, so hard edges still shade
 * hard — only the hull expansion uses the smoothed set.
 *
 * We compute a curvature estimate in the same pass. Curvature at a vertex is
 * how much its neighbours' smoothed normals disagree with its own: near zero on
 * a flat panel, near one on a tight crease. The hull shader uses it to swell
 * the stroke on creases and taper it on flats, which is the single detail that
 * separates a drawn line from a uniform toon offset.
 */

import { BufferAttribute, BufferGeometry, Vector3 } from 'three';

const _a = new Vector3();
const _b = new Vector3();
const _c = new Vector3();
const _ab = new Vector3();
const _ac = new Vector3();
const _cross = new Vector3();

export interface OutlinePrepOptions {
  /** Weld tolerance in world units. Smaller = fewer accidental merges. */
  tolerance?: number;
  /**
   * Creases sharper than this angle (degrees) are NOT welded — the hull keeps
   * a genuine break there. Set high (180) to weld everything, which is what
   * you want for organic shapes; set around 75 for hard-surface props so an
   * intentional sharp silhouette corner stays sharp.
   */
  maxWeldAngle?: number;
  /** Multiplies the computed curvature. Tune per asset class. */
  curvatureGain?: number;
  /** Overwrite existing attributes if present. */
  force?: boolean;
}

/**
 * Add `aSmoothNormal` (vec3) and `aCurvature` (float) to a geometry, in place.
 * Idempotent unless `force` is set. Returns the same geometry for chaining.
 */
export function prepareOutlineGeometry(
  geo: BufferGeometry,
  options: OutlinePrepOptions = {},
): BufferGeometry {
  const { tolerance = 1e-4, maxWeldAngle = 180, curvatureGain = 1.0, force = false } = options;

  if (!force && geo.getAttribute('aSmoothNormal') && geo.getAttribute('aCurvature')) return geo;

  const posAttr = geo.getAttribute('position') as BufferAttribute;
  if (!posAttr) throw new Error('[OutlineGeometry] geometry has no position attribute');
  if (!geo.getAttribute('normal')) geo.computeVertexNormals();
  const nrmAttr = geo.getAttribute('normal') as BufferAttribute;

  const count = posAttr.count;
  const pos = posAttr.array as Float32Array;
  const nrm = nrmAttr.array as Float32Array;

  // ── 1. Weld map ───────────────────────────────────────────────────────────
  // Spatial hash on quantised position. Using a string key is fast enough here
  // (this runs once at load, over tens of thousands of verts at most) and is
  // far more robust than a float-tolerance grid with neighbour probing.
  const inv = 1 / tolerance;
  const weldKey = (i: number): string => {
    const x = Math.round(pos[i * 3] * inv);
    const y = Math.round(pos[i * 3 + 1] * inv);
    const z = Math.round(pos[i * 3 + 2] * inv);
    return `${x}|${y}|${z}`;
  };

  const groups = new Map<string, number[]>();
  for (let i = 0; i < count; i++) {
    const k = weldKey(i);
    const g = groups.get(k);
    if (g) g.push(i);
    else groups.set(k, [i]);
  }

  // ── 2. Area-weighted normal accumulation ──────────────────────────────────
  // Accumulate per-vertex triangle-area weights first, so the averaging step
  // reflects surface contribution rather than raw triangle count.
  const areaWeight = new Float32Array(count);
  const index = geo.getIndex();
  const triCount = index ? index.count / 3 : count / 3;
  const idxAt = index
    ? (t: number, k: number) => index.getX(t * 3 + k)
    : (t: number, k: number) => t * 3 + k;

  for (let t = 0; t < triCount; t++) {
    const i0 = idxAt(t, 0);
    const i1 = idxAt(t, 1);
    const i2 = idxAt(t, 2);
    _a.fromArray(pos, i0 * 3);
    _b.fromArray(pos, i1 * 3);
    _c.fromArray(pos, i2 * 3);
    _ab.subVectors(_b, _a);
    _ac.subVectors(_c, _a);
    const area = _cross.crossVectors(_ab, _ac).length() * 0.5;
    areaWeight[i0] += area;
    areaWeight[i1] += area;
    areaWeight[i2] += area;
  }

  const smooth = new Float32Array(count * 3);
  const curvature = new Float32Array(count);
  const cosLimit = Math.cos((maxWeldAngle * Math.PI) / 180);

  const acc = new Vector3();
  const n0 = new Vector3();
  const nk = new Vector3();

  for (const groupIndices of groups.values()) {
    if (groupIndices.length === 1) {
      const i = groupIndices[0];
      smooth[i * 3] = nrm[i * 3];
      smooth[i * 3 + 1] = nrm[i * 3 + 1];
      smooth[i * 3 + 2] = nrm[i * 3 + 2];
      // A truly isolated vertex sits on a boundary edge — treat it as a crease
      // so open edges (canopy cards, cloth panels) get a full-weight stroke.
      curvature[i] = 0.5;
      continue;
    }

    // Average, honouring the crease limit relative to each member in turn.
    for (const i of groupIndices) {
      n0.fromArray(nrm, i * 3);
      acc.set(0, 0, 0);
      let wsum = 0;
      let maxDivergence = 0;

      for (const k of groupIndices) {
        nk.fromArray(nrm, k * 3);
        const d = n0.dot(nk);
        maxDivergence = Math.max(maxDivergence, 1 - d);
        if (maxWeldAngle < 180 && d < cosLimit) continue; // too sharp — don't weld
        const w = areaWeight[k] + 1e-8;
        acc.addScaledVector(nk, w);
        wsum += w;
      }

      if (wsum > 0) acc.divideScalar(wsum);
      if (acc.lengthSq() < 1e-12) acc.copy(n0);
      acc.normalize();

      smooth[i * 3] = acc.x;
      smooth[i * 3 + 1] = acc.y;
      smooth[i * 3 + 2] = acc.z;

      // Curvature: how far the shading normal diverges across this weld group.
      // 1 - dot maps [0°,180°] to [0,2]; halve it and shape it so a 30° crease
      // already reads as substantially curved.
      curvature[i] = Math.min(1, Math.pow((maxDivergence * 0.5) * 3.2, 0.65) * curvatureGain);
    }
  }

  // ── 3. Smooth the curvature field once across welded neighbours ───────────
  // Raw per-vertex curvature is noisy on dense meshes and makes the stroke
  // stutter. One averaging pass over the triangle graph settles it without
  // washing out genuine creases.
  const curvSmooth = new Float32Array(curvature);
  const neighbourSum = new Float32Array(count);
  const neighbourCount = new Float32Array(count);
  for (let t = 0; t < triCount; t++) {
    const i0 = idxAt(t, 0);
    const i1 = idxAt(t, 1);
    const i2 = idxAt(t, 2);
    neighbourSum[i0] += curvature[i1] + curvature[i2];
    neighbourCount[i0] += 2;
    neighbourSum[i1] += curvature[i0] + curvature[i2];
    neighbourCount[i1] += 2;
    neighbourSum[i2] += curvature[i0] + curvature[i1];
    neighbourCount[i2] += 2;
  }
  for (let i = 0; i < count; i++) {
    if (neighbourCount[i] > 0) {
      const avg = neighbourSum[i] / neighbourCount[i];
      curvSmooth[i] = curvature[i] * 0.6 + avg * 0.4;
    }
  }

  geo.setAttribute('aSmoothNormal', new BufferAttribute(smooth, 3));
  geo.setAttribute('aCurvature', new BufferAttribute(curvSmooth, 1));
  return geo;
}

/**
 * Attach a flat `aCurvature` of a fixed value without doing the weld pass.
 * For geometry that is already smooth-normalled (spheres, tubes, lathes) and
 * where we know the stroke weight we want.
 */
export function setFlatOutlineAttributes(geo: BufferGeometry, curvature = 0.5): BufferGeometry {
  if (!geo.getAttribute('normal')) geo.computeVertexNormals();
  const nrm = geo.getAttribute('normal') as BufferAttribute;
  const count = nrm.count;
  if (!geo.getAttribute('aSmoothNormal')) {
    geo.setAttribute('aSmoothNormal', new BufferAttribute((nrm.array as Float32Array).slice(), 3));
  }
  if (!geo.getAttribute('aCurvature')) {
    const c = new Float32Array(count);
    c.fill(curvature);
    geo.setAttribute('aCurvature', new BufferAttribute(c, 1));
  }
  return geo;
}

/**
 * Add a `aSway` attribute for wind, derived from height above the geometry's
 * lowest point. Foliage roots stay planted; tips move.
 */
export function addSwayAttribute(geo: BufferGeometry, power = 1.6, originY?: number): BufferGeometry {
  const pos = geo.getAttribute('position') as BufferAttribute;
  const count = pos.count;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < count; i++) {
    const y = pos.getY(i);
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const base = originY ?? minY;
  const span = Math.max(maxY - base, 1e-5);
  const sway = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    sway[i] = Math.pow(Math.max(0, (pos.getY(i) - base) / span), power);
  }
  geo.setAttribute('aSway', new BufferAttribute(sway, 1));
  return geo;
}

/**
 * Add a vertex-AO attribute by casting the normal against a crude ambient
 * estimate: how enclosed the vertex is relative to the geometry's bounding
 * volume, blended with downward-facing bias. Cheap, and enough to keep the
 * undersides of rocks and the insides of wheel wells from glowing.
 */
export function addVertexAo(geo: BufferGeometry, strength = 1, downBias = 0.5): BufferGeometry {
  const pos = geo.getAttribute('position') as BufferAttribute;
  if (!geo.getAttribute('normal')) geo.computeVertexNormals();
  const nrm = geo.getAttribute('normal') as BufferAttribute;
  geo.computeBoundingBox();
  const bb = geo.boundingBox!;
  const count = pos.count;
  const ao = new Float32Array(count);
  const centre = new Vector3();
  bb.getCenter(centre);
  const extent = new Vector3();
  bb.getSize(extent);
  const radius = Math.max(extent.length() * 0.5, 1e-5);

  const p = new Vector3();
  const n = new Vector3();
  const toCentre = new Vector3();
  for (let i = 0; i < count; i++) {
    p.fromBufferAttribute(pos, i);
    n.fromBufferAttribute(nrm, i);
    toCentre.subVectors(centre, p);
    const dist = toCentre.length() / radius;
    toCentre.normalize();
    // Facing outward from the centre and far from it = open. Facing inward or
    // near the centre = occluded.
    const openness = 0.5 - 0.5 * n.dot(toCentre);
    const down = 0.5 - 0.5 * n.y;
    const occ = 1 - (openness * 0.55 + dist * 0.45) * (1 - down * downBias);
    ao[i] = 1 - Math.min(1, Math.max(0, occ)) * strength;
  }
  geo.setAttribute('aAo', new BufferAttribute(ao, 1));
  return geo;
}

/**
 * One-call preparation for any mesh that will be cel-shaded and outlined.
 * Every geometry builder in the project ends with this.
 */
export function finalizeGeometry(
  geo: BufferGeometry,
  opts: OutlinePrepOptions & { sway?: boolean; swayPower?: number; ao?: boolean; aoStrength?: number } = {},
): BufferGeometry {
  prepareOutlineGeometry(geo, opts);
  if (opts.sway) addSwayAttribute(geo, opts.swayPower ?? 1.6);
  if (opts.ao) addVertexAo(geo, opts.aoStrength ?? 0.6);
  geo.computeBoundingSphere();
  return geo;
}
