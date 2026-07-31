/**
 * RiderMesh — the rider's body and clothing, every vertex generated here.
 *
 * There are no imported models in this project, so the rider is lathes, lofted
 * elliptical tubes and box forms, merged per material and skinned to the bones
 * built in Skeleton.ts.
 *
 * The three things that make this read as a character rather than as a set of
 * primitives stuck together:
 *
 *  1. CROSS-SECTIONS ARE ELLIPSES, NOT CIRCLES. A torso is wide and shallow, a
 *     forearm is oval, a shin is flat at the front. Every loft here carries a
 *     separate half-width and half-depth per ring, and the ring's X axis is
 *     locked to the rig's X axis rather than to a Frenet frame, so nothing
 *     corkscrews and the silhouette stays deliberate from every angle.
 *
 *  2. WEIGHTS COME FROM DISTANCE TO BONE SEGMENTS, WITH AN EXPLICIT CANDIDATE
 *     SET PER PART. A global "nearest bone" pass puts helmet vertices on the
 *     clavicle and shorts vertices on the forearm the moment two parts pass
 *     close to each other. Every part here declares which bones it is allowed to
 *     be influenced by, and the Gaussian falloff over segment distance does the
 *     blending inside that set. That is what keeps the elbow and knee creases
 *     smooth while the helmet stays perfectly rigid on the skull.
 *
 *  3. PARTS ARE NOT NORMAL-WELDED TO EACH OTHER. Each part computes its own
 *     normals before merging, so a sleeve cuff against a bare arm shades as a
 *     hard edge — which is what a cel ramp needs to make clothing read as
 *     clothing. `finalizeGeometry` still welds POSITIONS for the outline hull,
 *     so the ink never tears at those same seams.
 *
 * Everything is authored in bind space, which is rig space with the skeleton at
 * rest — see Skeleton.ts.
 */

import {
  BufferAttribute,
  BufferGeometry,
  Color,
  Group,
  Matrix4,
  Mesh,
  Quaternion,
  SkinnedMesh,
  SphereGeometry,
  Vector2,
  Vector3,
} from 'three';

import { CelMaterial, attachOutline, registerNprMesh, type CelOptions } from '../npr/CelMaterial';
import { finalizeGeometry } from '../npr/OutlineGeometry';
import { RAMPS, type RampPreset } from '../npr/Palette';
import { clamp01, lerp } from '../core/MathX';
import {
  BONE_INDEX,
  REST,
  RIDER_DIMS as D,
  RiderSkeleton,
  restPos,
  type BoneName,
} from './Skeleton';

// ─────────────────────────────────────────────────────────────────────────────
// Small geometry kit
// ─────────────────────────────────────────────────────────────────────────────

const _m4 = new Matrix4();
const _q = new Quaternion();
const _vA = new Vector3();
const _vB = new Vector3();
const _vC = new Vector3();

const X_AXIS = new Vector3(1, 0, 0);

/** One cross-section of a loft: an ellipse in the plane spanned by ax/ay. */
interface LoftRing {
  c: Vector3;
  ax: Vector3;
  ay: Vector3;
  rx: number;
  ry: number;
}

function makeRing(c: Vector3, ax: Vector3, ay: Vector3, rx: number, ry: number): LoftRing {
  return { c: c.clone(), ax: ax.clone(), ay: ay.clone(), rx, ry };
}

/**
 * Build rings along a polyline with the ring X axis locked to `xRef`.
 *
 * Locking the reference rather than parallel-transporting is the right choice
 * for a biped: every limb axis here is far from ±X, so the projection is always
 * well conditioned, and the result is perfectly reproducible — the seam on a
 * sleeve is in the same place on both arms.
 */
function ringsAlongPath(
  path: Vector3[],
  rx: number[],
  ry: number[],
  xRef: Vector3 = X_AXIS,
): LoftRing[] {
  const n = path.length;
  const rings: LoftRing[] = [];
  for (let i = 0; i < n; i++) {
    const a = path[Math.max(0, i - 1)];
    const b = path[Math.min(n - 1, i + 1)];
    _vA.subVectors(b, a);
    if (_vA.lengthSq() < 1e-12) _vA.set(0, 1, 0);
    _vA.normalize();

    _vB.copy(xRef).addScaledVector(_vA, -xRef.dot(_vA));
    if (_vB.lengthSq() < 1e-10) _vB.set(0, 0, 1).addScaledVector(_vA, -_vA.z);
    _vB.normalize();
    _vC.crossVectors(_vA, _vB).normalize();

    rings.push(makeRing(path[i], _vB, _vC, rx[i], ry[i]));
  }
  return rings;
}

/** Append rounded cap rings at one or both ends of a ring list, in place. */
function roundCaps(rings: LoftRing[], startRounds: number, endRounds: number): LoftRing[] {
  const out: LoftRing[] = [];
  if (startRounds > 0 && rings.length >= 2) {
    const r0 = rings[0];
    _vA.subVectors(rings[0].c, rings[1].c).normalize(); // outward at the start
    for (let k = startRounds; k >= 1; k--) {
      const t = k / (startRounds + 0.35);
      const th = t * Math.PI * 0.5;
      const s = Math.cos(th);
      const push = Math.sin(th) * Math.max(r0.rx, r0.ry) * 0.85;
      out.push(
        makeRing(_vB.copy(r0.c).addScaledVector(_vA, push), r0.ax, r0.ay, r0.rx * s, r0.ry * s),
      );
    }
  }
  for (const r of rings) out.push(r);
  if (endRounds > 0 && rings.length >= 2) {
    const rn = rings[rings.length - 1];
    _vA.subVectors(rings[rings.length - 1].c, rings[rings.length - 2].c).normalize();
    for (let k = 1; k <= endRounds; k++) {
      const t = k / (endRounds + 0.35);
      const th = t * Math.PI * 0.5;
      const s = Math.cos(th);
      const push = Math.sin(th) * Math.max(rn.rx, rn.ry) * 0.85;
      out.push(
        makeRing(_vB.copy(rn.c).addScaledVector(_vA, push), rn.ax, rn.ay, rn.rx * s, rn.ry * s),
      );
    }
  }
  return out;
}

/** Triangulate a ring list into a closed (or capped) tube. */
function buildLoft(rings: LoftRing[], segs: number, capStart: boolean, capEnd: boolean): BufferGeometry {
  const rowCount = rings.length;
  const vertCount = rowCount * segs + (capStart ? 1 : 0) + (capEnd ? 1 : 0);
  const pos = new Float32Array(vertCount * 3);
  const uv = new Float32Array(vertCount * 2);
  const idx: number[] = [];

  let p = 0;
  let t = 0;
  for (let r = 0; r < rowCount; r++) {
    const ring = rings[r];
    const vCoord = r / Math.max(1, rowCount - 1);
    for (let s = 0; s < segs; s++) {
      const a = (s / segs) * Math.PI * 2;
      const cs = Math.cos(a);
      const sn = Math.sin(a);
      pos[p++] = ring.c.x + ring.ax.x * cs * ring.rx + ring.ay.x * sn * ring.ry;
      pos[p++] = ring.c.y + ring.ax.y * cs * ring.rx + ring.ay.y * sn * ring.ry;
      pos[p++] = ring.c.z + ring.ax.z * cs * ring.rx + ring.ay.z * sn * ring.ry;
      uv[t++] = s / segs;
      uv[t++] = vCoord;
    }
  }

  for (let r = 0; r < rowCount - 1; r++) {
    for (let s = 0; s < segs; s++) {
      const s2 = (s + 1) % segs;
      const a = r * segs + s;
      const b = r * segs + s2;
      const c = (r + 1) * segs + s2;
      const d = (r + 1) * segs + s;
      idx.push(a, b, c, a, c, d);
    }
  }

  let cursor = rowCount * segs;
  if (capStart) {
    const ring = rings[0];
    pos[p++] = ring.c.x;
    pos[p++] = ring.c.y;
    pos[p++] = ring.c.z;
    uv[t++] = 0.5;
    uv[t++] = 0;
    const centre = cursor++;
    for (let s = 0; s < segs; s++) {
      const s2 = (s + 1) % segs;
      idx.push(centre, s2, s);
    }
  }
  if (capEnd) {
    const ring = rings[rowCount - 1];
    pos[p++] = ring.c.x;
    pos[p++] = ring.c.y;
    pos[p++] = ring.c.z;
    uv[t++] = 0.5;
    uv[t++] = 1;
    const centre = cursor++;
    const base = (rowCount - 1) * segs;
    for (let s = 0; s < segs; s++) {
      const s2 = (s + 1) % segs;
      idx.push(centre, base + s, base + s2);
    }
  }

  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(pos, 3));
  geo.setAttribute('uv', new BufferAttribute(uv, 2));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return geo;
}

/** A lofted limb: path + radius profile, with optional rounded ends. */
function loftLimb(
  path: Vector3[],
  rx: number[],
  ry: number[],
  segs: number,
  startRounds = 0,
  endRounds = 0,
  xRef: Vector3 = X_AXIS,
): BufferGeometry {
  const rings = roundCaps(ringsAlongPath(path, rx, ry, xRef), startRounds, endRounds);
  return buildLoft(rings, segs, startRounds === 0, endRounds === 0);
}

/** An ellipsoid (or a dome, with `thetaLength` < π), placed and oriented. */
function sphereForm(
  centre: Vector3,
  rx: number,
  ry: number,
  rz: number,
  wSeg: number,
  hSeg: number,
  thetaLength = Math.PI,
  basis?: Matrix4,
): BufferGeometry {
  const g = new SphereGeometry(1, wSeg, hSeg, 0, Math.PI * 2, 0, thetaLength);
  _m4.makeScale(rx, ry, rz);
  g.applyMatrix4(_m4);
  if (basis) g.applyMatrix4(basis);
  _m4.makeTranslation(centre.x, centre.y, centre.z);
  g.applyMatrix4(_m4);
  return g;
}

/**
 * A revolved profile (x = radius, y = height), placed by `basis` then `origin`.
 * Built by hand rather than with LatheGeometry so the caller controls the ring
 * count independently of the profile resolution and gets UVs that run along the
 * profile — which is what the cel hatch expects.
 */
function latheForm(profile: Vector2[], segs: number, origin: Vector3, basis?: Matrix4): BufferGeometry {
  const rows = profile.length;
  const pos: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];
  for (let r = 0; r < rows; r++) {
    for (let s = 0; s < segs; s++) {
      const a = (s / segs) * Math.PI * 2;
      pos.push(Math.cos(a) * profile[r].x, profile[r].y, Math.sin(a) * profile[r].x);
      uv.push(s / segs, r / Math.max(1, rows - 1));
    }
  }
  for (let r = 0; r < rows - 1; r++) {
    for (let s = 0; s < segs; s++) {
      const s2 = (s + 1) % segs;
      const a = r * segs + s;
      const b = r * segs + s2;
      const c = (r + 1) * segs + s2;
      const d = (r + 1) * segs + s;
      idx.push(a, b, c, a, c, d);
    }
  }
  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(new Float32Array(pos), 3));
  g.setAttribute('uv', new BufferAttribute(new Float32Array(uv), 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  if (basis) g.applyMatrix4(basis);
  _m4.makeTranslation(origin.x, origin.y, origin.z);
  g.applyMatrix4(_m4);
  return g;
}

/** A rounded box. Used for shoe soles, helmet vents, glove knuckles. */
function boxForm(
  w: number,
  h: number,
  d: number,
  centre: Vector3,
  rot?: Quaternion,
  bevel = 0.25,
): BufferGeometry {
  // Two stacked lofts give a chamfered box with far fewer verts than a real
  // bevel and a much better cel silhouette than a hard-edged cube.
  const hx = w * 0.5;
  const hy = h * 0.5;
  const hz = d * 0.5;
  const b = clamp01(bevel);
  const path = [
    new Vector3(0, -hy, 0),
    new Vector3(0, -hy * (1 - b * 0.5), 0),
    new Vector3(0, hy * (1 - b * 0.5), 0),
    new Vector3(0, hy, 0),
  ];
  const rxs = [hx * (1 - b), hx, hx, hx * (1 - b)];
  const rys = [hz * (1 - b), hz, hz, hz * (1 - b)];
  const rings = ringsAlongPath(path, rxs, rys, new Vector3(1, 0, 0));
  // Square the ellipse off: 8 segments with a superellipse push.
  const geo = buildLoft(rings, 8, true, true);
  const pos = geo.getAttribute('position') as BufferAttribute;
  for (let i = 0; i < pos.count; i++) {
    // Push each ring vertex out toward the box corner — a cheap superellipse.
    const x = pos.getX(i);
    const z = pos.getZ(i);
    const m = Math.max(Math.abs(x) / (hx || 1), Math.abs(z) / (hz || 1));
    if (m > 1e-5) {
      const k = lerp(1, 1 / m, 0.55);
      pos.setX(i, x * k);
      pos.setZ(i, z * k);
    }
  }
  geo.computeVertexNormals();
  if (rot) {
    _m4.makeRotationFromQuaternion(rot);
    geo.applyMatrix4(_m4);
  }
  _m4.makeTranslation(centre.x, centre.y, centre.z);
  geo.applyMatrix4(_m4);
  return geo;
}

/** Orthonormal basis with +Y along `up` and +Z on the `zHint` side of it. */
function basisMatrix(up: Vector3, zHint: Vector3, out: Matrix4): Matrix4 {
  _vA.copy(up).normalize();
  _vC.copy(zHint).addScaledVector(_vA, -zHint.dot(_vA));
  if (_vC.lengthSq() < 1e-10) _vC.set(0, 0, 1);
  _vC.normalize();
  _vB.crossVectors(_vA, _vC).normalize(); // x
  return out.makeBasis(_vB, _vA, _vC);
}

/** Points on an ellipse around an axis frame — goggle straps, chin bars. */
function arcPath(
  centre: Vector3,
  basis: Matrix4,
  rx: number,
  rz: number,
  a0: number,
  a1: number,
  steps: number,
  yOffsetAt?: (t: number) => number,
): Vector3[] {
  const pts: Vector3[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const a = lerp(a0, a1, t);
    const p = new Vector3(Math.sin(a) * rx, yOffsetAt ? yOffsetAt(t) : 0, Math.cos(a) * rz);
    p.applyMatrix4(basis);
    p.add(centre);
    pts.push(p);
  }
  return pts;
}

// ─────────────────────────────────────────────────────────────────────────────
// Merge
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Merge parts that already carry position/normal/uv/skinIndex/skinWeight.
 * Written here rather than pulled from an addon so the project keeps its "no
 * dependency beyond three core" property.
 */
function mergeParts(list: BufferGeometry[]): BufferGeometry {
  let vtx = 0;
  let ind = 0;
  for (const g of list) {
    const p = g.getAttribute('position');
    vtx += p.count;
    const i = g.getIndex();
    ind += i ? i.count : p.count;
  }

  const pos = new Float32Array(vtx * 3);
  const nor = new Float32Array(vtx * 3);
  const uv = new Float32Array(vtx * 2);
  const si = new Uint16Array(vtx * 4);
  const sw = new Float32Array(vtx * 4);
  const index = new Uint32Array(ind);

  let vo = 0;
  let io = 0;
  for (const g of list) {
    const p = g.getAttribute('position') as BufferAttribute;
    const n = g.getAttribute('normal') as BufferAttribute;
    const t = g.getAttribute('uv') as BufferAttribute;
    const a = g.getAttribute('skinIndex') as BufferAttribute;
    const w = g.getAttribute('skinWeight') as BufferAttribute;
    const count = p.count;

    pos.set(p.array as Float32Array, vo * 3);
    nor.set(n.array as Float32Array, vo * 3);
    uv.set(t.array as Float32Array, vo * 2);
    si.set(a.array as Uint16Array, vo * 4);
    sw.set(w.array as Float32Array, vo * 4);

    const gi = g.getIndex();
    if (gi) {
      for (let k = 0; k < gi.count; k++) index[io + k] = gi.getX(k) + vo;
      io += gi.count;
    } else {
      for (let k = 0; k < count; k++) index[io + k] = k + vo;
      io += count;
    }
    vo += count;
  }

  const out = new BufferGeometry();
  out.setAttribute('position', new BufferAttribute(pos, 3));
  out.setAttribute('normal', new BufferAttribute(nor, 3));
  out.setAttribute('uv', new BufferAttribute(uv, 2));
  out.setAttribute('skinIndex', new BufferAttribute(si, 4));
  out.setAttribute('skinWeight', new BufferAttribute(sw, 4));
  out.setIndex(new BufferAttribute(index, 1));
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Skinning
// ─────────────────────────────────────────────────────────────────────────────

interface Binding {
  bone: BoneName;
  /** Segment the vertex distance is measured to. Defaults to the rest bone. */
  head?: Vector3;
  tail?: Vector3;
  /** Gaussian falloff radius, metres. Roughly the local limb radius. */
  falloff: number;
  /** Multiplier before normalisation. Lower = this bone yields to its neighbours. */
  bias?: number;
}

const _seg = new Vector3();
const _pv = new Vector3();

/** Squared distance from p to the segment [a,b]. */
function distToSegment(p: Vector3, a: Vector3, b: Vector3): number {
  _seg.subVectors(b, a);
  const len2 = _seg.lengthSq();
  if (len2 < 1e-12) return p.distanceTo(a);
  _pv.subVectors(p, a);
  const t = clamp01(_pv.dot(_seg) / len2);
  _pv.copy(a).addScaledVector(_seg, t);
  return p.distanceTo(_pv);
}

/** Resolve a binding's segment from the rest table when not given explicitly. */
function bindingSegment(b: Binding, outA: Vector3, outB: Vector3): void {
  const i = BONE_INDEX[b.bone];
  if (b.head) outA.copy(b.head);
  else outA.copy(REST.pos[i]);
  if (b.tail) outB.copy(b.tail);
  else outB.copy(REST.pos[i]).addScaledVector(REST.dir[i], Math.max(REST.length[i], 0.04));
}

const _segA = new Vector3();
const _segB = new Vector3();
const _vp = new Vector3();

/**
 * Compute four-bone skin weights for one part.
 *
 * The falloff is Gaussian in distance-to-segment, which gives a C¹ blend across
 * a joint: the weight of the parent decays exactly as fast as the child's grows,
 * so an elbow at 90° keeps its volume instead of pinching. Only the four
 * strongest bones survive, which is the hardware limit anyway.
 */
function skinPart(geo: BufferGeometry, bindings: Binding[]): BufferGeometry {
  const pos = geo.getAttribute('position') as BufferAttribute;
  const count = pos.count;
  const si = new Uint16Array(count * 4);
  const sw = new Float32Array(count * 4);

  const nb = bindings.length;
  const heads: Vector3[] = [];
  const tails: Vector3[] = [];
  for (const b of bindings) {
    const a = new Vector3();
    const t = new Vector3();
    bindingSegment(b, a, t);
    heads.push(a);
    tails.push(t);
  }

  const w = new Float64Array(nb);
  for (let v = 0; v < count; v++) {
    _vp.set(pos.getX(v), pos.getY(v), pos.getZ(v));

    let best = -1;
    let bestD = Infinity;
    for (let b = 0; b < nb; b++) {
      _segA.copy(heads[b]);
      _segB.copy(tails[b]);
      const d = distToSegment(_vp, _segA, _segB);
      if (d < bestD) {
        bestD = d;
        best = b;
      }
      const f = bindings[b].falloff;
      const x = d / f;
      w[b] = (bindings[b].bias ?? 1) * Math.exp(-x * x * 1.35);
    }

    // Top four.
    let i0 = -1;
    let i1 = -1;
    let i2 = -1;
    let i3 = -1;
    let w0 = 0;
    let w1 = 0;
    let w2 = 0;
    let w3 = 0;
    for (let b = 0; b < nb; b++) {
      const x = w[b];
      if (x > w0) {
        i3 = i2; w3 = w2; i2 = i1; w2 = w1; i1 = i0; w1 = w0; i0 = b; w0 = x;
      } else if (x > w1) {
        i3 = i2; w3 = w2; i2 = i1; w2 = w1; i1 = b; w1 = x;
      } else if (x > w2) {
        i3 = i2; w3 = w2; i2 = b; w2 = x;
      } else if (x > w3) {
        i3 = b; w3 = x;
      }
    }

    let sum = w0 + w1 + w2 + w3;
    if (sum < 1e-9) {
      // Everything underflowed — the vertex is far outside every falloff. Fall
      // back to hard-binding the nearest bone so it can never be left at the
      // origin, which is the classic "one stray vertex spike" artefact.
      i0 = best;
      w0 = 1;
      i1 = i2 = i3 = -1;
      w1 = w2 = w3 = 0;
      sum = 1;
    }
    const inv = 1 / sum;
    const o = v * 4;
    si[o] = i0 >= 0 ? BONE_INDEX[bindings[i0].bone] : 0;
    si[o + 1] = i1 >= 0 ? BONE_INDEX[bindings[i1].bone] : 0;
    si[o + 2] = i2 >= 0 ? BONE_INDEX[bindings[i2].bone] : 0;
    si[o + 3] = i3 >= 0 ? BONE_INDEX[bindings[i3].bone] : 0;
    sw[o] = w0 * inv;
    sw[o + 1] = w1 * inv;
    sw[o + 2] = w2 * inv;
    sw[o + 3] = w3 * inv;
  }

  geo.setAttribute('skinIndex', new BufferAttribute(si, 4));
  geo.setAttribute('skinWeight', new BufferAttribute(sw, 4));
  return geo;
}

/** Hard-bind a whole part to one bone. Helmets, lenses, shoes. */
function rigidPart(geo: BufferGeometry, bone: BoneName): BufferGeometry {
  const count = (geo.getAttribute('position') as BufferAttribute).count;
  const si = new Uint16Array(count * 4);
  const sw = new Float32Array(count * 4);
  const b = BONE_INDEX[bone];
  for (let v = 0; v < count; v++) {
    si[v * 4] = b;
    sw[v * 4] = 1;
  }
  geo.setAttribute('skinIndex', new BufferAttribute(si, 4));
  geo.setAttribute('skinWeight', new BufferAttribute(sw, 4));
  return geo;
}

// ─────────────────────────────────────────────────────────────────────────────
// The rider's parts
// ─────────────────────────────────────────────────────────────────────────────

/** Rest positions, pulled once so the builders read cleanly. */
const P = {
  pelvis: restPos('pelvis'),
  spine1: restPos('spine1'),
  spine2: restPos('spine2'),
  chest: restPos('chest'),
  neck: restPos('neck'),
  head: restPos('head'),
  headEnd: restPos('headEnd'),
  clavL: restPos('clavL'),
  clavR: restPos('clavR'),
  shoulderL: restPos('upperArmL'),
  shoulderR: restPos('upperArmR'),
  elbowL: restPos('forearmL'),
  elbowR: restPos('forearmR'),
  handL: restPos('handL'),
  handR: restPos('handR'),
  handEndL: restPos('handEndL'),
  handEndR: restPos('handEndR'),
  hipL: restPos('thighL'),
  hipR: restPos('thighR'),
  kneeL: restPos('shinL'),
  kneeR: restPos('shinR'),
  ankleL: restPos('footL'),
  ankleR: restPos('footR'),
  toeL: restPos('toeL'),
  toeR: restPos('toeR'),
  hem: restPos('hem'),
  shortsL: restPos('shortsL'),
  shortsR: restPos('shortsR'),
};

function mix(a: Vector3, b: Vector3, t: number): Vector3 {
  return new Vector3().lerpVectors(a, b, t);
}

function offset(a: Vector3, x: number, y: number, z: number): Vector3 {
  return new Vector3(a.x + x, a.y + y, a.z + z);
}

/** The head's own frame: +Y up the skull, +Z out of the face. */
const HEAD_BASIS = (() => {
  const up = new Vector3().subVectors(P.headEnd, P.head).normalize();
  return basisMatrix(up, new Vector3(0, 0, 1), new Matrix4());
})();
const HEAD_CENTRE = new Vector3().lerpVectors(P.head, P.headEnd, 0.40);

// ── Skin ─────────────────────────────────────────────────────────────────────

function buildSkinParts(): BufferGeometry[] {
  const parts: BufferGeometry[] = [];

  // Skull + face. The skull is mostly hidden by the helmet; what matters is the
  // jaw and cheek line under the goggle strap, so that is where the detail is.
  parts.push(
    rigidPart(sphereForm(HEAD_CENTRE, D.headRadius * 0.95, D.headRadius * 1.06, D.headRadius, 16, 12, Math.PI, HEAD_BASIS), 'head'),
  );
  // Jaw / chin wedge, pushed forward and down out of the skull sphere.
  const jaw = new Vector3(0, -0.052, 0.026).applyMatrix4(HEAD_BASIS).add(HEAD_CENTRE);
  parts.push(rigidPart(sphereForm(jaw, 0.070, 0.060, 0.082, 12, 8, Math.PI, HEAD_BASIS), 'head'));
  // Nose — tiny, but a face without one reads as a mannequin at close range.
  const nose = new Vector3(0, -0.020, 0.092).applyMatrix4(HEAD_BASIS).add(HEAD_CENTRE);
  parts.push(rigidPart(sphereForm(nose, 0.020, 0.026, 0.024, 8, 6, Math.PI, HEAD_BASIS), 'head'));

  // Neck.
  const neckPath = [offset(P.neck, 0, -0.030, -0.010), P.neck, mix(P.neck, P.head, 0.75)];
  parts.push(
    skinPart(
      loftLimb(neckPath, [D.neckRadius * 1.12, D.neckRadius, D.neckRadius * 0.94], [D.neckRadius * 1.18, D.neckRadius * 1.02, D.neckRadius * 0.96], 12),
      [
        { bone: 'chest', falloff: 0.075, bias: 0.7 },
        { bone: 'neck', falloff: 0.085 },
        { bone: 'head', falloff: 0.070, bias: 0.9 },
      ],
    ),
  );

  // Arms: bare from mid-upper-arm to the wrist.
  for (const side of [1, -1]) {
    const sh = side > 0 ? P.shoulderL : P.shoulderR;
    const el = side > 0 ? P.elbowL : P.elbowR;
    const wr = side > 0 ? P.handL : P.handR;
    const upper: BoneName = side > 0 ? 'upperArmL' : 'upperArmR';
    const fore: BoneName = side > 0 ? 'forearmL' : 'forearmR';
    const hand: BoneName = side > 0 ? 'handL' : 'handR';

    const path = [
      mix(sh, el, 0.46),
      mix(sh, el, 0.78),
      el,
      mix(el, wr, 0.30),
      mix(el, wr, 0.68),
      mix(el, wr, 0.95),
    ];
    const rx = [D.upperArmTop * 0.96, D.upperArmBottom * 1.04, D.upperArmBottom, D.forearmTop * 1.02, D.forearmTop * 0.82, D.forearmBottom];
    const ry = [D.upperArmTop * 0.90, D.upperArmBottom * 0.98, D.upperArmBottom * 0.96, D.forearmTop * 0.94, D.forearmTop * 0.78, D.forearmBottom * 0.94];
    parts.push(
      skinPart(loftLimb(path, rx, ry, 12, 1, 1), [
        { bone: upper, falloff: 0.095 },
        { bone: fore, falloff: 0.095 },
        { bone: hand, falloff: 0.045, bias: 0.8 },
      ]),
    );
  }

  // Calves: knee to the top of the shoe.
  for (const side of [1, -1]) {
    const kn = side > 0 ? P.kneeL : P.kneeR;
    const an = side > 0 ? P.ankleL : P.ankleR;
    const shin: BoneName = side > 0 ? 'shinL' : 'shinR';
    const thigh: BoneName = side > 0 ? 'thighL' : 'thighR';
    const foot: BoneName = side > 0 ? 'footL' : 'footR';
    const path = [mix(kn, an, 0.10), mix(kn, an, 0.34), mix(kn, an, 0.62), mix(kn, an, 0.88)];
    const rx = [D.shinTop, D.shinTop * 0.92, D.shinBottom * 1.20, D.shinBottom];
    const ry = [D.shinTop * 1.06, D.shinTop * 0.96, D.shinBottom * 1.24, D.shinBottom];
    parts.push(
      skinPart(loftLimb(path, rx, ry, 12, 1, 1), [
        { bone: thigh, falloff: 0.070, bias: 0.6 },
        { bone: shin, falloff: 0.115 },
        { bone: foot, falloff: 0.055, bias: 0.7 },
      ]),
    );
  }

  return parts;
}

// ── Jersey ───────────────────────────────────────────────────────────────────

function buildJerseyParts(): BufferGeometry[] {
  const parts: BufferGeometry[] = [];
  const g = D.clothGap;

  // Torso. Rings follow the spine; the shoulder ring is the widest thing on the
  // rider and is what makes the silhouette read as "athlete on a bike" rather
  // than "cylinder".
  const shoulderMid = mix(P.clavL, P.clavR, 0.5);
  const path = [
    offset(P.hem, 0, -0.010, 0.010),
    mix(P.pelvis, P.spine1, 0.25),
    P.spine1,
    P.spine2,
    P.chest,
    mix(shoulderMid, P.neck, 0.55),
  ];
  const rx = [
    D.hipHalfWidth * 0.94 + g,
    D.hipHalfWidth + g,
    D.waistHalfWidth + g,
    D.waistHalfWidth * 1.12 + g,
    D.chestHalfWidth + g,
    D.chestHalfWidth * 0.68 + g,
  ];
  const ry = [
    D.hipDepth * 0.94 + g,
    D.hipDepth + g,
    D.waistDepth + g,
    D.waistDepth * 1.05 + g,
    D.chestDepth + g,
    D.chestDepth * 0.72 + g,
  ];
  parts.push(
    skinPart(loftLimb(path, rx, ry, 20, 0, 1), [
      { bone: 'hem', falloff: 0.085, bias: 0.85 },
      { bone: 'pelvis', falloff: 0.125 },
      { bone: 'spine1', falloff: 0.115 },
      { bone: 'spine2', falloff: 0.115 },
      { bone: 'chest', falloff: 0.135 },
      { bone: 'neck', falloff: 0.070, bias: 0.7 },
      { bone: 'clavL', falloff: 0.090, bias: 0.8 },
      { bone: 'clavR', falloff: 0.090, bias: 0.8 },
    ]),
  );

  // Deltoid caps + sleeves.
  for (const side of [1, -1]) {
    const sh = side > 0 ? P.shoulderL : P.shoulderR;
    const el = side > 0 ? P.elbowL : P.elbowR;
    const clav: BoneName = side > 0 ? 'clavL' : 'clavR';
    const upper: BoneName = side > 0 ? 'upperArmL' : 'upperArmR';

    parts.push(
      skinPart(sphereForm(offset(sh, side * 0.006, 0.012, 0), 0.078, 0.080, 0.082, 12, 10), [
        { bone: clav, falloff: 0.090, bias: 0.55 },
        { bone: upper, falloff: 0.110 },
        { bone: 'chest', falloff: 0.090, bias: 0.5 },
      ]),
    );

    const sleeve = [mix(sh, el, 0.06), mix(sh, el, 0.28), mix(sh, el, 0.50)];
    const srx = [D.upperArmTop * 1.34, D.upperArmTop * 1.22, D.upperArmTop * 1.14];
    const sry = [D.upperArmTop * 1.30, D.upperArmTop * 1.18, D.upperArmTop * 1.10];
    parts.push(
      skinPart(loftLimb(sleeve, srx, sry, 12, 0, 0), [
        { bone: clav, falloff: 0.075, bias: 0.4 },
        { bone: upper, falloff: 0.130 },
        { bone: 'chest', falloff: 0.075, bias: 0.35 },
      ]),
    );
  }

  // Collar.
  const collarPath = [mix(P.chest, P.neck, 0.72), mix(P.chest, P.neck, 1.02)];
  parts.push(
    skinPart(
      loftLimb(collarPath, [D.neckRadius * 1.50, D.neckRadius * 1.36], [D.neckRadius * 1.56, D.neckRadius * 1.40], 14),
      [
        { bone: 'chest', falloff: 0.085 },
        { bone: 'neck', falloff: 0.075 },
      ],
    ),
  );

  return parts;
}

// ── Shorts ───────────────────────────────────────────────────────────────────

function buildClothParts(): BufferGeometry[] {
  const parts: BufferGeometry[] = [];
  const g = D.clothGap;

  // Seat / waistband. Sits over the jersey hem.
  const seatPath = [offset(P.pelvis, 0, 0.070, 0.012), P.pelvis, offset(P.pelvis, 0, -0.062, -0.012)];
  parts.push(
    skinPart(
      loftLimb(
        seatPath,
        [D.hipHalfWidth * 0.98 + g * 1.6, D.hipHalfWidth + g * 1.8, D.hipHalfWidth * 1.02 + g * 1.8],
        [D.hipDepth * 0.98 + g * 1.6, D.hipDepth + g * 1.8, D.hipDepth * 1.06 + g * 1.8],
        18,
        0,
        0,
      ),
      [
        { bone: 'pelvis', falloff: 0.150 },
        { bone: 'spine1', falloff: 0.100, bias: 0.6 },
        { bone: 'thighL', falloff: 0.110, bias: 0.7 },
        { bone: 'thighR', falloff: 0.110, bias: 0.7 },
      ],
    ),
  );

  // Legs of the shorts — baggy, ending just above the knee, with the hem bone
  // owning the last two rings so it can flap.
  for (const side of [1, -1]) {
    const hip = side > 0 ? P.hipL : P.hipR;
    const knee = side > 0 ? P.kneeL : P.kneeR;
    const thigh: BoneName = side > 0 ? 'thighL' : 'thighR';
    const hem: BoneName = side > 0 ? 'shortsL' : 'shortsR';
    const path = [
      mix(hip, knee, -0.06),
      mix(hip, knee, 0.22),
      mix(hip, knee, 0.48),
      mix(hip, knee, 0.66),
      mix(hip, knee, 0.72),
    ];
    const rx = [
      D.thighTop * 1.18 + g,
      D.thighTop * 1.14 + g,
      D.thighBottom * 1.42 + g,
      D.thighBottom * 1.52 + g,
      D.thighBottom * 1.44 + g,
    ];
    const ry = [
      D.thighTop * 1.14 + g,
      D.thighTop * 1.12 + g,
      D.thighBottom * 1.40 + g,
      D.thighBottom * 1.56 + g,
      D.thighBottom * 1.46 + g,
    ];
    parts.push(
      skinPart(loftLimb(path, rx, ry, 16, 0, 0), [
        { bone: 'pelvis', falloff: 0.100, bias: 0.7 },
        { bone: thigh, falloff: 0.150 },
        { bone: hem, falloff: 0.105, bias: 1.25 },
      ]),
    );
  }

  return parts;
}

// ── Rubber: gloves, shoes, straps, pads ──────────────────────────────────────

function buildRubberParts(): BufferGeometry[] {
  const parts: BufferGeometry[] = [];

  for (const side of [1, -1]) {
    const wrist = side > 0 ? P.handL : P.handR;
    const tip = side > 0 ? P.handEndL : P.handEndR;
    const elbow = side > 0 ? P.elbowL : P.elbowR;
    const hand: BoneName = side > 0 ? 'handL' : 'handR';
    const fore: BoneName = side > 0 ? 'forearmL' : 'forearmR';

    // The glove is a closed fist around the bar: a fat ellipsoid straddling the
    // grip axis, four knuckle blocks on top, and a cuff that blends onto the
    // forearm so the wrist does not shear when the bars turn.
    const barDir = new Vector3().subVectors(tip, wrist).normalize();
    const gripUp = new Vector3(0, 1, 0).addScaledVector(barDir, -barDir.y).normalize();
    const gripFwd = new Vector3().crossVectors(barDir, gripUp).normalize();

    const palm = new Vector3().copy(wrist).addScaledVector(barDir, side * 0.004);
    const gloveBasis = new Matrix4().makeBasis(barDir, gripUp, gripFwd);
    parts.push(rigidPart(sphereForm(palm, 0.062, 0.055, 0.050, 12, 10, Math.PI, gloveBasis), hand));

    for (let k = 0; k < 3; k++) {
      const along = (k - 1) * 0.030;
      const knuckle = new Vector3()
        .copy(palm)
        .addScaledVector(barDir, along)
        .addScaledVector(gripUp, 0.030)
        .addScaledVector(gripFwd, 0.014);
      _q.setFromRotationMatrix(gloveBasis);
      parts.push(rigidPart(boxForm(0.026, 0.020, 0.044, knuckle, _q.clone(), 0.4), hand));
    }

    const cuffA = new Vector3().copy(wrist).addScaledVector(new Vector3().subVectors(elbow, wrist).normalize(), 0.060);
    const cuffB = new Vector3().copy(wrist).addScaledVector(new Vector3().subVectors(elbow, wrist).normalize(), 0.014);
    parts.push(
      skinPart(loftLimb([cuffA, cuffB], [D.forearmBottom * 1.30, D.forearmBottom * 1.42], [D.forearmBottom * 1.26, D.forearmBottom * 1.38], 12), [
        { bone: fore, falloff: 0.070 },
        { bone: hand, falloff: 0.070 },
      ]),
    );
  }

  // Shoes. Built along the ankle→toe line with a flat sole slab, because a shoe
  // that is a tapered tube reads as a sock.
  for (const side of [1, -1]) {
    const ankle = side > 0 ? P.ankleL : P.ankleR;
    const toe = side > 0 ? P.toeL : P.toeR;
    const foot: BoneName = side > 0 ? 'footL' : 'footR';
    const toeB: BoneName = side > 0 ? 'toeL' : 'toeR';

    const fdir = new Vector3().subVectors(toe, ankle).normalize();
    const heel = new Vector3().copy(ankle).addScaledVector(fdir, -0.085).add(new Vector3(0, -0.030, 0));
    const path = [
      heel,
      offset(ankle, 0, -0.030, 0),
      mix(ankle, toe, 0.55).add(new Vector3(0, -0.032, 0)),
      new Vector3().copy(toe).add(new Vector3(0, -0.014, 0.012)),
    ];
    const rx = [D.shoeWidth * 0.86, D.shoeWidth, D.shoeWidth * 0.98, D.shoeWidth * 0.74];
    const ry = [D.shoeHeight * 0.62, D.shoeHeight * 0.66, D.shoeHeight * 0.50, D.shoeHeight * 0.34];
    parts.push(
      skinPart(loftLimb(path, rx, ry, 12, 1, 1), [
        { bone: foot, falloff: 0.130 },
        { bone: toeB, falloff: 0.110 },
        { bone: side > 0 ? 'shinL' : 'shinR', falloff: 0.055, bias: 0.5 },
      ]),
    );
    // Sole slab — the flat that sits on the pedal and catches the key light.
    const solePath = [
      new Vector3().copy(heel).add(new Vector3(0, -0.016, 0)),
      mix(ankle, toe, 0.5).add(new Vector3(0, -0.062, 0)),
      new Vector3().copy(toe).add(new Vector3(0, -0.036, 0.006)),
    ];
    parts.push(
      skinPart(loftLimb(solePath, [D.shoeWidth * 0.90, D.shoeWidth * 1.02, D.shoeWidth * 0.76], [0.016, 0.017, 0.013], 8), [
        { bone: foot, falloff: 0.130 },
        { bone: toeB, falloff: 0.110 },
      ]),
    );
    // Ankle collar.
    parts.push(
      skinPart(
        loftLimb([offset(ankle, 0, 0.006, -0.004), offset(ankle, 0, 0.052, -0.008)], [D.shinBottom * 1.32, D.shinBottom * 1.24], [D.shinBottom * 1.36, D.shinBottom * 1.26], 10),
        [
          { bone: foot, falloff: 0.080 },
          { bone: side > 0 ? 'shinL' : 'shinR', falloff: 0.080 },
        ],
      ),
    );
  }

  // Knee pads.
  for (const side of [1, -1]) {
    const knee = side > 0 ? P.kneeL : P.kneeR;
    const hip = side > 0 ? P.hipL : P.hipR;
    const ankle = side > 0 ? P.ankleL : P.ankleR;
    const up = new Vector3().subVectors(hip, knee).normalize();
    const fwd = new Vector3().subVectors(knee, mix(hip, ankle, 0.5)).normalize();
    const b = basisMatrix(up, fwd, new Matrix4());
    parts.push(
      skinPart(sphereForm(new Vector3().copy(knee).addScaledVector(fwd, 0.018), 0.070, 0.088, 0.072, 12, 10, Math.PI, b), [
        { bone: side > 0 ? 'thighL' : 'thighR', falloff: 0.110 },
        { bone: side > 0 ? 'shinL' : 'shinR', falloff: 0.110 },
      ]),
    );
  }

  // Goggle strap around the back of the helmet.
  const strapCentre = new Vector3(0, -0.028, 0).applyMatrix4(HEAD_BASIS).add(HEAD_CENTRE);
  const strap = arcPath(strapCentre, HEAD_BASIS, D.helmetRadius * 1.00, D.helmetRadius * 1.02, 0.70, Math.PI * 2 - 0.70, 18, (t) =>
    Math.sin(t * Math.PI) * 0.006,
  );
  const strapR = strap.map(() => 0.020);
  const strapD = strap.map(() => 0.009);
  parts.push(rigidPart(loftLimb(strap, strapR, strapD, 6, 0, 0, new Vector3(0, 1, 0)), 'head'));

  return parts;
}

// ── Helmet ───────────────────────────────────────────────────────────────────

function buildHelmetParts(): BufferGeometry[] {
  const parts: BufferGeometry[] = [];
  const R = D.helmetRadius;

  // Shell: a revolved profile rather than a squashed sphere, because the shape
  // that reads as a helmet is the tuck under the brow line and the flat over the
  // crown — neither of which a sphere has. Scaled slightly long in Z inside the
  // head's own frame so the lid is egg-shaped, not round.
  const shellProfile: Vector2[] = [
    new Vector2(0.0, R * 1.03),
    new Vector2(R * 0.30, R * 1.00),
    new Vector2(R * 0.58, R * 0.90),
    new Vector2(R * 0.80, R * 0.72),
    new Vector2(R * 0.94, R * 0.46),
    new Vector2(R * 1.00, R * 0.14),
    new Vector2(R * 0.99, R * -0.20),
    new Vector2(R * 0.93, R * -0.46),
    new Vector2(R * 0.86, R * -0.56),
  ];
  const shellBasis = new Matrix4().copy(HEAD_BASIS).multiply(new Matrix4().makeScale(0.97, 1, 1.06));
  parts.push(rigidPart(latheForm(shellProfile, 18, HEAD_CENTRE, shellBasis), 'head'));

  // Occipital scoop at the back.
  const backCentre = new Vector3(0, -0.030, -0.052).applyMatrix4(HEAD_BASIS).add(HEAD_CENTRE);
  parts.push(rigidPart(sphereForm(backCentre, R * 0.86, R * 0.72, R * 0.80, 12, 10, Math.PI * 0.72, HEAD_BASIS), 'head'));

  // Ear/temple pads: they close the gap between the shell edge and the jaw, and
  // they are what stops the lid reading as a bowl balanced on a head.
  for (const side of [1, -1]) {
    const pad = new Vector3(side * (R * 0.88), -0.050, -0.010).applyMatrix4(HEAD_BASIS).add(HEAD_CENTRE);
    parts.push(rigidPart(sphereForm(pad, 0.030, 0.042, 0.052, 10, 8, Math.PI, HEAD_BASIS), 'head'));
  }

  // Peak / visor: a swept plate tilted down over the goggles.
  const peakCentre = new Vector3(0, 0.044, 0.014).applyMatrix4(HEAD_BASIS).add(HEAD_CENTRE);
  const peak = arcPath(peakCentre, HEAD_BASIS, R * 0.86, R * 1.42, -1.22, 1.22, 14, (t) => -Math.sin(t * Math.PI) * 0.036);
  parts.push(
    rigidPart(
      loftLimb(
        peak,
        // Ring X is world-up here, so the first radius is the visor's THICKNESS
        // and the second is how far it reaches fore-and-aft. A peak that is
        // thick and narrow reads as a rod stuck to the lid; this reads as a plate.
        peak.map(() => 0.0055),
        peak.map((_, i, arr) => lerp(0.020, 0.042, Math.sin((i / (arr.length - 1)) * Math.PI))),
        6,
        0,
        0,
        new Vector3(0, 1, 0),
      ),
      'head',
    ),
  );

  // Two vent ridges over the crown — they catch a band edge and stop the shell
  // reading as one undifferentiated blob under a hard ramp.
  for (const side of [1, -1]) {
    const v0 = new Vector3(side * 0.038, 0.082, -0.036).applyMatrix4(HEAD_BASIS).add(HEAD_CENTRE);
    const v1 = new Vector3(side * 0.048, 0.058, 0.048).applyMatrix4(HEAD_BASIS).add(HEAD_CENTRE);
    parts.push(rigidPart(loftLimb([v0, v1], [0.011, 0.010], [0.006, 0.005], 6, 1, 1), 'head'));
  }

  return parts;
}

// ── Goggle lens ──────────────────────────────────────────────────────────────

function buildLensParts(): BufferGeometry[] {
  const parts: BufferGeometry[] = [];
  const R = D.helmetRadius;
  const centre = new Vector3(0, -0.028, -0.004).applyMatrix4(HEAD_BASIS).add(HEAD_CENTRE);
  const arc = arcPath(centre, HEAD_BASIS, R * 0.88, R * 1.06, -1.16, 1.16, 14, (t) => Math.sin(t * Math.PI) * 0.010);
  parts.push(
    rigidPart(
      loftLimb(
        arc,
        arc.map((_, i, a2) => lerp(0.024, 0.034, Math.sin((i / (a2.length - 1)) * Math.PI))),
        arc.map(() => 0.012),
        6,
        0,
        0,
        new Vector3(0, 1, 0),
      ),
      'head',
    ),
  );
  return parts;
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometry cache — one build for every rider on the mountain
// ─────────────────────────────────────────────────────────────────────────────

export type RiderPart = 'skin' | 'jersey' | 'cloth' | 'rubber' | 'helmet' | 'lens';

const PART_RAMP: Record<RiderPart, keyof typeof RAMPS> = {
  skin: 'skin',
  jersey: 'jerseyPlayer',
  cloth: 'cloth',
  rubber: 'rubber',
  helmet: 'helmet',
  lens: 'lens',
};

let geoCache: Record<RiderPart, BufferGeometry> | null = null;

/** Build (once) the six merged, skinned, outline-ready geometries. */
export function getRiderGeometries(): Record<RiderPart, BufferGeometry> {
  if (geoCache) return geoCache;
  const build = (parts: BufferGeometry[], ao: boolean): BufferGeometry =>
    finalizeGeometry(mergeParts(parts), {
      tolerance: 3e-4,
      maxWeldAngle: 180,
      curvatureGain: 0.9,
      ao,
      aoStrength: 0.45,
    });

  geoCache = {
    skin: build(buildSkinParts(), false),
    jersey: build(buildJerseyParts(), true),
    cloth: build(buildClothParts(), true),
    rubber: build(buildRubberParts(), false),
    helmet: build(buildHelmetParts(), false),
    lens: build(buildLensParts(), false),
  };
  return geoCache;
}

// ─────────────────────────────────────────────────────────────────────────────
// Materials
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Re-hue a ramp toward a rider's committed colour, preserving each band's
 * lightness. Same rule the bike frame uses: we never invent a colour, we move
 * the committed ramp onto the committed hue.
 */
function reHueRamp(base: RampPreset, target: Color, amount = 0.7): RampPreset {
  const th = { h: 0, s: 0, l: 0 };
  target.getHSL(th);
  const colors = base.colors.map((c) => {
    const o = { h: 0, s: 0, l: 0 };
    c.getHSL(o);
    return new Color().setHSL(th.h, lerp(o.s, th.s, amount), o.l);
  });
  return { ...base, colors };
}

/** Re-hue an existing material's bands in place. Used by setJerseyColor. */
export function reHueMaterial(mat: CelMaterial, base: RampPreset, target: Color, amount = 0.7): void {
  const th = { h: 0, s: 0, l: 0 };
  target.getHSL(th);
  const bands = mat.uniforms.uBandColor.value as Color[];
  for (let i = 0; i < bands.length; i++) {
    const src = base.colors[Math.min(i, base.colors.length - 1)];
    const o = { h: 0, s: 0, l: 0 };
    src.getHSL(o);
    bands[i].setHSL(th.h, lerp(o.s, th.s, amount), o.l);
  }
}

/** Materials with no per-rider variation are built once and shared. */
const sharedMaterials = new Map<RiderPart, CelMaterial>();

function celOptionsFor(part: RiderPart): CelOptions {
  const o: CelOptions = { skinned: true, idName: `rider-${part}`, name: `rider:${part}` };
  if (part === 'lens') {
    o.matcapMix = 0.55;
  } else if (part === 'helmet') {
    o.matcapMix = 0.18;
  }
  return o;
}

function sharedMaterial(part: RiderPart): CelMaterial {
  let m = sharedMaterials.get(part);
  if (!m) {
    m = new CelMaterial(RAMPS[PART_RAMP[part]], celOptionsFor(part));
    sharedMaterials.set(part, m);
  }
  return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Assembly
// ─────────────────────────────────────────────────────────────────────────────

export interface RiderMeshOptions {
  /** Jersey colour, sRGB hex. */
  jersey?: number;
  /** Accent colour, sRGB hex — helmet and shorts trim. */
  accent?: number;
  name?: string;
}

export interface RiderMeshSet {
  group: Group;
  meshes: SkinnedMesh[];
  hulls: Mesh[];
  /** Per-rider materials. Shared ones are not in this list and never disposed. */
  owned: CelMaterial[];
  jerseyMaterial: CelMaterial;
  helmetMaterial: CelMaterial;
}

/**
 * Build one rider's six skinned meshes, each with an inverted-hull sibling bound
 * to the SAME skeleton — which is the only way the ink can follow a deforming
 * character without peeling off at speed.
 */
export function buildRiderMeshes(skel: RiderSkeleton, opts: RiderMeshOptions = {}): RiderMeshSet {
  const geos = getRiderGeometries();
  const group = new Group();
  group.name = opts.name ? `${opts.name}:body` : 'rider:body';

  const meshes: SkinnedMesh[] = [];
  const hulls: Mesh[] = [];
  const owned: CelMaterial[] = [];

  const jerseyColor = new Color().setHex(opts.jersey ?? 0xe0574c);
  const accentColor = new Color().setHex(opts.accent ?? 0xff9a5c);

  const jerseyMaterial = new CelMaterial(
    reHueRamp(RAMPS.jerseyPlayer, jerseyColor),
    celOptionsFor('jersey'),
  );
  const helmetMaterial = new CelMaterial(
    reHueRamp(RAMPS.helmet, accentColor, 0.40),
    celOptionsFor('helmet'),
  );
  owned.push(jerseyMaterial, helmetMaterial);

  const order: RiderPart[] = ['skin', 'jersey', 'cloth', 'rubber', 'helmet', 'lens'];
  for (const part of order) {
    const mat =
      part === 'jersey' ? jerseyMaterial : part === 'helmet' ? helmetMaterial : sharedMaterial(part);

    const mesh = new SkinnedMesh(geos[part], mat);
    mesh.name = `${opts.name ?? 'rider'}:${part}`;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    // A rider's pose leaves the bind-pose bounds constantly (superman, crashes).
    // Four characters is nothing to draw; a character that vanishes mid-trick
    // because a stale bounding sphere left the frustum is unforgivable.
    mesh.frustumCulled = false;
    group.add(mesh);
    mesh.bind(skel.skeleton, new Matrix4());

    registerNprMesh(mesh, mat);
    const hull = attachOutline(mesh, mat.preset, { ...mat.celOptions, skinned: true });
    if (hull) {
      hull.frustumCulled = false;
      group.add(hull);
      hulls.push(hull);
    }
    meshes.push(mesh);
  }

  return { group, meshes, hulls, owned, jerseyMaterial, helmetMaterial };
}

/** Apply a rider's identity colours to an already-built set. */
export function applyRiderColors(set: RiderMeshSet, jersey: number, accent: number): void {
  reHueMaterial(set.jerseyMaterial, RAMPS.jerseyPlayer, new Color().setHex(jersey), 0.7);
  reHueMaterial(set.helmetMaterial, RAMPS.helmet, new Color().setHex(accent), 0.40);
}
