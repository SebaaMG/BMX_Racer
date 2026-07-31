/**
 * Scatter — where the loose material on the mountain goes, and how it gets
 * drawn without costing anything.
 *
 * This file owns two things that look unrelated and are not:
 *
 *  • The PLACEMENT machinery — a deterministic, zone-and-slope-driven sampler
 *    that both this module and Foliage.ts use. Placement is seeded from the
 *    world seed and from a named stream, so adding a new scatter pass cannot
 *    move a single existing rock. That property is what makes captured frames
 *    comparable across builds; without it every tweak reshuffles the world.
 *
 *  • The STREAMING instance pools. Every scattered object in the game lives in
 *    a flat typed-array field, bucketed into a coarse spatial grid, and a small
 *    fixed-size InstancedMesh is refilled from it whenever the camera has moved
 *    far enough to matter. One draw call per object type per LOD tier, no
 *    per-frame allocation, and no per-cell mesh explosion.
 *
 * WHY REFILL RATHER THAN PARTITION.
 * The obvious design is one InstancedMesh per spatial cell, made visible when
 * the cell is in range. It works, and it produces sixty to a hundred draw calls
 * of a dozen instances each, which is the worst possible shape for a GPU. The
 * refill costs about a millisecond every half second of riding — the camera has
 * to travel ten metres before anything is rewritten — and in exchange every
 * rock on the mountain is one draw call. That trade is not close.
 *
 * WHY DISTANCE CULLING AND NOT FRUSTUM CULLING.
 * Refilling on rotation as well as translation would mean rebuilding the buffers
 * during every corner, which is exactly when the frame budget is tightest. The
 * instances are cheap enough (a far rock is twenty triangles) that keeping the
 * ones behind the camera costs less than the rebuild would.
 */

import {
  BufferAttribute,
  BufferGeometry,
  DynamicDrawUsage,
  Group,
  IcosahedronGeometry,
  InstancedBufferAttribute,
  InstancedMesh,
  Matrix4,
  Object3D,
  Quaternion,
  Vector3,
} from 'three';

import { Noise2D } from '../core/Noise';
import { Rng, WORLD_SEED } from '../core/RNG';
import { clamp, clamp01, lerp, smoothstep } from '../core/MathX';
import { SurfaceKind, TrackCarve } from '../game/Contracts';
import { ROUTE, WORLD_HALF } from '../game/WorldConstants';
import { CelMaterial, CelOptions, attachOutline, registerNprMesh } from '../npr/CelMaterial';
import { RAMPS, RampName } from '../npr/Palette';
import { finalizeGeometry } from '../npr/OutlineGeometry';
import type { CarvedBoulder } from './Heightfield';

// Module-scope scratch — the refill path allocates nothing.
const _m = new Matrix4();
const _p = new Vector3();
const _q = new Quaternion();
const _qy = new Quaternion();
const _s = new Vector3(1, 1, 1);
const _n = new Vector3();
const _up = new Vector3(0, 1, 0);
const _tilt = new Vector3();

// ─────────────────────────────────────────────────────────────────────────────
// What the placer needs to know about the mountain
// ─────────────────────────────────────────────────────────────────────────────

export interface ScatterSource {
  heightAt(x: number, z: number): number;
  normalAt(x: number, z: number, out: Vector3): Vector3;
  zoneAt(x: number, z: number): SurfaceKind;
  /** Normalised erosion, [-1, 1]. Negative = cut, positive = deposited. */
  erosionAt(x: number, z: number): number;
  /** Distance to the route centreline, capped by the corridor field. */
  corridorDistAt(x: number, z: number): number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Placement
// ─────────────────────────────────────────────────────────────────────────────

export interface ScatterRule {
  /** Nominal candidate spacing, metres. One candidate per cell, jittered. */
  spacing: number;
  /** Acceptance weight per SurfaceKind, 0..1, indexed by the enum. */
  zoneWeight: number[];
  /** Accepted slope window in radians, with a soft shoulder at each end. */
  slopeMin: number;
  slopeMax: number;
  slopeFeather: number;
  /** Never place closer than this to the route centreline. */
  routeClear: number;
  /** Never place further than this from the route centreline. */
  routeMax: number;
  /**
   * Preference for eroded ground. Positive favours deposition (fans, silt),
   * negative favours freshly cut ground (gully floors, scarps).
   */
  erosionBias: number;
  /** 0 = statistically even, 1 = strongly clumped into patches. */
  clumping: number;
  /** Clump patch size, metres. */
  clumpScale: number;
  /** Master acceptance multiplier. */
  density: number;
  /** Uniform scale range. */
  scaleMin: number;
  scaleMax: number;
  /** How far the instance leans to match the ground normal. 1 = flush. */
  alignToNormal: number;
  /** Vertical offset applied after grounding — negative buries the object. */
  sink: number;
  /** How many distinct geometry variants the pool has. */
  variants: number;
  maxCount: number;
}

export function makeRule(p: Partial<ScatterRule>): ScatterRule {
  return {
    spacing: 8,
    zoneWeight: [0, 0, 0, 0, 0, 0, 0],
    slopeMin: 0,
    slopeMax: 0.7,
    slopeFeather: 0.12,
    routeClear: 0,
    routeMax: Infinity,
    erosionBias: 0,
    clumping: 0.5,
    clumpScale: 60,
    density: 1,
    scaleMin: 0.8,
    scaleMax: 1.4,
    alignToNormal: 0.6,
    sink: 0,
    variants: 1,
    maxCount: 20000,
    ...p,
  };
}

/**
 * A flat field of placed instances plus a coarse spatial index over it.
 *
 * Structure-of-arrays rather than an array of objects: the refill walks tens of
 * thousands of these per rebuild and an array of objects would be a cache miss
 * per instance and a GC root per instance.
 */
export class InstanceField {
  count = 0;
  readonly pos: Float32Array;
  readonly quat: Float32Array;
  readonly scale: Float32Array;
  readonly tint: Float32Array;
  readonly phase: Float32Array;
  readonly variant: Uint8Array;
  /** Cleared to 0 when the track carve removes an instance. */
  readonly alive: Uint8Array;

  /** Spatial buckets, counting-sorted. */
  cellSize = 96;
  nx = 1;
  nz = 1;
  cellStart: Int32Array = new Int32Array(1);
  cellItems: Int32Array = new Int32Array(0);

  constructor(capacity: number) {
    this.pos = new Float32Array(capacity * 3);
    this.quat = new Float32Array(capacity * 4);
    this.scale = new Float32Array(capacity);
    this.tint = new Float32Array(capacity * 3);
    this.phase = new Float32Array(capacity);
    this.variant = new Uint8Array(capacity);
    this.alive = new Uint8Array(capacity);
  }

  /** Build the bucket index. Call once after placement. */
  index(cellSize: number): void {
    this.cellSize = cellSize;
    this.nx = Math.max(1, Math.ceil((WORLD_HALF * 2) / cellSize));
    this.nz = this.nx;
    const cells = this.nx * this.nz;
    const counts = new Int32Array(cells + 1);

    const cellOf = (i: number): number => {
      const cx = clamp(((this.pos[i * 3] + WORLD_HALF) / cellSize) | 0, 0, this.nx - 1);
      const cz = clamp(((this.pos[i * 3 + 2] + WORLD_HALF) / cellSize) | 0, 0, this.nz - 1);
      return cz * this.nx + cx;
    };

    for (let i = 0; i < this.count; i++) counts[cellOf(i) + 1]++;
    for (let c = 0; c < cells; c++) counts[c + 1] += counts[c];
    this.cellStart = counts;
    this.cellItems = new Int32Array(this.count);
    const cursor = counts.slice(0, cells);
    for (let i = 0; i < this.count; i++) {
      const c = cellOf(i);
      this.cellItems[cursor[c]++] = i;
    }
  }

  /** Kill every live instance inside a world-space disc. Returns how many. */
  cullDisc(cx: number, cz: number, radius: number): number {
    let killed = 0;
    const r2 = radius * radius;
    const c0x = clamp((((cx - radius) + WORLD_HALF) / this.cellSize) | 0, 0, this.nx - 1);
    const c1x = clamp((((cx + radius) + WORLD_HALF) / this.cellSize) | 0, 0, this.nx - 1);
    const c0z = clamp((((cz - radius) + WORLD_HALF) / this.cellSize) | 0, 0, this.nz - 1);
    const c1z = clamp((((cz + radius) + WORLD_HALF) / this.cellSize) | 0, 0, this.nz - 1);
    for (let cz2 = c0z; cz2 <= c1z; cz2++) {
      const row = cz2 * this.nx;
      for (let cx2 = c0x; cx2 <= c1x; cx2++) {
        const c = row + cx2;
        const s = this.cellStart[c];
        const e = this.cellStart[c + 1];
        for (let k = s; k < e; k++) {
          const i = this.cellItems[k];
          if (this.alive[i] === 0) continue;
          const dx = this.pos[i * 3] - cx;
          const dz = this.pos[i * 3 + 2] - cz;
          if (dx * dx + dz * dz <= r2) {
            this.alive[i] = 0;
            killed++;
          }
        }
      }
    }
    return killed;
  }

  liveCount(): number {
    let n = 0;
    for (let i = 0; i < this.count; i++) n += this.alive[i];
    return n;
  }
}

/** World-space XZ bounds of the route, expanded by a margin. */
function routeBounds(margin: number): { x0: number; z0: number; x1: number; z1: number } {
  let x0 = Infinity;
  let x1 = -Infinity;
  let z0 = Infinity;
  let z1 = -Infinity;
  for (const r of ROUTE) {
    if (r.x < x0) x0 = r.x;
    if (r.x > x1) x1 = r.x;
    if (r.z < z0) z0 = r.z;
    if (r.z > z1) z1 = r.z;
  }
  return {
    x0: Math.max(-WORLD_HALF, x0 - margin),
    x1: Math.min(WORLD_HALF, x1 + margin),
    z0: Math.max(-WORLD_HALF, z0 - margin),
    z1: Math.min(WORLD_HALF, z1 + margin),
  };
}

/**
 * Place instances by rejection sampling on a jittered grid.
 *
 * A jittered grid rather than pure random points: pure random points clump and
 * gap by chance, and the gaps read as bald patches on a hillside. A jittered
 * grid gives an approximately Poisson distribution — no two instances closer
 * than about half the spacing — and then the CLUMP field puts the clustering
 * back deliberately, at a scale that reads as terrain rather than as noise.
 * Rocks gather in fields; grass gathers in meadows; neither is uniform, and
 * neither is random.
 */
export function generateInstances(
  src: ScatterSource,
  rule: ScatterRule,
  seed: string,
): InstanceField {
  const clumpNoise = new Noise2D(`${seed}:clump`);
  const sizeNoise = new Noise2D(`${seed}:size`);

  const bounds = isFinite(rule.routeMax)
    ? routeBounds(rule.routeMax + rule.spacing * 2)
    : { x0: -WORLD_HALF, z0: -WORLD_HALF, x1: WORLD_HALF, z1: WORLD_HALF };

  const step = rule.spacing;
  const nx = Math.max(1, Math.floor((bounds.x1 - bounds.x0) / step));
  const nz = Math.max(1, Math.floor((bounds.z1 - bounds.z0) / step));
  const clumpFreq = 1 / Math.max(rule.clumpScale, 1);

  /**
   * One sweep of the candidate grid.
   *
   * `thin` scales every acceptance probability, and `field` may be null for a
   * counting-only pass. The two-pass structure exists because truncating at the
   * instance cap mid-sweep would fill the low-Z half of the mountain and leave
   * the high-Z half completely bare — the population has to be thinned
   * uniformly, which means knowing the total before placing anything. Both
   * passes drive a freshly-seeded Rng, so the second pass sees the identical
   * random sequence and its output is a genuine subset of the first pass's.
   */
  const sweep = (thin: number, field: InstanceField | null): number => {
    const rng = new Rng(WORLD_SEED).fork(seed);
    let accepted = 0;

    for (let j = 0; j < nz; j++) {
      const bz = bounds.z0 + j * step;
      for (let i = 0; i < nx; i++) {
        const bx = bounds.x0 + i * step;

        // Jitter inside the cell. Kept at 0.8 of the cell so the minimum
        // spacing guarantee survives — at full jitter two neighbours coincide.
        const x = bx + (rng.next() - 0.5) * step * 0.8 + step * 0.5;
        const z = bz + (rng.next() - 0.5) * step * 0.8 + step * 0.5;
        const rSpin = rng.next();
        const rSize = rng.next();
        const rTint = rng.next();
        const rTintG = rng.next();
        const rTintB = rng.next();
        const rPhase = rng.next();
        const rVariant = rng.next();
        const rAccept = rng.next();

        // ── Corridor gate: cheapest rejection, so it goes first ────────────
        const rd = src.corridorDistAt(x, z);
        if (rd < rule.routeClear || rd > rule.routeMax) continue;

        // ── Zone ──────────────────────────────────────────────────────────
        const zone = src.zoneAt(x, z);
        const zw = rule.zoneWeight[zone] ?? 0;
        if (zw <= 0.0005) continue;

        // ── Slope ─────────────────────────────────────────────────────────
        src.normalAt(x, z, _n);
        const slope = Math.acos(clamp(_n.y, -1, 1));
        const sw =
          smoothstep(rule.slopeMin - rule.slopeFeather, rule.slopeMin + rule.slopeFeather, slope) *
          (1 -
            smoothstep(rule.slopeMax - rule.slopeFeather, rule.slopeMax + rule.slopeFeather, slope));
        if (sw <= 0.0005) continue;

        // ── Erosion preference ────────────────────────────────────────────
        const ero = src.erosionAt(x, z);
        let ew = 1;
        if (rule.erosionBias !== 0) {
          const aligned = ero * Math.sign(rule.erosionBias);
          ew = clamp01(0.5 + aligned * Math.abs(rule.erosionBias));
        }

        // ── Clumping ──────────────────────────────────────────────────────
        const cl = clumpNoise.fbm(x * clumpFreq, z * clumpFreq, 3) * 0.5 + 0.5;
        const cw = lerp(1, clamp01(cl * 1.9 - 0.35), rule.clumping);

        const p = zw * sw * ew * cw * rule.density * thin;
        if (rAccept >= p) continue;

        accepted++;
        if (!field || field.count >= rule.maxCount) continue;

        const y = src.heightAt(x, z);
        if (!isFinite(y)) continue;

        const k = field.count;
        // Size varies coherently as well as randomly, so a boulder field has
        // big rocks near big rocks rather than a uniform salt of sizes.
        const coherent = sizeNoise.fbm(x * 0.02, z * 0.02, 2) * 0.5 + 0.5;
        const sc = lerp(rule.scaleMin, rule.scaleMax, clamp01(coherent * 0.6 + rSize * 0.6));

        field.pos[k * 3] = x;
        field.pos[k * 3 + 1] = y + rule.sink * sc;
        field.pos[k * 3 + 2] = z;

        // Lean toward the ground normal, then spin about the world up.
        _tilt.copy(_up).lerp(_n, rule.alignToNormal);
        if (_tilt.lengthSq() < 1e-8) _tilt.copy(_up);
        _tilt.normalize();
        _q.setFromUnitVectors(_up, _tilt);
        _qy.setFromAxisAngle(_up, rSpin * Math.PI * 2);
        _q.multiply(_qy);

        field.quat[k * 4] = _q.x;
        field.quat[k * 4 + 1] = _q.y;
        field.quat[k * 4 + 2] = _q.z;
        field.quat[k * 4 + 3] = _q.w;
        field.scale[k] = sc;

        // A few percent of value variation so a rock field is not a stamp.
        const v = 0.90 + rTint * 0.20;
        field.tint[k * 3] = v;
        field.tint[k * 3 + 1] = v * (0.985 + rTintG * 0.03);
        field.tint[k * 3 + 2] = v * (0.98 + rTintB * 0.04);

        field.phase[k] = rPhase * Math.PI * 2;
        field.variant[k] =
          rule.variants > 1 ? Math.min(rule.variants - 1, (rVariant * rule.variants) | 0) : 0;
        field.alive[k] = 1;
        field.count = k + 1;
      }
    }
    return accepted;
  };

  const wanted = sweep(1, null);
  const thin = wanted > rule.maxCount ? rule.maxCount / wanted : 1;
  const field = new InstanceField(Math.min(rule.maxCount, wanted + 8));
  sweep(thin, field);

  field.index(96);
  return field;
}

// ─────────────────────────────────────────────────────────────────────────────
// Instance groups and streaming pools
// ─────────────────────────────────────────────────────────────────────────────

/**
 * One geometry variant's worth of GPU instances.
 *
 * `meshes` may hold several InstancedMeshes — a tree's trunk and its canopy use
 * different ramps and therefore different materials, but they are the SAME
 * instances. They share the instanceMatrix and the per-instance attribute
 * buffers by reference, so a refill writes each number exactly once and the
 * trunk can never end up a frame out of step with its own foliage.
 */
export class InstanceGroup {
  readonly meshes: InstancedMesh[] = [];
  readonly max: number;
  count = 0;

  private matrixArray: Float32Array;
  private tintAttr: InstancedBufferAttribute;
  private fadeAttr: InstancedBufferAttribute;
  private phaseAttr: InstancedBufferAttribute;
  private primary: InstancedMesh;

  constructor(primary: InstancedMesh, max: number) {
    this.primary = primary;
    this.max = max;
    this.meshes.push(primary);
    primary.instanceMatrix.setUsage(DynamicDrawUsage);
    this.matrixArray = primary.instanceMatrix.array as Float32Array;
    this.tintAttr = primary.geometry.getAttribute('aInstanceTint') as InstancedBufferAttribute;
    this.fadeAttr = primary.geometry.getAttribute('aInstanceFade') as InstancedBufferAttribute;
    this.phaseAttr = primary.geometry.getAttribute('aInstancePhase') as InstancedBufferAttribute;
  }

  /** Add a mesh that draws the same instances with a different material. */
  addSibling(mesh: InstancedMesh): void {
    mesh.instanceMatrix = this.primary.instanceMatrix;
    this.meshes.push(mesh);
  }

  begin(): void {
    this.count = 0;
  }

  push(
    px: number,
    py: number,
    pz: number,
    qx: number,
    qy: number,
    qz: number,
    qw: number,
    scale: number,
    tr: number,
    tg: number,
    tb: number,
    fade: number,
    phase: number,
  ): void {
    const i = this.count;
    if (i >= this.max) return;
    _p.set(px, py, pz);
    _q.set(qx, qy, qz, qw);
    _s.set(scale, scale, scale);
    _m.compose(_p, _q, _s);
    _m.toArray(this.matrixArray, i * 16);

    const t = this.tintAttr.array as Float32Array;
    t[i * 3] = tr;
    t[i * 3 + 1] = tg;
    t[i * 3 + 2] = tb;
    (this.fadeAttr.array as Float32Array)[i] = fade;
    (this.phaseAttr.array as Float32Array)[i] = phase;
    this.count = i + 1;
  }

  end(): void {
    for (const m of this.meshes) {
      m.count = this.count;
      m.visible = this.count > 0;
    }
    this.primary.instanceMatrix.needsUpdate = true;
    this.tintAttr.needsUpdate = true;
    this.fadeAttr.needsUpdate = true;
    this.phaseAttr.needsUpdate = true;
  }

  dispose(): void {
    for (const m of this.meshes) {
      m.geometry.dispose();
      const mat = m.material;
      if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
      else mat.dispose();
    }
    this.meshes.length = 0;
  }
}

/** One LOD tier: a distance window, and one group per geometry variant. */
export interface PoolTier {
  groups: InstanceGroup[];
  /** Instances closer than this belong to a finer tier. 0 for the finest. */
  near: number;
  far: number;
  /** Width of the dithered dissolve at each end of the window. */
  fade: number;
}

/**
 * The streaming refill.
 *
 * Rebuilt only when the camera has travelled `refillDistance`, which at race
 * speed is roughly twice a second. The dithered fade at each end of every tier
 * window is what makes that acceptable: an instance never appears or vanishes,
 * it dissolves over tens of metres, so a rebuild half a second late is
 * invisible rather than a pop.
 */
export class ScatterPool {
  private lastX = Infinity;
  private lastZ = Infinity;
  private dirty = true;

  readonly stats = { instances: 0, refills: 0 };

  constructor(
    readonly field: InstanceField,
    readonly tiers: PoolTier[],
    /** Camera travel that forces a rebuild, metres. */
    readonly refillDistance = 10,
  ) {}

  invalidate(): void {
    this.dirty = true;
  }

  update(camX: number, camZ: number): void {
    if (!this.dirty) {
      const dx = camX - this.lastX;
      const dz = camZ - this.lastZ;
      if (dx * dx + dz * dz < this.refillDistance * this.refillDistance) return;
    }
    this.lastX = camX;
    this.lastZ = camZ;
    this.dirty = false;
    this.refill(camX, camZ);
  }

  private refill(camX: number, camZ: number): void {
    const f = this.field;
    let total = 0;

    for (const tier of this.tiers) {
      for (const g of tier.groups) g.begin();

      const reach = tier.far;
      const c0x = clamp((((camX - reach) + WORLD_HALF) / f.cellSize) | 0, 0, f.nx - 1);
      const c1x = clamp((((camX + reach) + WORLD_HALF) / f.cellSize) | 0, 0, f.nx - 1);
      const c0z = clamp((((camZ - reach) + WORLD_HALF) / f.cellSize) | 0, 0, f.nz - 1);
      const c1z = clamp((((camZ + reach) + WORLD_HALF) / f.cellSize) | 0, 0, f.nz - 1);

      const nearIn = Math.max(0, tier.near - tier.fade);
      const farOut = tier.far;
      const farIn = tier.far - tier.fade;

      for (let cz = c0z; cz <= c1z; cz++) {
        const row = cz * f.nx;
        for (let cx = c0x; cx <= c1x; cx++) {
          const c = row + cx;
          const s = f.cellStart[c];
          const e = f.cellStart[c + 1];
          for (let k = s; k < e; k++) {
            const i = f.cellItems[k];
            if (f.alive[i] === 0) continue;
            const dx = f.pos[i * 3] - camX;
            const dz = f.pos[i * 3 + 2] - camZ;
            const d = Math.sqrt(dx * dx + dz * dz);
            if (d < nearIn || d > farOut) continue;

            // Dissolve in at the near edge, dissolve out at the far edge.
            let fade = 1;
            if (tier.near > 0 && d < tier.near) fade = (d - nearIn) / Math.max(tier.fade, 1e-3);
            if (d > farIn) fade = Math.min(fade, (farOut - d) / Math.max(tier.fade, 1e-3));
            fade = clamp01(fade);
            if (fade <= 0.004) continue;

            const g = tier.groups[Math.min(f.variant[i], tier.groups.length - 1)];
            if (g.count >= g.max) continue;
            g.push(
              f.pos[i * 3],
              f.pos[i * 3 + 1],
              f.pos[i * 3 + 2],
              f.quat[i * 4],
              f.quat[i * 4 + 1],
              f.quat[i * 4 + 2],
              f.quat[i * 4 + 3],
              f.scale[i],
              f.tint[i * 3],
              f.tint[i * 3 + 1],
              f.tint[i * 3 + 2],
              fade,
              f.phase[i],
            );
          }
        }
      }

      for (const g of tier.groups) {
        g.end();
        total += g.count;
      }
    }

    this.stats.instances = total;
    this.stats.refills++;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Instanced mesh construction
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The dithered LOD dissolve, shared by every instanced scatter material.
 *
 * A distance fade that scales alpha needs blending, and blending needs sorting,
 * and sorted transparency over ten thousand instances is not happening. A
 * screen-space ordered dither instead: each pixel discards if its threshold in
 * a Bayer pattern exceeds the instance's fade. The result is opaque, order
 * independent, depth-correct, and — because the pattern is fixed in screen
 * space — reads as a stipple dissolve, which is a drawn convention rather than
 * a rendering artefact.
 *
 * The discard sits after the shading call rather than before it. That wastes
 * work on the fragments being dissolved, which are a thin band of the frame,
 * and buys the ability to do the whole thing through CelMaterial's documented
 * extension points rather than forking the material.
 */
export const SCATTER_FADE_VARYINGS = /* glsl */ `
  out float vScatterFade;
`;

export const SCATTER_FADE_VERTEX = /* glsl */ `
  #ifdef USE_INSTANCING
    vScatterFade = aInstanceFade;
  #else
    vScatterFade = 1.0;
  #endif
`;

export const SCATTER_FADE_FRAGMENT = /* glsl */ `
  if (vScatterFade < 0.996) {
    // 8x8 ordered Bayer, evaluated arithmetically so no texture is needed.
    ivec2 dp = ivec2(gl_FragCoord.xy) & ivec2(7);
    int bx = dp.x ^ dp.y;
    int by = dp.y;
    int bits = 0;
    bits |= ((bx >> 2) & 1) << 0;
    bits |= ((by >> 2) & 1) << 1;
    bits |= ((bx >> 1) & 1) << 2;
    bits |= ((by >> 1) & 1) << 3;
    bits |= ((bx >> 0) & 1) << 4;
    bits |= ((by >> 0) & 1) << 5;
    if (float(bits) * (1.0 / 64.0) > vScatterFade) discard;
  }
`;

export interface InstancedBuildOptions extends CelOptions {
  preset: RampName;
  max: number;
  name: string;
  /** Attach an inverted-hull outline. */
  outline?: boolean;
}

/**
 * Wire a geometry up as a streaming InstancedMesh.
 *
 * All three per-instance channels are bound whether or not this asset varies
 * them. The cel vertex shader reads all of them whenever three defines
 * USE_INSTANCING, and an unbound attribute reads as (0,0,0,1) — which would
 * tint every instance in the world to black.
 */
export function buildInstanced(
  geo: BufferGeometry,
  parent: Object3D,
  opts: InstancedBuildOptions,
  shared?: { tint: InstancedBufferAttribute; fade: InstancedBufferAttribute; phase: InstancedBufferAttribute },
): { mesh: InstancedMesh; hull: InstancedMesh | null } {
  const max = opts.max;
  const tint = shared?.tint ?? new InstancedBufferAttribute(new Float32Array(max * 3).fill(1), 3);
  const fade = shared?.fade ?? new InstancedBufferAttribute(new Float32Array(max).fill(1), 1);
  const phase = shared?.phase ?? new InstancedBufferAttribute(new Float32Array(max), 1);
  tint.setUsage(DynamicDrawUsage);
  fade.setUsage(DynamicDrawUsage);
  phase.setUsage(DynamicDrawUsage);
  geo.setAttribute('aInstanceTint', tint);
  geo.setAttribute('aInstanceFade', fade);
  geo.setAttribute('aInstancePhase', phase);

  const celOpts: CelOptions = {
    ...opts,
    instanced: true,
    varyings: `${SCATTER_FADE_VARYINGS}${opts.varyings ?? ''}`,
    vertexBody: `${SCATTER_FADE_VERTEX}${opts.vertexBody ?? ''}`,
    fragmentBody: `${opts.fragmentBody ?? ''}${SCATTER_FADE_FRAGMENT}`,
  };

  const material = new CelMaterial(RAMPS[opts.preset], celOpts);
  if (opts.vertexColors) material.vertexColors = true;

  const mesh = new InstancedMesh(geo, material, max);
  mesh.name = opts.name;
  mesh.count = 0;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  // Culled by the pool's distance window; the bounding sphere of a streaming
  // instanced mesh is meaningless because its contents change every rebuild.
  mesh.frustumCulled = false;
  registerNprMesh(mesh, material);

  let hull: InstancedMesh | null = null;
  if (opts.outline) {
    const h = attachOutline(mesh, RAMPS[opts.preset], celOpts);
    if (h) {
      h.frustumCulled = false;
      hull = h as InstancedMesh;
      parent.add(h);
    }
  }
  parent.add(mesh);
  return { mesh, hull };
}

// ─────────────────────────────────────────────────────────────────────────────
// Rock geometry
// ─────────────────────────────────────────────────────────────────────────────

const _v = new Vector3();

/**
 * A boulder.
 *
 * Built by pushing an icosahedron around with noise, then flattening its base.
 * Two details matter more than the silhouette:
 *
 *  • The geometry stays FACETED. An icosahedron subdivision is non-indexed, so
 *    every triangle keeps its own normals and the cel ramp bands per facet.
 *    A smooth-shaded rock under a hard ramp reads as a potato; a faceted one
 *    reads as stone, because the value steps follow planes.
 *  • The base is flattened rather than left round, so the instance sits IN the
 *    ground instead of tangent to it. A sphere resting on a slope shows a
 *    crescent of daylight underneath from any low camera, and the whole scatter
 *    then reads as decals floating on the terrain.
 */
export function buildRockGeometry(seed: string, detail: number, squash: number): BufferGeometry {
  const geo = new IcosahedronGeometry(1, detail);
  const pos = geo.getAttribute('position') as BufferAttribute;
  const nA = new Noise2D(`${seed}:a`);
  const nB = new Noise2D(`${seed}:b`);

  for (let i = 0; i < pos.count; i++) {
    _v.fromBufferAttribute(pos, i);
    // Two projections of the same 3D point through 2D noise. Duplicated
    // vertices at a shared corner get identical input and therefore identical
    // displacement, which keeps the mesh watertight.
    const a = nA.fbm(_v.x * 1.6 + _v.y * 0.7, _v.z * 1.6 - _v.y * 0.5, 3);
    const b = nB.fbm(_v.z * 3.1 - _v.x * 0.4, _v.y * 3.1 + _v.x * 0.8, 2);
    const r = 1 + a * 0.34 + b * 0.13;
    _v.multiplyScalar(r);
    _v.y *= squash;
    if (_v.y < -0.34) _v.y = -0.34 + (_v.y + 0.34) * 0.12;
    pos.setXYZ(i, _v.x, _v.y, _v.z);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  // maxWeldAngle 180: a boulder wants a continuous hull even across its hard
  // facet creases, or the outline tears open at every edge.
  return finalizeGeometry(geo, {
    tolerance: 1e-3,
    maxWeldAngle: 180,
    curvatureGain: 1.15,
    ao: true,
    aoStrength: 0.55,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// The scatter system
// ─────────────────────────────────────────────────────────────────────────────

export interface ScatterOptions {
  source: ScatterSource;
  /** Half-buried domes the rock-garden carve already put in the heightfield. */
  boulders?: CarvedBoulder[];
  /** Scale every population. 0 disables scatter entirely. */
  quality?: number;
}

const ROCK_VARIANTS = 3;

/**
 * Rocks, boulders and stone litter.
 *
 * Three populations with different jobs:
 *   • FIELD ROCKS — the mid-distance texture of the mountain. Everywhere the
 *     ground is rock or scree, sparse, clumped into fields.
 *   • TRAIL ROCKS — dense, small, close to the course. These are what the
 *     rider actually reads as "technical", and they are deliberately kept just
 *     outside the ribbon so they frame the line rather than block it.
 *   • CARVED BOULDERS — the rock garden. These are NOT scattered; their
 *     positions come from the heightfield carve, so the visual rock and the
 *     physical bump are the same rock.
 */
export class Scatter {
  readonly object: Object3D = new Group();
  readonly fields: InstanceField[] = [];
  readonly pools: ScatterPool[] = [];
  private groups: InstanceGroup[] = [];

  constructor(opts: ScatterOptions) {
    this.object.name = 'terrain-scatter';
    const quality = opts.quality ?? 1;
    if (quality <= 0) return;

    // ── Geometry ────────────────────────────────────────────────────────────
    const lod0: BufferGeometry[] = [];
    for (let v = 0; v < ROCK_VARIANTS; v++) {
      lod0.push(buildRockGeometry(`rock${v}`, 1, 0.72 + v * 0.09));
    }
    const lod1 = buildRockGeometry('rockfar', 0, 0.78);

    // ── Field rocks ─────────────────────────────────────────────────────────
    const fieldRule = makeRule({
      spacing: 11,
      // Rock and scree carry almost all of it; a little on dirt, none on grass
      // meadow (a meadow with rocks all over it is a scree slope).
      zoneWeight: [0.85, 0.16, 0.05, 0.72, 0.10, 0.0, 0.0],
      slopeMin: 0,
      slopeMax: 0.86,
      slopeFeather: 0.16,
      routeClear: 0,
      routeMax: Infinity,
      erosionBias: -0.55,
      clumping: 0.75,
      clumpScale: 90,
      density: 0.60 * quality,
      scaleMin: 0.7,
      scaleMax: 3.4,
      alignToNormal: 0.55,
      sink: -0.30,
      variants: ROCK_VARIANTS,
      maxCount: Math.round(26000 * quality),
    });
    const fieldRocks = generateInstances(opts.source, fieldRule, 'rocks:field');

    // ── Trail rocks ─────────────────────────────────────────────────────────
    const trailRule = makeRule({
      spacing: 4.2,
      zoneWeight: [0.9, 0.55, 0.14, 0.8, 0.05, 0.0, 0.0],
      slopeMin: 0,
      slopeMax: 0.72,
      slopeFeather: 0.14,
      routeClear: 9,
      routeMax: 70,
      erosionBias: -0.3,
      clumping: 0.55,
      clumpScale: 34,
      density: 0.34 * quality,
      scaleMin: 0.28,
      scaleMax: 1.05,
      alignToNormal: 0.75,
      sink: -0.22,
      variants: ROCK_VARIANTS,
      maxCount: Math.round(16000 * quality),
    });
    const trailRocks = generateInstances(opts.source, trailRule, 'rocks:trail');

    // ── Carved boulders ─────────────────────────────────────────────────────
    const carved = this.buildCarvedField(opts.boulders ?? [], opts.source);

    this.fields.push(fieldRocks, trailRocks, carved);

    // ── Pools ───────────────────────────────────────────────────────────────
    this.pools.push(
      this.makeRockPool(fieldRocks, lod0, lod1, 'rock-field', 3400, 1600, 200, 480),
      this.makeRockPool(trailRocks, lod0, lod1, 'rock-trail', 2600, 900, 150, 300),
      this.makeRockPool(carved, lod0, lod1, 'rock-carved', 700, 400, 220, 420),
    );
  }

  /**
   * Turn the rock garden's heightfield domes into visible stone.
   *
   * The carve raised a smooth ellipsoid cap in the terrain at each of these
   * positions; the mesh placed here sits in the same spot at the same radius,
   * so the silhouette the rider reads and the bump the wheel hits are the same
   * object. Rock gardens where those two disagree are the single most
   * infuriating thing a mountain-bike game can do.
   */
  private buildCarvedField(boulders: CarvedBoulder[], src: ScatterSource): InstanceField {
    const field = new InstanceField(Math.max(1, boulders.length));
    const rng = new Rng(WORLD_SEED).fork('rocks:carved');
    for (const b of boulders) {
      const k = field.count;
      src.normalAt(b.x, b.z, _n);
      _tilt.copy(_up).lerp(_n, 0.5).normalize();
      _q.setFromUnitVectors(_up, _tilt);
      _qy.setFromAxisAngle(_up, b.rotation);
      _q.multiply(_qy);

      field.pos[k * 3] = b.x;
      // Sunk by a third of its radius: the carve already raised the ground, so
      // the mesh only has to account for its own base.
      field.pos[k * 3 + 1] = src.heightAt(b.x, b.z) - b.radius * 0.34;
      field.pos[k * 3 + 2] = b.z;
      field.quat[k * 4] = _q.x;
      field.quat[k * 4 + 1] = _q.y;
      field.quat[k * 4 + 2] = _q.z;
      field.quat[k * 4 + 3] = _q.w;
      field.scale[k] = b.radius;
      const v = 0.92 + rng.next() * 0.16;
      field.tint[k * 3] = v;
      field.tint[k * 3 + 1] = v;
      field.tint[k * 3 + 2] = v * 1.01;
      field.phase[k] = rng.next() * Math.PI * 2;
      field.variant[k] = rng.int(0, ROCK_VARIANTS - 1);
      field.alive[k] = 1;
      field.count = k + 1;
    }
    field.index(96);
    return field;
  }

  private makeRockPool(
    field: InstanceField,
    lod0: BufferGeometry[],
    lod1: BufferGeometry,
    name: string,
    maxNear: number,
    maxFar: number,
    nearRange: number,
    farRange: number,
  ): ScatterPool {
    const nearGroups: InstanceGroup[] = [];
    for (let v = 0; v < lod0.length; v++) {
      const geo = lod0[v].clone();
      const built = buildInstanced(geo, this.object, {
        preset: 'rock',
        idName: 'rock',
        max: maxNear,
        name: `${name}:lod0:${v}`,
        vertexAo: true,
        outline: true,
        outlineWidth: 0.011,
      });
      const g = new InstanceGroup(built.mesh, maxNear);
      if (built.hull) g.addSibling(built.hull);
      nearGroups.push(g);
      this.groups.push(g);
    }

    // The far tier collapses all three variants onto one shape. At 260m a
    // twenty-triangle lump and a hand-sculpted boulder are the same four
    // pixels, and the draw call saved is worth more than the difference.
    const farGeo = lod1.clone();
    const farBuilt = buildInstanced(farGeo, this.object, {
      preset: 'rock',
      idName: 'rock',
      max: maxFar,
      name: `${name}:lod1`,
      vertexAo: true,
      outline: false,
    });
    farBuilt.mesh.castShadow = false;
    farBuilt.mesh.userData.skipShadow = true;
    farBuilt.mesh.userData.skipPrepass = true;
    const farGroup = new InstanceGroup(farBuilt.mesh, maxFar);
    this.groups.push(farGroup);

    const fade = Math.max(20, nearRange * 0.18);
    return new ScatterPool(field, [
      { groups: nearGroups, near: 0, far: nearRange, fade },
      { groups: [farGroup], near: nearRange - fade, far: farRange, fade },
    ]);
  }

  /** Remove everything the trail ribbon now occupies. */
  cullCarve(carve: TrackCarve, margin = 1.1): void {
    for (const field of this.fields) {
      for (let i = 0; i < carve.points.length; i++) {
        const p = carve.points[i];
        field.cullDisc(p.x, p.z, carve.halfWidths[i] + margin);
      }
    }
    for (const pool of this.pools) pool.invalidate();
  }

  update(camX: number, camZ: number): void {
    for (const pool of this.pools) pool.update(camX, camZ);
  }

  dispose(): void {
    for (const g of this.groups) g.dispose();
    this.groups.length = 0;
    this.pools.length = 0;
    this.fields.length = 0;
    this.object.clear();
  }
}
