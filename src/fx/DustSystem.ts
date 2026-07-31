/**
 * DustSystem — hard-edged cel dust.
 *
 * The single most common way a stylised game betrays itself is its particles.
 * Everything else can be banded, inked and flattened, and then a soft radial
 * alpha sprite drifts across the frame and the whole illusion collapses: soft
 * gradients are a *photographic* signal, and there is exactly one of them in
 * the picture.
 *
 * So none of this is a soft sprite. Every puff here is:
 *
 *  • a CLOSED DRAWN SHAPE with an ink contour (dustPuffSet() from
 *    GeneratedTextures — cartoon-cloud lobes with a black ring), never a blur;
 *  • LIT with the same banded ramp as everything else in the world, using a
 *    sphere-impostor normal reconstructed in the fragment shader, so a puff on
 *    the shadow side of the ridge goes violet exactly like the rock does;
 *  • ANIMATED ON A STEP GRID. Scale, rotation and opacity are all quantised to
 *    4–7 discrete holds across the puff's life, with a per-instance phase
 *    offset so they don't all cut on the same frame. An animator does not fade
 *    smoke smoothly — they hold a drawing, then cut to the next one. Continuous
 *    interpolation is the tell.
 *
 * The whole system is ONE draw call: an InstancedBufferGeometry quad with
 * per-instance origin / velocity / life / shape / tint, integrated analytically
 * in the vertex shader (closed-form linear drag), so a burst costs a handful of
 * float writes and zero per-frame CPU simulation.
 *
 * Note on the mesh type: this is an InstancedBufferGeometry on a plain Mesh
 * rather than an InstancedMesh. three renders both with drawElementsInstanced,
 * but InstancedMesh would force a 16-float instanceMatrix per particle that we
 * would never read — the billboard basis comes from the view matrix and the
 * transform is three floats. Same one draw call, a third of the bandwidth.
 */

import {
  CanvasTexture,
  Color,
  DoubleSide,
  DynamicDrawUsage,
  Float32BufferAttribute,
  GLSL3,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  LinearFilter,
  LinearMipmapLinearFilter,
  Mesh,
  NormalBlending,
  Object3D,
  PerspectiveCamera,
  ShaderMaterial,
  Sphere,
  SRGBColorSpace,
  Texture,
  Vector2,
  Vector3,
} from 'three';

import { SurfaceKind, type SurfaceProperties } from '../game/Contracts';
import { Rng } from '../core/RNG';
import { clamp, clamp01 } from '../core/MathX';
import { dustPuffSet } from '../npr/GeneratedTextures';
import { globalUniformBlock } from '../npr/NprGlobals';
import { INK, RAMPS } from '../npr/Palette';
import { GLSL_COMMON, GLSL_FOG, GLSL_FRAG_OUT, GLSL_GLOBAL_UNIFORMS, GLSL_SHADOW } from '../npr/ShaderChunks';

// ─────────────────────────────────────────────────────────────────────────────
// Module-scope scratch. Nothing in an update path may allocate.
// ─────────────────────────────────────────────────────────────────────────────
const _n = new Vector3();
const _t = new Vector3();
const _b = new Vector3();
const _d = new Vector3();
const _v = new Vector3();
const _camPos = new Vector3();

/** Puff kinds. The kind selects step count, drag and buoyancy in the shader. */
export const DUST_SKID = 0;
export const DUST_SPRAY = 1;
export const DUST_PLUME = 2;

// ─────────────────────────────────────────────────────────────────────────────
// Shared GLSL: the billboard basis, pulled straight out of the view matrix.
// ─────────────────────────────────────────────────────────────────────────────
/**
 * viewMatrix is the world->view transform, so the ROWS of its upper 3x3 are the
 * camera's world-space axes. Reading them here means a camera-facing quad costs
 * no uniforms and no CPU work at all, and can never drift out of sync with the
 * camera the frame is actually being rendered from.
 *
 * camFwdWS points from the scene TOWARD the camera (+Z in view space), which is
 * exactly the direction a sphere-impostor normal should bulge.
 */
export const GLSL_BILLBOARD = /* glsl */ `
  vec3 camRightWS() { return vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]); }
  vec3 camUpWS()    { return vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]); }
  vec3 camFwdWS()   { return vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]); }
`;

/**
 * The same anti-aliased hard band edge the whole game uses, restated locally.
 *
 * It is deliberately a copy of bandStep() from GLSL_CEL_CORE rather than an
 * import of that chunk: GLSL_CEL_CORE also declares evalRamp/applyHatch, which
 * would drag in the entire GLSL_RAMP_UNIFORMS block (uBandColor[4], the hatch
 * atlas, the matcaps) that a two-band particle has no use for. The maths is
 * identical on purpose — dust must alias exactly the way terrain aliases, or it
 * reads as a different material system laid over the top.
 */
export const GLSL_BAND_STEP = /* glsl */ `
  float fxBandStep(float x, float threshold, float softness) {
    float w = max(fwidth(x) * 0.75, softness);
    return smoothstep(threshold - w, threshold + w, x);
  }
`;

// ─────────────────────────────────────────────────────────────────────────────
// Instance pool
// ─────────────────────────────────────────────────────────────────────────────

export interface AttribSpec {
  name: string;
  /** Components per instance: 1, 2, 3 or 4. */
  size: number;
}

/**
 * A fixed-capacity ring of instanced quads.
 *
 * Emission writes floats into pre-allocated Float32Arrays and records the
 * touched range; flush() uploads only that range. Zero allocation per emission,
 * and a 20-puff burst uploads ~1.4 kB rather than the whole 80 kB buffer.
 */
export class QuadParticlePool {
  readonly capacity: number;
  readonly geometry: InstancedBufferGeometry;
  private readonly arrays = new Map<string, Float32Array>();
  private readonly attribs = new Map<string, InstancedBufferAttribute>();

  private head = 0;
  /** High-water mark — instanceCount only ever needs to cover what we've used. */
  private used = 0;
  private lo = Number.POSITIVE_INFINITY;
  private hi = -1;

  constructor(capacity: number, specs: AttribSpec[]) {
    this.capacity = capacity;

    const geo = new InstancedBufferGeometry();
    // Unit quad in XY, centred. Corners are +/-1 so the vertex shader can treat
    // position.xy directly as the billboard offset.
    geo.setAttribute('position', new Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3));
    geo.setAttribute('uv', new Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
    geo.setIndex([0, 1, 2, 0, 2, 3]);

    for (const s of specs) {
      const arr = new Float32Array(capacity * s.size);
      const attr = new InstancedBufferAttribute(arr, s.size);
      attr.setUsage(DynamicDrawUsage);
      geo.setAttribute(s.name, attr);
      this.arrays.set(s.name, arr);
      this.attribs.set(s.name, attr);
    }

    geo.instanceCount = 0;
    // Particles live wherever the action is; a static bounding sphere would be
    // a lie. We cull them by emission distance on the CPU instead.
    geo.boundingSphere = new Sphere(new Vector3(), 1e6);
    this.geometry = geo;
  }

  array(name: string): Float32Array {
    const a = this.arrays.get(name);
    if (!a) throw new Error(`[fx] no particle attribute "${name}"`);
    return a;
  }

  /** Claim the next ring slot. Oldest live particle is silently recycled. */
  allocRing(): number {
    const i = this.head;
    this.head = (this.head + 1) % this.capacity;
    if (i >= this.used) this.used = i + 1;
    this.touch(i);
    return i;
  }

  touch(i: number): void {
    if (i < this.lo) this.lo = i;
    if (i > this.hi) this.hi = i;
  }

  markAll(): void {
    this.lo = 0;
    this.hi = this.capacity - 1;
  }

  /**
   * Publish a live instance count directly. Ring-allocated systems get this
   * for free from allocRing(); systems that own their own free list (Debris)
   * set the high-water mark themselves.
   */
  setUsed(n: number): void {
    this.used = Math.min(Math.max(n, this.used), this.capacity);
  }

  /** Upload the dirty range and publish the live instance count. */
  flush(): void {
    this.geometry.instanceCount = this.used;
    if (this.hi < this.lo) return;
    const count = this.hi - this.lo + 1;
    for (const attr of this.attribs.values()) {
      attr.clearUpdateRanges();
      attr.addUpdateRange(this.lo * attr.itemSize, count * attr.itemSize);
      attr.needsUpdate = true;
    }
    this.lo = Number.POSITIVE_INFINITY;
    this.hi = -1;
  }

  dispose(): void {
    this.geometry.dispose();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Puff atlas
// ─────────────────────────────────────────────────────────────────────────────

let _puffAtlas: Texture | null = null;

/**
 * Pack the four generated puff variants into one 2x2 atlas so the whole dust
 * system stays a single draw call.
 *
 * Each 128px puff is blitted at NATIVE SIZE into the centre of a 256px cell.
 * That padding matters twice over: it stops mip generation bleeding one puff's
 * ink ring into its neighbour, and it means no resampling happens — the puff's
 * alpha is a hard binary cut and any scaling here would soften the contour,
 * which is the one thing this system exists to avoid.
 */
export function puffAtlas(): Texture {
  if (_puffAtlas) return _puffAtlas;

  const CELL = 256;
  const SRC = 128;
  const cv = document.createElement('canvas');
  cv.width = CELL * 2;
  cv.height = CELL * 2;
  const ctx = cv.getContext('2d')!;
  ctx.clearRect(0, 0, cv.width, cv.height);

  const puffs = dustPuffSet();
  for (let i = 0; i < 4; i++) {
    const src = puffs[i].image as HTMLCanvasElement;
    const cx = (i % 2) * CELL + (CELL - SRC) * 0.5;
    const cy = Math.floor(i / 2) * CELL + (CELL - SRC) * 0.5;
    ctx.drawImage(src, cx, cy);
  }

  const tex = new CanvasTexture(cv);
  tex.name = 'fx:puff-atlas';
  tex.colorSpace = SRGBColorSpace;
  tex.minFilter = LinearMipmapLinearFilter;
  tex.magFilter = LinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = 4;
  _puffAtlas = tex;
  return tex;
}

// ─────────────────────────────────────────────────────────────────────────────
// Surface tinting
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Dust colour per surface, taken from the committed ramps — nothing invented.
 *
 * The tint is HUE-NORMALISED (divided by its own luminance) before it reaches
 * the shader, because it multiplies over a ramp that already carries the value
 * structure. Multiplying a mid-value band by a mid-value colour twice is how
 * you get mud; normalising means the surface decides the hue and the ramp keeps
 * deciding the light.
 */
const _tintCache = new Map<SurfaceKind, Color>();
function normalisedTint(src: Color): Color {
  const c = src.clone();
  const l = Math.max(0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b, 1e-3);
  c.multiplyScalar(1 / l);
  c.r = Math.min(c.r, 1.7);
  c.g = Math.min(c.g, 1.7);
  c.b = Math.min(c.b, 1.7);
  return c;
}

export function dustTintFor(kind: SurfaceKind): Color {
  let c = _tintCache.get(kind);
  if (c) return c;
  switch (kind) {
    case SurfaceKind.Rock: c = normalisedTint(RAMPS.rock.colors[2]); break;
    case SurfaceKind.Dirt: c = normalisedTint(RAMPS.dirt.colors[2]); break;
    case SurfaceKind.Grass: c = normalisedTint(RAMPS.grass.colors[2]); break;
    case SurfaceKind.Scree: c = normalisedTint(RAMPS.scree.colors[2]); break;
    case SurfaceKind.Snow: c = normalisedTint(RAMPS.snow.colors[1]); break;
    case SurfaceKind.Water: c = normalisedTint(RAMPS.water.colors[2]); break;
    case SurfaceKind.Trail: c = normalisedTint(RAMPS.trail.colors[2]); break;
    default: c = normalisedTint(RAMPS.dirt.colors[2]); break;
  }
  _tintCache.set(kind, c);
  return c;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shaders
// ─────────────────────────────────────────────────────────────────────────────

const DUST_VERT = /* glsl */ `
  precision highp float;
  ${GLSL_COMMON}
  ${GLSL_GLOBAL_UNIFORMS}
  ${GLSL_BILLBOARD}

  // The FX clock is NOT uTime. uTime is engine-elapsed and keeps running during
  // an impact-frame freeze; ages driven by it would let dust drift through a
  // hold that is supposed to stop the world dead. uFxTime advances by the
  // SCALED dt, so a freeze freezes the smoke too.
  uniform float uFxTime;
  uniform float uSizeScale;

  in vec3 aOrigin;
  in vec3 aVel;
  in vec4 aParams;   // x birthTime, y 1/life, z seed01, w kind
  in vec4 aShape;    // x startScale, y growth, z spinTotal, w spin0
  in vec3 aTint;

  out vec2  vUv;
  out vec2  vQuad;
  out vec3  vTint;
  out float vAlpha;
  out vec3  vWorldPos;
  out float vViewDist;

  void main() {
    // Defaults so a culled particle never leaves a varying undefined.
    vUv = vec2(0.0);
    vQuad = vec2(0.0);
    vTint = aTint;
    vAlpha = 0.0;
    vWorldPos = aOrigin;
    vViewDist = 1.0;

    float t = uFxTime - aParams.x;
    float age = t * aParams.y;
    if (age < 0.0 || age >= 1.0) {
      // Collapse dead instances outside the clip volume. Cheaper than any
      // CPU-side compaction and completely branch-free downstream.
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      return;
    }

    float kind = aParams.w;
    float isPlume = step(1.5, kind);
    float isSpray = step(0.5, kind) * (1.0 - isPlume);

    // ── The step grid ────────────────────────────────────────────────────────
    // Scale, rotation and opacity all read from the SAME quantised age, so each
    // puff visibly holds a drawing and then cuts to the next one. The per-
    // instance phase offset is what stops a whole burst cutting in unison,
    // which would read as a strobe rather than as animation.
    float steps = mix(mix(5.0, 4.0, isSpray), 7.0, isPlume);
    float phase = fract(aParams.z * 17.13);
    float q = clamp(floor(age * steps + phase * 0.7) / steps, 0.0, 1.0);

    // ── Motion ───────────────────────────────────────────────────────────────
    // Closed-form linear drag: p = p0 + v0 * (1 - e^-kt) / k. Analytic rather
    // than integrated so there is no per-frame CPU cost and no drift between
    // a 60Hz and a 120Hz machine.
    float drag = mix(2.9, 1.15, isPlume);
    vec3 p = aOrigin + aVel * ((1.0 - exp(-drag * t)) / drag);
    // Dust lifts on its own turbulence and then gives up.
    float rise = mix(0.45, 1.25, aParams.z) * mix(0.7, 1.5, isPlume);
    p.y += rise * t * (1.0 - 0.55 * age);

    float sz = aShape.x * (1.0 + aShape.y * q) * uSizeScale;
    float ang = aShape.w + aShape.z * q;

    // ── Opacity ──────────────────────────────────────────────────────────────
    // Squared so the puff sits at full opacity for most of its life and then
    // leaves quickly, then ceil()-quantised to four hard levels: 1, .75, .5, .25
    // and cut. Never a smooth ramp.
    float fade = 1.0 - q;
    fade *= fade;
    vAlpha = ceil(clamp(fade, 0.0, 1.0) * 4.0) * 0.25;

    // ── Billboard ────────────────────────────────────────────────────────────
    vec3 R = camRightWS();
    vec3 U = camUpWS();
    float ca = cos(ang), sa = sin(ang);
    vec2 rq = vec2(position.x * ca - position.y * sa, position.x * sa + position.y * ca);
    vec3 wpos = p + (R * rq.x + U * rq.y) * sz;

    // vQuad is the ROTATED offset: the impostor normal must follow the puff's
    // actual screen footprint, while vUv stays unrotated so the drawn shape
    // spins with the sprite.
    vQuad = rq;

    float tile = floor(fract(aParams.z * 4.0 + 0.13) * 3.999);
    vec2 tileOff = vec2(mod(tile, 2.0), floor(tile * 0.5)) * 0.5;
    vUv = tileOff + uv * 0.5;

    vTint = aTint;
    vWorldPos = wpos;
    vViewDist = length(wpos - uCameraPos);

    gl_Position = projectionMatrix * viewMatrix * vec4(wpos, 1.0);
  }
`;

const DUST_FRAG = /* glsl */ `
  precision highp float;
  ${GLSL_COMMON}
  ${GLSL_GLOBAL_UNIFORMS}
  ${GLSL_SHADOW}
  ${GLSL_FOG}
  ${GLSL_BILLBOARD}
  ${GLSL_BAND_STEP}
  ${GLSL_FRAG_OUT}

  uniform sampler2D uAtlas;
  uniform vec3  uBand0;
  uniform vec3  uBand1;
  uniform vec3  uBand2;
  uniform vec2  uBandT;
  uniform vec3  uInk;
  uniform float uOpacity;

  in vec2  vUv;
  in vec2  vQuad;
  in vec3  vTint;
  in float vAlpha;
  in vec3  vWorldPos;
  in float vViewDist;

  void main() {
    vec4 tex = texture(uAtlas, vUv);
    // The puff's alpha is a hard binary cut by construction. Discarding rather
    // than blending the edge is what keeps the contour a drawn line.
    if (tex.a < 0.5) discard;

    float texLum = luma(tex.rgb);
    // The generated puff carries its own ink ring (near-black) and a single
    // hard shadow lobe (mid grey). We keep the ring as ink and use the lobe as
    // a BIAS on the lighting term rather than as a second multiply, so the
    // drawn shading and the world lighting agree instead of stacking.
    float isInk = 1.0 - step(0.035, texLum);
    float drawnShade = (1.0 - isInk) * (1.0 - smoothstep(0.10, 0.60, texLum));

    // Sphere impostor: reconstruct a bulging normal from the quad footprint so
    // the puff has a real terminator instead of reading as a flat decal.
    vec2 qd = vQuad;
    float r2 = min(dot(qd, qd), 1.0);
    float nz = sqrt(max(0.0, 1.0 - r2));
    vec3 R = camRightWS();
    vec3 U = camUpWS();
    vec3 F = camFwdWS();
    // The 0.35 flattens the bulge. A full unit sphere puts the terminator right
    // on the silhouette and the puff reads as a ball bearing; drawn smoke is
    // much flatter than that.
    vec3 N = normalize(R * qd.x + U * qd.y + F * (nz * 0.9 + 0.35));

    float ndlRaw = dot(N, uSunDir);
    float ndl = ndlRaw * 0.5 + 0.5;
    ndl -= drawnShade * 0.30;

    float shadow = sunShadow(vWorldPos, N, vViewDist, saturate1(ndlRaw));
    ndl *= mix(0.62, 1.0, shadow);

    vec3 col = uBand0;
    col = mix(col, uBand1, fxBandStep(ndl, uBandT.x, 0.014));
    col = mix(col, uBand2, fxBandStep(ndl, uBandT.y, 0.014));
    col *= vTint;
    col = mix(col, uInk, isInk);

    vec3 viewDir = normalize(vWorldPos - uCameraPos);
    col = applyQuantizedFog(col, vViewDist, viewDir, gl_FragCoord.xy);

    fragColor = vec4(col, vAlpha * uOpacity);
  }
`;

// ─────────────────────────────────────────────────────────────────────────────
// DustSystem
// ─────────────────────────────────────────────────────────────────────────────

export interface DustOptions {
  capacity?: number;
  rng?: Rng;
  /** Emission beyond this distance from the camera is skipped entirely. */
  cullDistance?: number;
}

export class DustSystem {
  readonly object: Object3D;
  readonly mesh: Mesh;
  readonly material: ShaderMaterial;

  private pool: QuadParticlePool;
  private rng: Rng;
  private fxTime = 0;
  private cullDistanceSq: number;
  /** Emission culling is disabled until update() has told us where the camera is. */
  private haveCamera = false;

  // Direct handles on the instance arrays, so emission is float writes only.
  private aOrigin: Float32Array;
  private aVel: Float32Array;
  private aParams: Float32Array;
  private aShape: Float32Array;
  private aTint: Float32Array;

  /** Live-ish count for the perf overlay. Approximate: ring slots issued. */
  emittedThisFrame = 0;

  constructor(opts: DustOptions = {}) {
    const capacity = opts.capacity ?? 1100;
    this.rng = opts.rng ?? new Rng('fx:dust');
    const cull = opts.cullDistance ?? 170;
    this.cullDistanceSq = cull * cull;

    this.pool = new QuadParticlePool(capacity, [
      { name: 'aOrigin', size: 3 },
      { name: 'aVel', size: 3 },
      { name: 'aParams', size: 4 },
      { name: 'aShape', size: 4 },
      { name: 'aTint', size: 3 },
    ]);

    this.aOrigin = this.pool.array('aOrigin');
    this.aVel = this.pool.array('aVel');
    this.aParams = this.pool.array('aParams');
    this.aShape = this.pool.array('aShape');
    this.aTint = this.pool.array('aTint');

    // Every slot starts dead (birth far in the past, life 1) so nothing renders
    // before the first emission.
    for (let i = 0; i < capacity; i++) this.aParams[i * 4 + 1] = 1;

    // Bands come from the scree ramp: the most neutral, least chromatic of the
    // terrain ramps, so a per-surface hue tint can push it anywhere without
    // fighting a colour that was already committed.
    const b0 = RAMPS.scree.colors[0];
    const b1 = RAMPS.scree.colors[2];
    const b2 = RAMPS.scree.colors[3];

    this.material = new ShaderMaterial({
      glslVersion: GLSL3,
      uniforms: {
        ...globalUniformBlock(),
        uFxTime: { value: 0 },
        // The atlas cell is twice the puff's footprint (the padding that keeps
        // mips from bleeding), so every quad is scaled x2 to compensate.
        uSizeScale: { value: 2.0 },
        uAtlas: { value: puffAtlas() },
        uBand0: { value: b0.clone() },
        uBand1: { value: b1.clone() },
        uBand2: { value: b2.clone() },
        // Band thresholds on the half-Lambert term. Only two edges: dust is a
        // three-tone shape in every animated film that has ever drawn it.
        uBandT: { value: new Vector2(0.30, 0.66) },
        uInk: { value: INK.clone() },
        uOpacity: { value: 0.94 },
      },
      vertexShader: DUST_VERT,
      fragmentShader: DUST_FRAG,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: NormalBlending,
      side: DoubleSide,
    });
    this.material.name = 'fx:dust';

    this.mesh = new Mesh(this.pool.geometry, this.material);
    this.mesh.name = 'fx:dust';
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    // Drawn after all opaque geometry and after the outline hulls.
    this.mesh.renderOrder = 20;
    // Signals to the post pipeline that this must not enter the G-buffer
    // prepass: dust has no meaningful normal or material id, and letting it
    // write there would make the Sobel pass draw interior lines around smoke.
    this.mesh.userData.fxTransparent = true;
    this.mesh.userData.nprSkipPrepass = true;

    this.object = new Object3D();
    this.object.name = 'fx:dust:group';
    this.object.add(this.mesh);
  }

  /** Advance the FX clock. `dt` must be the SCALED dt so freezes freeze dust. */
  update(dt: number, camera: PerspectiveCamera): void {
    this.fxTime += dt;
    this.material.uniforms.uFxTime.value = this.fxTime;
    camera.getWorldPosition(_camPos);
    this.haveCamera = true;
    this.pool.flush();
    this.emittedThisFrame = 0;
  }

  /** Current FX clock, for callers that stamp their own births. */
  get time(): number {
    return this.fxTime;
  }

  private tooFar(x: number, y: number, z: number): boolean {
    if (!this.haveCamera) return false;
    const dx = x - _camPos.x;
    const dy = y - _camPos.y;
    const dz = z - _camPos.z;
    return dx * dx + dy * dy + dz * dz > this.cullDistanceSq;
  }

  /**
   * Write one puff. All parameters are already resolved — this is the only
   * place that touches the instance arrays.
   */
  private write(
    px: number, py: number, pz: number,
    vx: number, vy: number, vz: number,
    life: number, kind: number,
    scale: number, growth: number, spinTotal: number,
    tint: Color,
  ): void {
    const i = this.pool.allocRing();
    const i3 = i * 3;
    const i4 = i * 4;

    this.aOrigin[i3] = px;
    this.aOrigin[i3 + 1] = py;
    this.aOrigin[i3 + 2] = pz;

    this.aVel[i3] = vx;
    this.aVel[i3 + 1] = vy;
    this.aVel[i3 + 2] = vz;

    this.aParams[i4] = this.fxTime;
    this.aParams[i4 + 1] = 1 / Math.max(life, 0.05);
    this.aParams[i4 + 2] = this.rng.next();
    this.aParams[i4 + 3] = kind;

    this.aShape[i4] = scale;
    this.aShape[i4 + 1] = growth;
    this.aShape[i4 + 2] = spinTotal;
    this.aShape[i4 + 3] = this.rng.range(0, Math.PI * 2);

    this.aTint[i3] = tint.r;
    this.aTint[i3 + 1] = tint.g;
    this.aTint[i3 + 2] = tint.b;

    this.emittedThisFrame++;
  }

  /** Build an orthonormal basis around `n` into the module scratch _t / _b. */
  private basis(n: Vector3): void {
    _n.copy(n);
    if (_n.lengthSq() < 1e-8) _n.set(0, 1, 0);
    else _n.normalize();
    // Pick the least-aligned cardinal axis so the cross product never degenerates.
    if (Math.abs(_n.y) < 0.9) _t.set(0, 1, 0);
    else _t.set(1, 0, 0);
    _b.crossVectors(_n, _t).normalize();
    _t.crossVectors(_b, _n).normalize();
  }

  // ── Public emission ────────────────────────────────────────────────────────

  /**
   * A discrete burst — the pop of dust off a skid, or the radial spray of a
   * landing. `amount` 0..1 selects between the two: anything above 0.55 becomes
   * a landing spray (faster, denser, flatter, shorter-lived), below it a skid
   * puff (slower, fewer, larger, lingering).
   */
  burst(position: Vector3, normal: Vector3, velocity: Vector3, amount: number, surface: SurfaceProperties): void {
    const a = clamp01(amount);
    if (a <= 0.01) return;
    if (this.tooFar(position.x, position.y, position.z)) return;

    const dustScale = surface.dustAmount;
    if (dustScale <= 0.02) return;

    const spray = a > 0.55;
    const kind = spray ? DUST_SPRAY : DUST_SKID;
    const tint = dustTintFor(surface.kind);

    // Count scales with both the hit strength and how dusty the ground is, but
    // is capped hard — 24 puffs is already a wall of smoke at these sizes, and
    // the ring only holds 1100.
    const count = clamp(Math.round((2 + a * 16) * dustScale), 1, 24);

    this.basis(normal);
    const vy = velocity.y;
    const vLen = velocity.length();

    for (let i = 0; i < count; i++) {
      const ang = this.rng.range(0, Math.PI * 2);
      const radial = spray ? this.rng.range(0.75, 1.0) : this.rng.range(0.35, 0.95);
      const up = spray ? this.rng.range(0.18, 0.55) : this.rng.range(0.35, 0.95);
      const speed = (spray ? 2.4 + a * 8.5 : 1.1 + a * 3.4) * this.rng.range(0.62, 1.35);

      // Radial component in the surface plane, plus a lift along the normal.
      _d.copy(_t).multiplyScalar(Math.cos(ang) * radial)
        .addScaledVector(_b, Math.sin(ang) * radial)
        .addScaledVector(_n, up);
      _d.multiplyScalar(speed);

      // A landing throws its dust forward with the bike; a skid drags it back.
      _d.addScaledVector(velocity, spray ? 0.16 : -0.10);
      // Never let the seed velocity drive a puff into the ground.
      if (_d.y < 0 && vy < 0) _d.y *= 0.25;

      // Emission point jittered across the contact patch, not from a point.
      const jr = this.rng.range(0, spray ? 0.55 : 0.30);
      const ja = this.rng.range(0, Math.PI * 2);
      _v.copy(position)
        .addScaledVector(_t, Math.cos(ja) * jr)
        .addScaledVector(_b, Math.sin(ja) * jr)
        .addScaledVector(_n, this.rng.range(0.02, 0.22));

      const life = spray ? this.rng.range(0.55, 0.95) : this.rng.range(0.85, 1.35);
      const scale = (spray ? this.rng.range(0.22, 0.42) : this.rng.range(0.26, 0.48)) * (0.8 + dustScale * 0.35);
      const growth = spray ? this.rng.range(1.9, 2.8) : this.rng.range(1.2, 1.9);
      const spin = this.rng.signed() * (spray ? 1.6 : 0.9);

      this.write(_v.x, _v.y, _v.z, _d.x, _d.y, _d.z, life, kind, scale, growth, spin, tint);
    }

    // A fast landing on loose ground also throws a slower, much larger plume
    // behind the contact — the tail that sells the impact after the spray has
    // already gone.
    if (spray && dustScale > 0.9 && vLen > 6) {
      const n2 = clamp(Math.round(a * 4 * dustScale), 1, 6);
      for (let i = 0; i < n2; i++) {
        _d.copy(velocity).multiplyScalar(-0.16);
        _d.addScaledVector(_n, this.rng.range(0.6, 1.6));
        _d.addScaledVector(_t, this.rng.signed() * 0.9);
        this.write(
          position.x + this.rng.signed() * 0.5, position.y + 0.15, position.z + this.rng.signed() * 0.5,
          _d.x, _d.y, _d.z,
          this.rng.range(1.5, 2.3), DUST_PLUME,
          this.rng.range(0.55, 0.95) * dustScale, this.rng.range(1.6, 2.4),
          this.rng.signed() * 0.55, tint,
        );
      }
    }
  }

  /**
   * Continuous emission from a rolling or sliding contact.
   *
   * `rate` is puffs per second before the surface multiplier. The fractional
   * remainder is resolved stochastically against the seeded FX stream rather
   * than carried in per-call-site state: the contract has no call-site id, and
   * a shared accumulator would silently mis-rate the moment a second wheel
   * started calling in. Stochastic is correct in expectation, exactly
   * reproducible for a given call sequence, and needs no state at all.
   */
  trail(
    position: Vector3,
    normal: Vector3,
    velocity: Vector3,
    rate: number,
    dt: number,
    surface: SurfaceProperties,
  ): void {
    if (rate <= 0 || dt <= 0) return;
    if (this.tooFar(position.x, position.y, position.z)) return;

    const dustScale = surface.dustAmount;
    if (dustScale <= 0.02) return;

    const expected = rate * dustScale * dt;
    let count = Math.floor(expected);
    if (this.rng.next() < expected - count) count++;
    if (count <= 0) return;
    if (count > 8) count = 8;

    // Loose ground (scree, snow, water) throws a genuine plume: bigger, slower
    // to disperse, and persistent enough to hang behind the rider as a trail
    // rather than popping and vanishing.
    const plume = dustScale >= 1.15;
    const kind = plume ? DUST_PLUME : DUST_SKID;
    const tint = dustTintFor(surface.kind);

    this.basis(normal);

    for (let i = 0; i < count; i++) {
      const ang = this.rng.range(0, Math.PI * 2);
      const radial = this.rng.range(0.2, 0.8);
      _d.copy(_t).multiplyScalar(Math.cos(ang) * radial)
        .addScaledVector(_b, Math.sin(ang) * radial)
        .addScaledVector(_n, this.rng.range(0.5, 1.2));
      _d.multiplyScalar(this.rng.range(0.9, 2.1) * (plume ? 1.5 : 1.0));
      // Dragged backward out of the contact patch.
      _d.addScaledVector(velocity, -0.13);

      const jr = this.rng.range(0, 0.26);
      const ja = this.rng.range(0, Math.PI * 2);
      _v.copy(position)
        .addScaledVector(_t, Math.cos(ja) * jr)
        .addScaledVector(_b, Math.sin(ja) * jr)
        .addScaledVector(_n, this.rng.range(0.03, 0.18));

      const life = plume ? this.rng.range(1.6, 2.4) : this.rng.range(0.7, 1.15);
      const scale = (plume ? this.rng.range(0.45, 0.85) : this.rng.range(0.20, 0.36)) * (0.85 + dustScale * 0.3);
      const growth = plume ? this.rng.range(1.7, 2.5) : this.rng.range(1.1, 1.7);
      const spin = this.rng.signed() * (plume ? 0.5 : 1.1);

      this.write(_v.x, _v.y, _v.z, _d.x, _d.y, _d.z, life, kind, scale, growth, spin, tint);
    }
  }

  setOpacity(v: number): void {
    this.material.uniforms.uOpacity.value = clamp01(v);
  }

  dispose(): void {
    this.pool.dispose();
    this.material.dispose();
    this.object.removeFromParent();
  }
}
