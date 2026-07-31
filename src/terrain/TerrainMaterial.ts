/**
 * TerrainMaterial — the mountain's surface, and the three passes that draw it.
 *
 * Five decisions define this file. Each one exists because the obvious
 * alternative fails in a way that is visible in a screenshot.
 *
 * 1. HEIGHT LIVES IN A TEXTURE, NOT IN THE VERTEX BUFFER.
 *    The clipmap's rings are flat grids that slide over the world every frame.
 *    If height were baked into their vertices, every ring would have to have its
 *    positions rewritten and re-uploaded on every re-centre — megabytes per
 *    frame, and a guaranteed hitch. Fetching the height in the vertex shader
 *    from a single 2048² R32F texture makes a re-centre a change to one
 *    translation matrix.
 *
 * 2. NORMALS COME FROM A TEXTURE, NOT FROM THE GEOMETRY.
 *    This is the more important one. If shading normals came from the
 *    tessellation, the shading detail of the mountain would collapse as the LOD
 *    coarsened, and — because the cel ramp quantises — the band boundaries would
 *    visibly JUMP at every LOD transition. Sampling a full-resolution normal
 *    map instead means the terrain shades identically at 5m and at 2km, and the
 *    LOD becomes purely a silhouette question. Nothing else lets a hard-banded
 *    shader survive a geometry clipmap.
 *
 * 3. PER-ZONE RAMPS, SELECTED PER PIXEL, WITH NO BLEND.
 *    Seven complete cel ramps live in uniform arrays and one is chosen per
 *    fragment from the zone map. Choosing — not blending — is the whole point:
 *    a blend between two cel ramps produces intermediate colours that belong to
 *    neither palette, which is the single fastest way to make a stylised
 *    terrain look like a badly-lit realistic one.
 *
 * 4. THE SHADING ENTRY POINT IS REPLACED, NOT WRAPPED.
 *    `celShade()` in ShaderChunks reads ONE ramp from uniforms. Terrain needs
 *    seven. The material therefore defines `terrainShade()` and redirects the
 *    call site to it with a preprocessor macro (see TERRAIN_FRAGMENT below).
 *    The alternative — letting celShade run and then overwriting its result —
 *    would waste a full shadow-map lookup on every pixel of the largest surface
 *    in the frame, which at retina is tens of millions of texture fetches a
 *    frame thrown away.
 *
 * 5. THE PREPASS AND SHADOW MATERIALS ARE HAND-WRITTEN.
 *    `createPrepassMaterial` in CelMaterial.ts does not know about our vertex
 *    displacement, so the stock companions would rasterise a flat plane at y=0
 *    while the main pass draws a mountain. Both are rebuilt here from the same
 *    displacement chunk, which is the only way to guarantee they can never
 *    disagree.
 */

import {
  ClampToEdgeWrapping,
  Color,
  DataTexture,
  FloatType,
  FrontSide,
  GLSL3,
  IUniform,
  LinearFilter,
  LinearMipmapLinearFilter,
  NearestFilter,
  RGBAFormat,
  RedFormat,
  ShaderMaterial,
  Texture,
  UnsignedByteType,
  Vector4,
} from 'three';

import {
  GLSL_COMMON,
  GLSL_FRAG_OUT,
  GLSL_GLOBAL_UNIFORMS,
  GLSL_PREPASS_OUTPUTS,
  GLSL_VERTEX_TRANSFORM,
} from '../npr/ShaderChunks';
import { CelMaterial, materialIdFor } from '../npr/CelMaterial';
import { globalUniformBlock } from '../npr/NprGlobals';
import { RAMPS, RampPreset } from '../npr/Palette';
import { barkTexture, paperGrain, trailSurface } from '../npr/GeneratedTextures';
import { SurfaceKind } from '../game/Contracts';
import { WORLD_HALF } from '../game/WorldConstants';
import { ZONE_KIND_COUNT } from './Zones';

// ─────────────────────────────────────────────────────────────────────────────
// Data textures
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The height field as an R32F texture.
 *
 * NearestFilter is deliberate: WebGL2 only guarantees linear filtering of float
 * textures behind `OES_texture_float_linear`, so the vertex shader does its own
 * bilinear from four `texelFetch` taps. That is exact, extension-free, and it
 * matches `bilinearSample` on the CPU bit-for-bit in the cases that matter —
 * which is what keeps the wheels on the ground the render draws.
 */
export function createHeightTexture(height: Float32Array, size: number): DataTexture {
  const tex = new DataTexture(height, size, size, RedFormat, FloatType);
  tex.minFilter = NearestFilter;
  tex.magFilter = NearestFilter;
  tex.wrapS = ClampToEdgeWrapping;
  tex.wrapT = ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.name = 'terrain:height';
  tex.needsUpdate = true;
  return tex;
}

/**
 * RGB = world-space normal, A = erosion remapped to [0,1].
 *
 * Packing erosion into the spare channel is not a space saving — it is a
 * fetch saving. The fragment shader wants both at exactly the same texel on
 * every pixel of the mountain, and one RGBA fetch is half the bandwidth of two.
 */
export function buildNormalErosionData(
  height: Float32Array,
  erosion: Float32Array,
  size: number,
  cellSize: number,
): Uint8Array {
  const data = new Uint8Array(size * size * 4);
  writeNormalRegion(data, height, erosion, size, cellSize, 0, 0, size - 1, size - 1);
  return data;
}

/** Recompute a sub-rectangle of the normal/erosion data. Used after a carve. */
export function writeNormalRegion(
  data: Uint8Array,
  height: Float32Array,
  erosion: Float32Array,
  size: number,
  cellSize: number,
  ix0: number,
  iz0: number,
  ix1: number,
  iz1: number,
): void {
  const inv2c = 1 / (2 * cellSize);
  const x0 = Math.max(0, ix0);
  const z0 = Math.max(0, iz0);
  const x1 = Math.min(size - 1, ix1);
  const z1 = Math.min(size - 1, iz1);

  for (let iz = z0; iz <= z1; iz++) {
    const row = iz * size;
    const rowUp = (iz > 0 ? iz - 1 : 0) * size;
    const rowDn = (iz < size - 1 ? iz + 1 : size - 1) * size;
    for (let ix = x0; ix <= x1; ix++) {
      const xl = ix > 0 ? ix - 1 : 0;
      const xr = ix < size - 1 ? ix + 1 : size - 1;
      const dhdx = (height[row + xr] - height[row + xl]) * inv2c;
      const dhdz = (height[rowDn + ix] - height[rowUp + ix]) * inv2c;
      // The surface normal of h(x,z) is (-dh/dx, 1, -dh/dz), normalised.
      let nx = -dhdx;
      const ny = 1;
      let nz = -dhdz;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      nx /= len;
      nz /= len;
      const i4 = (row + ix) * 4;
      data[i4] = Math.round((nx * 0.5 + 0.5) * 255);
      data[i4 + 1] = Math.round((1 / len) * 0.5 * 255 + 127.5);
      data[i4 + 2] = Math.round((nz * 0.5 + 0.5) * 255);
      data[i4 + 3] = Math.round((erosion[row + ix] * 0.5 + 0.5) * 255);
    }
  }
}

export function createNormalTexture(data: Uint8Array, size: number): DataTexture {
  const tex = new DataTexture(data, size, size, RGBAFormat, UnsignedByteType);
  // Linear + mipmaps: this is the ONLY thing standing between the mountain and
  // a shimmering mess at distance, because the cel ramp turns a half-texel of
  // normal aliasing into a full band flip.
  tex.minFilter = LinearMipmapLinearFilter;
  tex.magFilter = LinearFilter;
  tex.wrapS = ClampToEdgeWrapping;
  tex.wrapT = ClampToEdgeWrapping;
  tex.generateMipmaps = true;
  tex.anisotropy = 8;
  tex.name = 'terrain:normal';
  tex.needsUpdate = true;
  return tex;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared GLSL
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Declarations smuggled into the vertex stage through `CelOptions.varyings`.
 *
 * CelMaterial has exactly one insertion point that lands above `main()` in the
 * VERTEX shader, and this is it. The same text is also spliced into the
 * fragment shader (with the token `out` rewritten to `in`), which is why this
 * block contains no varying declarations, no `out` parameters, and nothing
 * whose meaning changes between the two stages — a uniform is a uniform and a
 * function is a function in both. `uHeightParams` is genuinely wanted in both
 * stages, so the duplication is useful rather than merely harmless.
 *
 * uHeightParams = (worldHalf, 1 / metresPerTexel, texelCount, metresPerTexel)
 */
export const TERRAIN_SHARED_DECLS = /* glsl */ `
  uniform sampler2D uHeightTex;
  uniform vec4 uHeightParams;

  /** World XZ -> normalised [0,1] texture coordinate on any terrain field. */
  vec2 terrainUv(vec2 wxz) {
    return (wxz + uHeightParams.x) * uHeightParams.y / uHeightParams.z;
  }

  /**
   * Bilinear height fetch, done by hand from four integer taps.
   * See the note on createHeightTexture for why this is not a texture() call.
   */
  float terrainHeight(vec2 wxz) {
    float n = uHeightParams.z;
    vec2 g = clamp((wxz + uHeightParams.x) * uHeightParams.y, vec2(0.0), vec2(n - 1.001));
    vec2 gi = floor(g);
    vec2 f = g - gi;
    ivec2 i0 = ivec2(gi);
    ivec2 i1 = min(i0 + ivec2(1), ivec2(int(n) - 1));
    float h00 = texelFetch(uHeightTex, ivec2(i0.x, i0.y), 0).r;
    float h10 = texelFetch(uHeightTex, ivec2(i1.x, i0.y), 0).r;
    float h01 = texelFetch(uHeightTex, ivec2(i0.x, i1.y), 0).r;
    float h11 = texelFetch(uHeightTex, ivec2(i1.x, i1.y), 0).r;
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
  }
`;

/**
 * The clipmap displacement, identical in all three passes.
 *
 * The geometry arriving here is a FLAT grid centred on its own origin, and the
 * mesh's model matrix is a pure XZ translation to the ring's snapped position.
 * Two constants that differ per ring travel in the unused `uv` attribute:
 *   uv.x = this ring's vertex spacing in metres
 *   uv.y = this ring's half extent in metres
 *
 * THE MORPH. Every second vertex of a ring does not exist on the next-coarser
 * ring. Left alone, that produces a T-junction crack at every level boundary
 * and a visible pop whenever a level re-centres. The fix is to slide those
 * vertices onto their coarse neighbour as they approach the ring's outer edge,
 * and — critically — to resample the height AFTER sliding, so the vertex stays
 * exactly on the surface throughout the transition rather than shearing off it.
 * By the outermost two rows the slide is complete, the ring's boundary edge is
 * literally a coarse-grid edge, and the join is exact rather than approximate.
 */
export const TERRAIN_ZONE_LOOKUP = /* glsl */ `
  uniform sampler2D uZoneTex;
  uniform float uZoneJitter;

  /**
   * The zone lookup, shared by the MAIN and PREPASS fragment shaders.
   *
   * It is ONE function because the two passes must agree exactly, and they did
   * not: the main pass jittered and block-snapped the lookup while the prepass
   * read the raw texel. The Sobel ID edge was therefore inked on the 2 m grid
   * staircase while the paint sat on blocks up to 64 m wide — hard zigzag lines
   * drawn across uniformly-coloured terrain, marking a boundary that was not
   * there. Two copies of a lookup this fiddly will always drift apart.
   *
   * It lives in its own chunk rather than in TERRAIN_SHARED_DECLS because that
   * block is injected into the VERTEX stage as well, and fwidth() is a fragment
   * builtin.
   *
   * Snap to a power-of-two texel block sized from the screen-space derivative
   * (this is what kills far-field salt-and-pepper on a nearest-sampled map that
   * can never have a mip chain, because the average of rock and grass is dirt),
   * but jitter BEFORE the snap, in units of the block, so the block grid itself
   * is ragged rather than axis-aligned at every distance. Jittering after the
   * snap only frays the sample position; the grid survives and the far field
   * breaks into rectangles that read as compression artefacts.
   */
  int sampleZone(vec2 worldXZ, vec2 uvT, int kindCount) {
    float texels = uHeightParams.z;
    vec2 tc = uvT * texels;
    float footprint = max(max(fwidth(tc.x), fwidth(tc.y)), 1.0);
    float zStep = clamp(exp2(ceil(log2(footprint))), 1.0, 32.0);
    float noiseFreq = 1.6 / zStep;
    vec2 jit = vec2(fbm2(worldXZ * noiseFreq, 2), fbm2(worldXZ * noiseFreq + 47.3, 2)) - 0.5;
    // About a third of a block of wander: enough to break the grid, small
    // enough that the boundary still reads as one drawn line rather than a wave.
    vec2 jitTexels = jit * (uZoneJitter * texels) * max(1.0, zStep * 0.30);
    vec2 zoneTc = (floor((tc + jitTexels) / zStep) + 0.5) * zStep;
    return clamp(int(texture(uZoneTex, zoneTc / texels).r * 255.0 + 0.5), 0, kindCount - 1);
  }
`;

export const TERRAIN_DISPLACE = /* glsl */ `
  float ringSpacing = uv.x;
  float ringHalf    = uv.y;

  // Chebyshev distance from the ring centre, normalised. The ring is snapped to
  // the camera every frame, so this is also (to within one snap quantum) the
  // distance from the camera — which is what the morph wants to key on.
  float ringR = max(abs(localPos.x), abs(localPos.z)) / ringHalf;
  float morphK = clamp((ringR - 0.76) / 0.20, 0.0, 1.0);

  // 1 on grid indices that the coarser level does not have, 0 on shared ones.
  vec2 parity = mod(floor(abs(localPos.xz) / ringSpacing + 0.5), 2.0);
  localPos.xz -= parity * ringSpacing * morphK;

  vec2 terrainXZ = localPos.xz + vec2(modelMatrix[3].x, modelMatrix[3].z);
  localPos.y = terrainHeight(terrainXZ);
  localNrm = vec3(0.0, 1.0, 0.0);
`;

/**
 * The fragment program. Everything below runs once per terrain pixel, so the
 * costs are called out where they are paid.
 */
const TERRAIN_FRAGMENT = /* glsl */ `
  uniform sampler2D uNormalTex;      // rgb = world normal, a = erosion
  // uZoneTex / uZoneJitter are declared in TERRAIN_ZONE_LOOKUP, which this
  uniform sampler2D uDetailRock;     // stretched worley borders — rock bedding
  uniform sampler2D uDetailGravel;   // trail-surface generator, g = gravel
  uniform sampler2D uDetailGrain;    // paper grain — the finest tier of tooth

  uniform vec3  uZoneColor[${ZONE_KIND_COUNT * 4}];
  uniform vec4  uZoneThresh[${ZONE_KIND_COUNT}];
  uniform vec4  uZoneMiscA[${ZONE_KIND_COUNT}];  // bandCount, edgeSoftness, spec, specPower
  uniform vec4  uZoneMiscB[${ZONE_KIND_COUNT}];  // hatch, hatchScale, rim, rimPower
  // preamble is concatenated after. Declaring them twice fails to link.
  uniform float uDetailScale;

  /** Band select with the same one-pixel edge as bandStep, per-zone softness. */
  float zoneBandStep(float x, float t, float soft) {
    float w = max(fwidth(x) * 0.75, soft);
    return smoothstep(t - w, t + w, x);
  }

  /**
   * Evaluate zone z's ramp. Composited low-to-high exactly like evalRamp, and
   * three-band ramps simply weight their fourth step to zero — no branch, so
   * the fwidth() inside zoneBandStep is never evaluated under divergent flow.
   */
  vec3 evalZoneRamp(int z, float ndl) {
    int b = z * 4;
    vec4 th = uZoneThresh[z];
    float soft = uZoneMiscA[z].y;
    float has4 = step(3.5, uZoneMiscA[z].x);
    vec3 c = uZoneColor[b];
    c = mix(c, uZoneColor[b + 1], zoneBandStep(ndl, th.x, soft));
    c = mix(c, uZoneColor[b + 2], zoneBandStep(ndl, th.y, soft));
    c = mix(c, uZoneColor[b + 3], zoneBandStep(ndl, th.z, soft) * has4);
    return c;
  }

  float zoneBandIndex(int z, float ndl) {
    vec4 th = uZoneThresh[z];
    float soft = uZoneMiscA[z].y;
    float has4 = step(3.5, uZoneMiscA[z].x);
    return zoneBandStep(ndl, th.x, soft)
         + zoneBandStep(ndl, th.y, soft)
         + zoneBandStep(ndl, th.z, soft) * has4;
  }

  /**
   * Screen-space hatch, parameterised per zone.
   *
   * A copy of applyHatch with the strength and scale passed in rather than read
   * from uniforms, because the whole reason this material exists is that its
   * seven surfaces need seven different sets of those constants. Scree gets a
   * dense tight screen; snow gets almost none. Everything else is identical to
   * the shared implementation, deliberately, so terrain hatching and rider
   * hatching are visibly the same pen.
   */
  vec3 terrainHatch(vec3 col, float bandIdx, vec2 fragCoord, float strength, float scale) {
    float amount = strength * uHatchGlobal;
    float deep = 1.0 - saturate1(bandIdx);
    float mid  = saturate1(1.0 - abs(bandIdx - 1.0));
    float cover = deep + mid * 0.42;
    vec2 huv = fragCoord / (34.0 * scale);
    const float a = 0.5934;                       // ~34 degrees off the pixel grid
    mat2 rot = mat2(cos(a), -sin(a), sin(a), cos(a));
    vec3 atlas = texture(uHatchTex, rot * huv).rgb;
    float tier = mix(atlas.g, atlas.r, deep);
    float breakup = fbm2(fragCoord * 0.004, 2);
    float h = tier * mix(0.72, 1.0, breakup);
    return mix(col, col * 0.74, h * cover * amount * saturate1(cover));
  }

  vec3 terrainShade(CelInput s) {
    vec3 wp = s.worldPos;
    vec2 w = wp.xz;
    vec2 uvT = terrainUv(w);

    // ── Surface data: one fetch for normal + erosion, one for the zone ──────
    vec4 nx = texture(uNormalTex, uvT);
    vec3 N = normalize(nx.xyz * 2.0 - 1.0);
    float erosion = nx.w * 2.0 - 1.0;

    // ── Zone lookup ────────────────────────────────────────────────────────
    // Two problems, opposite ends of the distance range, one lookup.
    //
    // NEAR: the zone map is 2m per texel and sampled NEAREST, so every material
    // boundary is a 2m staircase — unmistakable pixelation from a low camera.
    // Jittering the LOOKUP by a couple of texels of high-frequency noise turns
    // that staircase into a torn edge without softening it by a single pixel.
    // The boundary is still a hard step between two ramps; it just stops
    // running along the grid.
    //
    // FAR: a nearest-sampled map with no mip chain — and it cannot have one,
    // because the average of rock (0) and grass (2) is dirt (1) — turns into
    // salt-and-pepper the moment a screen pixel spans more than a texel.
    // Snapping the lookup to a power-of-two texel block sized from the
    // screen-space derivative collapses that noise into large flat patches:
    // stable under camera motion, and exactly the cut-paper read the far
    // distance is supposed to have.
    int zone = sampleZone(w, uvT, ${ZONE_KIND_COUNT});

    // Everything with a highlight in it has to calm down at range, or the
    // normal map's minification turns a banded specular into a field of
    // crawling white sparks against the flat fog plateaus.
    // QUANTISED. On flat ground the lit term is constant by construction, so
    // this and the detail fade below were the ONLY terms varying across the
    // surface — which
    // means the smooth ramp the reviewer saw on every foreground plane was
    // being drawn by the two distance fades, not by the lighting. A stepped
    // fade keeps the intent (calm the highlights down at range) without laying
    // a gradient over the top of the hard bands.
    float farFade = floor((1.0 - smoothstep(150.0, 780.0, s.viewDist)) * 3.0 + 0.5) / 3.0;

    float isRock  = float(zone == ${SurfaceKind.Rock});
    float isDirt  = float(zone == ${SurfaceKind.Dirt});
    float isGrass = float(zone == ${SurfaceKind.Grass});
    float isScree = float(zone == ${SurfaceKind.Scree});
    float isSnow  = float(zone == ${SurfaceKind.Snow});
    float isWater = float(zone == ${SurfaceKind.Water});

    // Water is the one terrain surface with motion in it. Two crossed waves,
    // amplitude small enough that the banded specular breaks into moving
    // shapes rather than sliding as a sheet.
    float ripple = sin(w.x * 2.4 + uTime * 1.6) + sin(w.y * 3.1 - uTime * 2.2);
    N = normalize(N + vec3(ripple * 0.055, 0.0, ripple * 0.041) * (isWater * farFade));

    // ── Lighting. Identical in structure to celShade; only the ramp differs ─
    vec3 V = s.viewDir;
    vec3 L = uSunDir;
    float ndlRaw = dot(N, L);

    // The committed sun sits at 21.5 degrees, so ridable ground occupies only
    // N.L in roughly [-0.15, 0.7]. A plain half-Lambert maps that narrow window
    // into the middle of the ramp and squaring it compresses it further, so an
    // entire dune lands inside ONE band and the largest surface in the frame
    // reads as a smooth gradient. Expanding the range the terrain genuinely
    // occupies onto the full ramp is what makes a dune show three plateaus.
    float lit = saturate1(ndlRaw * 1.15 + 0.16);

    // The shadow steps the ramp DOWN rather than scaling the term that feeds
    // it. Multiplying before the lookup sweeps the ramp continuously wherever
    // the penumbra is partial, re-introducing a gradient at exactly the edges
    // that are supposed to be the sharpest thing in the frame.
    float shadow = sunShadow(wp, N, s.viewDist, saturate1(ndlRaw));
    lit = max(lit - (1.0 - shadow) * 0.42, 0.0);

    vec3 col = evalZoneRamp(zone, lit);
    float bIdx = zoneBandIndex(zone, lit);

    // Ambient bounce, QUANTISED. N.y varies smoothly across every rounded
    // landform, so tinting by it continuously lays a soft gradient over the top
    // of the hard bands and undoes them.
    float upness = floor((N.y * 0.5 + 0.5) * 3.0 + 0.5) / 3.0;
    vec3 bounce = mix(uGroundBounce, uSkyBounce, upness);
    col = mix(col, col * bounce * 1.55, uAmbient * (1.0 - 0.55 * saturate1(bIdx / 3.0)));
    col *= s.albedoTint;

    // ── Surface detail ─────────────────────────────────────────────────────
    // Two vertical projections plus a horizontal one, blended by how the face
    // is oriented. A pure top-down projection smears into vertical streaks on
    // any cliff, which is the single most obvious tell of a cheap terrain
    // shader; a full three-axis triplanar costs a third more fetches than this
    // for a difference nobody can see on a heightfield, which by construction
    // never has a face steeper than vertical.
    float wallness = 1.0 - smoothstep(0.42, 0.82, abs(N.y));
    float kx = abs(N.x) / max(abs(N.x) + abs(N.z), 1e-4);
    float ds = uDetailScale;

    float rockZ = texture(uDetailRock, vec2(wp.z, wp.y) * ds).r;
    float rockX = texture(uDetailRock, vec2(wp.x, wp.y) * ds).r;
    float rockY = texture(uDetailRock, w * ds * 0.72).r;
    float bedding = mix(rockY, mix(rockZ, rockX, kx), wallness);

    float gravel = texture(uDetailGravel, w * ds * 2.4).g;
    float grain  = texture(uDetailGrain, w * ds * 5.0).r;

    // High-frequency detail is faded out with distance rather than left to the
    // mip chain alone: past a few hundred metres the ridges have to read as
    // flat paper shapes, and any residual texture in them fights the fog bands.
    float detail = floor((1.0 - smoothstep(80.0, 430.0, s.viewDist)) * 4.0 + 0.5) / 4.0;

    // Every one of these is a HARD step. Nothing in this block is allowed to
    // introduce a gradient across the largest surface in the frame.
    float bed = bandStep(bedding, 0.46, 0.025);
    col = mix(col, col * 0.87, bed * (isRock * 0.60 + isScree * 0.20 + isWater * 0.35) * detail);

    float pebble = 1.0 - bandStep(gravel, 0.30, 0.02);
    col = mix(col, col * vec3(1.07, 1.05, 1.0), pebble * (isScree * 0.34 + isDirt * 0.16) * detail);

    float clump = bandStep(grain, 0.53, 0.02);
    col = mix(col, col * vec3(0.91, 1.04, 0.89), clump * isGrass * 0.32 * detail);

    float drift = bandStep(rockY, 0.56, 0.02);
    col = mix(col, col * 1.06, drift * isSnow * 0.38 * detail);

    // ── Erosion read ───────────────────────────────────────────────────────
    // The shading agrees with the shape: cut ground goes cold and dark, the
    // fan below it goes pale and warm. Both quantised, both narrow.
    float cut  = bandStep(-erosion, 0.34, 0.03);
    float fill = bandStep(erosion, 0.30, 0.03);
    col = mix(col, col * vec3(0.86, 0.88, 0.95), cut * 0.50);
    col = mix(col, col * vec3(1.05, 1.02, 0.96), fill * 0.34);

    // ── Hatch, specular, rim — all per zone ────────────────────────────────
    col = terrainHatch(col, bIdx, s.fragCoord, uZoneMiscB[zone].x, uZoneMiscB[zone].y);

    float spec = bandedSpecular(N, V, L, uZoneMiscA[zone].w);
    col += uSpecColor * spec * uZoneMiscA[zone].z * shadow * saturate1(ndlRaw * 2.0)
         * mix(0.12, 1.0, farFade);

    float rim = celRim(N, V, uZoneMiscB[zone].w);
    float sunSide = saturate1(dot(normalize(N + L * 0.35), V) * 0.5 + 0.75);
    col += uRimColor * rim * uZoneMiscB[zone].z * mix(0.35, 1.0, sunSide);

    return col;
  }

  // Redirect celShade's single call site in CelMaterial's main() to the
  // seven-ramp version above. The macro is declared AFTER celShade's own
  // definition, so that definition is untouched and simply becomes dead code.
  #define celShade(inputSurface) terrainShade(inputSurface)
`;

// ─────────────────────────────────────────────────────────────────────────────
// Zone ramp packing
// ─────────────────────────────────────────────────────────────────────────────

/** Which ramp preset paints each SurfaceKind. */
export const ZONE_RAMPS: Record<SurfaceKind, RampPreset> = {
  [SurfaceKind.Rock]: RAMPS.rock,
  [SurfaceKind.Dirt]: RAMPS.dirt,
  [SurfaceKind.Grass]: RAMPS.grass,
  [SurfaceKind.Scree]: RAMPS.scree,
  [SurfaceKind.Snow]: RAMPS.snow,
  [SurfaceKind.Water]: RAMPS.water,
  [SurfaceKind.Trail]: RAMPS.trail,
};

function packZoneUniforms(): {
  colors: Color[];
  thresholds: Vector4[];
  miscA: Vector4[];
  miscB: Vector4[];
} {
  const colors: Color[] = [];
  const thresholds: Vector4[] = [];
  const miscA: Vector4[] = [];
  const miscB: Vector4[] = [];

  for (let k = 0; k < ZONE_KIND_COUNT; k++) {
    const p = ZONE_RAMPS[k as SurfaceKind];
    // Pad short ramps by repeating the last colour and pushing the unused
    // threshold out of reach, exactly the way rampUniforms does for CelMaterial.
    const c = p.colors.map((x) => x.clone());
    while (c.length < 4) c.push(c[c.length - 1].clone());
    colors.push(c[0], c[1], c[2], c[3]);

    const t = [...p.thresholds];
    while (t.length < 3) t.push(1.5);
    thresholds.push(new Vector4(t[0], t[1], t[2], 0));

    miscA.push(new Vector4(p.colors.length, p.edgeSoftness, p.specStrength, p.specPower));
    miscB.push(new Vector4(p.hatchStrength, p.hatchScale, p.rimStrength, p.rimPower));
  }

  return { colors, thresholds, miscA, miscB };
}

// ─────────────────────────────────────────────────────────────────────────────
// The material set
// ─────────────────────────────────────────────────────────────────────────────

export interface TerrainMaterialOptions {
  heightTexture: DataTexture;
  normalTexture: DataTexture;
  zoneTexture: DataTexture;
  size: number;
  /** Metres per height texel. */
  cellSize: number;
}

export interface TerrainMaterialSet {
  main: CelMaterial;
  prepass: ShaderMaterial;
  shadow: ShaderMaterial;
  /** Shared uniform objects — written once, seen by all three programs. */
  shared: Record<string, IUniform>;
  dispose(): void;
}

/**
 * Build the main / prepass / shadow trio.
 *
 * They share the height texture uniform OBJECT, not just its value, so a
 * re-upload after a track carve cannot leave one pass a frame behind another.
 */
export function createTerrainMaterials(o: TerrainMaterialOptions): TerrainMaterialSet {
  const heightParams = new Vector4(WORLD_HALF, 1 / o.cellSize, o.size, o.cellSize);

  const shared: Record<string, IUniform> = {
    uHeightTex: { value: o.heightTexture as Texture },
    uHeightParams: { value: heightParams },
    uNormalTex: { value: o.normalTexture as Texture },
    uZoneTex: { value: o.zoneTexture as Texture },
  };

  // Material ids, one per zone, so the Sobel pass inks every material boundary
  // on the mountain as an interior line. Depth and normal are both continuous
  // across a grass/scree boundary — the id channel is the only thing that can
  // see it, and seeing it is the difference between a painted edge and a
  // texture change.
  const zoneIds: number[] = [];
  for (let k = 0; k < ZONE_KIND_COUNT; k++) {
    zoneIds.push(materialIdFor(`terrain:${ZONE_RAMPS[k as SurfaceKind].name}`));
  }

  const packed = packZoneUniforms();
  const detailScale = 0.055;

  const fragUniforms: Record<string, IUniform> = {
    uDetailRock: { value: barkTexture() },
    uDetailGravel: { value: trailSurface() },
    uDetailGrain: { value: paperGrain() },
    uZoneColor: { value: packed.colors },
    uZoneThresh: { value: packed.thresholds },
    uZoneMiscA: { value: packed.miscA },
    uZoneMiscB: { value: packed.miscB },
    // 2.5 texels of raggedness, expressed in uv so the shader needs no divide.
    uZoneJitter: { value: 2.5 / o.size },
    uDetailScale: { value: detailScale },
  };

  // The base preset is only used for the handful of uniforms terrainShade still
  // reads from the shared ramp block (uSpecColor, uRimColor) and for the
  // material's name. Every band colour comes from the per-zone arrays.
  const main = new CelMaterial(RAMPS.rock, {
    name: 'terrain',
    idName: 'terrain',
    outlineWidth: 0,
    fog: true,
    varyings: TERRAIN_SHARED_DECLS,
    vertexBody: TERRAIN_DISPLACE,
    fragmentPreamble: TERRAIN_ZONE_LOOKUP + TERRAIN_FRAGMENT,
    uniforms: { ...shared, ...fragUniforms },
  });
  main.side = FrontSide;

  const prepass = createTerrainPrepassMaterial(shared, zoneIds);
  const shadow = createTerrainShadowMaterial(shared);

  return {
    main,
    prepass,
    shadow,
    shared,
    dispose(): void {
      main.dispose();
      prepass.dispose();
      shadow.dispose();
    },
  };
}

/**
 * The G-buffer prepass.
 *
 * Note what it does NOT do: it does not read the geometry normal. The normal it
 * writes is the same full-resolution texture normal the main pass shades with,
 * so the interior-line pass sees the creases the eye sees rather than the
 * tessellation of whichever clipmap ring happened to cover that pixel. Without
 * that, every LOD boundary would draw itself as a line across the mountain.
 */
function createTerrainPrepassMaterial(
  shared: Record<string, IUniform>,
  zoneIds: number[],
): ShaderMaterial {
  const mat = new ShaderMaterial({
    glslVersion: GLSL3,
    side: FrontSide,
    uniforms: {
      ...globalUniformBlock(),
      ...shared,
      uZoneIds: { value: zoneIds },
    },
    vertexShader: /* glsl */ `
      precision highp float;
      ${GLSL_COMMON}
      ${GLSL_GLOBAL_UNIFORMS}
      ${GLSL_VERTEX_TRANSFORM}
      ${TERRAIN_SHARED_DECLS}

      out vec3  vWorld;
      out float vViewDepth;

      void main() {
        vec3 localPos = position;
        vec3 localNrm = normal;
        ${TERRAIN_DISPLACE}

        vec3 wpos, wnrm;
        toWorld(localPos, localNrm, wpos, wnrm);
        vec4 vpos = viewMatrix * vec4(wpos, 1.0);
        vWorld = wpos;
        vViewDepth = -vpos.z;
        gl_Position = projectionMatrix * vpos;
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;
      ${GLSL_COMMON}
      ${GLSL_PREPASS_OUTPUTS}
      ${TERRAIN_SHARED_DECLS}
      ${TERRAIN_ZONE_LOOKUP}

      uniform sampler2D uNormalTex;
      uniform float uZoneIds[${ZONE_KIND_COUNT}];

      in vec3  vWorld;
      in float vViewDepth;

      void main() {
        vec2 uvT = terrainUv(vWorld.xz);
        vec3 n = normalize(texture(uNormalTex, uvT).xyz * 2.0 - 1.0);
        vec3 vn = normalize((viewMatrix * vec4(n, 0.0)).xyz);

        int zone = sampleZone(vWorld.xz, uvT, ${ZONE_KIND_COUNT});

        // Curvature from the screen-space rate of change of the normal. On a
        // heightfield this is a better estimate than anything bakeable: it
        // already accounts for how obliquely the surface is being viewed, so a
        // ridge seen edge-on thickens its own stroke.
        float curv = clamp(length(fwidth(n)) * 5.5, 0.0, 1.0);

        fragColor = vec4(vn, vViewDepth);
        gAux = vec4(uZoneIds[zone], curv, 0.0, 1.0);
      }
    `,
  });
  mat.name = 'terrain:prepass';
  return mat;
}

/**
 * The sun-cascade depth material.
 *
 * FrontSide, not the usual BackSide. The stock shadow material culls front
 * faces to kill acne on closed solids; a heightfield is an open sheet with
 * every triangle facing up, so front-face culling would cull the entire
 * mountain and it would cast nothing at all.
 */
function createTerrainShadowMaterial(shared: Record<string, IUniform>): ShaderMaterial {
  const mat = new ShaderMaterial({
    glslVersion: GLSL3,
    side: FrontSide,
    uniforms: {
      ...globalUniformBlock(),
      ...shared,
    },
    vertexShader: /* glsl */ `
      precision highp float;
      ${GLSL_COMMON}
      ${GLSL_GLOBAL_UNIFORMS}
      ${GLSL_VERTEX_TRANSFORM}
      ${TERRAIN_SHARED_DECLS}

      out float vDepth;

      void main() {
        vec3 localPos = position;
        vec3 localNrm = normal;
        ${TERRAIN_DISPLACE}

        vec3 wpos, wnrm;
        toWorld(localPos, localNrm, wpos, wnrm);
        vec4 clip = projectionMatrix * viewMatrix * vec4(wpos, 1.0);
        vDepth = clip.z / clip.w * 0.5 + 0.5;
        gl_Position = clip;
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;
      ${GLSL_FRAG_OUT}
      in float vDepth;
      void main() {
        fragColor = vec4(vDepth, vDepth, vDepth, 1.0);
      }
    `,
  });
  mat.name = 'terrain:shadow';
  return mat;
}
