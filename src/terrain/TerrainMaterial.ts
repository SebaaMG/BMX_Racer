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
    // The jitter noise has to have a WAVELENGTH COMPARABLE TO THE BLOCK, not to
    // the texel. At 1.6/zStep the wavelength was zStep/1.6 metres — a third of
    // the block it was supposed to be fraying — so on a steeply oblique face,
    // where one screen pixel already spans several metres of ground, the block
    // index resolved differently in adjacent pixels and the far field broke
    // into orange salt-and-pepper over grey rock. The zone map is 2 m per
    // texel, so a block is 2*zStep metres across; this puts the wander at
    // roughly two blocks per cycle, which frays the grid without ever
    // resolving below a screen pixel.
    float noiseFreq = 0.24 / zStep;
    vec2 jit = vec2(fbm2(worldXZ * noiseFreq, 2), fbm2(worldXZ * noiseFreq + 47.3, 2)) - 0.5;
    // About a third of a block of wander: enough to break the grid, small
    // enough that the boundary still reads as one drawn line rather than a wave.
    vec2 jitTexels = jit * (uZoneJitter * texels) * max(1.0, zStep * 0.30);

    // ── THE LATTICE IS ROTATED, AND THAT IS NOT A REFINEMENT ────────────────
    // A snapped boundary is a staircase; there is no way round that, because the
    // block IS a screen pixel by construction and anything that frays it at the
    // block scale aliases at the pixel scale. What CAN be fixed is the
    // staircase's ORIENTATION. Axis-aligned, it produces long right-angled runs
    // that the material-id Sobel then draws as a hard rectilinear zigzag across
    // the snowfields — unmistakably a grid, and the single most computer-looking
    // mark left on the mountain.
    //
    // Rotating the lattice by a very low-frequency angle field breaks that: the
    // boundary is still made of straight snapped segments, but they run at an
    // angle that drifts across the map, so what the eye reads is cut paper
    // rather than a raster. The angle field's wavelength is twenty blocks, so it
    // is constant to within a few percent across any one block and the lattice
    // never tears; and because both the main pass and the prepass call this one
    // function, the ink and the paint stay on the same boundary.
    float ang = (fbm2(worldXZ * (0.012 / zStep) + 91.7, 2) - 0.5) * 2.6;
    float ca = cos(ang);
    float sa = sin(ang);
    mat2 rot = mat2(ca, -sa, sa, ca);
    vec2 lat = rot * (tc + jitTexels);
    vec2 latSnap = (floor(lat / zStep) + 0.5) * zStep;
    // Inverse of a rotation is its transpose; written out so no matrix inverse
    // is generated for a 2x2 orthonormal frame.
    vec2 zoneTc = mat2(ca, sa, -sa, ca) * latSnap;
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
   * A stroke centred on one band boundary, about one and a half pixels wide.
   *
   * A terminator on terrain is invisible to every line system in the project.
   * The Sobel pass reads normal, depth and material id, and all three are
   * CONTINUOUS across a band boundary — the surface does not bend there, it
   * does not step there, and it is the same material on both sides. So the
   * largest shapes in the picture changed value without one stroke marking the
   * change, which is the single thing a painted background never does. The only
   * place in the pipeline that knows where the boundary is, is here.
   *
   * THE WIDTH IS PURELY THE SCREEN-SPACE DERIVATIVE, with no floor under it.
   * A floor expressed in shading units is a line of unbounded SCREEN width: on
   * near-flat ground fwidth(lit) goes to nothing and the shading term drifts
   * across a threshold over tens of metres, so a fixed 0.0016 of value became a
   * stroke metres wide and the valley floor filled with fat nested contour
   * arcs. Keyed to the derivative alone the stroke is one and a half pixels
   * wherever it exists, and on ground genuinely flat enough to have no
   * terminator at all it correctly draws nothing.
   */
  float terminatorInk(float x, float t) {
    float w = max(fwidth(x) * 1.5, 1e-7);
    return 1.0 - smoothstep(0.0, w, abs(x - t));
  }

  /** All three boundaries of zone z's ramp, drawn. */
  float zoneTerminatorInk(int z, float lit) {
    vec4 th = uZoneThresh[z];
    float has4 = step(3.5, uZoneMiscA[z].x);
    float ink = terminatorInk(lit, th.x);
    ink = max(ink, terminatorInk(lit, th.y));
    ink = max(ink, terminatorInk(lit, th.z) * has4);
    return ink;
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
    // The mid tier used to be weighted 0.42, which — multiplied by a palette
    // hatch strength of 0.18 and then again by saturate1(cover) — put the
    // maximum darkening on packed dirt at four percent. Four percent is not a
    // hatch, it is a rounding error, and the shadow band on the largest plane
    // in the frame was left as flat paint.
    float cover = deep + mid * 0.58;
    vec2 huv = fragCoord / (34.0 * scale);
    const float a = 0.5934;                       // ~34 degrees off the pixel grid
    mat2 rot = mat2(cos(a), -sin(a), sin(a), cos(a));
    vec3 atlas = texture(uHatchTex, rot * huv).rgb;
    // TERRAIN SKIPS THE COARSE TIER. The atlas packs coarse/medium/fine in
    // R/G/B and applyHatch reaches for R in the deepest band, which is right on
    // a rider: his shadow side is a couple of hundred pixels across and wants a
    // stroke you can see the direction of. The same band on terrain is half the
    // frame, and a screen whose cell is visible at that size stops reading as
    // tone and starts reading as a texture stuck to the mountain. One tier
    // finer throughout — the biggest shape on the page takes the finest screen.
    float tier = mix(atlas.b, atlas.g, deep);
    float breakup = fbm2(fragCoord * 0.004, 2);
    float h = tier * mix(0.72, 1.0, breakup);
    return mix(col, col * 0.70, h * cover * amount * saturate1(cover));
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

    // Water is the one terrain surface with motion in it.
    //
    // The amplitude used to be 0.055 and scaled by farFade, and both were
    // wrong. A ripple that only tilts the normal by three degrees cannot move a
    // highlight across a surface, and killing it at range meant the one shot in
    // the game that is mostly water had none of it at all. Two crossed waves
    // plus a slow chop, at an amplitude that genuinely breaks the mirror, and
    // NO distance fade — the stream bed is looked at from thirty metres and
    // from three hundred.
    float ripple = sin(w.x * 2.4 + uTime * 1.6) + sin(w.y * 3.1 - uTime * 2.2);
    float chop   = sin(w.x * 0.62 - w.y * 0.51 + uTime * 0.9);
    N = normalize(N + vec3(ripple * 0.17 + chop * 0.09, 0.0, ripple * 0.13 - chop * 0.07) * isWater);

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

    // ── The drawn terminator ───────────────────────────────────────────────
    // See zoneTerminatorInk. This is the stroke that turns a band boundary
    // into a drawn edge instead of a raw colour step, and it is the only line
    // system in the project that can see one.
    col = mix(col, col * 0.58, zoneTerminatorInk(zone, lit) * 0.80);

    // ── Aerial plates ──────────────────────────────────────────────────────
    // The defect this exists to kill, stated exactly: on flat ground lit is
    // constant BY CONSTRUCTION. Once every distance fade in this shader is
    // quantised there is nothing left varying across a flat surface at all, so
    // the largest plane in the frame renders as one enormous flat wash with a
    // smooth fog ramp laid over the top — which is precisely the "smooth
    // continuous ramp, nothing else" the review kept finding.
    //
    // A background painter does not paint receding ground as one plane. They
    // paint it as a STACK of flat plates, each a step cooler and hazier than
    // the one in front, with a boundary between them you can point at. On
    // receding ground an iso-distance contour runs roughly horizontally across
    // the frame, which is exactly where those boundaries belong.
    //
    // Logarithmic, so the plates are roughly even in SCREEN space rather than
    // in world space: eight plateaus from 4 m to 2 km, one step every 1.29
    // octaves, which lands two hard steps inside the first forty metres.
    // The plate boundaries land at 6, 15, 37 and 90 metres, and the stack
    // saturates there.
    float aerialT = saturate1(log2(max(s.viewDist, 4.0) * 0.25) / 9.0);
    float aerial  = floor(aerialT * 7.0 + 0.5) / 7.0;
    // SATURATING, and that matters as much as the quantisation. The plates are
    // carrying the near and middle field, which is where the shading term is
    // constant and the fog is still sitting on plateau zero with a strength of
    // 0.16. Past a couple of hundred metres the fog bands take the job over
    // completely, and stacking more haze on top of them only greys the far
    // distance into the wash the fog quantisation exists to prevent. Fog band 1
    // begins at 223 m, so the two systems hand over cleanly with no overlap.
    float plate = min(aerial, 0.43);
    col = mix(col, uSkyBounce * 1.28, plate * 0.62);
    col *= mix(0.88, 1.06, plate / 0.43);

    // ── Surface detail ─────────────────────────────────────────────────────
    // Two vertical projections plus a horizontal one, blended by how the face
    // is oriented. A pure top-down projection smears into vertical streaks on
    // any cliff, which is the single most obvious tell of a cheap terrain
    // shader; a full three-axis triplanar costs a third more fetches than this
    // for a difference nobody can see on a heightfield, which by construction
    // never has a face steeper than vertical.
    // Both blend weights are QUANTISED to three steps. A continuous blend
    // between two orthogonal projections is what produces the vertical
    // smearing and the faint concentric moire a cliff face shows in these
    // frames: the two samples slide past each other pixel by pixel. Three
    // steps keeps the projection change from reading as a seam while stopping
    // it from sweeping.
    float wallness = floor((1.0 - smoothstep(0.42, 0.82, abs(N.y))) * 3.0 + 0.5) / 3.0;
    float kx = floor((abs(N.x) / max(abs(N.x) + abs(N.z), 1e-4)) * 3.0 + 0.5) / 3.0;
    float ds = uDetailScale;

    float rockZ = texture(uDetailRock, vec2(wp.z, wp.y) * ds).r;
    float rockX = texture(uDetailRock, vec2(wp.x, wp.y) * ds).r;
    // THE TOP-DOWN PROJECTION USES A DIFFERENT TEXTURE, and that is the whole
    // point of writing it out rather than reusing rockZ. uDetailRock is the
    // bark generator and bark is CONCENTRIC. On a vertical face those rings
    // read correctly as bedding, which is why the two side projections keep
    // them; projected straight down onto open ground they lay literal tree
    // rings across the valley floor — the nested arcs visible on every flat
    // foreground plane in these frames, which no amount of domain warping
    // removes because a warp bends a ring, it does not open it. The paper
    // grain has no preferred centre and no preferred direction, which is
    // exactly what ground seen from above wants.
    float rockY = texture(uDetailGrain, w * ds * 0.85).r;
    float bedding = mix(rockY, mix(rockZ, rockX, kx), wallness);

    float gravel = texture(uDetailGravel, w * ds * 2.4).g;
    float grain  = texture(uDetailGrain, w * ds * 5.0).r;

    // High-frequency detail is faded out with distance rather than left to the
    // mip chain alone: past a few hundred metres the ridges have to read as
    // flat paper shapes, and any residual texture in them fights the fog bands.
    float detail = floor((1.0 - smoothstep(80.0, 430.0, s.viewDist)) * 4.0 + 0.5) / 4.0;

    // Rock bedding is a LARGE-scale feature and gets its own, far longer fade.
    // A canyon wall is normally seen from three to eight hundred metres, so on
    // the short fade above the one surface feature that could have broken the
    // wall up had already gone to zero by the time the wall was in frame — and
    // what was left, on a face where N.y is near zero and lit is therefore
    // constant across the whole plane, was lit concrete.
    float detailBed = floor((1.0 - smoothstep(420.0, 1500.0, s.viewDist)) * 3.0 + 0.5) / 3.0;

    // Every one of these is a HARD step. Nothing in this block is allowed to
    // introduce a gradient across the largest surface in the frame.
    float bed = bandStep(bedding, 0.46, 0.025);
    col = mix(col, col * 0.87, bed * (isRock * 0.70 + isScree * 0.30 + isWater * 0.35) * detailBed);

    // ── Strata ─────────────────────────────────────────────────────────────
    // Two hard bedding planes, sampled almost purely on world height so they
    // run level across a face the way real strata do, with just enough plan
    // drift that two neighbouring walls do not line up. This is the value break
    // that gives a vertical face plateaus to read; without it a wall has no
    // shading variation available to it at all, because every other term in
    // this shader is either a function of N (constant on a plane) or of view
    // distance (nearly constant across a face seen side-on).
    float strata = fbm2(vec2((wp.x + wp.z) * 0.010, wp.y * 0.085), 2);
    float wallMask = wallness * detailBed
                   * (isRock * 1.0 + isScree * 0.8 + isSnow * 0.35 + isDirt * 0.55 + isGrass * 0.3);
    col = mix(col, col * 0.845, bandStep(strata, 0.46, 0.02) * wallMask * 0.85);
    col = mix(col, col * 1.11,  bandStep(strata, 0.61, 0.02) * wallMask * 0.60);

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
    col += uSpecColor * spec * uZoneMiscA[zone].z * (1.0 - isWater) * shadow
         * saturate1(ndlRaw * 2.0) * mix(0.12, 1.0, farFade);

    float rim = celRim(N, V, uZoneMiscB[zone].w);
    float sunSide = saturate1(dot(normalize(N + L * 0.35), V) * 0.5 + 0.75);
    col += uRimColor * rim * uZoneMiscB[zone].z * (1.0 - isWater) * mix(0.35, 1.0, sunSide);

    // ── Water ──────────────────────────────────────────────────────────────
    // RAMPS.water asks for specPower 160 and rimPower 4, and on this course
    // NEITHER CAN EVER FIRE. The sun sits at 21.5 degrees and the camera looks
    // down a valley at maybe 25, so the half-vector on a level water surface
    // stands about 23 degrees off the horizontal and N.H tops out near 0.4 —
    // pow(0.4, 160) is zero to every float in the machine. The rim is the same
    // story from the other side: N.V on that surface is about 0.42, so the
    // Fresnel term reaches 0.11 against a 0.42 threshold. The stream bed
    // therefore rendered as a flat painted puddle no matter what the palette
    // asked for, and no amount of turning the strengths up would have changed
    // it. Both terms are rebuilt here at exponents the geometry can actually
    // reach, which is why they are not simply read from the ramp block.
    //
    // Everything stays HARD: two glint tiers and one rim step, so the result is
    // a drawn glitter track broken up by the ripple rather than a wet sheen.
    vec3  Hw   = normalize(L + V);
    float glint = pow(saturate1(dot(N, Hw)), 4.0);
    float fleck = bandStep(glint, 0.052, 0.004) * 0.55
                + bandStep(glint, 0.108, 0.003) * 0.45;
    float wrim = bandStep(pow(1.0 - saturate1(dot(N, V)), 1.6), 0.40, 0.03);
    // Flow contours: the current, drawn as banded bright lines running with the
    // ripple. Water in a cel background is drawn, not simulated.
    float flow = bandStep(sin(w.x * 1.35 + w.y * 0.9 + uTime * 0.85 + ripple * 0.7), 0.72, 0.03);
    col += isWater * (uSpecColor * fleck * 1.05 * shadow
                    + uRimColor * wrim * 0.34
                    + uSpecColor * flow * 0.16);

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

/**
 * Terrain-only gain on the palette's hatch strengths.
 *
 * The ramp presets are shared with the rider, the bike and the vegetation,
 * where a hatch strength of 0.18 is a whisper laid over a shape the eye is
 * already reading from its silhouette. Terrain has no silhouette to lean on: it
 * is one plane covering half the frame, and at 0.18 — multiplied again by the
 * mid-tier cover weight — its shadow band came out four percent darker than its
 * lit band and read as flat paint. The hatch is the ONLY texture a shadowed
 * plane gets, so terrain takes the same pen pressed harder — but only about
 * two-thirds again. Past roughly 0.45 the screen cell becomes legible at the
 * size terrain occupies and the hatch reads as a texture stuck to the mountain
 * rather than as tone laid over the picture.
 *
 * This lives here rather than in Palette because it is a statement about how
 * big the surface is, not about what the surface is made of.
 */
const TERRAIN_HATCH_GAIN = 1.7;

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
    miscB.push(
      new Vector4(p.hatchStrength * TERRAIN_HATCH_GAIN, p.hatchScale, p.rimStrength, p.rimPower),
    );
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
