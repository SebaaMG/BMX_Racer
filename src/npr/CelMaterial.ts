/**
 * CelMaterial — the one material factory the whole game builds surfaces from.
 *
 * Everything visible in DESCENT is a `CelMaterial` built from a `RampPreset`.
 * There is no MeshStandardMaterial anywhere in the project, and no code path
 * that could introduce one. That is the structural guarantee behind the "if any
 * surface reads as physically based, the art direction has failed" constraint —
 * there is no PBR to leak in.
 *
 * Each material also builds two silent companions:
 *   • a G-buffer prepass material (normal + depth + id + curvature)
 *   • a shadow-depth material
 * both sharing its defines, so instanced and skinned geometry can never fall
 * out of sync between passes. Outline hulls detaching from their mesh is
 * almost always a prepass/main mismatch; making them the same object removes
 * the failure mode.
 */

import {
  AddEquation,
  BackSide,
  Color,
  CustomBlending,
  DoubleSide,
  FrontSide,
  IUniform,
  Material,
  Mesh,
  OneFactor,
  ShaderMaterial,
  Side,
  Texture,
  Vector2,
  Vector3,
  ZeroFactor,
  GLSL3,
  Object3D,
  BufferGeometry,
  InstancedMesh,
  SkinnedMesh,
} from 'three';

import {
  GLSL_CEL_CORE,
  GLSL_CEL_SURFACE,
  GLSL_COMMON,
  GLSL_FOG,
  GLSL_FRAG_OUT,
  GLSL_GLOBAL_UNIFORMS,
  GLSL_HULL_VERTEX,
  GLSL_PREPASS_OUTPUTS,
  GLSL_RAMP_UNIFORMS,
  GLSL_SHADOW,
  GLSL_VERTEX_TRANSFORM,
} from './ShaderChunks';
import { globalUniformBlock, NPR } from './NprGlobals';
import { INK, LINES, RampPreset, RAMPS } from './Palette';

// ─────────────────────────────────────────────────────────────────────────────
// Material IDs. The Sobel pass draws an interior line wherever this value
// changes between neighbouring pixels, which catches the edges depth and
// normal both miss — an arm crossing a torso at the same depth and angle.
// ─────────────────────────────────────────────────────────────────────────────
let nextMaterialId = 1;
const idByName = new Map<string, number>();
export function materialIdFor(name: string): number {
  let id = idByName.get(name);
  if (id === undefined) {
    id = nextMaterialId++;
    idByName.set(name, id);
  }
  return id;
}

export interface CelOptions {
  /** Rendered as an InstancedMesh — enables the per-instance attribute block. */
  instanced?: boolean;
  /** Rendered as a SkinnedMesh — enables the bone-texture skinning path. */
  skinned?: boolean;
  /** Geometry carries a `color` attribute used as an albedo tint. */
  vertexColors?: boolean;
  /** Geometry carries an `aAo` float attribute (1 = open, 0 = occluded). */
  vertexAo?: boolean;
  /** Foliage wind: geometry carries an `aSway` float (0 at root, 1 at tip). */
  wind?: boolean;
  /** Alpha-tested cut-out — canopy cards, grass tufts. */
  alphaTest?: number;
  /** Generated albedo/mask texture. Multiplied over the ramp result. */
  map?: Texture | null;
  side?: Side;
  transparent?: boolean;
  depthWrite?: boolean;
  /** 0 = pure ramp, 1 = pure matcap. Metals sit around 0.45. */
  matcapMix?: number;
  tint?: Color;
  tintStrength?: number;
  /** Override the preset's outline width. 0 disables the hull entirely. */
  outlineWidth?: number;
  /** Extra GLSL injected into the fragment shader before `celShade` is called. */
  fragmentPreamble?: string;
  /** Extra GLSL run inside main() after the base colour is computed.
   *  Has access to `celCol` (inout vec3), `vWorldPos`, `vNormal`, `vUv`. */
  fragmentBody?: string;
  /** Extra GLSL run in the vertex shader; may modify `localPos`/`localNrm`. */
  vertexBody?: string;
  /** Extra uniforms merged into the material. */
  uniforms?: Record<string, IUniform>;
  /** Extra #defines. */
  defines?: Record<string, string | number>;
  /** Extra varyings declared in both stages (e.g. `out float vFoo;`). */
  varyings?: string;
  /** Written into the G-buffer for ID-based interior edges. */
  idName?: string;
  /** Skip the G-buffer prepass (sky dome, fully transparent FX). */
  noPrepass?: boolean;
  /** Skip shadow casting. */
  noShadow?: boolean;
  /** Receives fog. Off for the sky dome and for HUD-space geometry. */
  fog?: boolean;
  name?: string;
}

export class CelMaterial extends ShaderMaterial {
  declare uniforms: Record<string, IUniform>;
  readonly preset: RampPreset;
  readonly celOptions: CelOptions;
  /** Companion materials — built lazily by `ensurePassMaterials`. */
  prepassMaterial: ShaderMaterial | null = null;
  shadowMaterial: ShaderMaterial | null = null;
  hullMaterial: ShaderMaterial | null = null;

  constructor(preset: RampPreset, opts: CelOptions = {}) {
    const defines = buildDefines(opts);
    const varyings = opts.varyings ?? '';

    super({
      glslVersion: GLSL3,
      defines,
      uniforms: {
        ...globalUniformBlock(),
        ...rampUniforms(preset, opts),
        ...(opts.uniforms ?? {}),
      },
      vertexShader: buildVertexShader(opts, varyings),
      fragmentShader: buildFragmentShader(opts, varyings),
      side: opts.side ?? FrontSide,
      transparent: opts.transparent ?? false,
      depthWrite: opts.depthWrite ?? true,
      alphaTest: opts.alphaTest ?? 0,
      lights: false,
      fog: false,
    });

    this.preset = preset;
    this.celOptions = opts;
    this.name = opts.name ?? `cel:${preset.name}`;
  }

  /** Live-tunable band colours — used by the palette debug overlay. */
  setBandColor(index: number, color: Color): void {
    (this.uniforms.uBandColor.value as Color[])[index].copy(color);
  }

  setTint(color: Color, strength = 1): void {
    (this.uniforms.uTint.value as Color).copy(color);
    this.uniforms.uTintStrength.value = strength;
  }
}

function buildDefines(opts: CelOptions): Record<string, string | number> {
  const d: Record<string, string | number> = { ...(opts.defines ?? {}) };
  // USE_INSTANCING and USE_SKINNING are injected by three itself based on the
  // object type; we only declare the *extra* channels we add on top.
  if (opts.instanced) d.NPR_INSTANCED = 1;
  if (opts.skinned) d.NPR_SKINNED = 1;
  if (opts.vertexColors) d.NPR_VERTEX_COLOR = 1;
  if (opts.vertexAo) d.NPR_VERTEX_AO = 1;
  if (opts.wind) d.NPR_WIND = 1;
  if (opts.map) d.NPR_MAP = 1;
  if (opts.fog !== false) d.NPR_FOG = 1;
  if ((opts.alphaTest ?? 0) > 0) d.NPR_ALPHATEST = 1;
  return d;
}

function rampUniforms(p: RampPreset, opts: CelOptions): Record<string, IUniform> {
  const colors = [...p.colors.map((c) => c.clone())];
  while (colors.length < 4) colors.push(colors[colors.length - 1].clone());
  const thresholds = [...p.thresholds];
  while (thresholds.length < 3) thresholds.push(1.5); // never reached

  return {
    uBandColor: { value: colors },
    uBandThreshold: { value: thresholds },
    uBandCount: { value: p.colors.length },
    uEdgeSoftness: { value: p.edgeSoftness },
    uSpecStrength: { value: p.specStrength },
    uSpecPower: { value: p.specPower },
    uSpecColor: { value: p.specColor.clone() },
    uRimStrength: { value: p.rimStrength },
    uRimPower: { value: p.rimPower },
    uRimColor: { value: p.rimColor.clone() },
    uHatchStrength: { value: p.hatchStrength },
    uHatchScale: { value: p.hatchScale },
    uTint: { value: (opts.tint ?? new Color(1, 1, 1)).clone() },
    uTintStrength: { value: opts.tintStrength ?? 0 },
    uMatcapMix: { value: opts.matcapMix ?? 0 },
    uMaterialId: { value: materialIdFor(opts.idName ?? p.name) },
    uOpacity: { value: 1 },
    uMap: { value: opts.map ?? null },
    uWindStrength: { value: opts.wind ? 1 : 0 },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Vertex shader
// ─────────────────────────────────────────────────────────────────────────────
function buildVertexShader(opts: CelOptions, varyings: string): string {
  return /* glsl */ `
    precision highp float;
    ${GLSL_COMMON}
    ${GLSL_GLOBAL_UNIFORMS}
    ${GLSL_VERTEX_TRANSFORM}

    uniform float uWindStrength;

    #ifdef NPR_VERTEX_AO
      in float aAo;
    #endif
    #ifdef NPR_WIND
      in float aSway;
    #endif

    out vec3  vWorldPos;
    out vec3  vNormal;
    out vec2  vUv;
    out float vViewDist;
    out vec3  vAlbedoTint;
    out float vAo;
    ${varyings}

    void main() {
      vec3 localPos = position;
      vec3 localNrm = normal;

      ${opts.vertexBody ?? ''}

      resolveLocal(localPos, localNrm);

      #ifdef NPR_WIND
        // Wind is applied in OBJECT space before instancing so an instanced
        // forest sways coherently rather than each tree sliding sideways.
        float phase = 0.0;
        #ifdef USE_INSTANCING
          phase = aInstancePhase;
        #endif
        float t = uTime * 1.35 + phase;
        // Two-band motion: a slow trunk lean plus a faster branch flutter.
        float slow = sin(t * 0.55) * 0.6 + sin(t * 0.31 + 1.7) * 0.4;
        float fast = sin(t * 3.1 + localPos.y * 2.3) * 0.35;
        float amt = aSway * aSway * uWindStrength;
        localPos.x += (slow * 0.34 + fast * 0.10) * amt;
        localPos.z += (slow * 0.20 - fast * 0.13) * amt;
      #endif

      vec3 wpos, wnrm;
      toWorld(localPos, localNrm, wpos, wnrm);

      vWorldPos = wpos;
      vNormal   = wnrm;
      vUv       = uv;
      vViewDist = length(wpos - uCameraPos);

      vec3 tint = vec3(1.0);
      #ifdef NPR_VERTEX_COLOR
        tint *= color;
      #endif
      #ifdef USE_INSTANCING
        #ifdef NPR_INSTANCED
          tint *= aInstanceTint;
        #endif
      #endif
      vAlbedoTint = tint;

      #ifdef NPR_VERTEX_AO
        vAo = aAo;
      #else
        vAo = 1.0;
      #endif

      gl_Position = projectionMatrix * viewMatrix * vec4(wpos, 1.0);
    }
  `;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fragment shader
// ─────────────────────────────────────────────────────────────────────────────
function buildFragmentShader(opts: CelOptions, varyings: string): string {
  const inVaryings = varyings.replace(/\bout\b/g, 'in');
  return /* glsl */ `
    precision highp float;
    ${GLSL_COMMON}
    ${GLSL_GLOBAL_UNIFORMS}
    ${GLSL_RAMP_UNIFORMS}
    ${GLSL_SHADOW}
    ${GLSL_CEL_CORE}
    ${GLSL_FOG}
    ${GLSL_CEL_SURFACE}
    ${GLSL_FRAG_OUT}

    uniform sampler2D uMap;

    in vec3  vWorldPos;
    in vec3  vNormal;
    in vec2  vUv;
    in float vViewDist;
    in vec3  vAlbedoTint;
    in float vAo;
    ${inVaryings}

    ${opts.fragmentPreamble ?? ''}

    void main() {
      vec3 N = normalize(vNormal);
      // Two-sided surfaces (canopy cards, grass blades) must flip their normal
      // or the back face shades as though it were in permanent shadow.
      if (!gl_FrontFacing) N = -N;

      vec3 V = normalize(uCameraPos - vWorldPos);

      vec3 tint = vAlbedoTint;
      float alpha = uOpacity;

      #ifdef NPR_MAP
        vec4 texel = texture(uMap, vUv);
        tint *= texel.rgb;
        alpha *= texel.a;
      #endif

      #ifdef NPR_ALPHATEST
        if (alpha < ${(opts.alphaTest ?? 0.5).toFixed(3)}) discard;
      #endif

      CelInput s;
      s.worldPos   = vWorldPos;
      s.normal     = N;
      s.viewDir    = V;
      s.fragCoord  = gl_FragCoord.xy;
      s.viewDist   = vViewDist;
      s.albedoTint = tint;
      s.aoTerm     = vAo;

      vec3 celCol = celShade(s);

      ${opts.fragmentBody ?? ''}

      #ifdef NPR_FOG
        celCol = applyQuantizedFog(celCol, vViewDist, -V, gl_FragCoord.xy);
      #endif

      fragColor = vec4(celCol, alpha);
    }
  `;
}

// ─────────────────────────────────────────────────────────────────────────────
// G-buffer prepass material
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Writes the two RGBA16F attachments the Sobel pass reads:
 *   0: rgb = view-space normal   a = linear view depth (metres)
 *   1: r  = material id          g = curvature   b = stylisation mask
 *
 * Built from the same options as its parent so instancing, skinning, wind and
 * alpha-test all behave identically — a mismatch here shows up as interior
 * lines that swim off the geometry when a rider animates.
 */
export function createPrepassMaterial(preset: RampPreset, opts: CelOptions): ShaderMaterial {
  const defines = buildDefines(opts);
  const mat = new ShaderMaterial({
    glslVersion: GLSL3,
    defines,
    uniforms: {
      ...globalUniformBlock(),
      uMaterialId: { value: materialIdFor(opts.idName ?? preset.name) },
      uStyleMask: { value: opts.skinned ? 1.0 : 0.0 },
      uMap: { value: opts.map ?? null },
      uWindStrength: { value: opts.wind ? 1 : 0 },
    },
    side: opts.side ?? FrontSide,
    vertexShader: /* glsl */ `
      precision highp float;
      ${GLSL_COMMON}
      ${GLSL_GLOBAL_UNIFORMS}
      ${GLSL_VERTEX_TRANSFORM}

      uniform float uWindStrength;
      #ifdef NPR_WIND
        in float aSway;
      #endif
      // The prepass reads curvature so interior lines can taper with the hull.
      in float aCurvature;

      out vec3  vViewNormal;
      out float vViewDepth;
      out vec2  vUv;
      out float vCurv;

      void main() {
        vec3 localPos = position;
        vec3 localNrm = normal;
        resolveLocal(localPos, localNrm);

        #ifdef NPR_WIND
          float phase = 0.0;
          #ifdef USE_INSTANCING
            phase = aInstancePhase;
          #endif
          float t = uTime * 1.35 + phase;
          float slow = sin(t * 0.55) * 0.6 + sin(t * 0.31 + 1.7) * 0.4;
          float fast = sin(t * 3.1 + localPos.y * 2.3) * 0.35;
          float amt = aSway * aSway * uWindStrength;
          localPos.x += (slow * 0.34 + fast * 0.10) * amt;
          localPos.z += (slow * 0.20 - fast * 0.13) * amt;
        #endif

        vec3 wpos, wnrm;
        toWorld(localPos, localNrm, wpos, wnrm);

        vec4 vpos = viewMatrix * vec4(wpos, 1.0);
        vViewNormal = normalize((viewMatrix * vec4(wnrm, 0.0)).xyz);
        vViewDepth  = -vpos.z;
        vUv = uv;
        vCurv = aCurvature;

        gl_Position = projectionMatrix * vpos;
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;
      ${GLSL_COMMON}
      ${GLSL_PREPASS_OUTPUTS}

      uniform float uMaterialId;
      uniform float uStyleMask;
      uniform sampler2D uMap;

      in vec3  vViewNormal;
      in float vViewDepth;
      in vec2  vUv;
      in float vCurv;

      void main() {
        #ifdef NPR_ALPHATEST
          if (texture(uMap, vUv).a < ${(opts.alphaTest ?? 0.5).toFixed(3)}) discard;
        #endif
        vec3 n = normalize(vViewNormal);
        if (!gl_FrontFacing) n = -n;
        fragColor = vec4(n, vViewDepth);
        gAux = vec4(uMaterialId, vCurv, uStyleMask, 1.0);
      }
    `,
  });
  mat.name = `prepass:${preset.name}`;
  return mat;
}

/** Depth-only material for the sun cascades. */
export function createShadowMaterial(preset: RampPreset, opts: CelOptions): ShaderMaterial {
  const defines = buildDefines(opts);
  const mat = new ShaderMaterial({
    glslVersion: GLSL3,
    defines,
    uniforms: {
      ...globalUniformBlock(),
      uMap: { value: opts.map ?? null },
      uWindStrength: { value: opts.wind ? 1 : 0 },
    },
    // Front-face culling in the shadow pass removes almost all acne without
    // needing a large bias, which keeps contact shadows tight under the tyres.
    side: BackSide,
    vertexShader: /* glsl */ `
      precision highp float;
      ${GLSL_COMMON}
      ${GLSL_GLOBAL_UNIFORMS}
      ${GLSL_VERTEX_TRANSFORM}
      uniform float uWindStrength;
      #ifdef NPR_WIND
        in float aSway;
      #endif
      out vec2 vUv;
      out float vDepth;
      void main() {
        vec3 localPos = position;
        vec3 localNrm = normal;
        resolveLocal(localPos, localNrm);
        #ifdef NPR_WIND
          float phase = 0.0;
          #ifdef USE_INSTANCING
            phase = aInstancePhase;
          #endif
          float t = uTime * 1.35 + phase;
          float slow = sin(t * 0.55) * 0.6 + sin(t * 0.31 + 1.7) * 0.4;
          float amt = aSway * aSway * uWindStrength;
          localPos.x += slow * 0.34 * amt;
          localPos.z += slow * 0.20 * amt;
        #endif
        vec3 wpos, wnrm;
        toWorld(localPos, localNrm, wpos, wnrm);
        vUv = uv;
        vec4 clip = projectionMatrix * viewMatrix * vec4(wpos, 1.0);
        vDepth = clip.z / clip.w * 0.5 + 0.5;
        gl_Position = clip;
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;
      ${GLSL_FRAG_OUT}
      uniform sampler2D uMap;
      in vec2 vUv;
      in float vDepth;
      void main() {
        #ifdef NPR_ALPHATEST
          if (texture(uMap, vUv).a < ${(opts.alphaTest ?? 0.5).toFixed(3)}) discard;
        #endif
        fragColor = vec4(vDepth, vDepth, vDepth, 1.0);
      }
    `,
  });
  mat.name = `shadow:${preset.name}`;
  return mat;
}

// ─────────────────────────────────────────────────────────────────────────────
// Inverted-hull outline material
// ─────────────────────────────────────────────────────────────────────────────
export interface HullOptions extends CelOptions {
  widthMultiplier?: number;
  jitter?: number;
  color?: Color;
}

export function createHullMaterial(preset: RampPreset, opts: HullOptions = {}): ShaderMaterial {
  const defines = buildDefines(opts);
  const width = opts.outlineWidth ?? preset.outlineWidth;

  const mat = new ShaderMaterial({
    glslVersion: GLSL3,
    defines,
    uniforms: {
      ...globalUniformBlock(),
      uOutlineWidth: { value: width * 100 * (opts.widthMultiplier ?? 1) },
      uTargetPixels: { value: LINES.targetPixels },
      uFalloffStart: { value: LINES.distanceFalloffStart },
      uFalloffEnd: { value: LINES.distanceFalloffEnd },
      uMinWidthScale: { value: LINES.minWidthScale },
      uCurvatureWeight: { value: LINES.curvatureWeight },
      uCurvatureFloor: { value: LINES.curvatureFloor },
      uWidthJitter: { value: opts.jitter ?? 0.07 },
      uInkColor: { value: (opts.color ?? preset.outlineColor ?? INK).clone() },
      uMap: { value: opts.map ?? null },
      uWindStrength: { value: opts.wind ? 1 : 0 },
    },
    // BackSide + front-face culling: only the expanded backfaces survive, and
    // they poke out around the silhouette as a stroke.
    side: BackSide,
    transparent: false,
    depthWrite: true,
    vertexShader: /* glsl */ `
      precision highp float;
      ${GLSL_COMMON}
      ${GLSL_GLOBAL_UNIFORMS}
      ${GLSL_VERTEX_TRANSFORM}
      ${GLSL_HULL_VERTEX}

      uniform float uWindStrength;
      #ifdef NPR_WIND
        in float aSway;
      #endif

      out vec2  vUvH;
      out float vDistH;

      void main() {
        vec3 localPos = position;
        vec3 localNrm = normal;
        vec3 smoothNrm = aSmoothNormal;

        #ifdef USE_SKINNING
          // Skin the smoothed normal with the same matrix as the position, or
          // the hull peels off the mesh the moment a joint rotates.
          mat4 boneMatX = getBoneMatrix(skinIndex.x);
          mat4 boneMatY = getBoneMatrix(skinIndex.y);
          mat4 boneMatZ = getBoneMatrix(skinIndex.z);
          mat4 boneMatW = getBoneMatrix(skinIndex.w);
          mat4 skinMatrix = skinWeight.x * boneMatX + skinWeight.y * boneMatY
                          + skinWeight.z * boneMatZ + skinWeight.w * boneMatW;
          skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
          localPos  = (skinMatrix * vec4(localPos, 1.0)).xyz;
          localNrm  = normalize((skinMatrix * vec4(localNrm, 0.0)).xyz);
          smoothNrm = normalize((skinMatrix * vec4(smoothNrm, 0.0)).xyz);
        #endif

        #ifdef NPR_WIND
          float phase = 0.0;
          #ifdef USE_INSTANCING
            phase = aInstancePhase;
          #endif
          float t = uTime * 1.35 + phase;
          float slow = sin(t * 0.55) * 0.6 + sin(t * 0.31 + 1.7) * 0.4;
          float fast = sin(t * 3.1 + localPos.y * 2.3) * 0.35;
          float amt = aSway * aSway * uWindStrength;
          localPos.x += (slow * 0.34 + fast * 0.10) * amt;
          localPos.z += (slow * 0.20 - fast * 0.13) * amt;
        #endif

        vec3 wpos, wnrm;
        toWorld(localPos, localNrm, wpos, wnrm);

        // Transform the smoothed normal to world by the same route.
        vec3 wsmooth;
        #ifdef USE_INSTANCING
          wsmooth = normalize(mat3(modelMatrix * instanceMatrix) * smoothNrm);
        #else
          wsmooth = normalize(mat3(modelMatrix) * smoothNrm);
        #endif

        float viewDist = length(wpos - uCameraPos);
        wpos += hullOffset(wpos, wsmooth, aCurvature, viewDist);

        vUvH = uv;
        vDistH = viewDist;
        gl_Position = projectionMatrix * viewMatrix * vec4(wpos, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;
      ${GLSL_COMMON}
      ${GLSL_GLOBAL_UNIFORMS}
      ${GLSL_FRAG_OUT}
      uniform vec3 uInkColor;
      uniform sampler2D uMap;
      in vec2  vUvH;
      in float vDistH;
      void main() {
        #ifdef NPR_ALPHATEST
          if (texture(uMap, vUvH).a < ${(opts.alphaTest ?? 0.5).toFixed(3)}) discard;
        #endif
        // The stroke picks up a little of the atmosphere at distance so a far
        // tree line doesn't read as a black scribble against pale haze.
        vec3 ink = uInkColor;
        float t = saturate1((vDistH - uFogNear) / max(uFogFar - uFogNear, EPS));
        t = pow(t, 0.78);
        float f = saturate1(t) * 3.999;
        int i = clamp(int(floor(f)), 0, 3);
        ink = mix(ink, uFogColors[i] * 0.55, uFogStrengths[i] * saturate1(t * 1.3));
        fragColor = vec4(ink, 1.0);
      }
    `,
  });
  mat.name = `hull:${preset.name}`;
  return mat;
}

// ─────────────────────────────────────────────────────────────────────────────
// Assembly helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Attach an inverted-hull outline to a mesh.
 *
 * The hull is a sibling mesh sharing the SAME geometry buffer (no duplication
 * in memory) and, for skinned meshes, the SAME skeleton — so it deforms in
 * lockstep. Rendered before the main pass so the depth buffer resolves the
 * stroke behind the surface.
 */
export function attachOutline(
  mesh: Mesh,
  preset: RampPreset,
  opts: HullOptions = {},
): Mesh | null {
  const width = opts.outlineWidth ?? preset.outlineWidth;
  if (width <= 0) return null;

  const geo = mesh.geometry as BufferGeometry;
  if (!geo.getAttribute('aSmoothNormal')) {
    console.warn(`[npr] ${mesh.name || preset.name} has no aSmoothNormal — outline skipped.`);
    return null;
  }

  const hullOpts: HullOptions = {
    ...opts,
    instanced: opts.instanced ?? (mesh as InstancedMesh).isInstancedMesh === true,
    skinned: opts.skinned ?? (mesh as SkinnedMesh).isSkinnedMesh === true,
  };
  const mat = createHullMaterial(preset, hullOpts);

  let hull: Mesh;
  if ((mesh as InstancedMesh).isInstancedMesh) {
    const src = mesh as InstancedMesh;
    const im = new InstancedMesh(geo, mat, src.count);
    im.instanceMatrix = src.instanceMatrix;
    im.count = src.count;
    hull = im;
  } else if ((mesh as SkinnedMesh).isSkinnedMesh) {
    const src = mesh as SkinnedMesh;
    const sm = new SkinnedMesh(geo, mat);
    sm.bind(src.skeleton, src.bindMatrix);
    hull = sm;
  } else {
    hull = new Mesh(geo, mat);
  }

  hull.name = `${mesh.name || preset.name}:hull`;
  hull.renderOrder = (mesh.renderOrder ?? 0) - 1;
  hull.frustumCulled = mesh.frustumCulled;
  hull.castShadow = false;
  hull.receiveShadow = false;
  hull.userData.isHull = true;
  hull.userData.hullOf = mesh;
  mesh.userData.hull = hull;
  return hull;
}

/**
 * Build a mesh's companion pass materials and stash them where the render
 * passes look for them. Call once per mesh after construction.
 */
export function registerNprMesh(mesh: Mesh, material: CelMaterial): void {
  const opts = material.celOptions;
  const isInstanced = (mesh as InstancedMesh).isInstancedMesh === true;
  const isSkinned = (mesh as SkinnedMesh).isSkinnedMesh === true;
  const passOpts: CelOptions = { ...opts, instanced: isInstanced, skinned: isSkinned };

  if (!opts.noPrepass) {
    material.prepassMaterial ??= createPrepassMaterial(material.preset, passOpts);
    mesh.userData.prepassMaterial = material.prepassMaterial;
  }
  if (!opts.noShadow) {
    material.shadowMaterial ??= createShadowMaterial(material.preset, passOpts);
    mesh.userData.shadowMaterial = material.shadowMaterial;
  }
}

/** Convenience: build material + register + attach outline in one call. */
export function makeCelMesh(
  geometry: BufferGeometry,
  presetName: keyof typeof RAMPS,
  opts: CelOptions = {},
): { mesh: Mesh; material: CelMaterial; hull: Mesh | null; group: Object3D } {
  const preset = RAMPS[presetName];
  const material = new CelMaterial(preset, opts);
  const mesh = new Mesh(geometry, material);
  mesh.name = opts.name ?? presetName;
  registerNprMesh(mesh, material);
  const hull = attachOutline(mesh, preset, opts);
  const group = new Object3D();
  group.name = `${mesh.name}:group`;
  if (hull) group.add(hull);
  group.add(mesh);
  return { mesh, material, hull, group };
}

/** Dispose everything a cel mesh owns. */
export function disposeCelMaterial(m: CelMaterial): void {
  m.prepassMaterial?.dispose();
  m.shadowMaterial?.dispose();
  m.hullMaterial?.dispose();
  m.dispose();
}
