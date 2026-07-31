/**
 * LUT — the grade, as a table, ready for the GPU.
 *
 * `gradeLUT()` in GeneratedTextures builds the 32-cube from the constants in
 * Palette.GRADE as a Float32 Data3DTexture. This module does two small but
 * necessary things to it:
 *
 *  1. Converts it to an 8-bit table. Linear filtering of a FLOAT 3D texture
 *     requires OES_texture_float_linear, which is an optional WebGL2 extension
 *     — present on Apple silicon, absent on several software rasterisers,
 *     including the one the headless capture harness can fall back to. When it
 *     is missing three silently drops to NEAREST and the grade posterises into
 *     32 visible steps per channel. An 8-bit RGBA 3D texture filters natively
 *     everywhere, is 128KB, and — since we are grading a DISPLAY-REFERRED
 *     image, where values above 1.0 have nowhere to go — loses nothing.
 *
 *  2. Owns the sampling GLSL, including the half-texel inset. Sampling a 3D LUT
 *     without the inset shifts every colour by half a cell toward black, which
 *     is subtle enough to survive review and wrong enough to matter.
 *
 * The grade is applied in DISPLAY space, after the sRGB encode. That is not an
 * arbitrary choice: GRADE.contrastPivot is 0.46 and the lift is 0.012, numbers
 * that only make sense as fractions of a display signal. Applied to linear
 * radiance a 0.46 pivot sits at 71% sRGB and the "subtle" lift would raise the
 * black point by a third of a stop.
 */

import {
  ClampToEdgeWrapping,
  Data3DTexture,
  LinearFilter,
  RGBAFormat,
  UnsignedByteType,
} from 'three';
import { gradeLUT } from './GeneratedTextures';

let _lut: Data3DTexture | null = null;

/** The grade table, built once. 32^3 RGBA8. */
export function gradeLutTexture(size = 32): Data3DTexture {
  if (_lut) return _lut;

  const src = gradeLUT(size);
  const data = src.image.data as Float32Array;
  const n = size * size * size;
  const bytes = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const r = data[i * 4 + 0];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    bytes[i * 4 + 0] = Math.round(Math.min(Math.max(r, 0), 1) * 255);
    bytes[i * 4 + 1] = Math.round(Math.min(Math.max(g, 0), 1) * 255);
    bytes[i * 4 + 2] = Math.round(Math.min(Math.max(b, 0), 1) * 255);
    bytes[i * 4 + 3] = 255;
  }

  const tex = new Data3DTexture(bytes, size, size, size);
  tex.format = RGBAFormat;
  tex.type = UnsignedByteType;
  tex.minFilter = LinearFilter;
  tex.magFilter = LinearFilter;
  tex.wrapS = ClampToEdgeWrapping;
  tex.wrapT = ClampToEdgeWrapping;
  tex.wrapR = ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  tex.name = 'grade-lut-8';
  _lut = tex;
  return tex;
}

export function disposeGradeLut(): void {
  _lut?.dispose();
  _lut = null;
}

/**
 * GLSL for the grade tail of the composite: the display shoulder, the LUT
 * lookup, vignette and paper grain. Kept together because the ORDER of these
 * four operations is the thing that has to stay fixed, and separating them
 * across files is how an order gets quietly changed.
 */
export const GLSL_GRADE = /* glsl */ `
  uniform highp sampler3D uLut;
  uniform float uLutSize;
  uniform sampler2D uGrain;
  uniform vec2  uGrainScale;
  uniform float uGrainStrength;
  uniform float uVignette;
  uniform float uVignetteSoftness;
  uniform vec3  uVignetteTint;   // GRADE.shadowTint — corners cool, never crush

  /**
   * Display shoulder.
   *
   * The renderer does no tone mapping — deliberately, because a filmic curve
   * would put a smooth roll across every flat cel band and the bands are the
   * whole point. But bloom and speculars do exceed 1.0, and hard-clipping them
   * flattens the halo into a disc. This is the minimum intervention: exactly
   * the identity below 'k', an asymptote to 1.0 above it. Flat bands live
   * entirely below k and are therefore untouched.
   */
  vec3 displayShoulder(vec3 c) {
    const float k = 0.86;
    vec3 over = max(c - k, vec3(0.0));
    return min(c, vec3(k)) + (1.0 - k) * (over / (over + (1.0 - k)));
  }

  /**
   * 3D LUT lookup with the half-texel inset. Without the (size-1)/size scale
   * and the +0.5 offset the lookup samples the CORNER of each cell instead of
   * its centre and the whole image drifts half a cell dark.
   */
  vec3 applyGradeLut(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    vec3 uvw = (c * (uLutSize - 1.0) + 0.5) / uLutSize;
    return texture(uLut, uvw).rgb;
  }

  /** Elliptical vignette. Subtle by construction — it keeps the eye on the trail. */
  vec3 applyVignette(vec3 c, vec2 uv, float aspect) {
    vec2 d = (uv - 0.5) * vec2(aspect, 1.0);
    float r = length(d) * 1.42;
    float v = uVignette * smoothstep(uVignetteSoftness, 1.25, r);
    // Toward the committed shadow hue, never toward black. Multiplying by
    // (1 - v) desaturates the corners to grey-black, which is the one thing
    // rule 2 of the palette forbids: dark is a hue-rotated violet cousin of the
    // lit colour, not an absence of light.
    vec3 shadow = uVignetteTint * (0.22 + 0.78 * luma(c));
    return mix(c, shadow, v);
  }

  /**
   * Paper grain. A whisper — 1.5% at most — and biased into the darks, because
   * that is where real paper tooth shows and because grain in a flat highlight
   * reads as sensor noise, which is the one thing this frame must not look like.
   */
  vec3 applyGrain(vec3 c, vec2 fragCoord) {
    if (uGrainStrength <= 0.0001) return c;
    float g = texture(uGrain, fragCoord * uGrainScale).r - 0.5;
    float weight = mix(1.0, 0.30, saturate1(luma(c)));
    return c + g * uGrainStrength * weight;
  }
`;
