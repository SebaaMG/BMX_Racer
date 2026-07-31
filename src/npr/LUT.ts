/**
 * LUT — the grade, as a table, ready for the GPU.
 *
 * This module evaluates the 32-cube from the constants in Palette.GRADE and
 * uploads it as an 8-bit RGBA 3D texture. Three things about it matter:
 *
 *  1. It is 8-bit, not float. Linear filtering of a FLOAT 3D texture
 *     requires OES_texture_float_linear, which is an optional WebGL2 extension
 *     — present on Apple silicon, absent on several software rasterisers,
 *     including the one the headless capture harness can fall back to. When it
 *     is missing three silently drops to NEAREST and the grade posterises into
 *     32 visible steps per channel. An 8-bit RGBA 3D texture filters natively
 *     everywhere, is 128KB, and — since we are grading a DISPLAY-REFERRED
 *     image, where values above 1.0 have nowhere to go — loses nothing.
 *
 *  2. It owns the sampling GLSL, including the half-texel inset. Sampling a 3D
 *     LUT without the inset shifts every colour by half a cell toward black,
 *     which is subtle enough to survive review and wrong enough to matter.
 *
 *  3. It owns the GRADE TRIM — see the constant below. The authored grade
 *     stacked four warm multipliers and clipped the red channel on the trail
 *     surface; the correction lives here, next to the transform it corrects.
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
import { GRADE } from './Palette';

/**
 * GRADE TRIM — the fix for the palette drift, applied here because Palette.ts
 * is not mine to edit.
 *
 * THE SYMPTOM. The ridden trail read as a saturated pumpkin orange, hotter and
 * higher-chroma than RAMPS.dirt's authored 0xb8825a / 0xd9a878.
 *
 * THE CAUSE, traced through the table by hand. Take the lit dirt plateau,
 * 0xd9a878 = (217, 168, 120), and run it through the untrimmed grade:
 *
 *   gain 1.06 on red         217 -> 230
 *   contrast 1.12 @ 0.46     230 -> 247   (everything above the pivot expands)
 *   split 0.22 -> gold       247 -> 258
 *   saturation 1.14          258 -> 267, CLIPPED TO 255
 *
 * Four warm multipliers stacked on a ramp that was already authored warm. The
 * last step is the one that actually does the damage: red clips, so the lit
 * plateau and the mid plateau BOTH land on red = 255 and the ramp's top step —
 * the whole reason for authoring four colours — collapses into one hue. What
 * survives is maximum-red, and maximum-red on a warm mid is pumpkin.
 *
 * THE TRIM. Three multipliers and a ceiling, chosen to keep the grade's warmth
 * and kill the clip:
 *
 *   gain   x (0.965, 1.0, 1.035)  -> effective (1.023, 1.0, 0.973)
 *   split  x 0.68                 -> effective 0.15
 *   sat    x 0.93                 -> effective 1.060, rolling to 0.85 of that
 *                                    in the highlights, which is where the
 *                                    over-chroma actually lived
 *   chroma ceiling at 0.985       -> nothing can clip a channel, ever
 *
 * Lit dirt now lands at (244, 179, 127) instead of (255, 178, 111): still
 * graded, still warm, but the top of the ramp is a colour again rather than a
 * clipped edge, and the step between the mid and lit plateaus survives.
 *
 * These are deltas rather than replacements on purpose — they stay locked to
 * whatever Palette.GRADE says, so a future move on the authored grade carries
 * through. The equivalent baked values are reported alongside.
 */
const TRIM = {
  gainR: 0.965,
  gainG: 1.0,
  gainB: 1.035,
  split: 0.68,
  saturation: 0.93,
  /** Saturation multiplier at the top of the range, reached over 0.55..0.95 luma. */
  highlightSat: 0.85,
  /** No channel may exceed this after grading. Guards the ramp's top step. */
  chromaCeiling: 0.985,
};

const LUMA_R = 0.2126;
const LUMA_G = 0.7152;
const LUMA_B = 0.0722;

function smoothstep01(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

let _lut: Data3DTexture | null = null;

/**
 * The grade table, built once. 32^3 RGBA8.
 *
 * Built here rather than pulled from GeneratedTextures.gradeLUT so the trim
 * above sits in the same function as the transform it corrects — a grade whose
 * correction lives in a different module is a grade that will be "fixed" twice.
 * The operator order is identical to the original: lift/gamma/gain, contrast
 * about a pivot, split tone by luminance, saturation last.
 */
export function gradeLutTexture(size = 32): Data3DTexture {
  if (_lut) return _lut;

  const n = size * size * size;
  const bytes = new Uint8Array(n * 4);

  const gainR = GRADE.gain.x * TRIM.gainR;
  const gainG = GRADE.gain.y * TRIM.gainG;
  const gainB = GRADE.gain.z * TRIM.gainB;
  const split = GRADE.splitStrength * TRIM.split;
  const satBase = GRADE.saturation * TRIM.saturation;
  const sh = GRADE.shadowTint;
  const hi = GRADE.highlightTint;

  let p = 0;
  for (let bi = 0; bi < size; bi++) {
    for (let gi = 0; gi < size; gi++) {
      for (let ri = 0; ri < size; ri++) {
        let cr = ri / (size - 1);
        let cg = gi / (size - 1);
        let cb = bi / (size - 1);

        // Lift / gamma / gain — the classic three-way grade.
        cr = Math.pow(Math.max(cr * gainR + GRADE.lift.x, 0), 1 / GRADE.gamma.x);
        cg = Math.pow(Math.max(cg * gainG + GRADE.lift.y, 0), 1 / GRADE.gamma.y);
        cb = Math.pow(Math.max(cb * gainB + GRADE.lift.z, 0), 1 / GRADE.gamma.z);

        // Contrast around a pivot below mid so the shadows stay rich.
        cr = (cr - GRADE.contrastPivot) * GRADE.contrast + GRADE.contrastPivot;
        cg = (cg - GRADE.contrastPivot) * GRADE.contrast + GRADE.contrastPivot;
        cb = (cb - GRADE.contrastPivot) * GRADE.contrast + GRADE.contrastPivot;

        // Split tone by luminance.
        const l = LUMA_R * cr + LUMA_G * cg + LUMA_B * cb;
        // THE CROSSOVER, and it was the palette rotation.
        //
        // These were `1 - l*1.6` and `(l - 0.55)*2.2`: the shadow tint reached
        // all the way to luma 0.625 while the gold did not begin until 0.55.
        // A sunlit dirt trail sits at roughly 0.50-0.55 — below the gold, still
        // inside the violet. And shadowTint - 0.5 is (-0.143, -0.210, +0.025),
        // which cuts green hardest and adds blue, so applied to an orange it
        // rotates it toward magenta. Measured across the review set, the lit
        // trail read hue 6 at 26% saturation against an authored 27-33 at
        // 44-45%, with two frames past red into magenta at 348 and 358. The
        // committed dawn-gold was being turned into dusk-pink by its own grade.
        //
        // Now the two weights hand over at ~0.35 with a small overlap: shadows
        // get the violet, mid and up get the gold, and no luminance is left in
        // a gap where only the cool tint applies.
        const sw = Math.max(0, 1 - l * 2.8);
        const hw = Math.max(0, (l - 0.34) * 1.6);
        cr += (sh.r - 0.5) * sw * split + (hi.r - 0.5) * hw * split;
        cg += (sh.g - 0.5) * sw * split + (hi.g - 0.5) * hw * split;
        cb += (sh.b - 0.5) * sw * split + (hi.b - 0.5) * hw * split;

        // Saturation last, so the split tone is amplified with everything else
        // — but rolled off in the highlights. The excess chroma was almost
        // entirely a highlight problem: a lit plateau is exactly the value that
        // has already been pushed up by gain, contrast AND the gold split, so
        // it is the one place a flat saturation boost has nothing left to give.
        const l2 = LUMA_R * cr + LUMA_G * cg + LUMA_B * cb;
        const sat = satBase * (1 - (1 - TRIM.highlightSat) * smoothstep01(0.55, 0.95, l2));
        cr = l2 + (cr - l2) * sat;
        cg = l2 + (cg - l2) * sat;
        cb = l2 + (cb - l2) * sat;

        // Chroma ceiling. Instead of letting a channel clip — which flattens
        // two authored plateaus onto one hue — pull the whole triplet back
        // toward its own luminance until the hottest channel just fits. Hue is
        // preserved exactly; only chroma gives.
        const m = Math.max(cr, cg, cb);
        if (m > TRIM.chromaCeiling) {
          const lc = LUMA_R * cr + LUMA_G * cg + LUMA_B * cb;
          const denom = m - lc;
          if (denom > 1e-5) {
            const k = Math.max(0, (TRIM.chromaCeiling - lc) / denom);
            cr = lc + (cr - lc) * k;
            cg = lc + (cg - lc) * k;
            cb = lc + (cb - lc) * k;
          }
        }

        bytes[p++] = Math.round(Math.min(Math.max(cr, 0), 1) * 255);
        bytes[p++] = Math.round(Math.min(Math.max(cg, 0), 1) * 255);
        bytes[p++] = Math.round(Math.min(Math.max(cb, 0), 1) * 255);
        bytes[p++] = 255;
      }
    }
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

  /**
   * Elliptical vignette. Subtle by construction — it keeps the eye on the trail.
   *
   * TWO THINGS HAD TO BE RIGHT HERE AND ONLY ONE OF THEM WAS.
   *
   * 1. COLOUR SPACE. uVignetteTint carries GRADE.shadowTint, and a three Color
   *    holds LINEAR values — 0x5b4a86 is (0.107, 0.072, 0.235) linear against
   *    (0.357, 0.290, 0.525) as the authored display value. This function runs
   *    AFTER the sRGB encode, so using the raw uniform tinted the corners with
   *    a colour three times darker and far more saturated than the one in the
   *    palette. It is encoded here, once, where it is used.
   *
   * 2. THE MOVE ITSELF. Mixing toward any fixed colour pulls every corner
   *    pixel toward the SAME triplet, which is a loss of chroma by definition:
   *    two different corner colours come out closer together than they went in.
   *    Multiplying by a tint that has been normalised to unit luminance cannot
   *    do that — it is a pure hue rotation, and how dark the corner gets is a
   *    separate, explicit term. Cool toward violet, then darken a little; never
   *    the one operation doing both jobs and desaturating as a side effect.
   */
  vec3 applyVignette(vec3 c, vec2 uv, float aspect) {
    vec2 d = (uv - 0.5) * vec2(aspect, 1.0);
    float r = length(d) * 1.42;
    float v = uVignette * smoothstep(uVignetteSoftness, 1.25, r);
    if (v <= 0.0002) return c;

    vec3 tint = linearToSrgb(uVignetteTint);
    tint /= max(luma(tint), 1e-3);

    vec3 cooled = c * mix(vec3(1.0), tint, saturate1(v * 1.15));
    // The darkening. Deliberately modest: a cel frame is flat plateaus, and a
    // heavy vignette puts a smooth ramp straight across every one of them.
    return cooled * (1.0 - v * 0.45);
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
