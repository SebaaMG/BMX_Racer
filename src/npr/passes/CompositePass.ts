/**
 * CompositePass — the last thing that happens to the frame.
 *
 * Order is the entire content of this file, and it is not negotiable:
 *
 *   1. lens        chromatic aberration + radial blur, on the captured image
 *   2. bloom       added in LINEAR light, where adding light belongs
 *   3. ink         the interior line field, painted OVER the bloom
 *   4. speed lines painted, not added — they are strokes, not exposure
 *   5. shoulder    the only highlight compression in the whole pipeline
 *   6. sRGB encode
 *   7. LUT         the grade, in display space, where its constants live
 *   8. ink flood   a wash toward the ink colour, for crashes
 *   9. impact flash the two-value anime frame
 *  10. desaturate
 *  11. vignette
 *  12. grain
 *
 * Step 3 is the one that is easy to get wrong. If the ink is applied to the
 * HDR image before bloom, the bloom halo from a hot band next to a line spills
 * ACROSS the line and softens it — a glowing, out-of-focus outline, which is
 * the exact opposite of ink. Ink goes on top. An animator inks last.
 *
 * Step 7 is the second. The renderer does no tone mapping, and the grade
 * constants in Palette.GRADE (a 0.46 contrast pivot, a 0.012 lift) are
 * display-referred numbers. Applying them to linear radiance would put the
 * pivot at 71% sRGB and lift the black point by a third of a stop.
 */

import {
  Color,
  IUniform,
  RepeatWrapping,
  Texture,
  Vector2,
  WebGLRenderer,
  WebGLRenderTarget,
} from 'three';
import { GLSL_COMMON, GLSL_FRAG_OUT, GLSL_POST_HELPERS } from '../ShaderChunks';
import { GLSL_GRADE, gradeLutTexture } from '../LUT';
import { paperGrain } from '../GeneratedTextures';
import { GRADE, HUD_PALETTE, INK } from '../Palette';
import { POST_STATE } from '../NprGlobals';
import { FullscreenPass } from './Fullscreen';

export const COMPOSITE_DEBUG = {
  off: 0,
  ink: 1,
  bloom: 2,
  ungraded: 3,
} as const;

const FRAGMENT = /* glsl */ `
  precision highp float;
  ${GLSL_COMMON}
  ${GLSL_POST_HELPERS}
  ${GLSL_GRADE}
  ${GLSL_FRAG_OUT}

  uniform sampler2D uScene;
  uniform sampler2D uBloom;
  uniform sampler2D uInk;

  uniform vec2  uResolution;
  uniform float uTime;

  uniform float uBloomIntensity;
  uniform float uLineOpacity;

  uniform float uSpeedIntensity;
  uniform vec2  uSpeedFocus;
  uniform vec3  uSpeedColor;

  uniform float uChroma;
  uniform float uRadialBlur;

  uniform float uImpactFlash;
  uniform vec3  uImpactTint;
  uniform float uInkFlood;
  uniform vec3  uInkColor;
  uniform float uDesaturate;

  uniform float uDebug;

  in vec2 vUv;

  void main() {
    vec2 uv = vUv;
    float aspect = uResolution.x / max(uResolution.y, 1.0);

    // ── 1. Lens ─────────────────────────────────────────────────────────────
    vec3 col;
    if (uChroma > 0.0005) {
      // Radial split, not a fixed XY offset: a lens disperses outward from the
      // optical centre, and a fixed offset reads as a broken screenshot.
      vec2 d = (uv - 0.5) * uChroma * 0.010;
      col.r = texture(uScene, uv + d).r;
      col.g = texture(uScene, uv).g;
      col.b = texture(uScene, uv - d).b;
    } else {
      col = texture(uScene, uv).rgb;
    }

    if (uRadialBlur > 0.0005) {
      // Six taps toward the focus point. Enough to read as speed, few enough
      // that it stays inside the budget when a boost and a landing land in the
      // same frame.
      vec2 toFocus = uSpeedFocus - uv;
      vec3 acc = col;
      for (int i = 1; i < 6; i++) {
        float t = float(i) / 5.0;
        acc += texture(uScene, uv + toFocus * t * uRadialBlur * 0.16).rgb;
      }
      col = acc * (1.0 / 6.0);
    }

    // ── 2. Bloom, in linear light ───────────────────────────────────────────
    col += texture(uBloom, uv).rgb * uBloomIntensity;

    // ── 3. Interior ink, over the bloom ─────────────────────────────────────
    vec4 ink = texture(uInk, uv);
    col = mix(col, srgbToLinear(ink.rgb), saturate1(ink.a * uLineOpacity));

    // ── 4. Speed lines ──────────────────────────────────────────────────────
    if (uSpeedIntensity > 0.001) {
      float sl = speedLines(uv, uSpeedFocus, uSpeedIntensity, uTime, aspect);
      // Painted, not added. A speed line in animation is a stroke of paint at
      // a flat value; adding light instead gives a glow that blows out the sky
      // and leaves nothing over dark trees.
      col = mix(col, uSpeedColor, saturate1(sl) * 0.92);
    }

    if (uDebug > 2.5 && uDebug < 3.5) { fragColor = vec4(linearToSrgb(col), 1.0); return; }
    if (uDebug > 0.5 && uDebug < 1.5) { fragColor = vec4(vec3(ink.a), 1.0); return; }
    if (uDebug > 1.5 && uDebug < 2.5) {
      fragColor = vec4(linearToSrgb(texture(uBloom, uv).rgb), 1.0);
      return;
    }

    // ── 5-6. Shoulder and encode ────────────────────────────────────────────
    vec3 disp = linearToSrgb(displayShoulder(max(col, vec3(0.0))));

    // ── 7. Grade ────────────────────────────────────────────────────────────
    disp = applyGradeLut(disp);

    // ── 8. Ink flood ────────────────────────────────────────────────────────
    if (uInkFlood > 0.001) {
      // Floods inward from the frame edge and spares the highlights, so it
      // reads as ink washing across the cel rather than as a fade to black.
      vec2 d = (uv - 0.5) * vec2(aspect, 1.0);
      float r = saturate1(length(d) * 1.35);
      float flood = saturate1(uInkFlood * (0.42 + r * 0.85));
      float keep = smoothstep(0.70, 0.96, luma(disp)) * 0.80;
      disp = mix(disp, linearToSrgb(uInkColor), flood * (1.0 - keep));
    }

    // ── 9. Impact flash ─────────────────────────────────────────────────────
    if (uImpactFlash > 0.001) {
      // The anime impact frame: the image collapses to TWO values on a hard
      // threshold. A simple additive white flash reads as a camera artefact;
      // this reads as a held drawing.
      float l = luma(disp);
      float k = smoothstep(0.33, 0.39, l);
      vec3 hi = linearToSrgb(uImpactTint);
      vec3 lo = linearToSrgb(uInkColor) * 0.8;
      vec3 punch = mix(lo, hi, k);
      float f = saturate1(uImpactFlash);
      disp = mix(disp, punch, f);
      // A short additive lift on the very hardest hits, so a big one still
      // blows the frame out rather than merely posterising it.
      disp += hi * f * f * 0.30;
    }

    // ── 10-12. Tail ─────────────────────────────────────────────────────────
    if (uDesaturate > 0.001) {
      disp = mix(disp, vec3(luma(disp)), saturate1(uDesaturate));
    }
    disp = applyVignette(disp, uv, aspect);
    disp = applyGrain(disp, gl_FragCoord.xy);

    fragColor = vec4(clamp(disp, 0.0, 1.0), 1.0);
  }
`;

export class CompositePass {
  private pass: FullscreenPass;
  readonly uniforms: Record<string, IUniform>;

  constructor(width: number, height: number) {
    const grain = paperGrain();
    grain.wrapS = grain.wrapT = RepeatWrapping;
    const lut = gradeLutTexture();

    this.uniforms = {
      uScene: { value: null },
      uBloom: { value: null },
      uInk: { value: null },
      uResolution: { value: new Vector2(width, height) },
      uTime: { value: 0 },
      uBloomIntensity: { value: GRADE.bloomIntensity },
      uLineOpacity: { value: 1 },
      uSpeedIntensity: { value: 0 },
      uSpeedFocus: { value: new Vector2(0.5, 0.52) },
      uSpeedColor: { value: new Color().copy(HUD_PALETTE.paper) },
      uChroma: { value: 0 },
      uRadialBlur: { value: 0 },
      uImpactFlash: { value: 0 },
      uImpactTint: { value: new Color(1, 1, 1) },
      uInkFlood: { value: 0 },
      uInkColor: { value: new Color().copy(INK) },
      uDesaturate: { value: 0 },
      uDebug: { value: 0 },
      // Grade tail (declared by GLSL_GRADE).
      uLut: { value: lut },
      uLutSize: { value: (lut.image as { width: number }).width },
      uGrain: { value: grain },
      uGrainScale: { value: new Vector2(1 / 512, 1 / 512) },
      uGrainStrength: { value: 0.016 },
      uVignette: { value: GRADE.vignetteStrength },
      uVignetteSoftness: { value: GRADE.vignetteSoftness },
      uVignetteTint: { value: GRADE.shadowTint.clone() },
    };

    this.pass = new FullscreenPass('npr:composite', FRAGMENT, this.uniforms);
  }

  setSize(width: number, height: number): void {
    (this.uniforms.uResolution.value as Vector2).set(width, height);
  }

  setDebug(mode: number): void {
    this.uniforms.uDebug.value = mode;
  }

  /** Pull the per-frame dials the game drives out of POST_STATE. */
  syncState(time: number): void {
    const u = this.uniforms;
    u.uTime.value = time;
    u.uBloomIntensity.value = POST_STATE.bloomIntensity;
    u.uSpeedIntensity.value = POST_STATE.speedLineIntensity;
    (u.uSpeedFocus.value as Vector2).copy(POST_STATE.speedLineFocus);
    u.uChroma.value = POST_STATE.chromaticAberration;
    u.uRadialBlur.value = POST_STATE.radialBlur;
    u.uImpactFlash.value = POST_STATE.impactFlash;
    (u.uImpactTint.value as Color).copy(POST_STATE.impactTint);
    u.uInkFlood.value = POST_STATE.inkFlood;
    u.uDesaturate.value = POST_STATE.desaturate;
    u.uVignette.value = POST_STATE.vignette;
    u.uLineOpacity.value = POST_STATE.lineOpacity;
    u.uGrainStrength.value = POST_STATE.grainStrength;
  }

  render(
    renderer: WebGLRenderer,
    scene: Texture,
    bloom: Texture,
    ink: Texture,
    target: WebGLRenderTarget | null,
  ): void {
    this.uniforms.uScene.value = scene;
    this.uniforms.uBloom.value = bloom;
    this.uniforms.uInk.value = ink;
    this.pass.render(renderer, target);
  }

  dispose(): void {
    this.pass.dispose();
  }
}
