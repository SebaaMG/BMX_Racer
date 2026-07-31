/**
 * ImpactFrames — the anime impact hold.
 *
 * The convention: on a hit that matters, the film stops. One or two frames of
 * a held, high-contrast, often near-monochrome drawing, then it cuts straight
 * back to full motion. It costs almost nothing and it is the single largest
 * multiplier on how heavy a hit feels, because the eye reads the discontinuity
 * as force rather than as a stutter.
 *
 * It is also the effect most easily ruined. Used on every landing it stops
 * being punctuation and becomes framerate. So this module is mostly a
 * gatekeeper:
 *
 *   SEVERITY THRESHOLD  0.42 for landings, 0.30 for crashes.
 *                       A crash is always worth a frame; a landing has to earn it.
 *   COOLDOWN            0.85 s. Below that, two hits blur into one long hitch.
 *   BURST LIMIT         3 holds in any 6 s window. A rider bouncing down a rock
 *                       garden would otherwise trigger on every impact and the
 *                       game would look broken rather than brutal.
 *   FRAME COUNT         1 frame under 0.72 severity, 2 above. Never 3 — at
 *                       60fps a third frame reads as a dropped frame.
 *
 * Below the threshold we still fire a FLASH without freezing, so smaller hits
 * get punctuation without the game stopping. That split is what lets the hold
 * stay rare.
 *
 * The flash itself decays as an explicit STAIRCASE — 1.0, 0.55, 0.22, cut —
 * rather than a curve. A smoothly fading white flash is a bloom artefact; a
 * flash that holds three discrete values and then stops is a drawn effect.
 */

import { Color } from 'three';
import { clamp01 } from '../core/MathX';
import { POST_STATE } from '../npr/NprGlobals';
import { HUD_PALETTE, SUN_RIM_COLOR } from '../npr/Palette';

/** Thresholds and timings, exported so the HUD/debug overlay can show them. */
export const IMPACT_TUNING = {
  landingThreshold: 0.42,
  crashThreshold: 0.30,
  /** Below the hold threshold but above this, we flash without freezing. */
  flashOnlyThreshold: 0.16,
  cooldown: 0.85,
  burstWindow: 6.0,
  burstLimit: 3,
  /** Severity at or above which we hold for two frames instead of one. */
  twoFrameSeverity: 0.72,
  /** Seconds of partial-speed recovery after the hold releases. */
  recoverTime: 0.10,
  recoverScale: 0.34,
  /**
   * Staircase step duration, seconds. Three steps then cut.
   *
   * 0.055 gave a 165 ms flash — 10 frames at 60fps against a brief that asks
   * for a 1-2 frame hold. At that length it stops reading as an impact accent
   * and starts reading as a broken post-process. 0.018 puts the full-strength
   * step on one frame and the two decay steps on one each.
   */
  flashStep: 0.018,
} as const;

const _tint = new Color();

export class ImpactFrames {
  /** 0 while the world is held, then a short partial-speed tail, then 1. */
  timeScale = 1;

  /** True on any frame the simulation is fully frozen. */
  get frozen(): boolean {
    return this.framesLeft > 0;
  }

  private framesLeft = 0;
  private cooldown = 0;
  private recover = 0;

  private flashT = -1;
  private flashPeak = 0;
  private inkPeak = 0;
  private flashTint = new Color(1, 1, 1);

  private clock = 0;
  /** Ring of the last few hold timestamps, for the burst limiter. */
  private recent = new Float32Array(IMPACT_TUNING.burstLimit);
  private recentHead = 0;

  constructor() {
    this.recent.fill(-1e6);
  }

  /**
   * Request an impact frame.
   *
   * `intensity` 0..1. `tint` is an optional sRGB hex for the flash colour —
   * defaults to the hot rim gold so the flash sits inside the palette instead
   * of punching a pure white hole in the frame.
   *
   * Returns true if the request produced a genuine hold (as opposed to a
   * flash-only, or nothing at all). The caller does not need the return value;
   * it exists so the audio system can decide whether to fire the big hit.
   */
  trigger(intensity: number, tint?: number, isCrash = false): boolean {
    const s = clamp01(intensity);
    const threshold = isCrash ? IMPACT_TUNING.crashThreshold : IMPACT_TUNING.landingThreshold;

    if (s < threshold || this.cooldown > 0 || this.burstExhausted()) {
      if (s >= IMPACT_TUNING.flashOnlyThreshold) this.flashOnly(s * 0.55, tint);
      return false;
    }

    this.framesLeft = s >= IMPACT_TUNING.twoFrameSeverity ? 2 : 1;
    this.cooldown = IMPACT_TUNING.cooldown;
    this.recent[this.recentHead] = this.clock;
    this.recentHead = (this.recentHead + 1) % this.recent.length;

    this.flashT = 0;
    this.flashPeak = 0.55 + 0.45 * s;
    // Ink flood is the other half of the convention: the frame collapses toward
    // the ink colour at the same moment it blows out, which is what gives the
    // held drawing its poster-like contrast instead of just making it bright.
    this.inkPeak = 0.30 + 0.55 * s * (isCrash ? 1.0 : 0.7);
    this.setTint(tint, isCrash);
    return true;
  }

  /**
   * A flash with no freeze. Boost ignition, checkpoint gates, a clean landing
   * that doesn't deserve to stop the world.
   */
  flashOnly(intensity: number, tint?: number): void {
    const s = clamp01(intensity);
    if (s <= 0.01) return;
    // Never let a small flash stomp a big one that is still playing.
    const peak = s * 0.7;
    if (this.flashT >= 0 && peak <= this.flashPeak * this.staircase(this.flashT)) return;
    this.flashT = 0;
    this.flashPeak = peak;
    this.inkPeak = s * 0.18;
    this.setTint(tint, false);
  }

  private setTint(tint: number | undefined, isCrash: boolean): void {
    if (tint !== undefined) {
      _tint.setHex(tint);
      this.flashTint.copy(_tint);
    } else if (isCrash) {
      // A crash flashes red-hot, not white — it should read as damage.
      this.flashTint.copy(HUD_PALETTE.red).lerp(SUN_RIM_COLOR, 0.45);
    } else {
      this.flashTint.copy(SUN_RIM_COLOR);
    }
  }

  private burstExhausted(): boolean {
    let n = 0;
    for (let i = 0; i < this.recent.length; i++) {
      if (this.clock - this.recent[i] < IMPACT_TUNING.burstWindow) n++;
    }
    return n >= IMPACT_TUNING.burstLimit;
  }

  /**
   * The three-step staircase. Values held flat for `flashStep` seconds each,
   * then cut to zero. Returns a multiplier in {1, 0.55, 0.22, 0}.
   */
  private staircase(t: number): number {
    if (t < 0) return 0;
    const step = Math.floor(t / IMPACT_TUNING.flashStep);
    if (step === 0) return 1;
    if (step === 1) return 0.55;
    if (step === 2) return 0.22;
    return 0;
  }

  /**
   * Advance. `dt` MUST be the real, unscaled frame delta — this is the clock
   * that ends the freeze, so feeding it the scaled dt would hold forever.
   *
   * Called once per RENDERED frame; the freeze is counted in frames, not
   * seconds, because "one frame" is the actual unit of the convention.
   */
  update(dt: number): void {
    this.clock += dt;
    if (this.cooldown > 0) this.cooldown -= dt;

    if (this.framesLeft > 0) {
      this.timeScale = 0;
      this.framesLeft--;
      if (this.framesLeft === 0) this.recover = IMPACT_TUNING.recoverTime;
    } else if (this.recover > 0) {
      // A short partial-speed tail out of the hold. Not a slow-mo — 100ms of
      // a third speed, which reads as the world taking the hit and shrugging
      // it off rather than as an effect.
      this.recover -= dt;
      const u = clamp01(1 - this.recover / IMPACT_TUNING.recoverTime);
      this.timeScale = IMPACT_TUNING.recoverScale + (1 - IMPACT_TUNING.recoverScale) * (u * u);
      if (this.recover <= 0) this.timeScale = 1;
    } else {
      this.timeScale = 1;
    }

    if (this.flashT >= 0) {
      this.flashT += dt;
      const s = this.staircase(this.flashT);
      if (s <= 0) {
        this.flashT = -1;
        POST_STATE.impactFlash = 0;
        POST_STATE.inkFlood = 0;
        POST_STATE.desaturate = 0;
      } else {
        POST_STATE.impactFlash = this.flashPeak * s;
        POST_STATE.impactTint.copy(this.flashTint);
        // Ink flood decays one step faster than the flash so the frame goes
        // white-hot first and graphic second, which is the order the eye reads.
        POST_STATE.inkFlood = this.inkPeak * this.staircase(this.flashT + IMPACT_TUNING.flashStep);
        // The blown-out frame loses colour before it loses brightness.
        POST_STATE.desaturate = this.flashPeak * s * 0.45;
      }
    }
  }

  /** Hard reset — used on race restart and by the capture harness. */
  reset(): void {
    this.framesLeft = 0;
    this.cooldown = 0;
    this.recover = 0;
    this.flashT = -1;
    this.timeScale = 1;
    this.recent.fill(-1e6);
    this.recentHead = 0;
    POST_STATE.impactFlash = 0;
    POST_STATE.inkFlood = 0;
    POST_STATE.desaturate = 0;
    POST_STATE.impactTint.copy(SUN_RIM_COLOR);
  }
}
