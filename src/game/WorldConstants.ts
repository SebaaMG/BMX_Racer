/**
 * WorldConstants — the canonical numbers, shared by terrain and track.
 *
 * Terrain generation and course layout are the one genuinely coupled pair in
 * this project: the mountain has to be shaped so the course is rideable, and
 * the course has to sit on the mountain so it looks carved rather than pasted.
 *
 * They are decoupled here by making the ROUTE the authority. The route control
 * points below are fixed. The terrain generator adds a "corridor prior" to its
 * heightfield before erosion — a gentle bias that guarantees a continuously
 * descending, rideable band of ground along this line — and then erosion runs
 * over the whole thing so the corridor ends up with real gullies and spurs
 * around it rather than looking like a bulldozed ramp. The track builder then
 * conforms its spline to the FINAL eroded surface.
 *
 * Neither system needs to know anything about the other's internals.
 */

import { Vector3 } from 'three';
import { TrackSectionKind } from './Contracts';

// ── World scale ──────────────────────────────────────────────────────────────
/** Heightfield resolution. 2048² over 4096m gives a 2m sample spacing. */
export const HEIGHTMAP_SIZE = 2048;
/** World extent of the heightfield, metres, centred on the origin. */
export const WORLD_SIZE = 4096;
export const WORLD_HALF = WORLD_SIZE / 2;
export const METRES_PER_SAMPLE = WORLD_SIZE / HEIGHTMAP_SIZE;

/** Vertical range of the mountain. */
export const SEA_LEVEL = 0;
export const SUMMIT_HEIGHT = 640;
export const VALLEY_HEIGHT = 55;

/** Altitude zone boundaries, metres. Hard cel transitions, not blends. */
export const ZONE = {
  waterLine: 62,
  grassTop: 210,
  dirtTop: 330,
  rockTop: 470,
  screeTop: 545,
  // Above screeTop is snow.
  /** Slopes steeper than this are always rock, regardless of altitude. */
  rockSlopeRad: 0.72,
  /** Slopes steeper than this are always scree above the grass line. */
  screeSlopeRad: 0.52,
};

// ── The route ────────────────────────────────────────────────────────────────
/**
 * Control points for the descent, summit to valley. XZ only — Y is taken from
 * the terrain at build time. Points are dense through technical sections and
 * sparse through fast ones, which is what shapes the pacing: the spline's own
 * curvature distribution becomes the course's rhythm.
 *
 * Reading the shape: a long raking traverse off the summit, a straight fast
 * plunge down the scree face, six tightening switchbacks through the tree line,
 * a broken diagonal through the rock garden, two jumps on a straight approach,
 * a knife-edge ridge, a drop into the stream, and a final open run-out.
 */
export interface RouteControl {
  x: number;
  z: number;
  /** Rideable half-width here, metres. Narrow = technical. */
  halfWidth: number;
  /** Requested bank, radians. Positive banks the outside of a left turn. */
  bank?: number;
  section: TrackSectionKind;
  /** Optional explicit height offset above the eroded terrain. */
  lift?: number;
}

export const ROUTE: RouteControl[] = [
  // ── 1. Technical start: exposed rock, tight, off-camber ──────────────────
  { x: -120, z: -1620, halfWidth: 3.6, section: TrackSectionKind.TechnicalStart },
  { x: -104, z: -1560, halfWidth: 3.2, section: TrackSectionKind.TechnicalStart },
  { x: -66, z: -1516, halfWidth: 2.9, section: TrackSectionKind.TechnicalStart },
  { x: -18, z: -1494, halfWidth: 2.8, section: TrackSectionKind.TechnicalStart },
  { x: 34, z: -1478, halfWidth: 3.0, section: TrackSectionKind.TechnicalStart },
  { x: 78, z: -1436, halfWidth: 3.2, bank: 0.18, section: TrackSectionKind.TechnicalStart },
  { x: 96, z: -1372, halfWidth: 3.6, bank: 0.22, section: TrackSectionKind.TechnicalStart },

  // ── 2. Scree run: open, straight, build speed ────────────────────────────
  { x: 88, z: -1280, halfWidth: 5.5, section: TrackSectionKind.ScreeRun },
  { x: 64, z: -1160, halfWidth: 7.0, section: TrackSectionKind.ScreeRun },
  { x: 40, z: -1030, halfWidth: 8.0, section: TrackSectionKind.ScreeRun },
  { x: 26, z: -900, halfWidth: 8.0, section: TrackSectionKind.ScreeRun },
  { x: 24, z: -790, halfWidth: 6.5, section: TrackSectionKind.ScreeRun },

  // ── 3. Switchbacks: six bermed corners, tightening ───────────────────────
  { x: 52, z: -710, halfWidth: 4.6, bank: -0.34, section: TrackSectionKind.Switchbacks },
  { x: 118, z: -680, halfWidth: 4.2, bank: -0.46, section: TrackSectionKind.Switchbacks },
  { x: 150, z: -630, halfWidth: 4.0, bank: -0.40, section: TrackSectionKind.Switchbacks },
  { x: 120, z: -578, halfWidth: 3.9, bank: 0.44, section: TrackSectionKind.Switchbacks },
  { x: 40, z: -556, halfWidth: 3.9, bank: 0.48, section: TrackSectionKind.Switchbacks },
  { x: -42, z: -536, halfWidth: 3.8, bank: 0.44, section: TrackSectionKind.Switchbacks },
  { x: -96, z: -496, halfWidth: 3.7, bank: -0.50, section: TrackSectionKind.Switchbacks },
  { x: -86, z: -434, halfWidth: 3.6, bank: -0.52, section: TrackSectionKind.Switchbacks },
  { x: -20, z: -404, halfWidth: 3.6, bank: -0.44, section: TrackSectionKind.Switchbacks },
  { x: 54, z: -388, halfWidth: 3.7, bank: 0.46, section: TrackSectionKind.Switchbacks },
  { x: 106, z: -344, halfWidth: 3.6, bank: 0.54, section: TrackSectionKind.Switchbacks },
  { x: 96, z: -282, halfWidth: 3.5, bank: 0.50, section: TrackSectionKind.Switchbacks },
  { x: 34, z: -252, halfWidth: 3.8, bank: -0.30, section: TrackSectionKind.Switchbacks },

  // ── 4. Rock garden: broken diagonal, punishes a bad line ─────────────────
  { x: -18, z: -212, halfWidth: 4.4, section: TrackSectionKind.RockGarden },
  { x: -54, z: -156, halfWidth: 4.8, section: TrackSectionKind.RockGarden },
  { x: -66, z: -96, halfWidth: 5.2, section: TrackSectionKind.RockGarden },
  { x: -48, z: -36, halfWidth: 5.0, section: TrackSectionKind.RockGarden },
  { x: -14, z: 12, halfWidth: 4.6, section: TrackSectionKind.RockGarden },

  // ── 5. Tabletop: straight, clear takeoff read ────────────────────────────
  { x: 10, z: 62, halfWidth: 5.0, section: TrackSectionKind.Tabletop },
  { x: 22, z: 122, halfWidth: 5.4, section: TrackSectionKind.Tabletop },
  { x: 28, z: 182, halfWidth: 5.4, section: TrackSectionKind.Tabletop },

  // ── 6. Ravine gap: a real hole in the ground ─────────────────────────────
  { x: 30, z: 246, halfWidth: 5.0, section: TrackSectionKind.RavineGap },
  { x: 32, z: 306, halfWidth: 4.6, section: TrackSectionKind.RavineGap },
  { x: 34, z: 362, halfWidth: 5.0, section: TrackSectionKind.RavineGap },

  // ── 7. Ridge sprint: exposure on both sides ──────────────────────────────
  { x: 40, z: 430, halfWidth: 2.6, section: TrackSectionKind.RidgeSprint },
  { x: 54, z: 510, halfWidth: 2.3, section: TrackSectionKind.RidgeSprint },
  { x: 74, z: 596, halfWidth: 2.2, section: TrackSectionKind.RidgeSprint },
  { x: 92, z: 684, halfWidth: 2.4, section: TrackSectionKind.RidgeSprint },
  { x: 100, z: 766, halfWidth: 2.9, section: TrackSectionKind.RidgeSprint },

  // ── 8. Stream bed: a drop in, wet rock, then out ─────────────────────────
  { x: 92, z: 838, halfWidth: 3.8, section: TrackSectionKind.StreamBed },
  { x: 66, z: 898, halfWidth: 4.4, section: TrackSectionKind.StreamBed },
  { x: 40, z: 962, halfWidth: 4.6, section: TrackSectionKind.StreamBed },
  { x: 30, z: 1032, halfWidth: 4.4, section: TrackSectionKind.StreamBed },

  // ── 9. Final sprint ──────────────────────────────────────────────────────
  { x: 44, z: 1110, halfWidth: 5.6, section: TrackSectionKind.FinalSprint },
  { x: 72, z: 1200, halfWidth: 6.4, section: TrackSectionKind.FinalSprint },
  { x: 96, z: 1300, halfWidth: 7.0, section: TrackSectionKind.FinalSprint },
  { x: 106, z: 1400, halfWidth: 7.0, section: TrackSectionKind.FinalSprint },
  { x: 108, z: 1490, halfWidth: 6.0, section: TrackSectionKind.FinalSprint },
];

/**
 * Explicit terrain features the mountain must contain for the course to work.
 * The terrain generator carves these into the heightfield after erosion so
 * they survive intact — erosion would otherwise silt up a ravine or round off
 * a jump lip, and "the jump got eroded away" is not a debuggable failure.
 */
export interface TerrainFeature {
  kind: 'ravine' | 'tabletop' | 'berm-bowl' | 'stream-channel' | 'ridge-narrow' | 'rock-garden' | 'start-plateau' | 'finish-flat';
  /** Route distance (metres) this feature is centred on, or explicit XZ. */
  at?: number;
  x?: number;
  z?: number;
  /** Feature-specific parameters. */
  params: Record<string, number>;
}

export const TERRAIN_FEATURES: TerrainFeature[] = [
  { kind: 'start-plateau', x: -120, z: -1620, params: { radius: 26, flatness: 0.92 } },
  // The tabletop: a raised flat-topped mound with a clean lip.
  { kind: 'tabletop', x: 25, z: 150, params: { length: 34, width: 16, height: 4.2, lipSharpness: 0.72 } },
  // The ravine: a genuine gap the rider must clear. Steep sides, 11m across.
  { kind: 'ravine', x: 32, z: 306, params: { width: 11.5, depth: 26, length: 240, angle: 1.44 } },
  // The ridge: narrow the crest and drop the flanks hard.
  { kind: 'ridge-narrow', x: 74, z: 596, params: { length: 360, halfWidth: 9, flankDrop: 42, angle: 0.24 } },
  // The stream: a carved channel with a wet floor.
  { kind: 'stream-channel', x: 55, z: 940, params: { width: 14, depth: 5.5, length: 260, angle: 0.86 } },
  { kind: 'rock-garden', x: -50, z: -100, params: { radius: 70, roughness: 1.35, boulderCount: 46 } },
  { kind: 'finish-flat', x: 108, z: 1490, params: { radius: 60, flatness: 0.85 } },
];

// ── Checkpoints ──────────────────────────────────────────────────────────────
/** Normalised positions along the track where a checkpoint gate stands. */
export const CHECKPOINT_TS = [0.0, 0.13, 0.28, 0.44, 0.58, 0.70, 0.82, 0.92, 1.0];

// ── Race ─────────────────────────────────────────────────────────────────────
export const RACER_COUNT = 4;
export const COUNTDOWN_SECONDS = 3.4;
/** Lateral spacing of the four riders on the start line, metres. */
export const START_SPACING = 2.1;

// ── Physics tuning shared between the bike and the AI ────────────────────────
export const BIKE = {
  mass: 92,               // rider + bike, kg
  wheelbase: 1.08,        // m
  wheelRadius: 0.267,     // 26" wheel
  comHeight: 1.02,        // centre of mass above the axle line
  /** Suspension travel, metres. */
  forkTravel: 0.13,
  shockTravel: 0.09,
  forkStiffness: 32000,   // N/m
  forkDamping: 2300,      // Ns/m
  shockStiffness: 46000,
  shockDamping: 3100,
  /** Peak drive force from pedalling, newtons. */
  pedalForce: 460,
  /** Speed beyond which pedalling adds nothing (spun out), m/s. */
  spinOutSpeed: 17.5,
  brakeForceRear: 1450,
  brakeForceFront: 2350,
  /** Aerodynamic drag coefficient — 0.5 * rho * Cd * A. */
  dragK: 0.42,
  /** Maximum steering angle at the bars, radians. */
  maxSteer: 0.62,
  /** How much steering authority falls off with speed. */
  steerSpeedFalloff: 0.055,
  maxLean: 0.86,
  /** Air rotation rates, rad/s. */
  airPitchRate: 4.6,
  airYawRate: 3.9,
  airRollRate: 3.4,
  /** Landing tolerance: angle mismatch above this crashes, radians. */
  landingAngleTolerance: 0.62,
  /** Vertical impact speed above this crashes, m/s. */
  landingSpeedTolerance: 15.5,
  /** Boost. */
  boostForce: 620,
  boostDrainPerSecond: 0.42,
  boostPerTrickPoint: 0.00042,
  gravity: 20.4,          // exaggerated — arcade weight, not 9.81
};

export const SURFACES = {
  trail: { grip: 1.0, rollingResistance: 0.24, drag: 0.0, dustAmount: 0.55, harshness: 0.7 },
  dirt: { grip: 0.88, rollingResistance: 0.42, drag: 0.08, dustAmount: 1.0, harshness: 0.8 },
  grass: { grip: 0.74, rollingResistance: 0.92, drag: 0.28, dustAmount: 0.35, harshness: 0.55 },
  rock: { grip: 0.92, rollingResistance: 0.36, drag: 0.12, dustAmount: 0.18, harshness: 1.45 },
  scree: { grip: 0.46, rollingResistance: 0.70, drag: 0.22, dustAmount: 1.35, harshness: 1.15 },
  snow: { grip: 0.58, rollingResistance: 1.05, drag: 0.30, dustAmount: 1.5, harshness: 0.45 },
  water: { grip: 0.52, rollingResistance: 1.6, drag: 0.55, dustAmount: 1.8, harshness: 0.9 },
};

// ── Derived helpers ──────────────────────────────────────────────────────────

/** Route control points as Vector3 with y = 0 (filled in from terrain later). */
export function routePoints(): Vector3[] {
  return ROUTE.map((r) => new Vector3(r.x, 0, r.z));
}

/** Total straight-line length of the route polyline — a rough course length. */
export function routePolylineLength(): number {
  let d = 0;
  for (let i = 1; i < ROUTE.length; i++) {
    const dx = ROUTE[i].x - ROUTE[i - 1].x;
    const dz = ROUTE[i].z - ROUTE[i - 1].z;
    d += Math.hypot(dx, dz);
  }
  return d;
}

/**
 * The descent profile the corridor prior must produce: a monotonic drop from
 * SUMMIT_HEIGHT at the start to VALLEY_HEIGHT at the finish, with the gradient
 * shaped so the scree run is steep, the switchbacks are moderate, and the final
 * sprint is gentle. Returns the target height at a normalised route position.
 */
export function corridorHeightAt(t: number): number {
  // Piecewise gradient profile, integrated. Keys are (t, relative steepness).
  const keys: [number, number][] = [
    [0.00, 0.55],  // technical start — moderate
    [0.09, 0.55],
    [0.13, 1.60],  // scree face — steep
    [0.26, 1.55],
    [0.30, 0.85],  // switchbacks — moderate, traversing
    [0.46, 0.80],
    [0.50, 1.10],  // rock garden
    [0.58, 1.05],
    [0.61, 0.70],  // tabletop approach — flatter so the lip reads
    [0.66, 0.62],
    [0.69, 1.30],  // ravine drop
    [0.73, 0.95],
    [0.76, 0.75],  // ridge — nearly level, exposure does the work
    [0.86, 0.72],
    [0.89, 1.25],  // stream drop-in
    [0.93, 0.80],
    [0.96, 0.45],  // run-out
    [1.00, 0.35],
  ];

  // Integrate the profile once to normalise it, then evaluate.
  const N = 512;
  let total = 0;
  const cum: number[] = new Array(N + 1);
  for (let i = 0; i <= N; i++) {
    const tt = i / N;
    cum[i] = total;
    total += sampleKeys(keys, tt) / N;
  }
  const idx = Math.min(N, Math.max(0, Math.floor(t * N)));
  const frac = cum[idx] / (total || 1);
  return SUMMIT_HEIGHT + (VALLEY_HEIGHT - SUMMIT_HEIGHT) * frac;
}

function sampleKeys(keys: [number, number][], t: number): number {
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    if (t <= keys[i][0]) {
      const [t0, v0] = keys[i - 1];
      const [t1, v1] = keys[i];
      const k = (t - t0) / Math.max(t1 - t0, 1e-6);
      return v0 + (v1 - v0) * (k * k * (3 - 2 * k));
    }
  }
  return keys[keys.length - 1][1];
}
