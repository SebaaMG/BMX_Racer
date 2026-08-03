/**
 * Rally camera profile.
 *
 * CameraDirector is a mature spring/boom solver and should not be forked. Its
 * exported tuning object is intentionally live, so this module applies a
 * car-scale profile before any director instance is constructed.
 *
 * The original values frame a 1.95 m upright rider and saturate the speed lens
 * at 26 m/s. The rally car is 1.46 m tall, 4.10 m long and reaches ~43 m/s.
 */

import { CAMERA_TUNING } from '../fx/CameraDirector';

const tuning = CAMERA_TUNING as unknown as Record<string, number>;

Object.assign(tuning, {
  // Lens opens over the car's actual performance range rather than being fully
  // saturated before 100 km/h.
  fovBase: 58,
  fovTop: 74,
  referenceSpeed: 37,
  fovSaturation: 1.25,

  // A longer, lower boom shows the full rear stance and leaves room for yaw.
  framingConstant: 3.78,
  chaseDistMin: 4.75,
  chaseDistMax: 6.90,
  chaseHeight: 1.68,
  chaseHeightSpeedGain: -0.10,
  subjectPivotHeight: 0.38,

  // Rally rotation should sweep across the frame, but camera roll must not
  // imitate the large lean of a bicycle.
  cornerSwing: 0.31,
  cornerSwingMax: 3.85,
  cornerLookArc: 0.88,
  rollGain: 0.014,
  rollMax: 0.18,
  lagHalfLifeSlow: 0.16,
  lagHalfLifeFast: 0.31,

  // Car-sized boom/occlusion volumes.
  bodyRadius: 1.18,
  subjectClearRadius: 1.58,
  occluderRadius: 1.24,
  occluderClearance: 1.46,
  occluderLiftRadius: 1.95,
  riderTop: 1.10,

  // Hand-framed close-ups use the car's height, not a standing rider span.
  orbitSubjectSpan: 1.46,
  framedMinDist: 3.55,
});

export const RALLY_CAMERA_PROFILE_APPLIED = true;
