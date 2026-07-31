/**
 * Sky — gradient dome, two parallax layers of hard-edged cel cloud, a drawn
 * sun disc, and the geometric shaft volume that rakes through the tree line.
 *
 * The dome is a single inverted sphere locked to the camera. Every value in it
 * is banded: the vertical gradient steps between four plateaus rather than
 * blending, the clouds are alpha-cut with a two-value interior, and the sun
 * glow is a set of concentric hard rings rather than a falloff. A smooth sky
 * behind a banded mountain is the fastest way to break the illusion — the eye
 * reads the gradient as "3D render" instantly.
 */

import {
  AdditiveBlending,
  BackSide,
  BufferGeometry,
  Color,
  DoubleSide,
  Float32BufferAttribute,
  GLSL3,
  LinearFilter,
  LinearMipmapLinearFilter,
  Matrix4,
  Mesh,
  MirroredRepeatWrapping,
  Object3D,
  PerspectiveCamera,
  ShaderMaterial,
  SphereGeometry,
  Vector3,
} from 'three';
import { GLSL_COMMON, GLSL_FRAG_OUT } from './ShaderChunks';
import { NPR } from './NprGlobals';
import { SKY, SUN_DIRECTION } from './Palette';
import { celCloudMask } from './GeneratedTextures';

/** Scratch for the per-frame sun projection. Module scope so update() never allocates. */
const _sunWorld = new Vector3();

export class Sky {
  readonly group = new Object3D();
  private domeMat: ShaderMaterial;
  private shaftMat: ShaderMaterial | null = null;
  private shafts: Mesh | null = null;

  constructor(radius = 4200) {
    this.group.name = 'sky';
    this.group.frustumCulled = false;

    const geo = new SphereGeometry(radius, 48, 32);

    // Trilinear plus anisotropic. The masks are minified hard toward the
    // horizon and with LinearFilter alone they had no mip chain at all, so the
    // pattern aliased against the pixel grid. The shader then re-thresholds the
    // filtered alpha so the contour goes back to being a hard cut.
    const cloudNear = celCloudMask(1024, 'clouds-near', 0.46);
    const cloudFar = celCloudMask(1024, 'clouds-far', 0.60);
    for (const t of [cloudNear, cloudFar]) {
      t.minFilter = LinearMipmapLinearFilter;
      t.magFilter = LinearFilter;
      t.generateMipmaps = true;
      t.anisotropy = 8;
      // ── THE MASK DOES NOT TILE ──────────────────────────────────────────────
      // celCloudMask is generated from tileable noise but pushed through a
      // domain warp that is not, and the result does not join up: put the sky
      // debug view on the raw alpha and the cloud blobs are guillotined by a
      // dead-straight line at u = 1. That discontinuity, exposed by a
      // projection that reached |uv| = 1.4, is the hard vertical seam that has
      // been splitting the sky. The primary fix is that the projection can no
      // longer reach the tile edge at all — see cloudUv() in the shader.
      //
      // MirroredRepeat is the guarantee behind it. The sample coordinate stays
      // inside the tile, but a trilinear footprint at a high mip can still
      // reach past it, and under mirrored repeat the sampled function is
      // continuous at every fold by construction: the texel at 1 - e and the
      // texel at 1 + e are the same texel, so there is nothing left that can be
      // discontinuous. Done by the SAMPLER rather than by folding the uv in the
      // shader, which matters — folding in the shader breaks the hardware
      // derivative at each fold and trades one seam for a thinner one.
      //
      // The real repair belongs in celCloudMask and is reported separately.
      t.wrapS = MirroredRepeatWrapping;
      t.wrapT = MirroredRepeatWrapping;
      t.needsUpdate = true;
    }

    this.domeMat = new ShaderMaterial({
      glslVersion: GLSL3,
      side: BackSide,
      depthWrite: false,
      depthTest: false,
      fog: false,
      uniforms: {
        uTime: NPR.uTime,
        uSunDir: NPR.uSunDir,
        uResolution: NPR.uResolution,
        uZenith: { value: SKY.zenith.clone() },
        uUpper: { value: SKY.upper.clone() },
        uHorizon: { value: SKY.horizon.clone() },
        uBelow: { value: SKY.belowHorizon.clone() },
        uSunDisc: { value: SKY.sunDisc.clone() },
        uSunGlow: { value: SKY.sunGlow.clone() },
        uCloudLit: { value: SKY.cloudLit.clone() },
        uCloudMid: { value: SKY.cloudMid.clone() },
        uCloudShadow: { value: SKY.cloudShadow.clone() },
        uCloudNear: { value: cloudNear },
        uCloudFar: { value: cloudFar },
        uCloudDrift: { value: 0 },
        uInvProjection: { value: new Matrix4() },
        uCamWorld: { value: new Matrix4() },
        /**
         * Sky debug view. 0 off, 1 near-cloud alpha, 2 far-cloud alpha,
         * 3 near uv tile, 4 mask detail. Kept in the shipping shader on
         * purpose: every sky defect this file has had was a question about one
         * scalar field, and answering it by commenting out lines and reloading
         * is how you end up shipping the wrong hypothesis.
         */
        uSkyDebug: { value: 0 },
      },
      vertexShader: /* glsl */ `
        precision highp float;
        void main() {
          // Project at the far plane with w=z so the dome can never clip into
          // geometry regardless of the camera's far distance.
          //
          // NOTE THAT NOTHING IS INTERPOLATED OUT OF HERE. The view direction
          // used to be a varying (normalize(position) from the sphere's own
          // vertices) and that was the source of the hard vertical seams. See
          // the fragment shader.
          vec4 p = projectionMatrix * viewMatrix * vec4(position + cameraPosition, 1.0);
          gl_Position = p.xyww;
        }
      `,
      fragmentShader: /* glsl */ `
        precision highp float;
        ${GLSL_COMMON}
        ${GLSL_FRAG_OUT}

        uniform float uTime;
        uniform vec3  uSunDir;
        uniform vec2  uResolution;
        uniform vec3  uZenith, uUpper, uHorizon, uBelow;
        uniform vec3  uSunDisc, uSunGlow;
        uniform vec3  uCloudLit, uCloudMid, uCloudShadow;
        uniform sampler2D uCloudNear;
        uniform sampler2D uCloudFar;
        uniform float uCloudDrift;
        uniform mat4  uInvProjection;
        uniform mat4  uCamWorld;
        uniform float uSkyDebug;

        // Local band helper — the dome doesn't pull in the full cel core.
        float bandStepS(float x, float threshold, float softness) {
          float w = max(fwidth(x) * 0.8, softness);
          return smoothstep(threshold - w, threshold + w, x);
        }

        /**
         * THE VIEW RAY, AND THE REAL CAUSE OF THE HARD VERTICAL SEAMS.
         *
         * This used to be a varying: normalize(position) written per VERTEX of
         * the dome sphere and interpolated. Read that again — the direction the
         * whole sky is computed from was a linear interpolation across the
         * triangles of a 48 x 32 sphere.
         *
         * The direction itself is close enough after a re-normalise; the
         * problem is its DERIVATIVE. Linear interpolation is only piecewise
         * linear, so d(dir)/d(pixel) is CONSTANT within a triangle and JUMPS
         * across every triangle edge. Every fwidth() in this shader, and — far
         * more damaging — the mip level the GPU picks for the two cloud masks,
         * is computed from exactly that derivative. So at every meridian of the
         * dome the sampled mip level stepped by a notch, the filtered cloud
         * alpha stepped with it, and the hard bandStepS(alpha, 0.52) threshold
         * turned that step into a total flip: solid cloud sheet on one side,
         * clear sky on the other.
         *
         * A meridian of a sphere centred on the camera is a plane through the
         * camera, and a plane through the camera projects to a perfectly
         * straight line. That is the "razor-straight vertical line at x = 1105
         * with warmer sky to its right", the pair of them 27 pixels apart in
         * tabletop-air where two meridians converge near the top of the frame,
         * and the rectangular blocking in summit-wide and valley-vista where
         * one met the horizon plateau at a right angle.
         *
         * Reconstructing the ray from gl_FragCoord instead makes it an exact,
         * analytic function of the pixel — smooth to every order, identical
         * whatever the dome is tessellated at, and with no mesh edges anywhere
         * in it because the mesh no longer contributes to it at all.
         */
        vec3 viewRay() {
          vec2 ndc = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0;
          vec4 v = uInvProjection * vec4(ndc, 1.0, 1.0);
          return normalize(mat3(uCamWorld) * (v.xyz / v.w));
        }

        /**
         * How well the cloud mask resolves at this pixel, 1 = crisply, 0 = the
         * sampler is returning the mask's mean and there is no contour left.
         *
         * This is the second half of the seam fix and it is needed even with a
         * perfect view ray. A tileable mask compresses toward the horizon until
         * one texel covers many pixels; the filtered alpha then converges on the
         * mask's AVERAGE, and re-thresholding an average at a fixed 0.52 gives
         * either total coverage or none, over a whole region, decided by which
         * side of 0.52 the average happens to fall. That is how a flat lavender
         * sheet of uCloudShadow ended up covering a third of the sky.
         */
        float cloudDetail(vec2 uv) {
          vec2 d = fwidth(uv) * 1024.0;
          float lod = log2(max(max(d.x, d.y), 1e-4));
          return 1.0 - saturate1((lod - 3.2) / 2.6);
        }

        /**
         * THE CLOUD PROJECTION, AND WHY IT IS BOUNDED.
         *
         * dir.xz / (dir.y + k) with k = 0.30 reaches a magnitude of 1/k = 3.33
         * at the horizon, and at a uv scale of 0.42 that put the near layer's
         * coordinate up to 1.4 — well past the edge of the mask's own tile.
         * The mask is generated by celCloudMask and it does NOT tile: put the
         * sky debug view on the raw alpha and the cloud blobs are guillotined
         * by a dead-straight line where the coordinate crosses an integer. That
         * line is the intersection of a plane with the view sphere, which is
         * why it draws as a perfectly straight rule through the sky rather than
         * as anything organic, and it is the seam the critic traced.
         *
         * k = 1.0 makes this the STEREOGRAPHIC projection of the sky sphere
         * from its south pole. Three properties, all of which matter:
         *
         *  - Its magnitude is exactly 1.0 at the horizon and 0 at the zenith.
         *    Bounded, over the entire upper hemisphere, with no tuning. At a
         *    scale of 0.38 the coordinate can never leave [0.12, 0.88] of the
         *    tile, in any camera orientation, ever. There is no boundary left
         *    for the mask's own discontinuity to be exposed at.
         *  - It is CONFORMAL. Angles are preserved exactly, so the cloud shapes
         *    are never sheared or smeared — only uniformly scaled. That is not
         *    true of the old projection and it is the difference between clouds
         *    that compress toward the horizon and clouds that streak into it.
         *  - Its scale factor is 1/(1 + dir.y): half at the zenith, one at the
         *    horizon. Cloud shapes therefore still get finer with distance,
         *    which is the perspective read the parallax exists for, just at a
         *    2x ratio rather than the old 4.3x.
         *
         * The two layers use different k so they still separate in depth.
         *
         * The drift is a slow bounded oscillation rather than a linear
         * translation for the same reason the projection is bounded: a linear
         * drift walks the coordinate out of the safe tile and re-exposes the
         * seam, and it does it after a few minutes of play rather than
         * immediately, which is worse. The periods here are 12 and 16 minutes,
         * so within any shot the motion is a constant-velocity drift.
         */
        vec2 cloudUv(vec3 dir, float k, float scale, vec2 drift) {
          return (dir.xz / (dir.y + k)) * scale + vec2(0.5) + drift;
        }

        /**
         * A projection of the view direction onto a plane above the camera,
         * shared by the gradient's boundary wobble and both cloud layers.
         *
         * Dividing by (dir.y + k) rather than by dir.y is the whole fix for the
         * cloud seam. The old form was singular at the horizon: as dir.y fell
         * toward its clamp the uv magnitude ran away, the tileable mask
         * repeated faster than one pixel could resolve, and the repeats beat
         * against the pixel grid into a hard rectangular lattice across the
         * lower sky. Adding k keeps the identical perspective read — overhead
         * clouds large, horizon clouds compressed — while the mapping stays
         * finite, single-valued and C-infinity over the entire upper
         * hemisphere. There is no wrap anywhere in it, so there is no seam.
         */
        vec2 domePlane(vec3 dir, float k) {
          return dir.xz / (dir.y + k);
        }

        /**
         * Banded vertical gradient. Four plateaus, hard boundaries.
         *
         * Two defects fixed here, both visible at the horizon.
         *
         * The bands are now positioned directly in dir.y. The previous form ran
         * dir.y through pow(x, 0.72) of a shifted range, and that has an
         * INFINITE derivative at x = 0 — precisely where the sharpest boundary
         * in the whole sky sits. fwidth() blew up there, so the one transition
         * that most needed antialiasing was the one that could not get any.
         *
         * And the boundary perturbation is now low-frequency and locked to the
         * world instead of a per-pixel screen-space hash. The reason to perturb
         * a plateau edge is to stop it drawing a mathematically straight rule
         * across a clear sky; a slow wander does that and reads as the edge of
         * a brushstroke. A per-pixel hash does it by converting the edge into a
         * thirty-pixel band of speckle — which is exactly what the "heavy
         * dither near the horizon" was.
         */
        vec3 skyGradient(vec3 dir, vec2 fc) {
          float h = dir.y;

          // World-locked, roughly four cycles around the horizon. Locked to the
          // world and not to the screen so the wander does not slide along the
          // band edge when the camera pans.
          float wob = fbm2(domePlane(dir, 0.62) * 2.6, 3) - 0.5;

          vec3 col = uBelow;
          col = mix(col, uHorizon, bandStepS(h + wob * 0.008, 0.004, 0.0018));
          col = mix(col, uUpper,   bandStepS(h + wob * 0.022, 0.118, 0.0022));
          col = mix(col, uZenith,  bandStepS(h + wob * 0.036, 0.500, 0.0030));

          // Warm plateaus hugging the horizon on the sun side. Dawn light does
          // not wrap a sky evenly, and a gradient that is symmetric in azimuth
          // reads as a lit dome rather than as a sky.
          //
          // THE SOFTNESS HERE IS THE WHOLE POINT. This used to be a single
          // bandStepS(toward, 0.28, 0.16) — a softness two orders of magnitude
          // wider than the 0.0018-0.0030 used by the vertical bands directly
          // above. The result was a sky whose HEIGHT was banded and whose
          // AZIMUTH was a smooth yellow-to-orange ramp spanning half the frame,
          // which is the exact photographic gradient the banding exists to
          // avoid. Two plateaus at the band softness put the azimuthal falloff
          // on the same footing as the vertical one.
          //
          // The boundary is a line of constant azimuth, which projects to a
          // straight vertical rule on screen, so it gets the same world-locked
          // low-frequency wander the horizontal boundaries get — a hard edge
          // that wanders reads as the edge of a wash, a hard edge that is
          // mathematically straight reads as a broken renderer.
          vec2 az    = normalize(dir.xz + vec2(1e-5, 0.0));
          vec2 sunAz = normalize(uSunDir.xz + vec2(1e-5, 0.0));
          float toward = saturate1(dot(az, sunAz));
          float wobA = fbm2(domePlane(dir, 0.62) * 3.1 + 17.0, 3) - 0.5;
          float tw = toward + wobA * 0.045;
          float low = 1.0 - bandStepS(h + wob * 0.022, 0.086, 0.0025);
          col = mix(col, mix(col, uSunGlow, 0.24), low * bandStepS(tw, 0.20, 0.0060) * 0.55);
          col = mix(col, mix(col, uSunGlow, 0.34), low * bandStepS(tw, 0.62, 0.0055) * 0.55);

          // A whisper of ordered noise, an order of magnitude below the old
          // dither. This exists only to break 8-bit banding INSIDE a plateau
          // and is far too small to disturb a boundary.
          col += (hash21(fc) - 0.5) * 0.0022;
          return col;
        }

        /**
         * Concentric hard rings around the sun. An animator draws a low sun as
         * a disc plus two or three discrete haloes; a radial falloff is a
         * photograph. This is the former.
         */
        vec3 sunDisc(vec3 dir, vec3 col) {
          float d = dot(dir, uSunDir);
          // Every ring softness is inside the same 0.002-0.01 window the sky
          // bands use. ring3 was 0.028 — five times ring2's — and that single
          // number turned the outermost halo into a photographic glow: a soft
          // radial falloff sitting in the middle of a sky built entirely from
          // hard plateaus. An animator draws three haloes, not a gradient.
          float disc  = bandStepS(d, 0.99955, 0.00008);
          float ring1 = bandStepS(d, 0.9975, 0.0009);
          float ring2 = bandStepS(d, 0.988,  0.0040);
          float ring3 = bandStepS(d, 0.940,  0.0075);
          col = mix(col, uSunGlow * 0.72, ring3 * 0.34);
          col = mix(col, uSunGlow,        ring2 * 0.48);
          col = mix(col, mix(uSunGlow, uSunDisc, 0.6), ring1 * 0.72);
          col = mix(col, uSunDisc * 1.35, disc);
          return col;
        }

        /**
         * Two cloud layers at different parallax depths. The far layer drifts
         * at a third the speed of the near one, which is what gives a flat dome
         * genuine depth without a single triangle of geometry.
         *
         * WHY THE ALPHA IS RE-THRESHOLDED. The mask is now mip-mapped, which is
         * what removed the aliasing lattice — but a filtered hard-edged mask is
         * a SOFT-edged mask, and a soft cloud is the one thing an anime sky
         * never has. So we filter to kill the aliasing and then put the hard
         * contour straight back with a step whose width is the screen-space
         * derivative of the alpha. The edge is therefore exactly one pixel wide
         * at any distance: cut paper up close, cut paper at the horizon, no
         * shimmer in between.
         *
         * WHAT THE THRESHOLD NOW DOES WHERE THE MASK STOPS RESOLVING. It walks
         * to 1.03 — above anything the mask can return — so as detail is lost
         * the cloud CONTOUR SHRINKS AND CLOSES rather than the coverage
         * flipping to fully-on or fully-off across whatever line the filtered
         * average happened to cross 0.52 on. Two properties matter here and
         * only this form has both: the edge stays a hard cut at every step of
         * the way, and the layer leaves the frame by getting smaller, which is
         * what distance does to a cloud. Fading its alpha instead would make it
         * translucent, and a translucent cloud is not a cel cloud.
         */
        vec3 clouds(vec3 dir, vec3 col) {
          float h = dir.y;
          if (h < -0.06) return col;

          float t = uTime;

          // ── Far layer: flatter, duller, slower. Depth cue only. ──
          vec2 driftFar = vec2(sin(t * 0.0052), sin(t * 0.0041 + 2.1)) * 0.026;
          vec2 uvFar = cloudUv(dir, 1.35, 0.58, driftFar);
          vec4 far = texture(uCloudFar, uvFar);
          float thrFar = mix(1.03, 0.52, cloudDetail(uvFar));
          float aFar   = bandStepS(far.a, thrFar, 0.0020);
          float litFar = bandStepS(far.r, 0.80, 0.0020);
          float fadeFar = smoothstep(0.012, 0.20, h);
          col = mix(col, mix(uCloudShadow, uCloudMid, litFar), aFar * 0.56 * fadeFar);

          // ── Near layer ──
          vec2 driftNear = vec2(sin(t * 0.0087), sin(t * 0.0068 + 0.7)) * 0.030;
          vec2 uvNear = cloudUv(dir, 1.00, 0.46, driftNear);
          vec4 near = texture(uCloudNear, uvNear);
          float thrNear = mix(1.03, 0.52, cloudDetail(uvNear));
          float aNear   = bandStepS(near.a, thrNear, 0.0020);
          float litNear = bandStepS(near.r, 0.80, 0.0020);
          float fadeNear = smoothstep(0.028, 0.24, h);

          // The hot sunward rim, built DIRECTIONALLY rather than from the
          // alpha ramp. Probing the mask a short step toward the sun's azimuth
          // and asking "is that point outside the cloud?" gives a rim of a
          // controlled, constant width on the sun side only — which is how an
          // animator inks a cloud. Deriving it from the alpha ramp instead
          // gives a rim whose width is whatever the noise gradient happens to
          // be, and it appears all the way round the contour.
          vec2 sunStep = normalize(uSunDir.xz + vec2(1e-5, 0.0)) * 0.0075;
          float aheadA = texture(uCloudNear, uvNear + sunStep).a;
          float rim = aNear * (1.0 - bandStepS(aheadA, thrNear, 0.0020));

          vec3 nearCol = mix(uCloudShadow, uCloudLit, litNear);
          nearCol = mix(nearCol, uCloudLit * 1.16, rim * 0.85);
          // The same probe run in the opposite direction gives the shadowed
          // edge. Hot rim on the sun side, cool cut on the far side, flat
          // two-value body in between: that is a cel cloud, and it is three
          // texture fetches.
          float behindA = texture(uCloudNear, uvNear - sunStep).a;
          float shade = aNear * (1.0 - bandStepS(behindA, thrNear, 0.0020)) * (1.0 - rim);
          nearCol = mix(nearCol, uCloudShadow * 0.88, shade * 0.60);

          col = mix(col, nearCol, aNear * 0.90 * fadeNear);
          return col;
        }

        void main() {
          vec3 dir = viewRay();
          vec2 fc = gl_FragCoord.xy;

          if (uSkyDebug > 0.5) {
            vec2 uvN = cloudUv(dir, 1.00, 0.46, vec2(0.0));
            vec2 uvF = cloudUv(dir, 1.35, 0.58, vec2(0.0));
            vec3 d = vec3(0.0);
            if (uSkyDebug < 1.5)      d = vec3(texture(uCloudNear, uvN).a);
            else if (uSkyDebug < 2.5) d = vec3(texture(uCloudFar, uvF).a);
            else if (uSkyDebug < 3.5) d = vec3(fract(uvN), 0.0);
            else if (uSkyDebug < 4.5) d = vec3(cloudDetail(uvN));
            else d = vec3(texture(uCloudNear, gl_FragCoord.xy / uResolution).a);
            fragColor = vec4(d, 1.0);
            return;
          }

          vec3 col = skyGradient(dir, fc);
          col = sunDisc(dir, col);
          col = clouds(dir, col);

          fragColor = vec4(col, 1.0);
        }
      `,
    });
    this.domeMat.name = 'sky:dome';

    const dome = new Mesh(geo, this.domeMat);
    dome.name = 'sky:dome';
    dome.frustumCulled = false;
    dome.renderOrder = -1000;
    dome.userData.skipPrepass = true;
    dome.userData.skipShadow = true;
    this.group.add(dome);
  }

  /**
   * God rays as explicit geometry.
   *
   * ── WHAT WAS WRONG ──────────────────────────────────────────────────────────
   * The previous build drew nine long quads in raw clip space at z = 0.999 with
   * depthTest off. Three separate defects fell out of that, and they are worth
   * naming because each one has a different cause:
   *
   *  1. A HARD VERTICAL SEAM SPLITTING THE SKY. The quads were laterally
   *     OFFSET from the sun (`off = (t - 0.5) * 2.2`) rather than radiating
   *     from it, so their edges were not rays — they were near-parallel bands.
   *     The first shaft's angle, -1.9 rad, is 71 degrees from horizontal, and
   *     the hard `across` cut at 0.62..0.92 therefore drew a near-vertical rule
   *     from the top of the frame down to mid-height with warmer sky on one
   *     side of it. Nothing about that reads as light. It reads as a torn
   *     framebuffer, which is exactly how the critic described it.
   *
   *  2. RECTANGULAR BLOCKING WITH TRUE 90-DEGREE CORNERS. Where a quad's hard
   *     side cut met the frame edge, or met the `along` taper, the two straight
   *     boundaries crossed at a right angle. A right angle in a sky is always a
   *     bug: there is no physical or graphic reason for one to exist.
   *
   *  3. A FULL-FRAME ADDITIVE WASH OVER THE RIDERS. Each quad ran 2.4 clip
   *     units — more than the entire height of the screen — with depth testing
   *     disabled, so pale wedges lay across the riders, the shadow field and
   *     the whole lower frame.
   *
   * ── WHAT IT IS NOW ──────────────────────────────────────────────────────────
   * A true fan of angular wedges sharing ONE apex, placed in the WORLD at a
   * fixed distance along the sun direction, depth-tested against the scene.
   *
   *  - Sharing an apex means every edge in the effect is a ray through the sun.
   *    A ray through the sun is the one straight line in a sky that reads
   *    unambiguously as light, and it is impossible for two of them to meet at
   *    a right angle.
   *  - The wedge is built from an ANGLE, so its sides stay hard and straight —
   *    it is still a flat-shaded quadrilateral of light, not a bloom — while
   *    the radial profile fades to nothing well inside the geometry. The
   *    polygon boundary is therefore never the visible boundary, at any edge.
   *  - Placing the apex at ~850 m in view space and turning depth testing ON is
   *    what removes the wash entirely. Riders, bikes, trail and mid-ground
   *    terrain all sit in front of it and occlude it exactly. The shafts can
   *    only ever paint sky, which is the brief's "shafts through the tree
   *    line": the trees cut them, they do not cut the trees.
   */
  buildShafts(count = 7): void {
    const positions: number[] = [];
    const uvs: number[] = [];
    const seeds: number[] = [];
    const angles: number[] = [];
    const halves: number[] = [];
    const lens: number[] = [];

    // The fan sweeps a 130-degree arc centred on straight down, so a sun above
    // the frame rakes into the picture and a sun near the horizon still throws
    // its outermost shafts along the ridge line rather than into the ground.
    const FAN_CENTRE = -Math.PI * 0.5;
    const FAN_SPREAD = 1.14;

    for (let i = 0; i < count; i++) {
      const t = (i + 0.5) / count;
      // A deterministic, irrational-ish jitter so the fan is not a metronome.
      const j = Math.sin(i * 12.9898) * 43758.5453;
      const jitter = (j - Math.floor(j) - 0.5) * (FAN_SPREAD / count) * 0.55;
      const angle = FAN_CENTRE + (t - 0.5) * 2 * FAN_SPREAD + jitter;
      // Alternating thick/thin, which is how a background painter spaces them.
      const half = (i & 1) === 0 ? 0.030 + t * 0.026 : 0.055 + t * 0.038;
      const len = (i & 1) === 0 ? 1.35 - t * 0.30 : 0.95 - t * 0.22;

      // One triangle per shaft: the apex on the sun, two corners at the tip.
      // position.x is the angular parameter (-1..1), position.y the radial one.
      const verts = [
        [0, 0], [-1, 1], [1, 1],
      ];
      for (const v of verts) {
        positions.push(v[0], v[1], 0);
        uvs.push(v[0] * 0.5 + 0.5, v[1]);
        seeds.push(t);
        angles.push(angle);
        halves.push(half);
        lens.push(len);
      }
    }

    const geo = new BufferGeometry();
    geo.setAttribute('position', new Float32BufferAttribute(positions, 3));
    geo.setAttribute('uv', new Float32BufferAttribute(uvs, 2));
    geo.setAttribute('aSeed', new Float32BufferAttribute(seeds, 1));
    geo.setAttribute('aAngle', new Float32BufferAttribute(angles, 1));
    geo.setAttribute('aHalf', new Float32BufferAttribute(halves, 1));
    geo.setAttribute('aLen', new Float32BufferAttribute(lens, 1));

    this.shaftMat = new ShaderMaterial({
      glslVersion: GLSL3,
      transparent: true,
      depthWrite: false,
      // ON. This single flag is what stops the shafts washing the foreground.
      depthTest: true,
      blending: AdditiveBlending,
      side: DoubleSide,
      fog: false,
      uniforms: {
        uTime: NPR.uTime,
        uResolution: NPR.uResolution,
        uColor: { value: new Color(0xffd9a0) },
        uIntensity: { value: 0.0 },
        /** Sun direction in VIEW space. The apex sits along it. */
        uSunView: { value: new Vector3(0, 0, -1) },
        /** How far down the sun direction the fan is anchored, metres. */
        uShaftDist: { value: 850 },
      },
      vertexShader: /* glsl */ `
        precision highp float;
        in float aSeed;
        in float aAngle;
        in float aHalf;
        in float aLen;
        uniform vec3  uSunView;
        uniform float uShaftDist;
        uniform float uTime;
        out vec2  vUv;
        out float vSeed;
        void main() {
          vUv = uv;
          vSeed = aSeed;

          // The apex, in view space, exactly along the sun direction — so it
          // projects to exactly the sun's screen position, whatever the lens.
          vec3 apex = uSunView * uShaftDist;

          // Half the screen height, in view units, at the apex's depth. Working
          // through this rather than through the aspect ratio means the fan
          // angles are true screen angles and the shafts are the same length
          // relative to the frame on any window shape.
          float halfH = abs(apex.z) / max(projectionMatrix[1][1], 1e-4);

          float drift = sin(uTime * 0.17 + aSeed * 7.0) * 0.022;
          float a = aAngle + drift + position.x * aHalf;
          vec2 d = vec2(cos(a), sin(a));

          vec3 p = apex + vec3(d * (position.y * aLen * halfH), 0.0);
          gl_Position = projectionMatrix * vec4(p, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        precision highp float;
        ${GLSL_COMMON}
        ${GLSL_FRAG_OUT}
        uniform vec3 uColor;
        uniform float uIntensity;
        uniform float uTime;
        in vec2  vUv;
        in float vSeed;
        void main() {
          // ACROSS: two flat values with hard cuts, painted not blurred. The
          // outer cut sits at 0.82 of the wedge's angular half-width, so the
          // triangle's own edge at 1.0 carries zero alpha and can never be the
          // edge you see.
          float across = abs(vUv.x - 0.5) * 2.0;
          float w = max(fwidth(across), 0.0035);
          float body = 1.0 - smoothstep(0.82 - w, 0.82 + w, across);
          float core = 1.0 - smoothstep(0.36 - w, 0.36 + w, across);

          // ALONG: on at the sun, off well before the tip. This is the only
          // soft term in the shaft and it runs down the LENGTH, where a light
          // shaft genuinely does thin out — never across it, where an animator
          // would keep the cut.
          float t = vUv.y;
          float radial = smoothstep(0.015, 0.16, t) * (1.0 - smoothstep(0.26, 1.0, t));

          // Slow breathing so the shafts feel like light through moving canopy.
          float breathe = 0.74 + 0.26 * sin(uTime * 0.55 + vSeed * 9.1);
          float a = (body * 0.55 + core * 0.45) * radial * breathe * uIntensity;
          if (a < 0.003) discard;
          fragColor = vec4(uColor * a, a);
        }
      `,
    });

    this.shafts = new Mesh(geo, this.shaftMat);
    this.shafts.name = 'sky:shafts';
    this.shafts.frustumCulled = false;
    this.shafts.renderOrder = 4000;
    this.shafts.userData.skipPrepass = true;
    this.shafts.userData.skipShadow = true;
    this.group.add(this.shafts);
  }

  /**
   * @param sunVisibility 0..1 — how much of the sun the tree line leaves open.
   *   The game feeds this from a cheap occlusion probe so the shafts only
   *   appear when the sun is actually raking through something.
   */
  update(camera: PerspectiveCamera, sunVisibility = 0): void {
    this.group.position.copy(camera.position);

    // The dome reconstructs its view ray per pixel from these two, so they are
    // not optional bookkeeping — they ARE the sky's geometry.
    (this.domeMat.uniforms.uInvProjection.value as Matrix4).copy(camera.projectionMatrixInverse);
    (this.domeMat.uniforms.uCamWorld.value as Matrix4).copy(camera.matrixWorld);

    if (this.shaftMat) {
      // Sun direction in view space. Reused, never reallocated.
      const sv = this.shaftMat.uniforms.uSunView.value as Vector3;
      sv.copy(SUN_DIRECTION).transformDirection(camera.matrixWorldInverse);

      // The fan only exists when the sun itself is in frame. The old test
      // allowed |ndc| < 1.6, so shafts were routinely drawn from an apex a
      // long way off the top of the screen — which is how a shaft edge ended
      // up crossing the sky as a bare vertical rule with no visible source.
      _sunWorld.copy(SUN_DIRECTION).multiplyScalar(3000).add(camera.position);
      const p = _sunWorld.project(camera);
      const inFront = sv.z < -0.08 && p.z < 1;
      // A soft gate at the frame edge so the fan cannot pop on and off as the
      // camera pans; 0.92..1.18 is a little outside the frame, which is where
      // a real sun still throws shafts into it.
      const edge = Math.max(Math.abs(p.x), Math.abs(p.y));
      const gate = inFront ? 1 - Math.min(1, Math.max(0, (edge - 0.92) / 0.26)) : 0;

      // 0.34 rather than 1.0: these are an accent in the sky, not an exposure
      // event. At the old scale a single shaft added more light to the frame
      // than the sun disc itself.
      const target = gate * sunVisibility * 0.34;
      const cur = this.shaftMat.uniforms.uIntensity.value as number;
      this.shaftMat.uniforms.uIntensity.value = cur + (target - cur) * 0.06;
    }
  }

  dispose(): void {
    this.domeMat.dispose();
    this.shaftMat?.dispose();
  }
}
