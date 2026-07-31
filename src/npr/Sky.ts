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
  Mesh,
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

export class Sky {
  readonly group = new Object3D();
  private domeMat: ShaderMaterial;
  private shaftMat: ShaderMaterial | null = null;
  private shafts: Mesh | null = null;

  constructor(radius = 4200) {
    this.group.name = 'sky';
    this.group.frustumCulled = false;

    const geo = new SphereGeometry(radius, 48, 32);

    // The cloud masks are sampled through a projection that shears hard toward
    // the horizon, so a single texel can cover many pixels there. Built with
    // LinearFilter alone they had no mip chain at all, and the tileable mask
    // aliased against itself into a hard rectangular grid across the lower sky
    // — the "seam" that looked like a projection wrap and was in fact minified
    // repeats beating against the pixel grid. Trilinear + anisotropic fixes the
    // aliasing; the shader then re-thresholds the filtered alpha so the cloud
    // contour goes back to being a hard cut rather than a blurred one.
    const cloudNear = celCloudMask(1024, 'clouds-near', 0.46);
    const cloudFar = celCloudMask(1024, 'clouds-far', 0.60);
    for (const t of [cloudNear, cloudFar]) {
      t.minFilter = LinearMipmapLinearFilter;
      t.magFilter = LinearFilter;
      t.generateMipmaps = true;
      t.anisotropy = 8;
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
      },
      vertexShader: /* glsl */ `
        precision highp float;
        out vec3 vDir;
        void main() {
          vDir = normalize(position);
          // Project at the far plane with w=z so the dome can never clip into
          // geometry regardless of the camera's far distance.
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

        in vec3 vDir;

        // Local band helper — the dome doesn't pull in the full cel core.
        float bandStepS(float x, float threshold, float softness) {
          float w = max(fwidth(x) * 0.8, softness);
          return smoothstep(threshold - w, threshold + w, x);
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

          // A warm plateau hugging the horizon on the sun side. Dawn light does
          // not wrap a sky evenly, and a gradient that is symmetric in azimuth
          // reads as a lit dome rather than as a sky.
          vec2 az    = normalize(dir.xz + vec2(1e-5, 0.0));
          vec2 sunAz = normalize(uSunDir.xz + vec2(1e-5, 0.0));
          float toward = saturate1(dot(az, sunAz));
          float low = 1.0 - bandStepS(h + wob * 0.022, 0.086, 0.0025);
          col = mix(col, mix(col, uSunGlow, 0.40), low * bandStepS(toward, 0.28, 0.16) * 0.55);

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
          float disc  = bandStepS(d, 0.99955, 0.00008);
          float ring1 = bandStepS(d, 0.9975, 0.0009);
          float ring2 = bandStepS(d, 0.988,  0.006);
          float ring3 = bandStepS(d, 0.940,  0.028);
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
         */
        vec3 clouds(vec3 dir, vec3 col) {
          float h = dir.y;
          if (h < -0.06) return col;

          float drift = uTime;

          // ── Far layer: flatter, duller, slower. Depth cue only. ──
          vec2 uvFar = domePlane(dir, 0.62) * 0.30 + vec2(drift * 0.0011, drift * 0.0005);
          vec4 far = texture(uCloudFar, uvFar);
          float aFar   = bandStepS(far.a, 0.52, 0.0020);
          float litFar = bandStepS(far.r, 0.80, 0.0020);
          float fadeFar = smoothstep(0.012, 0.20, h);
          col = mix(col, mix(uCloudShadow, uCloudMid, litFar), aFar * 0.56 * fadeFar);

          // ── Near layer ──
          vec2 uvNear = domePlane(dir, 0.30) * 0.42 + vec2(drift * 0.0038, drift * 0.0016);
          vec4 near = texture(uCloudNear, uvNear);
          float aNear   = bandStepS(near.a, 0.52, 0.0020);
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
          float rim = aNear * (1.0 - bandStepS(aheadA, 0.52, 0.0020));

          vec3 nearCol = mix(uCloudShadow, uCloudLit, litNear);
          nearCol = mix(nearCol, uCloudLit * 1.16, rim * 0.85);
          // The same probe run in the opposite direction gives the shadowed
          // edge. Hot rim on the sun side, cool cut on the far side, flat
          // two-value body in between: that is a cel cloud, and it is three
          // texture fetches.
          float behindA = texture(uCloudNear, uvNear - sunStep).a;
          float shade = aNear * (1.0 - bandStepS(behindA, 0.52, 0.0020)) * (1.0 - rim);
          nearCol = mix(nearCol, uCloudShadow * 0.88, shade * 0.60);

          col = mix(col, nearCol, aNear * 0.90 * fadeNear);
          return col;
        }

        void main() {
          vec3 dir = normalize(vDir);
          vec2 fc = gl_FragCoord.xy;

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
   * Screen-space radial blur produces photographic shafts. What we want is the
   * animation convention: a small number of hard-edged quadrilateral wedges of
   * light, additive, with a visible boundary. Building them as actual camera-
   * facing quads anchored to the sun direction gives exactly that, costs
   * nothing, and lets us fade them by how much of the sun is occluded.
   */
  buildShafts(count = 9): void {
    const positions: number[] = [];
    const uvs: number[] = [];
    const seeds: number[] = [];

    for (let i = 0; i < count; i++) {
      const t = i / count;
      // Each shaft is a long thin quad in a local frame; the vertex shader
      // orients it along the sun direction in view space.
      const w0 = 0.5 + t * 0.4;
      const w1 = 2.4 + t * 3.2;
      const len = 1.0;
      const off = (t - 0.5) * 2.2;
      const quad = [
        [-w0 * 0.5 + off, 0, 0], [w0 * 0.5 + off, 0, 0],
        [w1 * 0.5 + off * 2.6, len, 0], [-w1 * 0.5 + off * 2.6, len, 0],
      ];
      const tri = [0, 1, 2, 0, 2, 3];
      for (const k of tri) {
        positions.push(quad[k][0], quad[k][1], quad[k][2]);
        uvs.push(k === 0 || k === 3 ? 0 : 1, k < 2 ? 0 : 1);
        seeds.push(t);
      }
    }

    const geo = new BufferGeometry();
    geo.setAttribute('position', new Float32BufferAttribute(positions, 3));
    geo.setAttribute('uv', new Float32BufferAttribute(uvs, 2));
    geo.setAttribute('aSeed', new Float32BufferAttribute(seeds, 1));

    this.shaftMat = new ShaderMaterial({
      glslVersion: GLSL3,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: AdditiveBlending,
      side: DoubleSide,
      fog: false,
      uniforms: {
        uTime: NPR.uTime,
        uSunDir: NPR.uSunDir,
        uResolution: NPR.uResolution,
        uColor: { value: new Color(0xffd9a0) },
        uIntensity: { value: 0.0 },
        uSunScreen: { value: new Vector3(0.5, 0.5, 1) },
      },
      vertexShader: /* glsl */ `
        precision highp float;
        in float aSeed;
        uniform vec3 uSunScreen;
        uniform vec2 uResolution;
        uniform float uTime;
        out vec2 vUv;
        out float vSeed;
        void main() {
          vUv = uv;
          vSeed = aSeed;
          // Build the shaft in normalised screen space anchored at the sun.
          float aspect = uResolution.x / max(uResolution.y, 1.0);
          vec2 origin = uSunScreen.xy * 2.0 - 1.0;
          // Shafts fan downward and outward from the sun.
          float ang = -1.9 + aSeed * 1.15 + sin(uTime * 0.19 + aSeed * 7.0) * 0.035;
          vec2 dir = vec2(cos(ang), sin(ang));
          vec2 side = vec2(-dir.y, dir.x);
          vec2 p = origin + side * position.x * 0.075 + dir * position.y * 2.4;
          p.x /= aspect;
          gl_Position = vec4(p, 0.999, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        precision highp float;
        ${GLSL_COMMON}
        ${GLSL_FRAG_OUT}
        uniform vec3 uColor;
        uniform float uIntensity;
        uniform float uTime;
        in vec2 vUv;
        in float vSeed;
        void main() {
          // Hard edges across the shaft, a long taper along it.
          float across = 1.0 - smoothstep(0.62, 0.92, abs(vUv.x - 0.5) * 2.0);
          float along  = pow(1.0 - vUv.y, 1.35);
          // Slow breathing so the shafts feel like light through moving canopy.
          float breathe = 0.72 + 0.28 * sin(uTime * 0.55 + vSeed * 9.1);
          float a = across * along * breathe * uIntensity;
          if (a < 0.004) discard;
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

    if (this.shaftMat) {
      const sunWorld = SUN_DIRECTION.clone().multiplyScalar(3000).add(camera.position);
      const p = sunWorld.project(camera);
      this.shaftMat.uniforms.uSunScreen.value.set(p.x * 0.5 + 0.5, p.y * 0.5 + 0.5, p.z);
      // Only draw when the sun is actually on screen and partially occluded.
      const onScreen = p.z < 1 && Math.abs(p.x) < 1.6 && Math.abs(p.y) < 1.6;
      const target = onScreen ? sunVisibility : 0;
      const cur = this.shaftMat.uniforms.uIntensity.value as number;
      this.shaftMat.uniforms.uIntensity.value = cur + (target - cur) * 0.06;
    }
  }

  dispose(): void {
    this.domeMat.dispose();
    this.shaftMat?.dispose();
  }
}
