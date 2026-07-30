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
        uCloudNear: { value: celCloudMask(1024, 'clouds-near', 0.46) },
        uCloudFar: { value: celCloudMask(1024, 'clouds-far', 0.60) },
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
         * Banded vertical gradient. Four plateaus with a one-pixel transition,
         * plus a dither on the boundary so the plateau edges never draw a hard
         * horizontal seam across a clear sky.
         */
        vec3 skyGradient(float h, vec2 fc) {
          float dither = (hash21(fc * 0.7) - 0.5) * 0.018;
          float t = saturate1(h * 0.5 + 0.5 + dither);
          // Compress the useful range into the upper hemisphere.
          float k = pow(saturate1((t - 0.48) / 0.52), 0.72);

          vec3 col = uBelow;
          col = mix(col, uHorizon, smoothstep(0.00, 0.012, k));
          col = mix(col, uUpper,   smoothstep(0.26, 0.275, k));
          col = mix(col, uZenith,  smoothstep(0.63, 0.645, k));
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
         * Two cloud layers at different parallax depths. The far layer moves at
         * a third the speed of the near one, which is what gives a flat dome
         * genuine depth without any geometry.
         */
        vec3 clouds(vec3 dir, vec3 col, vec2 fc) {
          if (dir.y < -0.02) return col;

          // Project the direction onto a plane above the camera. Dividing by y
          // is what produces the perspective foreshortening toward the horizon.
          float yy = max(dir.y, 0.035);

          // ── Far layer ──
          vec2 uvFar = dir.xz / (yy * 3.4) * 0.16 + vec2(uTime * 0.0016, uTime * 0.0007);
          vec4 far = texture(uCloudFar, uvFar);
          float fadeFar = smoothstep(0.0, 0.16, dir.y);
          vec3 farCol = mix(uCloudShadow, uCloudMid, far.r);
          col = mix(col, farCol, far.a * 0.62 * fadeFar);

          // ── Near layer ──
          vec2 uvNear = dir.xz / (yy * 1.5) * 0.14 + vec2(uTime * 0.0052, uTime * 0.0021);
          vec4 near = texture(uCloudNear, uvNear);
          float fadeNear = smoothstep(0.02, 0.22, dir.y);

          // Sun side of a cloud is a hard-edged hot rim, not a gradient.
          float sunAmt = saturate1(dot(dir, uSunDir));
          vec3 nearCol = mix(uCloudShadow, uCloudLit, near.r);
          nearCol = mix(nearCol, uCloudLit * 1.18, bandStepS(sunAmt, 0.86, 0.03) * near.r);
          col = mix(col, nearCol, near.a * 0.88 * fadeNear);

          return col;
        }

        void main() {
          vec3 dir = normalize(vDir);
          vec2 fc = gl_FragCoord.xy;

          vec3 col = skyGradient(dir.y, fc);
          col = sunDisc(dir, col);
          col = clouds(dir, col, fc);

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
