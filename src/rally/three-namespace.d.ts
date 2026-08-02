import type { ShaderMaterial as ThreeShaderMaterial } from 'three';

declare global {
  namespace THREE {
    type ShaderMaterial = ThreeShaderMaterial;
  }
}

export {};
