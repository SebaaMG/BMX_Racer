import {
  Color,
  OrthographicCamera,
  Scene,
  SRGBColorSpace,
  Vector3,
  WebGLRenderer,
} from 'three';

import { NPR, updateNprGlobals } from '../../src/npr/NprGlobals';
import { RallyCarVisual } from '../../src/rally/RallyCarVisualPolished';

const params = new URLSearchParams(location.search);
const view = params.get('view') ?? 'front-threequarter';
const pixelRatio = Math.max(1, Math.min(2, Number(params.get('pr') ?? 1)));

const renderer = new WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
renderer.outputColorSpace = SRGBColorSpace;
renderer.setPixelRatio(pixelRatio);
renderer.setClearColor(new Color(0xd9cfbf), 1);
document.body.append(renderer.domElement);

const scene = new Scene();
const camera = new OrthographicCamera(-3.2, 3.2, 2.4, -2.4, 0.1, 40);
scene.add(camera);

// Model-only review: no atmospheric wash or missing shadow maps are allowed to
// hide a silhouette defect. The same cel ramps, matcaps and inverted hulls are
// still used by the actual shipping model.
NPR.uFogStrengths.value = [0, 0, 0, 0];
NPR.uShadowStrength.value = 0;
NPR.uAmbient.value = 0.46;

const car = new RallyCarVisual({
  frameColor: new Color(0xc84f5f),
  accentColor: new Color(0xf2bd52),
  detail: 'full',
  name: 'orthographic-review-car',
});
scene.add(car.root);

const state = {
  steerAngle: 0.20,
  frontCompression: 0.38,
  rearCompression: 0.30,
  frontSpin: 0.42,
  rearSpin: 0.42,
  speed: 0,
  lateralSlip: 0,
  boosting: false,
  brake: 0.35,
  crashed: false,
};
car.update(state, 1 / 60, 6);
car.root.updateMatrixWorld(true);

const VIEWS: Record<string, { position: Vector3; target: Vector3; size: number }> = {
  front: {
    position: new Vector3(0, 0.40, 8),
    target: new Vector3(0, 0.04, 0.30),
    size: 2.55,
  },
  side: {
    position: new Vector3(8, 0.35, 0),
    target: new Vector3(0, 0.03, 0),
    size: 2.75,
  },
  rear: {
    position: new Vector3(0, 0.40, -8),
    target: new Vector3(0, 0.04, -0.20),
    size: 2.55,
  },
  'front-threequarter': {
    position: new Vector3(5.8, 2.35, 7.2),
    target: new Vector3(0, 0.02, 0.12),
    size: 2.85,
  },
  'rear-threequarter': {
    position: new Vector3(-5.8, 2.25, -7.2),
    target: new Vector3(0, 0.02, -0.12),
    size: 2.85,
  },
};

function configureCamera(width: number, height: number): void {
  const selected = VIEWS[view] ?? VIEWS['front-threequarter'];
  const aspect = width / Math.max(height, 1);
  camera.left = -selected.size * aspect;
  camera.right = selected.size * aspect;
  camera.top = selected.size;
  camera.bottom = -selected.size;
  camera.position.copy(selected.position);
  camera.lookAt(selected.target);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  const label = document.querySelector<HTMLDivElement>('#label');
  if (label) label.textContent = `Rally model — ${view.replaceAll('-', ' ')}`;
}

function render(): void {
  const width = innerWidth;
  const height = innerHeight;
  renderer.setSize(width, height, false);
  configureCamera(width, height);
  updateNprGlobals(0, camera, width * pixelRatio, height * pixelRatio);
  renderer.render(scene, camera);
}

addEventListener('resize', render);
render();
requestAnimationFrame(render);

Object.assign(window, {
  __RALLY_MODEL_REVIEW__: {
    ready: true,
    view,
    render,
  },
});
