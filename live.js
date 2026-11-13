import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/controls/OrbitControls.js";

const OBS_DIM = 7898;
const DEPTH_START = 698;
const DEPTH_SIZE = 2 * 45 * 80;
const TYPE_START = 696;
const diagnosticLaunchWait = new URLSearchParams(location.search).get("diagnosis_launch_wait_s");
const FIXED_LAUNCH_WAIT_S = diagnosticLaunchWait !== null && Number.isFinite(Number(diagnosticLaunchWait))
  ? Math.max(0, Math.min(5, Number(diagnosticLaunchWait)))
  : 2.7;
const MIN_REPEAT_LAUNCH_INTERVAL_S = 6.5;
const PREP_RETURN_DURATION_S = 1.5;
const diagnosticAblation = new URLSearchParams(location.search).get("ablation");
const diagnosticFastInterval = new URLSearchParams(location.search).has("fastInterval");
const diagnosticInitialStateEnabled = new URLSearchParams(location.search).has("diagnosis_initial_state");
const diagnosticBallPerturb = new URLSearchParams(location.search).has("diagnosis_ball_perturb");
const diagnosticWebPptReplay = new URLSearchParams(location.search).has("web_ppt_replay");
const diagnosticBallParameters = new URLSearchParams(location.search);

function perturbedLaunchPreset(preset) {
  const generated = globalThis.__diagnosticLaunchPreset;
  if (diagnosticBallPerturb && generated) {
    const position = generated.position_yaw_frame_m;
    const velocity = generated.velocity_yaw_frame_m_s;
    if (!Array.isArray(position) || !Array.isArray(velocity) || position.length !== 3 || velocity.length !== 3
      || ![...position, ...velocity].every(Number.isFinite)
      || position[0] < 0.8 || position[0] > 4 || Math.abs(position[1]) > 1
      || position[2] < 0.3 || position[2] > 2.5 || Math.hypot(...velocity) > 10 || velocity[0] >= 0) {
      throw new Error("Invalid diagnostic generated trajectory");
    }
    return { ...preset, position_yaw_frame_m: [...position], velocity_yaw_frame_m_s: [...velocity] };
  }
  const runtimeParameters = globalThis.__diagnosticBallParameters;
  if (!diagnosticBallPerturb && !runtimeParameters) return preset;
  const parameter = (name, fallback) => runtimeParameters?.[name]
    ?? diagnosticBallParameters.get(name)
    ?? fallback;
  const scale = Math.max(0.9, Math.min(1.1, Number(parameter("speed_scale", 1))));
  const yawDeg = Math.max(-5, Math.min(5, Number(parameter("yaw_deg", 0))));
  const pitchDeg = Math.max(-5, Math.min(5, Number(parameter("pitch_deg", 0))));
  const positionOffset = ["position_dx_m", "position_dy_m", "position_dz_m"].map((key) =>
    Math.max(-0.05, Math.min(0.05, Number(parameter(key, 0)))));
  const velocity = [...preset.velocity_yaw_frame_m_s];
  const horizontal = Math.hypot(velocity[0], velocity[1]);
  const speed = Math.hypot(horizontal, velocity[2]) * scale;
  const heading = Math.atan2(velocity[1], velocity[0]) + yawDeg * Math.PI / 180;
  const elevation = Math.atan2(velocity[2], horizontal) + pitchDeg * Math.PI / 180;
  return {
    ...preset,
    position_yaw_frame_m: preset.position_yaw_frame_m.map((value, index) => value + positionOffset[index]),
    velocity_yaw_frame_m_s: [
      speed * Math.cos(elevation) * Math.cos(heading),
      speed * Math.cos(elevation) * Math.sin(heading),
      speed * Math.sin(elevation),
    ],
  };
}

const startButton = document.querySelector("#live-start");
const resetButton = document.querySelector("#live-reset");
const statusText = document.querySelector("#live-status");
const statusDot = document.querySelector("#live-dot");
const pauseButton = document.querySelector("#live-pause");
const throwButton = document.querySelector("#live-throw");
const viewport = document.querySelector("#live-viewport");
const viewportPlaceholder = document.querySelector("#viewport-placeholder");
const depthPreview = document.querySelector("#depth-preview");
const depthPreviewContext = depthPreview.getContext("2d", { willReadFrequently: false });
const trajectorySelect = document.querySelector("#live-trajectory");
const clipSelect = document.querySelector("#live-clip");
const clipField = document.querySelector("#live-clip-field");
const directGridControls = document.querySelector("#live-direct-grid");
const gridPositionSelect = document.querySelector("#live-grid-position");
const gridHeightSelect = document.querySelector("#live-grid-height");
const gridStatus = document.querySelector("#live-grid-status");
let pendingLaunch = null;
let lastLaunchSimTime = null;
let activeTrajectory = "direct";
let readyJointPosition = null;
// Opt-in per-step record (?trace) for offline comparison against the local MuJoCo runner.
const rolloutTrace = new URLSearchParams(location.search).has("trace") ? [] : null;
if (rolloutTrace) window.liveRolloutTrace = rolloutTrace;
let policyWorker;
let physicsWorker;
let sessionGeneration = 0;
let sceneView;
let running = false;
let lastState;
let policyStepTimer = null;
let policyRequestPending = false;
let depthHistory = [];
let proprioHistory = [];
let observation = new Float32Array(OBS_DIM);
let ticks = 0;
let previousAction = new Float32Array(27);
let manifest;
let symmetricCatchGrid;
let bounceShowcaseGrid;
let lastPolicyWallTime = 0;
let policyDepthRangeM = [0, 2.5];
let pendingTraceSample = null;
let webPptReplayResolve;
let diagnosticResetPhysicsResolve = null;
let diagnosticResetPolicyResolve = null;

function notifyEmbeddedHeight() {
  if (window.parent === window) return;
  window.requestAnimationFrame(() => window.parent.postMessage({
    type: "l04-embed-height",
    height: document.body.scrollHeight,
  }, location.origin));
}

if (window.parent !== window && window.ResizeObserver) {
  const embeddedResizeObserver = new ResizeObserver(notifyEmbeddedHeight);
  embeddedResizeObserver.observe(document.body);
  const embeddedMain = document.querySelector(".live-main");
  if (embeddedMain) embeddedResizeObserver.observe(embeddedMain);
}

function makeGeomGeometry(geom, meshGeometries) {
  const [x, y, z] = geom.size;
  if (geom.type === 0) return new THREE.PlaneGeometry(200, 200);
  if (geom.type === 2) return new THREE.SphereGeometry(x, 24, 16);
  if (geom.type === 3) return new THREE.CapsuleGeometry(x, 2 * y, 6, 12).rotateX(Math.PI / 2);
  if (geom.type === 4) return new THREE.SphereGeometry(1, 24, 16).scale(x, y, z);
  if (geom.type === 5) return new THREE.CylinderGeometry(x, x, 2 * y, 16).rotateX(Math.PI / 2);
  if (geom.type === 6) return new THREE.BoxGeometry(2 * x, 2 * y, 2 * z);
  if (geom.type === 7) return meshGeometries.get(geom.meshId) ?? new THREE.SphereGeometry(0.03, 8, 6);
  return new THREE.SphereGeometry(0.025, 8, 6);
}

function makeCourtWoodTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const context = canvas.getContext("2d");
  const tones = ["#bd8b55", "#c2915b", "#b98752", "#c39460", "#b98752", "#c2915b", "#bd8b55", "#c39460"];
  for (let row = 0; row < 8; row++) {
    const height = 64;
    const offset = row % 2 ? -128 : 0;
    for (let column = -1; column < 3; column++) {
      const x = column * 256 + offset;
      context.fillStyle = tones[(row + column + 16) % tones.length];
      context.fillRect(x, row * height, 256, height);
      context.strokeStyle = "rgba(83, 49, 25, .16)";
      context.lineWidth = 2;
      context.strokeRect(x + 1, row * height + 1, 254, height - 2);
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(6, 6);
  texture.anisotropy = 4;
  return texture;
}

function addCourtPath(scene, points, material, width = 0.05) {
  const positions = [];
  const indices = [];
  const halfWidth = width / 2;
  for (let index = 0; index < points.length - 1; index++) {
    const [startX, startY] = points[index];
    const [endX, endY] = points[index + 1];
    const deltaX = endX - startX;
    const deltaY = endY - startY;
    const length = Math.hypot(deltaX, deltaY);
    if (length < 1e-6) continue;
    const normalX = -deltaY / length * halfWidth;
    const normalY = deltaX / length * halfWidth;
    const base = positions.length / 3;
    positions.push(
      startX + normalX, startY + normalY, 0.014,
      startX - normalX, startY - normalY, 0.014,
      endX + normalX, endY + normalY, 0.014,
      endX - normalX, endY - normalY, 0.014,
    );
    indices.push(base, base + 1, base + 2, base + 1, base + 3, base + 2);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  const stripe = new THREE.Mesh(geometry, material);
  stripe.frustumCulled = false;
  scene.add(stripe);
  return stripe;
}

function courtArcPoints(centerX, centerY, radius, start, end, samples = 48) {
  return Array.from({ length: samples + 1 }, (_, index) => {
    const angle = start + (end - start) * index / samples;
    return [centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius];
  });
}

function addHoop(scene, materials) {
  const rimX = -5.425;
  const boardX = -5.8;
  const board = new THREE.Mesh(new THREE.BoxGeometry(0.08, 1.83, 1.05), materials.backboard);
  board.position.set(boardX, 0, 3.425);
  board.castShadow = true;
  scene.add(board);

  const targetX = boardX + 0.047;
  const targetZ = 3.48;
  const targetBars = [
    [0.018, 0.61, 0.025, targetX, 0, targetZ + 0.2125],
    [0.018, 0.61, 0.025, targetX, 0, targetZ - 0.2125],
    [0.018, 0.025, 0.45, targetX, 0.2925, targetZ],
    [0.018, 0.025, 0.45, targetX, -0.2925, targetZ],
  ];
  for (const [sizeX, sizeY, sizeZ, x, y, z] of targetBars) {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(sizeX, sizeY, sizeZ), materials.target);
    bar.position.set(x, y, z);
    scene.add(bar);
  }

  const rim = new THREE.Mesh(new THREE.TorusGeometry(0.228, 0.018, 12, 48), materials.rim);
  rim.position.set(rimX, 0, 3.05);
  rim.castShadow = true;
  scene.add(rim);

  const rimMountX = rimX - Math.sqrt(0.228 ** 2 - 0.075 ** 2);
  const boardFrontX = boardX + 0.04;
  const mountLength = rimMountX - boardFrontX + 0.08;
  const mountCenterX = (boardFrontX + rimMountX) * 0.5;
  const visibleMountMaterial = new THREE.MeshStandardMaterial({
    color: "#87928b",
    roughness: 0.48,
    metalness: 0.42,
  });
  const centerMount = new THREE.Mesh(new THREE.BoxGeometry(mountLength, 0.075, 0.065), visibleMountMaterial);
  centerMount.position.set(mountCenterX, 0, 3.055);
  centerMount.castShadow = true;
  centerMount.receiveShadow = true;
  scene.add(centerMount);

  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.11, 3.3, 12), materials.support);
  post.position.set(boardX - 0.36, 0, 1.72);
  post.rotation.x = Math.PI / 2;
  scene.add(post);

  const netMaterial = new THREE.MeshStandardMaterial({
    color: "#f4f0e2",
    roughness: 0.72,
    metalness: 0.02,
  });
  const netTop = 0.211;
  const netBottom = 0.12;
  const netTopZ = 3.034;
  const netBottomZ = 2.63;
  const strandCount = 16;
  const addNetCord = (points, radius = 0.0045) => {
    const curve = new THREE.CatmullRomCurve3(points);
    const cord = new THREE.Mesh(new THREE.TubeGeometry(curve, 18, radius, 5, false), netMaterial);
    cord.castShadow = true;
    cord.receiveShadow = true;
    scene.add(cord);
  };

  for (let strand = 0; strand < strandCount; strand++) {
    const angle = (strand / strandCount) * Math.PI * 2;
    const points = [
      new THREE.Vector3(rimX + Math.cos(angle) * netTop, Math.sin(angle) * netTop, netTopZ),
      new THREE.Vector3(rimX + Math.cos(angle + 0.08) * 0.17, Math.sin(angle + 0.08) * 0.17, 2.84),
      new THREE.Vector3(rimX + Math.cos(angle + 0.14) * netBottom, Math.sin(angle + 0.14) * netBottom, netBottomZ),
    ];
    addNetCord(points);
  }

  const ringCount = 5;
  for (let ring = 0; ring < ringCount; ring++) {
    const blend = ring / (ringCount - 1);
    const radius = netTop + (netBottom - netTop) * blend;
    const z = netTopZ + (netBottomZ - netTopZ) * blend;
    const points = Array.from({ length: 49 }, (_, index) => {
      const angle = (index / 48) * Math.PI * 2;
      return new THREE.Vector3(rimX + Math.cos(angle) * radius, Math.sin(angle) * radius, z);
    });
    addNetCord(points, ring === 0 ? 0.005 : 0.0035);
  }
}

function addBasketballCourt(scene) {
  const outside = new THREE.Mesh(
    new THREE.PlaneGeometry(80, 80),
    new THREE.MeshStandardMaterial({ color: "#315844", roughness: 0.94 }),
  );
  outside.position.z = 0.001;
  outside.receiveShadow = true;
  scene.add(outside);

  const court = new THREE.Mesh(
    new THREE.PlaneGeometry(14, 15),
    new THREE.MeshStandardMaterial({ map: makeCourtWoodTexture(), color: "#ffffff", roughness: 0.82 }),
  );
  court.position.z = 0.004;
  court.receiveShadow = true;
  scene.add(court);

  const lineMaterial = new THREE.MeshBasicMaterial({ color: "#f6efd9", side: THREE.DoubleSide, toneMapped: false });
  const baselineX = -7;
  const midcourtX = 7;
  const sidelineY = 7.5;
  const rimX = -5.425;
  const keyHalfWidth = 2.45;
  const freeThrowX = -0.825;
  const threePointRadius = 6.75;
  const threePointCornerY = 6.6;
  const threePointJoinX = rimX + Math.sqrt(threePointRadius ** 2 - threePointCornerY ** 2);
  const threePointJoinAngle = Math.asin(threePointCornerY / threePointRadius);

  const courtBoundary = [
    [baselineX, -sidelineY], [midcourtX, -sidelineY], [midcourtX, sidelineY],
    [baselineX, sidelineY], [baselineX, -sidelineY],
  ];
  addCourtPath(scene, courtBoundary, lineMaterial);

  const key = [
    [baselineX, -keyHalfWidth], [freeThrowX, -keyHalfWidth],
    [freeThrowX, keyHalfWidth], [baselineX, keyHalfWidth],
  ];
  addCourtPath(scene, key, lineMaterial);
  addCourtPath(scene, courtArcPoints(freeThrowX, 0, 1.8, 0, Math.PI * 2, 96), lineMaterial);
  addCourtPath(scene, courtArcPoints(midcourtX, 0, 1.8, Math.PI / 2, Math.PI * 1.5, 64), lineMaterial);
  addCourtPath(scene, courtArcPoints(rimX, 0, threePointRadius, -threePointJoinAngle, threePointJoinAngle), lineMaterial);
  addCourtPath(scene, [[baselineX, -threePointCornerY], [threePointJoinX, -threePointCornerY]], lineMaterial);
  addCourtPath(scene, [[baselineX, threePointCornerY], [threePointJoinX, threePointCornerY]], lineMaterial);
  addCourtPath(scene, courtArcPoints(rimX, 0, 1.25, -Math.PI / 2, Math.PI / 2, 32), lineMaterial, 0.035);

  const materials = {
    backboard: new THREE.MeshStandardMaterial({ color: "#f4f5ee", roughness: 0.38, metalness: 0.05 }),
    target: new THREE.MeshStandardMaterial({ color: "#b95034", roughness: 0.65 }),
    rim: new THREE.MeshStandardMaterial({ color: "#df6735", roughness: 0.42, metalness: 0.18 }),
    support: new THREE.MeshStandardMaterial({ color: "#35423c", roughness: 0.6, metalness: 0.32 }),
  };
  addHoop(scene, materials);
  return { floor: "MuJoCo plane", court: "render-only half-court and hoop", dimensions: "14 x 15 m" };
}

function createSceneView(render) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#183126");
  scene.fog = new THREE.Fog("#183126", 18, 48);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.03, 120);
  camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  viewport.prepend(renderer.domElement);
  renderer.domElement.setAttribute("aria-hidden", "true");

  scene.add(new THREE.HemisphereLight("#ffffff", "#65766a", 2.0));
  const sun = new THREE.DirectionalLight("#fff5df", 2.6);
  sun.position.set(-4, -6, 10);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  scene.add(sun);

  const courtInfo = addBasketballCourt(scene);
  viewport.dataset.courtScene = courtInfo.court;
  viewport.dataset.courtDimensions = courtInfo.dimensions;

  const meshGeometries = new Map();
  for (const mesh of render.meshes) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(mesh.vertices, 3));
    geometry.setIndex(new THREE.BufferAttribute(mesh.faces, 1));
    geometry.computeVertexNormals();
    meshGeometries.set(mesh.id, geometry);
  }

  const visualMeshBodies = new Set(render.geoms
    .filter((geom) => geom.type === 7 && geom.contype === 0 && geom.conaffinity === 0 && geom.color[3] > 0.02)
    .map((geom) => geom.bodyId));
  viewport.dataset.visualMeshCount = String(render.geoms.filter(
    (geom) => geom.type === 7 && geom.contype === 0 && geom.conaffinity === 0,
  ).length);
  viewport.dataset.visualMeshBodies = String(visualMeshBodies.size);
  viewport.dataset.visualMeshGeometries = String(render.geoms.filter(
    (geom) => geom.type === 7 && geom.contype === 0 && geom.conaffinity === 0,
  ).length);
  viewport.dataset.s50HandGeometries = String(render.geoms.filter((geom) => geom.name.startsWith("s50_")).length);
  viewport.dataset.hiddenCollisionProxies = String(render.geoms.filter(
    (geom) => visualMeshBodies.has(geom.bodyId) && !(geom.type === 7 && geom.contype === 0 && geom.conaffinity === 0)
      && !geom.ball && geom.type !== 0,
  ).length);
  const objects = render.geoms.map((geom) => {
    const color = geom.ball
      ? new THREE.Color("#e77a38")
      : new THREE.Color().setRGB(geom.color[0], geom.color[1], geom.color[2]).multiplyScalar(0.72);
    const opacity = 1;
    const material = new THREE.MeshStandardMaterial({ color, roughness: geom.ball ? 0.72 : 0.68, metalness: 0.08, transparent: false, opacity });
    const mesh = new THREE.Mesh(makeGeomGeometry(geom, meshGeometries), material);
    const renderOnlyVisual = geom.type === 7 && geom.contype === 0 && geom.conaffinity === 0;
    const isS50HandGeom = geom.name.startsWith("s50_");
    mesh.visible = renderOnlyVisual || isS50HandGeom || geom.ball || (geom.type !== 0 && !visualMeshBodies.has(geom.bodyId));
    mesh.castShadow = geom.type !== 0;
    mesh.receiveShadow = true;
    scene.add(mesh);
    return mesh;
  });
  viewport.dataset.transparentGeomCount = String(render.geoms.filter((geom) => geom.color[3] <= 0.02).length);

  const depthScene = new THREE.Scene();
  const depthCamera = new THREE.PerspectiveCamera(
    manifest.depth_camera.vertical_fov_deg,
    Math.tan(THREE.MathUtils.degToRad(manifest.depth_camera.horizontal_fov_deg / 2))
      / Math.tan(THREE.MathUtils.degToRad(manifest.depth_camera.vertical_fov_deg / 2)),
    0.02,
    manifest.depth_camera.max_distance_m,
  );
  const depthMaterial = new THREE.ShaderMaterial({
    side: THREE.DoubleSide,
    depthWrite: true,
    uniforms: {},
    vertexShader: `
      varying float viewDepth;
      void main() {
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        viewDepth = -viewPosition.z;
        gl_Position = projectionMatrix * viewPosition;
      }
    `,
    fragmentShader: `
      varying float viewDepth;
      void main() {
        float normalized = clamp(viewDepth / 6.0, 0.0, 1.0);
        float encoded = floor(normalized * 65535.0 + 0.5);
        float highByte = floor(encoded / 256.0);
        float lowByte = mod(encoded, 256.0);
        gl_FragColor = vec4(highByte / 255.0, lowByte / 255.0, 0.0, 1.0);
      }
    `,
  });
  const depthObjects = objects.map((object, index) => {
    const geom = render.geoms[index];
    const proxy = new THREE.Mesh(object.geometry, depthMaterial);
    proxy.matrixAutoUpdate = false;
    proxy.visible = (geom.type === 7 && geom.contype === 0 && geom.conaffinity === 0)
      || geom.name.startsWith("s50_") || geom.name.startsWith("ball_") || geom.name === "sim_ground";
    depthScene.add(proxy);
    return proxy;
  });
  viewport.dataset.depthTargetCount = String(depthObjects.filter((object) => object.visible).length);
  const depthTarget = new THREE.WebGLRenderTarget(manifest.depth_camera.width, manifest.depth_camera.height, {
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    depthBuffer: true,
    stencilBuffer: false,
  });
  depthTarget.texture.colorSpace = THREE.NoColorSpace;
  const depthPixels = new Uint8Array(manifest.depth_camera.width * manifest.depth_camera.height * 4);
  const depthWidth = manifest.depth_camera.width;
  const depthHeight = manifest.depth_camera.height;
  const depthFrame = new Float32Array(depthWidth * depthHeight);
  const depthFar = manifest.depth_camera.max_distance_m;
  const depthNear = policyDepthRangeM[0];
  const depthRange = policyDepthRangeM[1] - depthNear;
  const rosToThree = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
  const mountQuaternion = new THREE.Quaternion(
    manifest.depth_camera.rotation_wxyz_ros[1],
    manifest.depth_camera.rotation_wxyz_ros[2],
    manifest.depth_camera.rotation_wxyz_ros[3],
    manifest.depth_camera.rotation_wxyz_ros[0],
  ).multiply(rosToThree);
  const mountPosition = new THREE.Vector3(...manifest.depth_camera.position_m);

  function captureDepth(state) {
    const cameraQuaternion = new THREE.Quaternion(
      state.camera.quaternionWxyz[1], state.camera.quaternionWxyz[2],
      state.camera.quaternionWxyz[3], state.camera.quaternionWxyz[0],
    );
    depthCamera.position.fromArray(state.camera.position).add(mountPosition.clone().applyQuaternion(cameraQuaternion));
    depthCamera.quaternion.copy(cameraQuaternion).multiply(mountQuaternion);
    depthCamera.updateMatrixWorld(true);

    renderer.setRenderTarget(depthTarget);
    renderer.setClearColor(0xffffff, 1);
    renderer.clear(true, true, true);
    renderer.render(depthScene, depthCamera);
    renderer.readRenderTargetPixels(depthTarget, 0, 0, depthWidth, depthHeight, depthPixels);
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);

    let hitPixels = 0;
    const preview = depthPreviewContext.createImageData(depthHeight, depthWidth);
    for (let row = 0; row < depthHeight; row++) {
      const sourceRow = depthHeight - row - 1;
      for (let column = 0; column < depthWidth; column++) {
        const pixelIndex = row * depthWidth + column;
        const source = (sourceRow * depthWidth + column) * 4;
        const encoded = depthPixels[source] * 256 + depthPixels[source + 1];
        const distance = encoded / 65535 * depthFar;
        const normalizedDepth = Math.max(0, Math.min(1, (distance - depthNear) / depthRange));
        depthFrame[pixelIndex] = normalizedDepth;
        hitPixels += distance < depthFar ? 1 : 0;
        const shade = Math.round((1 - normalizedDepth) * 255);
        const rotatedX = depthHeight - row - 1;
        const rotatedY = column;
        const displayY = depthWidth - rotatedY - 1;
        const target = (displayY * depthHeight + rotatedX) * 4;
        preview.data[target] = shade;
        preview.data[target + 1] = shade;
        preview.data[target + 2] = shade;
        preview.data[target + 3] = 255;
      }
    }
    depthPreviewContext.putImageData(preview, 0, 0);
    viewport.dataset.depthHitPixels = String(hitPixels);
    viewport.dataset.depthDrawCalls = String(renderer.info.render.calls);
    viewport.dataset.depthGlError = String(renderer.getContext().getError());
    return Float32Array.from(depthFrame);
  }

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(-2.0, 0, 1.9);
  controls.minDistance = 1.1;
  controls.maxDistance = 40;
  controls.maxPolarAngle = Math.PI / 2 - 0.04;
  let followRobot = true;
  let lastFocus = null;
  controls.addEventListener("start", () => { followRobot = false; });
  const recenter = () => {
    followRobot = true;
    lastFocus = null;
    controls.target.set(-0.8, 0, 1.95);
    camera.position.set(3.86, -1.7, 2.45);
    controls.update();
  };
  const setDiagnosticCamera = (view) => {
    followRobot = false;
    lastFocus = null;
    controls.enableDamping = false;
    renderer.setPixelRatio(1);
    renderer.setSize(1280, 800, false);
    camera.aspect = 1280 / 800;
    camera.updateProjectionMatrix();
    controls.target.set(0, 0, 0.98);
    if (view === "front") camera.position.set(4.4, 0, 1.38);
    else if (view === "front_reverse") camera.position.set(-4.4, 0, 1.38);
    else if (view === "side") camera.position.set(0, -4.4, 1.38);
    else throw new Error(`Unknown diagnostic camera view: ${view}`);
    controls.update();
  };
  recenter();

  const resizeObserver = new ResizeObserver(() => {
    const { width, height } = viewport.getBoundingClientRect();
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  });
  resizeObserver.observe(viewport);
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });

  return {
    update(state) {
      if (!state?.poses) return;
      viewport.dataset.root = state.root.map((value) => value.toFixed(3)).join(",");
      if (followRobot && !isRobotFallen(state) && state.root?.length >= 3) {
        const focus = new THREE.Vector3(state.root[0] + 0.5, state.root[1], Math.max(0.45, state.root[2]));
        if (lastFocus) {
          const shift = focus.clone().sub(lastFocus);
          controls.target.add(shift);
          camera.position.add(shift);
        }
        lastFocus = focus;
      }
      const matrix = new THREE.Matrix4();
      for (let index = 0; index < objects.length; index++) {
        const offset = index * 12;
        matrix.set(
          state.poses[offset + 3], state.poses[offset + 4], state.poses[offset + 5], state.poses[offset],
          state.poses[offset + 6], state.poses[offset + 7], state.poses[offset + 8], state.poses[offset + 1],
          state.poses[offset + 9], state.poses[offset + 10], state.poses[offset + 11], state.poses[offset + 2],
          0, 0, 0, 1,
        );
        objects[index].matrix.copy(matrix);
        objects[index].matrixAutoUpdate = false;
        depthObjects[index].matrix.copy(matrix);
        depthObjects[index].matrixAutoUpdate = false;
      }
      renderer.render(scene, camera);
    },
    captureDepth,
    recenter,
    setDiagnosticCamera,
    dispose() {
      renderer.setAnimationLoop(null);
      resizeObserver.disconnect();
      controls.dispose();
      for (const geometry of new Set(objects.map((object) => object.geometry))) geometry.dispose();
      depthTarget.dispose();
      depthMaterial.dispose();
      scene.traverse((object) => object.material?.dispose?.());
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}

function appendLog(message) {
  console.debug(`[L04] ${message}`);
}

function setStatus(message, ready = false) {
  statusText.textContent = message;
  statusDot.classList.toggle("ready", ready);
  notifyEmbeddedHeight();
}

function isRobotFallen(state) {
  if (!state?.root || !state?.qpos || state.root[2] < 0.48) return true;
  const [w, x, y, z] = state.qpos.slice(3, 7);
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x))));
  return Math.abs(roll) > 0.9 || Math.abs(pitch) > 0.9;
}

function rotateVectorWxyz(quaternion, vector) {
  const [w, x, y, z] = quaternion;
  const [vx, vy, vz] = vector;
  return [
    (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - w * z) * vy + 2 * (x * z + w * y) * vz,
    2 * (x * y + w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - w * x) * vz,
    2 * (x * z - w * y) * vx + 2 * (y * z + w * x) * vy + (1 - 2 * (x * x + y * y)) * vz,
  ];
}

function buildObservation(state) {
  observation.fill(0);
  const isBounce = activeTrajectory === "bounce";
  observation[TYPE_START] = isBounce ? 0 : 1;
  observation[TYPE_START + 1] = isBounce ? 1 : 0;
  const q = state.qpos;
  const qd = state.qvel;
  const [w, x, y, z] = q.slice(3, 7);
  const inverseQuaternion = [w, -x, -y, -z];
  const gravity = rotateVectorWxyz(inverseQuaternion, [0, 0, -1]);
  const jointPosition = new Float32Array(27);
  const jointVelocity = new Float32Array(27);
  for (let joint = 0; joint < 27; joint++) {
    jointPosition[joint] = q[manifest.action.qpos_addresses[joint]] - manifest.action.default_joint_pos[joint];
    jointVelocity[joint] = qd[manifest.action.qvel_addresses[joint]];
  }
  const currentFrame = {
    angularVelocity: Float32Array.from(qd.slice(3, 6)),
    gravity: Float32Array.from(gravity),
    jointPosition,
    jointVelocity,
    lastAction: Float32Array.from(previousAction),
  };
  if (!proprioHistory.length) {
    proprioHistory = Array.from({ length: 8 }, () => ({
      angularVelocity: currentFrame.angularVelocity.slice(),
      gravity: currentFrame.gravity.slice(),
      jointPosition: currentFrame.jointPosition.slice(),
      jointVelocity: currentFrame.jointVelocity.slice(),
      lastAction: currentFrame.lastAction.slice(),
    }));
  } else {
    proprioHistory.push(currentFrame);
    if (proprioHistory.length > 8) proprioHistory.shift();
  }
  for (let frame = 0; frame < proprioHistory.length; frame++) {
    const sample = proprioHistory[frame];
    observation.set(sample.angularVelocity, frame * 3);
    observation.set(sample.gravity, 24 + frame * 3);
    observation.set(sample.jointPosition, 48 + frame * 27);
    observation.set(sample.jointVelocity, 264 + frame * 27);
    observation.set(sample.lastAction, 480 + frame * 27);
  }
  observation.fill(0, DEPTH_START, DEPTH_START + DEPTH_SIZE);
  if (depthHistory.length >= 5) {
    observation.set(depthHistory[0], DEPTH_START);
    observation.set(depthHistory[2], DEPTH_START + DEPTH_SIZE / 2);
  } else {
    observation.fill(1, DEPTH_START, DEPTH_START + DEPTH_SIZE);
  }
  return observation;
}

function step(state, generation = sessionGeneration) {
  if (generation !== sessionGeneration || !running || policyRequestPending) return;
  lastState = state;
  observation = buildObservation(state);
  if (rolloutTrace) {
    pendingTraceSample = { time: state.time, qpos: Array.from(state.qpos), qvel: Array.from(state.qvel),
      observation: Array.from(observation) };
    rolloutTrace.push(pendingTraceSample);
  }
  policyRequestPending = true;
  policyWorker.postMessage({ type: "step", observation });
}

function schedulePolicyStep(state) {
  if (!running || policyRequestPending || (!pendingLaunch && lastLaunchSimTime === null)) return;
  const generation = sessionGeneration;
  clearTimeout(policyStepTimer);
  const now = performance.now();
  const delay = Math.max(0, manifest.physics.policy_dt_s * 1000 - (now - lastPolicyWallTime));
  lastPolicyWallTime = now + delay;
  policyStepTimer = setTimeout(() => {
    policyStepTimer = null;
    step(state, generation);
  }, delay);
}

async function initializeWorkers() {
  const generation = ++sessionGeneration;
  policyWorker?.terminate();
  physicsWorker?.terminate();
  const manifestResponse = await fetch("../assets/sim_bundle/l04_v61/manifest.json?v=sim2sim3");
  if (!manifestResponse.ok) throw new Error(`场景清单加载失败：HTTP ${manifestResponse.status}`);
  manifest = await manifestResponse.json();
  const gridResponse = await fetch("../assets/catch/direct_showcase_grid_v2.json?v=3");
  if (!gridResponse.ok) throw new Error(`直传展示候选加载失败：HTTP ${gridResponse.status}`);
  symmetricCatchGrid = await gridResponse.json();
  if (symmetricCatchGrid.schema !== 1 || symmetricCatchGrid.mode !== "direct"
    || symmetricCatchGrid.entries?.length !== 9
    || symmetricCatchGrid.entries.some((entry) => entry.cell?.length !== 2 || !entry.preset)) {
    throw new Error("直传展示候选结构不匹配");
  }
  const bounceGridResponse = await fetch("../assets/catch/bounce_showcase_grid_v2.json?v=4");
  if (!bounceGridResponse.ok) throw new Error(`击地展示候选加载失败：HTTP ${bounceGridResponse.status}`);
  bounceShowcaseGrid = await bounceGridResponse.json();
  if (bounceShowcaseGrid.schema !== 1 || bounceShowcaseGrid.mode !== "bounce"
    || bounceShowcaseGrid.entries?.length !== 9
    || bounceShowcaseGrid.entries.some((entry) => entry.cell?.length !== 2 || !entry.preset)) {
    throw new Error("击地展示候选结构不匹配");
  }
  return new Promise((resolve, reject) => {
    let policyReady = false;
    let physicsReady = false;
    let firstStepDone = false;

    policyWorker = new Worker(new URL("./policy-worker.js?v=22", import.meta.url), { type: "module" });
    physicsWorker = new Worker(new URL("./mujoco-worker.js?v=sim2sim15", import.meta.url), { type: "module" });

    const fail = (workerName, message) => {
      if (generation !== sessionGeneration) return;
      sessionGeneration++;
      running = false;
      clearTimeout(policyStepTimer);
      policyStepTimer = null;
      policyRequestPending = false;
      pendingLaunch = null;
      pendingTraceSample = null;
      policyWorker?.terminate();
      physicsWorker?.terminate();
      policyWorker = null;
      physicsWorker = null;
      pauseButton.disabled = true;
      throwButton.disabled = true;
      resetButton.disabled = false;
      startButton.disabled = true;
      statusText.title = `${workerName}: ${message}`;
      setStatus(firstStepDone ? "运行异常 · 请重置" : "加载失败 · 请重置");
      appendLog(`${workerName}：${message}`);
      reject(new Error(`${workerName}: ${message}`));
    };

    policyWorker.onerror = (error) => fail("095 Student policy", error.message || "worker error");
    physicsWorker.onerror = (error) => fail("MuJoCo", error.message || "worker error");
    policyWorker.onmessage = ({ data }) => {
      if (generation !== sessionGeneration) return;
      if (data.type === "error") return fail("095 Student policy", data.message);
      if (data.type === "ready") {
        if (!Array.isArray(data.depthRangeM) || data.depthRangeM.length !== 2 || data.depthRangeM[1] <= data.depthRangeM[0]) {
          return fail("095 Student policy", "策略清单缺少有效的深度归一化范围");
        }
        policyReady = true;
        if (diagnosticBallPerturb) globalThis.__diagnosticPolicyReady = true;
        policyDepthRangeM = data.depthRangeM;
        appendLog(`${data.policyLabel} ONNX 已加载：观测 ${data.observationDim} 维，动作 ${data.actionDim} 维；深度范围 ${data.depthRangeM[0]}–${data.depthRangeM[1]} m`);
      } else if (data.type === "reset") {
        diagnosticResetPolicyResolve?.();
        diagnosticResetPolicyResolve = null;
      } else if (data.type === "action") {
        policyRequestPending = false;
        if (!running) {
          const traceStep = rolloutTrace?.at(-1);
          if (traceStep) traceStep.discardedWhilePaused = true;
          return;
        }
        previousAction = data.action;
        const actionDisplay = document.querySelector("#live-action");
        if (actionDisplay) actionDisplay.textContent = Math.max(...previousAction.map(Math.abs)).toFixed(3);
        let appliedAction = previousAction;
        if (diagnosticAblation !== "no-prep"
          && pendingLaunch?.prepStartJointPosition && readyJointPosition) {
          const progress = Math.max(0, Math.min(1,
            (lastState.time - pendingLaunch.prepStartTime) / PREP_RETURN_DURATION_S));
          const blend = progress * progress * (3 - 2 * progress);
          appliedAction = Float32Array.from(previousAction);
          manifest.action.joint_names.forEach((name, joint) => {
            if (!/(shoulder|elbow|wrist)/.test(name)) return;
            const readyPosition = readyJointPosition[joint];
            const startPosition = pendingLaunch.prepStartJointPosition[joint];
            const targetPosition = startPosition + (readyPosition - startPosition) * blend;
            appliedAction[joint] = (targetPosition - manifest.action.default_joint_pos[joint]) / manifest.action.action_scale[joint];
          });
        }
        let launch;
        if (pendingLaunch && lastState && lastState.time >= pendingLaunch.launchAt - 1e-6) {
          launch = pendingLaunch;
          pendingLaunch = null;
          if (!readyJointPosition) {
            readyJointPosition = manifest.action.qpos_addresses.map((address) => lastState.qpos[address]);
          }
          lastLaunchSimTime = lastState.time;
          trajectorySelect.disabled = true;
          clipSelect.disabled = true;
          gridPositionSelect.disabled = true;
          gridHeightSelect.disabled = true;
          throwButton.disabled = true;
          throwButton.textContent = "本轮已发球";
          resetButton.textContent = "下一球 ↺";
          setStatus("来球已发出", true);
          appendLog(`出手：${launch.label}`);
        }
        physicsWorker.postMessage({ type: "step", action: appliedAction, launch: launch ? {
          ...launch,
          presetOverride: launch.preset,
        } : undefined });
        const traceStep = rolloutTrace?.at(-1);
        if (traceStep) Object.assign(traceStep, {
          action: Array.from(previousAction),
          appliedAction: appliedAction === previousAction ? undefined : Array.from(appliedAction),
          launched: Boolean(launch),
          launchPreset: launch ? {
            clip: launch.clip,
            mode: launch.preset.mode,
            gridCell: launch.gridCell,
            position_yaw_frame_m: launch.preset.position_yaw_frame_m,
            velocity_yaw_frame_m_s: launch.preset.velocity_yaw_frame_m_s,
            angular_velocity_yaw_frame_rad_s: launch.preset.angular_velocity_yaw_frame_rad_s,
          } : undefined,
        });
      }
      if (policyReady && physicsReady && !firstStepDone) {
        firstStepDone = true;
        resolve();
      }
    };
    physicsWorker.onmessage = ({ data }) => {
      if (generation !== sessionGeneration) return;
      if (data.type === "error") return fail("MuJoCo", data.message);
      if (data.type === "ready") {
        physicsReady = true;
        if (diagnosticBallPerturb) globalThis.__diagnosticPhysicsReady = true;
        if (data.diagnosticInitialStateMatchedJoints !== undefined) {
          viewport.dataset.diagnosticInitialStateMatchedJoints = String(data.diagnosticInitialStateMatchedJoints);
        }
        lastState = data.state;
        sceneView?.dispose();
        sceneView = createSceneView(data.render);
        if (diagnosticWebPptReplay) globalThis.__webPptSetCamera = (view) => sceneView?.setDiagnosticCamera(view);
        viewportPlaceholder.hidden = true;
        sceneView.update(data.state);
        const initialDepth = sceneView.captureDepth(data.state);
        depthHistory = Array.from({ length: 5 }, () => Float32Array.from(initialDepth));
        const modelDisplay = document.querySelector("#live-model");
        if (modelDisplay) modelDisplay.textContent = `${data.shape.nu} 关节 · ${data.shape.ngeom} 几何`;
        appendLog(`MuJoCo ${data.version} 已加载：nq=${data.shape.nq}, nv=${data.shape.nv}, nu=${data.shape.nu}`);
      } else if (data.type === "state") {
        lastState = data.state;
        if (pendingTraceSample) {
          pendingTraceSample.contacts = data.state.contacts;
          pendingTraceSample.catch = data.state.catch;
          pendingTraceSample = null;
        }
        sceneView?.update(data.state);
        const ballSlot = manifest.balls.slots[0];
        const ballPosition = data.state.qpos.slice(ballSlot.qpos_adr, ballSlot.qpos_adr + 3);
        const ballVelocityAddress = ballSlot.qvel_adr;
        const ballVelocity = data.state.qvel.slice(ballVelocityAddress, ballVelocityAddress + 3);
        viewport.dataset.ballPosition = ballPosition.map((value) => value.toFixed(3)).join(",");
        viewport.dataset.ballVelocity = ballVelocity.map((value) => value.toFixed(3)).join(",");
        viewport.dataset.root = Array.from(data.state.root.slice(0, 3), (value) => value.toFixed(3)).join(",");
        if (sceneView && !diagnosticWebPptReplay) {
          depthHistory.push(sceneView.captureDepth(data.state));
          if (depthHistory.length > 5) depthHistory.shift();
        }
        if (diagnosticWebPptReplay) {
          document.querySelector("#live-time").textContent = `${data.state.time.toFixed(3)} s`;
          webPptReplayResolve?.(data.state);
          webPptReplayResolve = null;
          return;
        }
        if (diagnosticResetPhysicsResolve) {
          const latestDepth = depthHistory.at(-1) ?? new Float32Array(DEPTH_SIZE / 2).fill(1);
          depthHistory = Array.from({ length: 5 }, () => Float32Array.from(latestDepth));
          proprioHistory = [];
          previousAction.fill(0);
          observation.fill(0);
          ticks = 0;
          document.querySelector("#live-time").textContent = "0.000 s";
          document.querySelector("#live-ball-speed").textContent = "0.0 m/s";
          diagnosticResetPhysicsResolve();
          diagnosticResetPhysicsResolve = null;
          return;
        }
        if (!running) return;
        ticks++;
        document.querySelector("#live-time").textContent = `${data.state.time.toFixed(3)} s`;
        document.querySelector("#live-ball-speed").textContent = `${Math.hypot(...ballVelocity).toFixed(2)} m/s`;
        if (isRobotFallen(data.state)) {
          running = false;
          pendingLaunch = null;
          throwButton.disabled = true;
          pauseButton.disabled = true;
          pauseButton.textContent = "机器人已失衡";
          startButton.textContent = "策略已停止";
          setStatus("机器人失衡 · 请重置模型");
          appendLog(`检测到机器人失衡（root z=${data.state.root[2].toFixed(2)} m）；策略已暂停`);
          return;
        }
        beginPendingPreparation(data.state);
        if (pendingLaunch) setStatus(pendingLaunchStatus(data.state.time), true);
        schedulePolicyStep(data.state);
      }
      if (policyReady && physicsReady && !firstStepDone) {
        firstStepDone = true;
        resolve();
      }
    };

    if (diagnosticWebPptReplay) {
      globalThis.__webPptReplayStatus = () => ({
        physicsReady,
        policyReady,
        sceneReady: Boolean(sceneView),
        workerReady: Boolean(physicsWorker),
        canvasCount: viewport.querySelectorAll("canvas").length,
      });
      globalThis.__webPptPauseStrategy = () => {
        running = false;
        clearTimeout(policyStepTimer);
        policyStepTimer = null;
        policyRequestPending = false;
      };
      globalThis.__webPptRenderFrame = (frame) => new Promise((resolve, reject) => {
        if (!physicsReady || !sceneView || !Array.isArray(frame?.qpos) || !Array.isArray(frame?.qvel)) {
          reject(new Error(`网页回放未就绪：${JSON.stringify({
            physicsReady, sceneReady: Boolean(sceneView), workerReady: Boolean(physicsWorker),
            qpos: Array.isArray(frame?.qpos) ? frame.qpos.length : typeof frame?.qpos,
            qvel: Array.isArray(frame?.qvel) ? frame.qvel.length : typeof frame?.qvel,
          })}`));
          return;
        }
        const timeout = setTimeout(() => {
          webPptReplayResolve = null;
          reject(new Error("网页场景轨迹帧渲染超时"));
        }, 5000);
        webPptReplayResolve = (state) => {
          clearTimeout(timeout);
          resolve(state);
        };
        physicsWorker.postMessage({ type: "replay_state", ...frame });
      });
    }

    policyWorker.postMessage({ type: "init" });
    physicsWorker.postMessage({ type: "init", initialState: diagnosticInitialStateEnabled
      ? globalThis.__diagnosticInitialState ?? null
      : null });
  });
}

startButton.addEventListener("click", async () => {
  startButton.disabled = true;
  activeTrajectory = trajectorySelect.value;
  setStatus("正在加载…");
  appendLog("开始初始化浏览器物理引擎和策略模型");
  try {
    await initializeWorkers();
    policyRequestPending = false;
    resetButton.disabled = false;
    pauseButton.disabled = false;
    pauseButton.textContent = "暂停策略 Ⅱ";
    throwButton.disabled = false;
    populateClipOptions();
    startButton.textContent = "已就绪";
    running = true;
    updateDirectGridAvailability();
    lastPolicyWallTime = performance.now();
    setStatus("等待发球", true);
    appendLog("已就绪；发球后开始准备与接球");
    schedulePolicyStep(lastState);
  } catch (error) {
    startButton.disabled = !resetButton.disabled;
    if (resetButton.disabled) setStatus("加载失败 · 可重试");
    appendLog(error.message);
  }
});

if (diagnosticBallPerturb) {
  globalThis.__resetDiagnosticTrial = async () => {
    if (!policyWorker || !physicsWorker || !sceneView || !globalThis.__diagnosticPolicyReady
      || !globalThis.__diagnosticPhysicsReady) {
      if (!resetButton.disabled) resetButton.click();
      return;
    }
    running = false;
    clearTimeout(policyStepTimer);
    policyStepTimer = null;
    pendingLaunch = null;
    pendingTraceSample = null;
    lastLaunchSimTime = null;
    readyJointPosition = null;
    policyRequestPending = false;
    activeTrajectory = trajectorySelect.value;
    if (rolloutTrace) rolloutTrace.length = 0;
    trajectorySelect.disabled = false;
    clipSelect.disabled = false;
    gridPositionSelect.disabled = false;
    gridHeightSelect.disabled = false;
    throwButton.disabled = false;
    throwButton.textContent = "发球 ↗";
    pauseButton.disabled = false;
    pauseButton.textContent = "暂停 Ⅱ";
    resetButton.disabled = false;
    setStatus("已重置 · 可重新发球", true);
    appendLog("诊断重置：复用已加载模型，机器人/GRU/动作与历史重新初始化");
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        diagnosticResetPhysicsResolve = null;
        diagnosticResetPolicyResolve = null;
        reject(new Error("Diagnostic worker reset timed out"));
      }, 10000);
      let physicsDone = false;
      let policyDone = false;
      const finish = () => {
        if (!physicsDone || !policyDone) return;
        clearTimeout(timeout);
        resolve();
      };
      diagnosticResetPhysicsResolve = () => { physicsDone = true; finish(); };
      diagnosticResetPolicyResolve = () => { policyDone = true; finish(); };
      physicsWorker.postMessage({ type: "reset" });
      policyWorker.postMessage({ type: "reset" });
    });
  };
}

function trajectoryLabel() {
  return trajectorySelect.value === "bounce" ? "击地反弹" : "直接来球";
}

function pendingLaunchStatus(simTime = lastState?.time ?? 0) {
  if (!pendingLaunch) return lastLaunchSimTime === null ? "等待发球" : "策略运行中";
  if (!pendingLaunch.prepStarted && simTime < pendingLaunch.prepStartAt) return "上一回合进行中";
  if (pendingLaunch.prepStartJointPosition && simTime < pendingLaunch.prepEndTime) return "准备中";
  if (simTime < pendingLaunch.recoveryReadyAt) return "回合间隔中";
  return "来球准备中";
}

function beginPendingPreparation(state) {
  const launch = pendingLaunch;
  if (!launch || launch.prepStarted || state.time < launch.prepStartAt - 1e-6) return;

  launch.prepStarted = true;
  activeTrajectory = launch.preset.mode;
  if (diagnosticAblation === "reset-policy-memory") {
    policyWorker.postMessage({ type: "reset" });
    previousAction.fill(0);
    proprioHistory = [];
    const latestDepth = depthHistory.at(-1) ?? new Float32Array(DEPTH_SIZE / 2).fill(1);
    depthHistory = Array.from({ length: 4 }, () => Float32Array.from(latestDepth));
  }
  launch.prepStartTime = state.time;
  launch.prepStartJointPosition = readyJointPosition
    ? manifest.action.qpos_addresses.map((address) => state.qpos[address])
    : null;
  launch.prepEndTime = launch.prepStartJointPosition
    ? state.time + PREP_RETURN_DURATION_S
    : state.time;
  launch.launchAt = state.time + FIXED_LAUNCH_WAIT_S;
  if (rolloutTrace) rolloutTrace.push({
    time: state.time,
    preparationStarted: true,
    clip: launch.clip,
    mode: launch.preset.mode,
    qpos: Array.from(state.qpos),
  });
}

function populateClipOptions() {
  if (!manifest) return;
  const gridMode = trajectorySelect.value === "direct" || trajectorySelect.value === "bounce";
  directGridControls.hidden = !gridMode;
  clipField.hidden = gridMode;
  const options = manifest.launch_presets
    .map((preset, index) => ({ preset, index }))
    .filter(({ preset }) => preset.mode === trajectorySelect.value)
    .map(({ preset, index }) => {
      const speed = Math.hypot(...preset.velocity_yaw_frame_m_s).toFixed(1);
      return new Option(`${preset.clip} · ${speed} m/s`, String(index));
    });
  clipSelect.replaceChildren(new Option("随机回合", "random"), ...options);
  clipSelect.value = "random";
  updateDirectGridAvailability();
}

function selectedDirectGridEntry() {
  const grid = trajectorySelect.value === "bounce" ? bounceShowcaseGrid : symmetricCatchGrid;
  if (!grid) return null;
  const cell = [Number(gridPositionSelect.value), Number(gridHeightSelect.value)];
  return grid.entries.find((entry) => entry.cell.every((value, index) => value === cell[index])) ?? null;
}

function updateDirectGridAvailability() {
  if (!gridStatus || !throwButton) return;
  const gridMode = trajectorySelect.value === "direct" || trajectorySelect.value === "bounce";
  if (!gridMode) {
    gridStatus.hidden = true;
    throwButton.disabled = !running || !physicsWorker || clipSelect.value === "";
    return;
  }
  gridStatus.hidden = false;
  const entry = selectedDirectGridEntry();
  const activeGrid = trajectorySelect.value === "bounce" ? bounceShowcaseGrid : symmetricCatchGrid;
  if (!activeGrid) {
    gridStatus.textContent = "当前模式展示球路未加载，暂不能发球。";
    gridStatus.dataset.state = "pending";
    throwButton.disabled = true;
    return;
  }
  const cellName = `${gridPositionSelect.selectedOptions[0].textContent} · ${gridHeightSelect.selectedOptions[0].textContent}`;
  if (!entry?.preset) {
    gridStatus.textContent = `${cellName} · 球路不存在，暂不可发球。`;
    gridStatus.dataset.state = "pending";
    throwButton.disabled = true;
    return;
  }
  const modeName = trajectorySelect.value === "bounce" ? "击地" : "直传";
  gridStatus.textContent = `${cellName} · ${modeName}：接球并持球约 1 秒后放下`;
  gridStatus.dataset.state = "pending";
  throwButton.disabled = !running || Boolean(pendingLaunch) || lastLaunchSimTime !== null;
}

function launchConfiguredBall() {
  if (pendingLaunch || lastLaunchSimTime !== null) return;
  if (!physicsWorker || !policyWorker || !running) return;
  const gridMode = trajectorySelect.value === "direct" || trajectorySelect.value === "bounce";
  if (!gridMode && clipSelect.value === "") return;
  const modePresets = manifest.launch_presets
    .map((preset, index) => ({ preset, index }))
    .filter(({ preset }) => preset.mode === trajectorySelect.value);
  if (!modePresets.length) return;
  let selectedPreset;
  let gridCell = null;
  if (gridMode) {
    const entry = selectedDirectGridEntry();
    if (!entry?.preset) {
      updateDirectGridAvailability();
      return;
    }
    const index = manifest.launch_presets.findIndex((preset) => preset.clip === entry.preset.clip
      && preset.mode === entry.preset.mode);
    if (index < 0) throw new Error(`Missing source launch preset: ${entry.preset.clip}`);
    selectedPreset = { index, preset: entry.preset };
    gridCell = entry.cell;
  } else {
    selectedPreset = clipSelect.value === "random"
      ? modePresets[Math.floor(Math.random() * modePresets.length)]
      : modePresets.find(({ index }) => index === Number(clipSelect.value));
  }
  if (!selectedPreset) return;
  const { preset, index: presetIndex } = selectedPreset;
  const effectivePreset = perturbedLaunchPreset(preset);
  const recoveryReadyAt = lastLaunchSimTime === null
    ? lastState.time
    : lastLaunchSimTime + (diagnosticFastInterval ? 0 : MIN_REPEAT_LAUNCH_INTERVAL_S);
  const prepStartAt = Math.max(lastState.time, recoveryReadyAt);
  const gridLabel = gridCell
    ? ` · ${gridPositionSelect.selectedOptions[0].textContent} · ${gridHeightSelect.selectedOptions[0].textContent}`
    : "";
  pendingLaunch = { ballId: 0, presetIndex, label: `${trajectoryLabel()} · ${preset.clip}${gridLabel}`,
    clip: preset.clip,
    preset: effectivePreset,
    gridCell,
    prepStartAt,
    launchAt: prepStartAt + FIXED_LAUNCH_WAIT_S,
    recoveryReadyAt,
    prepStarted: false,
    prepStartTime: null,
    prepEndTime: prepStartAt,
    prepStartJointPosition: null };
  if (rolloutTrace) rolloutTrace.push({
    time: lastState.time,
    request: true,
    ablation: diagnosticAblation ?? "default",
    clip: preset.clip,
    mode: preset.mode,
    qpos: Array.from(lastState.qpos),
  });
  trajectorySelect.disabled = true;
  clipSelect.disabled = true;
  gridPositionSelect.disabled = true;
  gridHeightSelect.disabled = true;
  throwButton.disabled = true;
  setStatus(pendingLaunchStatus(), true);
  lastPolicyWallTime = performance.now();
  schedulePolicyStep(lastState);
}

throwButton.addEventListener("click", launchConfiguredBall);
pauseButton.addEventListener("click", () => {
  if (running) {
    running = false;
    clearTimeout(policyStepTimer);
    policyStepTimer = null;
    pauseButton.textContent = "继续策略 ▶";
    setStatus("策略已暂停", true);
  } else if (lastState) {
    running = true;
    lastPolicyWallTime = performance.now();
    pauseButton.textContent = "暂停策略 Ⅱ";
    setStatus(pendingLaunchStatus(), true);
    if (!policyRequestPending) schedulePolicyStep(lastState);
  }
});


resetButton.addEventListener("click", () => {
  const startNextTrial = lastLaunchSimTime !== null;
  sessionGeneration++;
  running = false;
  clearTimeout(policyStepTimer);
  policyStepTimer = null;
  policyRequestPending = false;
  pendingLaunch = null;
  pendingTraceSample = null;
  lastLaunchSimTime = null;
  activeTrajectory = trajectorySelect.value;
  readyJointPosition = null;
  throwButton.textContent = "发球 ↗";
  trajectorySelect.disabled = false;
  clipSelect.disabled = false;
  gridPositionSelect.disabled = false;
  gridHeightSelect.disabled = false;
  ticks = 0;
  previousAction.fill(0);
  proprioHistory = [];
  depthHistory = [];
  sceneView?.dispose();
  sceneView = null;
  viewportPlaceholder.hidden = false;
  depthPreviewContext.clearRect(0, 0, depthPreview.width, depthPreview.height);
  policyWorker?.terminate();
  physicsWorker?.terminate();
  policyWorker = null;
  physicsWorker = null;
  startButton.disabled = false;
  startButton.textContent = "启动在线策略 ▶";
  pauseButton.disabled = true;
  pauseButton.textContent = "暂停策略 Ⅱ";
  throwButton.disabled = true;
  resetButton.disabled = true;
  resetButton.textContent = "重新开始 ↺";
  document.querySelector("#live-time").textContent = "0.000 s";
  document.querySelector("#live-ball-speed").textContent = "0.0 m/s";
  const actionDisplay = document.querySelector("#live-action");
  const modelDisplay = document.querySelector("#live-model");
  if (actionDisplay) actionDisplay.textContent = "—";
  if (modelDisplay) modelDisplay.textContent = "未加载";
  lastState = undefined;
  statusText.removeAttribute("title");
  setStatus("已重置 · 可重新启动", true);
  appendLog("GRU hidden 与 MuJoCo scene 已重置");
  if (startNextTrial) startButton.click();
});

window.addEventListener("pagehide", () => {
  running = false;
  clearTimeout(policyStepTimer);
  policyWorker?.terminate();
  physicsWorker?.terminate();
});

window.addEventListener("keydown", (event) => {
  if (event.repeat || event.target?.closest?.("input, select, textarea")) return;
  if (event.code === "Space") {
    event.preventDefault();
    if (!pauseButton.disabled) pauseButton.click();
    else if (!throwButton.disabled) throwButton.click();
    else if (!startButton.disabled) startButton.click();
  } else if (event.code === "KeyP") {
    event.preventDefault();
    if (!pauseButton.disabled) pauseButton.click();
    else if (!throwButton.disabled) throwButton.click();
  } else if (event.code === "KeyR") {
    event.preventDefault();
    if (!resetButton.disabled) resetButton.click();
    else if (!throwButton.disabled) throwButton.click();
  }
});

document.querySelector("#live-recenter").addEventListener("click", () => sceneView?.recenter());
document.querySelector("#live-fullscreen").addEventListener("click", async () => {
  if (document.fullscreenElement) await document.exitFullscreen();
  else await viewport.requestFullscreen();
});
document.addEventListener("fullscreenchange", () => {
  document.querySelector("#live-fullscreen").textContent = document.fullscreenElement ? "×" : "⛶";
});
trajectorySelect.addEventListener("change", () => {
  if (!pendingLaunch && lastLaunchSimTime === null) activeTrajectory = trajectorySelect.value;
  if (manifest?.launch_presets) populateClipOptions();
});
for (const select of [gridPositionSelect, gridHeightSelect]) {
  select.addEventListener("change", updateDirectGridAvailability);
}
