const canvas = document.querySelector("#hand-viewer-canvas");
const stage = canvas?.parentElement;
const fallback = stage?.querySelector(".hand-viewer-fallback");
const status = document.querySelector("#hand-viewer-status");
const buttons = [...document.querySelectorAll("[data-hand-view]")];

async function initializeHandViewer() {
  const [THREE, controlsModule, loaderModule] = await Promise.all([
    import("https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js"),
    import("https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/controls/OrbitControls.js"),
    import("https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/loaders/GLTFLoader.js"),
  ]);
  const { OrbitControls } = controlsModule;
  const { GLTFLoader } = loaderModule;

  try {
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.6));
    renderer.setSize(stage.clientWidth, stage.clientHeight, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.22;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34, stage.clientWidth / stage.clientHeight, 0.01, 20);
    camera.up.set(0, 1, 0);
    scene.add(new THREE.HemisphereLight(0xdff9ef, 0x18251f, 2.1));

    const keyLight = new THREE.DirectionalLight(0xffead1, 3.2);
    keyLight.position.set(-1.3, 2.2, 2.6);
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0x70dbc3, 3.8);
    rimLight.position.set(1.8, 1.2, -1.6);
    scene.add(rimLight);

    const grid = new THREE.GridHelper(0.7, 18, 0x6a9c88, 0x3c6355);
    grid.position.y = -0.132;
    grid.material.transparent = true;
    grid.material.opacity = 0.22;
    scene.add(grid);

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = false;
    controls.minDistance = 0.22;
    controls.maxDistance = 1.1;

    let viewDistance = 0.48;
    let cameraMotion;
    const directions = {
      palm: new THREE.Vector3(0, 0.12, 1.6),
      side: new THREE.Vector3(1.7, 0.12, 0.18),
      back: new THREE.Vector3(0, 0.12, -1.6),
      mount: new THREE.Vector3(1.15, 1.0, 1.45),
    };

    function setView(name) {
      const direction = directions[name];
      if (!direction) return;
      const destination = direction.clone().normalize().multiplyScalar(viewDistance);
      cameraMotion = { from: camera.position.clone(), to: destination, start: performance.now() };
      controls.target.set(0, 0, 0);
      controls.update();
      buttons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.handView === name)));
    }

    buttons.forEach((button) => button.addEventListener("click", () => setView(button.dataset.handView)));

    const resizeObserver = new ResizeObserver(() => {
      const width = Math.max(1, stage.clientWidth);
      const height = Math.max(1, stage.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    });
    resizeObserver.observe(stage);

    new GLTFLoader().load(
      "assets/models/s50_smooth_left.glb",
      (gltf) => {
        const model = gltf.scene;
        model.rotation.x = Math.PI / 2;
        model.traverse((object) => {
          if (!object.isMesh) return;
          object.castShadow = true;
          object.receiveShadow = true;
        });
        const bounds = new THREE.Box3().setFromObject(model);
        const center = bounds.getCenter(new THREE.Vector3());
        const size = bounds.getSize(new THREE.Vector3());
        model.position.sub(center);
        scene.add(model);

        viewDistance = Math.max(size.x, size.y, size.z) * 2.05;
        controls.minDistance = viewDistance * 0.58;
        controls.maxDistance = viewDistance * 3.2;
        camera.position.set(0, viewDistance * 0.08, viewDistance);
        controls.target.set(0, 0, 0);
        controls.update();
        grid.position.y = -size.y * 0.58;
        if (fallback) fallback.hidden = true;
        status.textContent = "S50 Smooth · GLB";
      },
      undefined,
      () => {
        status.textContent = "静态视图";
      },
    );

    function render(time) {
      if (cameraMotion) {
        const progress = Math.min(1, (time - cameraMotion.start) / 420);
        const eased = 1 - (1 - progress) ** 3;
        camera.position.lerpVectors(cameraMotion.from, cameraMotion.to, eased);
        if (progress === 1) cameraMotion = undefined;
      }
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
  } catch {
    status.textContent = "静态视图";
  }
}

if (stage && status && "IntersectionObserver" in window) {
  const viewerObserver = new IntersectionObserver((entries) => {
    if (!entries.some((entry) => entry.isIntersecting)) return;
    viewerObserver.disconnect();
    initializeHandViewer().catch(() => { status.textContent = "静态视图"; });
  }, { rootMargin: "240px 0px" });
  viewerObserver.observe(stage);
} else if (stage && status) {
  initializeHandViewer().catch(() => { status.textContent = "静态视图"; });
}