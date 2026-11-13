
const heroFilm = document.querySelector(".hero-film");
const heroToggle = document.querySelector("#hero-toggle");
heroToggle.addEventListener("click", async () => {
  if (heroFilm.paused) await heroFilm.play().catch(() => {});
  else heroFilm.pause();
});
function syncHeroToggle() {
  heroToggle.textContent = heroFilm.paused ? "播放" : "暂停";
  heroToggle.setAttribute("aria-label", `${heroFilm.paused ? "播放" : "暂停"}首屏视频`);
}
heroFilm.addEventListener("play", syncHeroToggle);
heroFilm.addEventListener("pause", syncHeroToggle);
if (matchMedia("(prefers-reduced-motion: reduce)").matches) heroFilm.pause();
syncHeroToggle();

const visionCases = {
  direct: {
    src: "assets/videos/student095_v8_D016m_rgbd.mp4",
    poster: "assets/videos/student095_v8_D016m_poster.jpg",
    caption: "STUDENT 095 · MODEL 325 · D016m / TEST",
    kicker: "TEST · DIRECT PASS",
    title: "直传接球",
    copy: "测试集直传示例：从来球进入视野、双手调整到持球稳定，RGB 与两帧深度同步记录。",
    hold: "1.48 s",
  },
  bounce: {
    src: "assets/videos/student095_v8_B028p_rgbd.mp4",
    poster: "assets/videos/student095_v8_B028p_poster.jpg",
    caption: "STUDENT 095 · MODEL 325 · B028p / TEST",
    kicker: "TEST · GROUND BOUNCE PASS",
    title: "击地反弹接球",
    copy: "球落地反弹后进入双手接球区；画中深度窗与仿真 RGB 来自同一时刻。",
    hold: "1.44 s",
  },
  directAlt: {
    src: "assets/videos/student095_v8_D061p_rgbd.mp4",
    poster: "assets/videos/student095_v8_D061p_poster.jpg",
    caption: "STUDENT 095 · MODEL 325 · D061p / TEST",
    kicker: "TEST · DIRECT PASS",
    title: "直传接球 · D061p",
    copy: "另一条测试集直传案例，来球横向偏移不同；仍使用同一深度延迟和控制策略。",
    hold: "1.50 s",
  },
  bounceAlt: {
    src: "assets/videos/student095_v8_B103m_rgbd.mp4",
    poster: "assets/videos/student095_v8_B103m_poster.jpg",
    caption: "STUDENT 095 · MODEL 325 · B103m / TEST",
    kicker: "TEST · GROUND BOUNCE PASS",
    title: "击地反弹接球 · B103m",
    copy: "第二条击地测试案例，反弹轨迹和横向来球不同；RGB 与深度保持同步。",
    hold: "1.44 s",
  },
};
const visionVideo = document.querySelector("#vision-loop-video");
function selectVisionCase(name) {
  const config = visionCases[name];
  if (!config || !visionVideo) return;
  visionVideo.pause();
  visionVideo.poster = config.poster;
  visionVideo.querySelector("source").src = config.src;
  visionVideo.load();
  visionVideo.play().catch(() => {});
  document.querySelector("#vision-loop-caption").textContent = config.caption;
  document.querySelector("#vision-case-kicker").textContent = config.kicker;
  document.querySelector("#vision-case-title").textContent = config.title;
  document.querySelector("#vision-case-copy").textContent = config.copy;
  document.querySelector("#vision-case-hold").textContent = config.hold;
  document.querySelectorAll("[data-vision-case]").forEach((button) => {
    const selected = button.dataset.visionCase === name;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  document.querySelector("#vision-loop-case-content").setAttribute("aria-labelledby", `vision-tab-${name}`);
}
document.querySelectorAll("[data-vision-case]").forEach((button, index, buttons) => {
  button.addEventListener("click", () => selectVisionCase(button.dataset.visionCase));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next].click();
    buttons[next].focus();
  });
});

const architecture = document.querySelector("#architecture-figure");
if (architecture && "IntersectionObserver" in window) {
  const architectureObserver = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      architecture.classList.add(entry.target.classList.contains("training-lane") ? "is-training-visible" : "is-inference-visible");
      architectureObserver.unobserve(entry.target);
    }
  }, { threshold: 0.18, rootMargin: "0px 0px -8% 0px" });
  architecture.querySelectorAll(".training-lane, .inference-lane").forEach((lane) => architectureObserver.observe(lane));
}

const diagramDialog = document.querySelector("#architecture-dialog");
document.querySelector("#expand-architecture").addEventListener("click", () => {
  const expanded = document.querySelector("#architecture-expanded");
  expanded.replaceChildren(document.querySelector("#architecture-figure").cloneNode(true));
  expanded.firstElementChild.removeAttribute("id");
  diagramDialog.showModal();
});
document.querySelector("#close-architecture").addEventListener("click", () => diagramDialog.close());
diagramDialog.addEventListener("click", (event) => {
  if (event.target === diagramDialog) {
    const bounds = diagramDialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) diagramDialog.close();
  }
});

const playModes = {
  replay: { url: "play/?embed=1", label: "Student 095 · 录制回合", button: "打开案例播放器", note: "8 个精选回合 · 4 直传 / 4 击地", description: "浏览已录制的直传与击地接球回合。" },
  live: { url: "play/live.html?embed=1", label: "在线接球演示", button: "打开在线演示", note: "可选择来球模式与片段", description: "进入在线场景，选择来球模式和接球回合。" },
};
let playMode = "replay";
let frame;
let frameResizeObserver;
const playPanel = document.querySelector("#play-panel");
const tabs = [...document.querySelectorAll("[data-play-mode]")];

function resizeEmbeddedFrame(target = frame) {
  if (!target?.isConnected) return;
  const child = target.contentDocument;
  if (!child?.body) return;
  const update = () => {
    if (frame !== target || !target.isConnected) return;
    target.style.height = `${Math.max(800, Math.ceil(child.body.scrollHeight))}px`;
  };
  update();
  requestAnimationFrame(() => requestAnimationFrame(update));
}

window.addEventListener("resize", () => {
  resizeEmbeddedFrame();
  setTimeout(() => resizeEmbeddedFrame(), 120);
});
window.addEventListener("message", (event) => {
  if (event.origin !== location.origin || event.source !== frame?.contentWindow) return;
  if (event.data?.type !== "l04-embed-height" || !Number.isFinite(event.data.height)) return;
  frame.style.height = `${Math.max(800, Math.ceil(event.data.height))}px`;
});

function selectMode(mode) {
  if (mode === playMode && frame?.isConnected) return;
  playMode = mode;
  frameResizeObserver?.disconnect();
  frameResizeObserver = undefined;
  frame?.remove();
  frame = undefined;
  playPanel.classList.remove("is-loaded");
  const config = playModes[mode];
  tabs.forEach((tab) => {
    const selected = tab.dataset.playMode === mode;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  playPanel.setAttribute("aria-labelledby", `tab-${mode}`);
  document.querySelector("#play-mode-note").textContent = config.description;
  document.querySelector("#open-play-page").href = config.url.replace("?embed=1", "");
  loadPlayFrame();
}
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectMode(tab.dataset.playMode));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    selectMode(tabs[next].dataset.playMode);
    tabs[next].focus();
  });
});
function loadPlayFrame() {
  const activeFrame = document.createElement("iframe");
  frame = activeFrame;
  activeFrame.title = playMode === "live" ? "L04 在线接球演示" : "Student 095 接球回放";
  activeFrame.allow = "fullscreen; autoplay";
  activeFrame.style.height = "900px";
  activeFrame.src = playModes[playMode].url;
  activeFrame.addEventListener("load", () => {
    if (frame !== activeFrame || !activeFrame.isConnected) return;
    const child = activeFrame?.contentDocument;
    if (!child) return;
    child.documentElement.classList.add("embedded-play");
    const stylesheet = child.createElement("link");
    stylesheet.rel = "stylesheet";
    stylesheet.href = new URL("embed.css?v=3", location.href).href;
    stylesheet.addEventListener("load", () => {
      if (frame !== activeFrame || !activeFrame.isConnected) return;
      resizeEmbeddedFrame(activeFrame);
      child.defaultView?.addEventListener("resize", () => resizeEmbeddedFrame(activeFrame));
      const ResizeObserverCtor = child.defaultView?.ResizeObserver;
      const main = child.querySelector(".play-main");
      const liveGrid = child.querySelector(".live-grid");
      if (ResizeObserverCtor && (main || liveGrid)) {
        frameResizeObserver?.disconnect();
        frameResizeObserver = new ResizeObserverCtor(() => resizeEmbeddedFrame(activeFrame));
        if (main) frameResizeObserver.observe(main);
        if (liveGrid) frameResizeObserver.observe(liveGrid);
      }
    }, { once: true });
    child.head.append(stylesheet);
    setTimeout(() => resizeEmbeddedFrame(activeFrame), 120);
    setTimeout(() => resizeEmbeddedFrame(activeFrame), 600);
  });
  playPanel.classList.add("is-loaded");
  playPanel.append(activeFrame);
}

selectMode("replay");