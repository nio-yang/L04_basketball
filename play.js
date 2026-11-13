const VIDEO_ROOT = "../assets/videos/";
const CASES = [
  { id: "D016m", type: "direct", label: "直接来球", split: "测试集", hold: "1.48 s", file: "student095_v8_D016m_rgbd.mp4", poster: "student095_v8_D016m_poster.jpg" },
  { id: "D035p", type: "direct", label: "直接来球", split: "验证集", hold: "1.48 s", file: "student095_v8_D035p_rgbd.mp4", poster: "student095_v8_D035p_poster.jpg" },
  { id: "D061p", type: "direct", label: "直接来球", split: "测试集", hold: "1.50 s", file: "student095_v8_D061p_rgbd.mp4", poster: "student095_v8_D061p_poster.jpg" },
  { id: "D079p", type: "direct", label: "直接来球", split: "测试集", hold: "1.48 s", file: "student095_v8_D079p_rgbd.mp4", poster: "student095_v8_D079p_poster.jpg" },
  { id: "B028p", type: "bounce", label: "击地反弹", split: "测试集", hold: "1.44 s", file: "student095_v8_B028p_rgbd.mp4", poster: "student095_v8_B028p_poster.jpg" },
  { id: "B103m", type: "bounce", label: "击地反弹", split: "测试集", hold: "1.44 s", file: "student095_v8_B103m_rgbd.mp4", poster: "student095_v8_B103m_poster.jpg" },
  { id: "B136m", type: "bounce", label: "击地反弹", split: "验证集", hold: "1.38 s", file: "student095_v8_B136m_rgbd.mp4", poster: "student095_v8_B136m_poster.jpg" },
  { id: "B155m", type: "bounce", label: "击地反弹", split: "验证集", hold: "1.28 s", file: "student095_v8_B155m_rgbd.mp4", poster: "student095_v8_B155m_poster.jpg" },
];

const video = document.querySelector("#rollout-video");
const source = document.querySelector("#rollout-source");
const list = document.querySelector("#case-list");
const overlay = document.querySelector("#video-overlay");
const playButton = document.querySelector("#play-button");
const timeline = document.querySelector("#timeline");
let selected = 0;
let activeMode = "all";

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "00:00";
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const remainder = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function visibleCases() {
  return CASES.map((item, index) => ({ ...item, index }))
    .filter((item) => activeMode === "all" || item.type === activeMode);
}

function renderCases() {
  list.replaceChildren();
  visibleCases().forEach((item) => {
    const button = document.createElement("button");
    button.className = `case-option${item.index === selected ? " active" : ""}`;
    button.type = "button";
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(item.index === selected));
    button.innerHTML = `<span class="case-kind">${item.type === "direct" ? "D" : "B"}</span><span class="case-copy"><strong>${item.id}</strong><small>${item.label} · ${item.split} · Student 095 v8</small></span><span class="case-state">保持 ${item.hold}</span>`;
    button.addEventListener("click", () => selectCase(item.index));
    list.append(button);
  });
}

function selectCase(index) {
  selected = index;
  const item = CASES[selected];
  video.pause();
  video.currentTime = 0;
  video.poster = `${VIDEO_ROOT}${item.poster}`;
  source.src = `${VIDEO_ROOT}${item.file}`;
  video.load();
  document.querySelector("#case-counter").textContent = `095 V8 · ${String(selected + 1).padStart(2, "0")} / ${String(CASES.length).padStart(2, "0")}`;
  document.querySelector("#stage-mode").textContent = item.type === "direct" ? "DIRECT BALL" : "GROUND BOUNCE";
  document.querySelector("#metric-mode").textContent = item.label;
  document.querySelector("#metric-status").textContent = "待开始";
  document.querySelector("#current-time").textContent = "00:00";
  document.querySelector("#metric-time").textContent = "0.0 s";
  timeline.value = "0";
  playButton.textContent = "▶";
  playButton.setAttribute("aria-label", "播放");
  overlay.classList.remove("hidden");
  renderCases();
  const url = new URL(window.location.href);
  url.searchParams.set("case", item.id);
  history.replaceState(null, "", url);
}

function togglePlayback() {
  if (video.paused) {
    video.play().catch(() => {
      document.querySelector("#metric-status").textContent = "播放失败";
    });
  } else {
    video.pause();
  }
}

document.querySelectorAll(".mode-button").forEach((button) => {
  button.addEventListener("click", () => {
    activeMode = button.dataset.mode;
    document.querySelectorAll(".mode-button").forEach((entry) => {
      const active = entry === button;
      entry.classList.toggle("active", active);
      entry.setAttribute("aria-pressed", String(active));
    });
    const available = visibleCases();
    if (!available.some((item) => item.index === selected)) selectCase(available[0].index);
    else renderCases();
  });
});

playButton.addEventListener("click", togglePlayback);
document.querySelector("#reset-button").addEventListener("click", () => {
  video.pause();
  video.currentTime = 0;
  togglePlayback();
});
document.querySelector("#random-button").addEventListener("click", () => {
  const available = visibleCases();
  const options = available.filter((item) => item.index !== selected);
  const choice = options[Math.floor(Math.random() * options.length)] ?? available[0];
  selectCase(choice.index);
  togglePlayback();
});
document.querySelector("#playback-speed").addEventListener("change", (event) => {
  video.playbackRate = Number(event.target.value);
});
document.querySelector("#fullscreen-button").addEventListener("click", async () => {
  const stage = document.querySelector(".video-stage");
  if (document.fullscreenElement) await document.exitFullscreen();
  else await stage.requestFullscreen?.();
});
timeline.addEventListener("input", () => {
  if (Number.isFinite(video.duration)) video.currentTime = video.duration * Number(timeline.value) / 1000;
});
video.addEventListener("loadedmetadata", () => {
  document.querySelector("#duration").textContent = formatTime(video.duration);
});
video.addEventListener("timeupdate", () => {
  const progress = video.duration ? video.currentTime / video.duration : 0;
  timeline.value = String(Math.round(progress * 1000));
  document.querySelector("#current-time").textContent = formatTime(video.currentTime);
  document.querySelector("#metric-time").textContent = `${video.currentTime.toFixed(1)} s`;
});
video.addEventListener("play", () => {
  playButton.textContent = "Ⅱ";
  playButton.setAttribute("aria-label", "暂停");
  overlay.classList.add("hidden");
  document.querySelector("#metric-status").textContent = "正在回放";
});
video.addEventListener("pause", () => {
  playButton.textContent = "▶";
  playButton.setAttribute("aria-label", "播放");
  if (video.currentTime > 0) document.querySelector("#metric-status").textContent = "已暂停";
});
video.addEventListener("ended", () => {
  document.querySelector("#metric-status").textContent = "回放完成";
  overlay.classList.remove("hidden");
});
video.addEventListener("error", () => {
  document.querySelector("#metric-status").textContent = "媒体暂不可用";
});

window.addEventListener("keydown", (event) => {
  if (event.target.closest?.("button, a, input, select, textarea, [role=option], [contenteditable=true]")) return;
  if (event.code === "Space") {
    event.preventDefault();
    togglePlayback();
  } else if (event.code === "KeyR") {
    video.currentTime = 0;
    video.pause();
    overlay.classList.remove("hidden");
    document.querySelector("#metric-status").textContent = "待开始";
  }
});

const initialCase = new URLSearchParams(location.search).get("case");
const initialIndex = CASES.findIndex((item) => item.id === initialCase);
renderCases();
if (initialIndex >= 0) selectCase(initialIndex);