<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

interface VideoInfo {
  path: string;
  width: number;
  height: number;
  fps: number;
  frames: number | null;
}
interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}
/** 视觉服务的一帧结果，全部在全幅坐标里。 */
interface Overlay {
  k: number;
  frame: number;
  aim: [number, number] | null;
  target: [number, number] | null;
  bbox: [number, number, number, number] | null;
  tracks: [number, number, number][];
  phase: string;
  tid: number | null;
  nDets: number;
  latencyMs: number;
  procMs: number;
  drove: boolean;
}
interface Preview {
  index: number;
  t: number;
  fps: number;
  full: string;
  window: string;
  rect: Rect;
  fullSize: [number, number];
  auto: boolean;
  tracking: boolean;
  overlays: Overlay[];
}

/** 轨迹叠加保留多少条结果。 */
const TRAIL = 72;
/** 指标曲线保留多少个点。 */
const CURVE = 240;

const path = ref("");
const info = ref<VideoInfo | null>(null);
const preview = ref<Preview | null>(null);
const playing = ref(false);
const auto = ref(false);
const message = ref("");
const status = ref("");
const last = ref<Overlay | null>(null);
const aimTrail = ref<[number, number][]>([]);
const targetTrail = ref<[number, number][]>([]);
const latency = ref<number[]>([]);
const proc = ref<number[]>([]);
const pushFps = ref<number[]>([]);
let unlisten: UnlistenFn[] = [];
let received = 0;
let statsTimer: number | undefined;

const rectStyle = computed(() => {
  const p = preview.value;
  if (!p) return {};
  const [fw, fh] = p.fullSize;
  return {
    left: `${(100 * p.rect.x) / fw}%`,
    top: `${(100 * p.rect.y) / fh}%`,
    width: `${(100 * p.rect.w) / fw}%`,
    height: `${(100 * p.rect.h) / fh}%`,
  };
});

const viewBox = computed(() => {
  const p = preview.value;
  return p ? `0 0 ${p.fullSize[0]} ${p.fullSize[1]}` : "0 0 1 1";
});
/** 叠加层的线宽、字号按全幅宽缩放，显示出来大致恒定。 */
const unit = computed(() => (preview.value ? preview.value.fullSize[0] / 640 : 1));
/** 轨迹折线：出画面或一步跳得太远（> 全幅宽的 1/6）处断开，各段分别画。 */
function segments(a: [number, number][]) {
  const p = preview.value;
  if (!p) return [];
  const [fw, fh] = p.fullSize;
  const out: string[] = [];
  let cur: string[] = [];
  let prev: [number, number] | null = null;
  for (const q of a) {
    const inside = q[0] >= 0 && q[0] <= fw && q[1] >= 0 && q[1] <= fh;
    const jump = prev !== null && Math.hypot(q[0] - prev[0], q[1] - prev[1]) > fw / 6;
    if (!inside || jump) {
      if (cur.length > 1) out.push(cur.join(" "));
      cur = [];
    }
    if (inside) cur.push(`${q[0].toFixed(1)},${q[1].toFixed(1)}`);
    prev = inside ? q : null;
  }
  if (cur.length > 1) out.push(cur.join(" "));
  return out;
}

function stats(a: number[]) {
  const v = a.filter((x) => Number.isFinite(x)).sort((p, q) => p - q);
  if (!v.length) return null;
  const q = (f: number) => v[Math.min(v.length - 1, Math.floor(f * v.length))];
  return { p50: q(0.5), p90: q(0.9), max: v[v.length - 1] };
}
const latencyStats = computed(() => stats(latency.value));
const procStats = computed(() => stats(proc.value));

/** 折线：`a` 映射到 0..W × 0..H（纵轴上限 `top`），最新的点在最右边。 */
function curve(a: number[], top: number, W = 300, H = 60) {
  if (a.length < 2) return "";
  const n = CURVE - 1;
  return a
    .map((v, i) => {
      const x = (W * (i + CURVE - a.length)) / n;
      const y = H - (H * Math.min(Number.isFinite(v) ? v : 0, top)) / top;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}
const msTop = computed(() => {
  const m = Math.max(10, ...latency.value.filter(Number.isFinite), ...proc.value.filter(Number.isFinite));
  return Math.ceil(m / 10) * 10;
});
const fpsTop = computed(() => Math.max(30, Math.ceil(Math.max(0, ...pushFps.value) / 10) * 10));

function pushCapped<T>(a: T[], v: T, n: number) {
  a.push(v);
  if (a.length > n) a.splice(0, a.length - n);
}

function resetTrails() {
  last.value = null;
  aimTrail.value = [];
  targetTrail.value = [];
  latency.value = [];
  proc.value = [];
  pushFps.value = [];
}

function onPreview(p: Preview) {
  received++;
  for (const o of p.overlays) {
    last.value = o;
    if (o.aim) pushCapped(aimTrail.value, o.aim, TRAIL);
    // 目标轨迹只记有新证据的帧（跟踪 / 印证后跳转）；等待中的估计只是跟着背景滑动
    if (o.target && o.phase !== "wait") pushCapped(targetTrail.value, o.target, TRAIL);
    pushCapped(latency.value, o.latencyMs, CURVE);
    pushCapped(proc.value, o.procMs, CURVE);
  }
  if (p.fps > 0) pushCapped(pushFps.value, p.fps, CURVE);
  preview.value = p;
}

function phaseName(p: string) {
  return ({ wait: "等待", track: "跟踪", jump: "印证后跳转" } as Record<string, string>)[p] ?? p;
}

async function open() {
  message.value = "";
  try {
    info.value = await invoke<VideoInfo>("open_video", { path: path.value.trim() });
    preview.value = null;
    playing.value = false;
    resetTrails();
  } catch (e) {
    message.value = String(e);
  }
}

async function start() {
  message.value = "";
  resetTrails();
  try {
    await invoke("set_auto", { on: auto.value });
    await invoke("start");
    playing.value = true;
  } catch (e) {
    message.value = String(e);
  }
}

async function stop() {
  await invoke("stop");
  playing.value = false;
}

async function toggleAuto() {
  await invoke("set_auto", { on: auto.value });
}

/** 点全景画面 ⇒ 把云台转到那一点（自动跟拍时也可人工干预）。 */
async function aimAt(ev: MouseEvent) {
  const p = preview.value;
  if (!p) return;
  const el = ev.currentTarget as HTMLElement;
  const box = el.getBoundingClientRect();
  const [fw, fh] = p.fullSize;
  const cx = ((ev.clientX - box.left) / box.width) * fw;
  const cy = ((ev.clientY - box.top) / box.height) * fh;
  try {
    const rect = await invoke<Rect>("set_gimbal", { cx, cy });
    p.rect = rect;
  } catch (e) {
    message.value = String(e);
  }
}

onMounted(async () => {
  unlisten.push(await listen<Preview>("preview", (e) => onPreview(e.payload)));
  unlisten.push(
    await listen<{ reason: string }>("ended", (e) => {
      playing.value = false;
      message.value = e.payload.reason;
    }),
  );
  unlisten.push(await listen<string>("status", (e) => (status.value = e.payload)));
  // 每秒报一次前端实际收到的预览帧率（宿主只在设了 MG_METRICS_LOG 时记下来）
  statsTimer = window.setInterval(() => {
    const n = received;
    received = 0;
    if (playing.value) invoke("ui_stats", { recvFps: n, index: preview.value?.index ?? 0 }).catch(() => {});
  }, 1000);
  // 测量与截图用：宿主环境变量 MG_AUTOPLAY 指定视频时自动打开并开始
  const cfg = await invoke<{ path: string | null; auto: boolean }>("autoplay_config");
  if (cfg.path) {
    path.value = cfg.path;
    auto.value = cfg.auto;
    await open();
    if (info.value) await start();
  }
});
onBeforeUnmount(() => {
  unlisten.forEach((u) => u());
  window.clearInterval(statsTimer);
});
</script>

<template>
  <main class="shell">
    <header class="bar">
      <h1>motion-gimbal</h1>
      <input
        v-model="path"
        class="path"
        placeholder="视频文件路径，例如 C:\data\moca\arctic_fox.mp4"
        @keydown.enter="open"
      />
      <button @click="open">打开</button>
      <label class="toggle" title="开始前勾选：接入视觉服务；播放中切换：结果是否驱动云台">
        <input v-model="auto" type="checkbox" @change="toggleAuto" />
        自动跟拍
      </label>
      <button :disabled="!info || playing" class="primary" @click="start">开始</button>
      <button :disabled="!playing" @click="stop">停止</button>
    </header>

    <p v-if="info" class="meta">
      {{ info.width }}×{{ info.height }} · {{ info.fps.toFixed(2) }} fps
      <template v-if="info.frames"> · {{ info.frames }} 帧</template>
      <template v-if="preview"> · 第 {{ preview.index }} 帧 · {{ preview.t.toFixed(2) }}s · 推送 {{ preview.fps.toFixed(1) }} fps</template>
      <template v-if="preview?.tracking">
        · {{ preview.auto ? "自动跟拍" : "仅显示（手动）" }}
        <template v-if="last"> · {{ phaseName(last.phase) }} · 候选 {{ last.nDets }} · 轨迹 {{ last.tracks.length }}</template>
      </template>
    </p>
    <p v-if="status" class="meta">{{ status }}</p>
    <p v-if="message" class="message">{{ message }}</p>

    <section class="panes">
      <figure class="pane">
        <figcaption>
          全景（点击把云台转过去）
          <span v-if="preview?.tracking" class="legend">
            <i class="sw aim"></i>瞄准点轨迹 <i class="sw target"></i>目标估计轨迹 <i class="sw box"></i>当前目标框
            <i class="sw trk"></i>运动轨迹点
          </span>
        </figcaption>
        <div class="stage" @click="aimAt">
          <img v-if="preview" :src="`data:image/jpeg;base64,${preview.full}`" alt="全景" />
          <div v-if="preview" class="rect" :style="rectStyle"></div>
          <svg v-if="preview && preview.tracking" class="overlay" :viewBox="viewBox" preserveAspectRatio="none">
            <polyline v-for="(s, i) in segments(targetTrail)" :key="`t${i}`" class="trail-target" :points="s" :stroke-width="2 * unit" />
            <polyline v-for="(s, i) in segments(aimTrail)" :key="`a${i}`" class="trail-aim" :points="s" :stroke-width="2.5 * unit" />
            <g v-if="last">
              <g v-for="[id, x, y] in last.tracks" :key="id">
                <circle class="trk" :cx="x" :cy="y" :r="3.5 * unit" />
                <text class="trk-id" :x="x + 5 * unit" :y="y - 5 * unit" :font-size="11 * unit">{{ id }}</text>
              </g>
              <rect
                v-if="last.bbox"
                class="box"
                :x="last.bbox[0]"
                :y="last.bbox[1]"
                :width="last.bbox[2] - last.bbox[0]"
                :height="last.bbox[3] - last.bbox[1]"
                :stroke-width="2 * unit"
              />
              <g v-if="last.aim" class="aim" :stroke-width="2 * unit">
                <circle :cx="last.aim[0]" :cy="last.aim[1]" :r="9 * unit" />
                <line :x1="last.aim[0] - 14 * unit" :y1="last.aim[1]" :x2="last.aim[0] + 14 * unit" :y2="last.aim[1]" />
                <line :x1="last.aim[0]" :y1="last.aim[1] - 14 * unit" :x2="last.aim[0]" :y2="last.aim[1] + 14 * unit" />
              </g>
            </g>
          </svg>
          <div v-if="!preview" class="empty">打开视频并开始</div>
        </div>
      </figure>
      <figure class="pane">
        <figcaption>云台画面（取景窗）</figcaption>
        <div class="stage">
          <img v-if="preview" :src="`data:image/jpeg;base64,${preview.window}`" alt="取景窗" />
          <div v-else class="empty">—</div>
        </div>
      </figure>
    </section>

    <section v-if="preview" class="pane metrics">
      <figure class="chart">
        <figcaption>
          延迟（ms，最近 {{ latency.length }} 帧）
          <span class="legend"><i class="sw lat"></i>写帧→云台动作 <i class="sw proc"></i>服务处理</span>
        </figcaption>
        <svg viewBox="0 0 300 60" preserveAspectRatio="none" class="curve">
          <line x1="0" x2="300" y1="30" y2="30" class="grid" />
          <polyline :points="curve(latency, msTop)" class="lat" />
          <polyline :points="curve(proc, msTop)" class="proc" />
        </svg>
        <p class="axis">
          纵轴上限 {{ msTop }} ms
          <template v-if="latencyStats"> · 端到端 中位 {{ latencyStats.p50.toFixed(1) }} / p90 {{ latencyStats.p90.toFixed(1) }}</template>
          <template v-if="procStats"> · 处理 中位 {{ procStats.p50.toFixed(1) }}</template>
        </p>
      </figure>
      <figure class="chart">
        <figcaption>推送帧率（fps）<span class="legend"><i class="sw fps"></i>宿主推送</span></figcaption>
        <svg viewBox="0 0 300 60" preserveAspectRatio="none" class="curve">
          <line x1="0" x2="300" y1="30" y2="30" class="grid" />
          <polyline :points="curve(pushFps, fpsTop)" class="fps" />
        </svg>
        <p class="axis">纵轴上限 {{ fpsTop }} fps · 当前 {{ preview.fps.toFixed(1) }}</p>
      </figure>
    </section>
  </main>
</template>

<style>
:root {
  --bg: #f6f7f8;
  --panel: #ffffff;
  --text: #1c2228;
  --muted: #66707a;
  --line: #d9dde1;
  --accent: #2f8f6f;
  --accent-text: #ffffff;
  --warn: #b5482f;
  --frame: #f0a03c;
  --aim: #e5484d;
  --target: #3e8ed0;
  --box: #f5d90a;
  --trk: #46d39a;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14181c;
    --panel: #1c2228;
    --text: #e6eaee;
    --muted: #93a0ab;
    --line: #2e363e;
    --accent: #5ac8a0;
    --accent-text: #0f1a15;
    --warn: #f08a70;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #14181c;
  --panel: #1c2228;
  --text: #e6eaee;
  --muted: #93a0ab;
  --line: #2e363e;
  --accent: #5ac8a0;
  --accent-text: #0f1a15;
  --warn: #f08a70;
  color-scheme: dark;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 system-ui, "Segoe UI", "Microsoft YaHei", sans-serif;
}
.shell {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.bar {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.bar h1 {
  font-size: 16px;
  margin: 0 8px 0 0;
}
.path {
  flex: 1 1 320px;
  min-width: 0;
  padding: 6px 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  color: var(--text);
}
button {
  padding: 6px 14px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  color: var(--text);
  cursor: pointer;
}
button.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: var(--accent-text);
}
button:disabled {
  opacity: 0.45;
  cursor: default;
}
.meta {
  margin: 0;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.message {
  margin: 0;
  color: var(--warn);
}
.panes {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 12px;
}
.pane {
  margin: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
}
.pane figcaption {
  color: var(--muted);
  margin-bottom: 8px;
}
.stage {
  position: relative;
  line-height: 0;
}
.stage img {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 4px;
}
.rect {
  position: absolute;
  border: 2px solid var(--frame);
  box-shadow: 0 0 0 9999px rgb(0 0 0 / 0.25);
  pointer-events: none;
}
.empty {
  line-height: 1.5;
  padding: 80px 0;
  text-align: center;
  color: var(--muted);
  border: 1px dashed var(--line);
  border-radius: 4px;
}
.toggle {
  display: inline-flex;
  gap: 4px;
  align-items: center;
  color: var(--text);
  user-select: none;
}
.legend {
  margin-left: 8px;
  font-size: 12px;
}
.sw {
  display: inline-block;
  width: 12px;
  height: 3px;
  margin: 0 3px 3px 8px;
  vertical-align: middle;
  border-radius: 2px;
}
.sw.aim,
.sw.lat {
  background: var(--aim);
}
.sw.target,
.sw.proc {
  background: var(--target);
}
.sw.box {
  background: var(--box);
}
.sw.trk {
  background: var(--trk);
}
.sw.fps {
  background: var(--accent);
}
.overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  overflow: hidden;
}
.overlay polyline,
.overlay .aim,
.overlay .box {
  fill: none;
  stroke-linejoin: round;
  stroke-linecap: round;
}
.overlay .trail-aim,
.overlay .aim {
  stroke: var(--aim);
}
.overlay .trail-target {
  stroke: var(--target);
  stroke-dasharray: 6 4;
}
.overlay .box {
  stroke: var(--box);
}
.overlay .trk {
  fill: var(--trk);
  stroke: rgb(0 0 0 / 0.5);
}
.overlay .trk-id {
  fill: var(--trk);
  paint-order: stroke;
  stroke: rgb(0 0 0 / 0.6);
  stroke-width: 2px;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 16px;
}
.chart {
  margin: 0;
}
.chart figcaption {
  color: var(--muted);
  margin-bottom: 4px;
}
.curve {
  width: 100%;
  height: 90px;
  background: var(--bg);
  border-radius: 4px;
}
.curve polyline {
  fill: none;
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
}
.curve .lat {
  stroke: var(--aim);
}
.curve .proc {
  stroke: var(--target);
}
.curve .fps {
  stroke: var(--accent);
}
.curve .grid {
  stroke: var(--line);
  stroke-dasharray: 3 3;
  vector-effect: non-scaling-stroke;
}
.axis {
  margin: 4px 0 0;
  color: var(--muted);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}
</style>
