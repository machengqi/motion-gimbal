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
interface Preview {
  index: number;
  t: number;
  fps: number;
  full: string;
  window: string;
  rect: Rect;
  fullSize: [number, number];
}

const path = ref("");
const info = ref<VideoInfo | null>(null);
const preview = ref<Preview | null>(null);
const playing = ref(false);
const message = ref("");
let unlisten: UnlistenFn[] = [];

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

async function open() {
  message.value = "";
  try {
    info.value = await invoke<VideoInfo>("open_video", { path: path.value.trim() });
    preview.value = null;
    playing.value = false;
  } catch (e) {
    message.value = String(e);
  }
}

async function start() {
  message.value = "";
  try {
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

/** 点全景画面 ⇒ 把云台转到那一点。 */
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
  unlisten.push(await listen<Preview>("preview", (e) => (preview.value = e.payload)));
  unlisten.push(
    await listen<{ reason: string }>("ended", (e) => {
      playing.value = false;
      message.value = e.payload.reason;
    }),
  );
});
onBeforeUnmount(() => unlisten.forEach((u) => u()));
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
      <button :disabled="!info || playing" class="primary" @click="start">开始</button>
      <button :disabled="!playing" @click="stop">停止</button>
    </header>

    <p v-if="info" class="meta">
      {{ info.width }}×{{ info.height }} · {{ info.fps.toFixed(2) }} fps
      <template v-if="info.frames"> · {{ info.frames }} 帧</template>
      <template v-if="preview"> · 第 {{ preview.index }} 帧 · {{ preview.t.toFixed(2) }}s · 推送 {{ preview.fps.toFixed(1) }} fps</template>
    </p>
    <p v-if="message" class="message">{{ message }}</p>

    <section class="panes">
      <figure class="pane">
        <figcaption>全景（点击把云台转过去）</figcaption>
        <div class="stage" @click="aimAt">
          <img v-if="preview" :src="`data:image/jpeg;base64,${preview.full}`" alt="全景" />
          <div v-if="preview" class="rect" :style="rectStyle"></div>
          <div v-else class="empty">打开视频并开始</div>
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
</style>
