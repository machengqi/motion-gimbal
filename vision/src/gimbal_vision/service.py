"""**视觉服务** —— 从帧环取帧 → 运动候选 → 跟踪 → 显著性选择 → 平滑瞄准 → 回传瞄准点。

    uv run --extra service python -m gimbal_vision.service --port 8765

# 两个平面

```text
数据面   共享内存帧环（`ring.py` / `src-tauri/src/ring.rs`）   宿主写帧，这里读
控制面   HTTP（FastAPI）                                       会话开始/停止、参数、状态
结果流   WebSocket `/results`（同一个 HTTP 服务上的升级连接）  每处理完一帧推一条 JSON
```

结果走 WebSocket 推送而不是逐帧 HTTP 请求：结果是服务**主动**产生的，逐帧轮询要么多一个轮询周期的
延迟，要么每帧一次请求的开销；推送在已建立的连接上发一帧 JSON，没有请求往返。

# 坐标

宿主写进帧环的是**全幅**帧和当时的取景窗矩形 `roi`。`perception="window"`（默认，闭环：相当于装在
云台上的相机）只把取景窗内的像素交给感知；`"full"`（开环）看全幅。处理前按 `proc_width` 缩放
（参数都按画面宽缩放，标定在 640 宽上做）。结果里：

- `aim`：瞄准点，坐标在**这一帧送去感知的画面**里（全分辨率像素），`origin` 是那块画面在全幅里的
  左上角 ⇒ 宿主 `VirtualGimbal::aim_in(origin, aim)`，即中心移到 `origin + aim`；
- `target` / `box` / `tracks`：全幅坐标，只给界面叠加用。
"""

# ⚠️ 不用 `from __future__ import annotations`：FastAPI 要在运行时解析路由函数的注解（WebSocket 等）
import argparse
import asyncio
import json
import os
import threading
import time
import zlib
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from . import core, ring
from .candidates.motion import MotionCandidates
from .core import timing


@dataclass
class Params:
    proc_width: int = 640
    """感知用的「全幅等效宽度」：帧先按 `proc_width / 全幅宽` 缩放。0 = 不缩放（原分辨率处理）。"""
    perception: str = "window"
    """`"window"`（闭环，只看取景窗）或 `"full"`（开环，看全幅）。"""
    gate: str = "uncertainty"
    screenfix: bool = True
    stale: bool = True
    shift_chal: bool = True
    checksum: bool = False
    """回传每帧像素的 CRC32（联调测试用：验证帧环传过来的像素逐字节一致）。"""
    fps_hint: float = timing.REF_FPS


class Processor:
    """一帧像素 → 一条结果。不碰 I/O，可以离线直接喂帧（测试与基准用）。"""

    def __init__(self, params: Params, full_w: int, full_h: int):
        self.p = params
        self.full = (full_w, full_h)
        self.s = 1.0 if not params.proc_width else params.proc_width / full_w
        self.ref_w = full_w * self.s
        self.gen = MotionCandidates(self.ref_w)
        self.clk = timing.OnlineClock(roi_w=self.ref_w, fps_hint=params.fps_hint)
        self.pipe = None
        self.roi_size = None
        self.bg_fail = 0

    def _scaled(self, img):
        if self.s == 1.0:
            return img
        h, w = img.shape[:2]
        size = (max(1, int(round(w * self.s))), max(1, int(round(h * self.s))))
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def __call__(self, fr: ring.Frame) -> dict:
        t0 = time.perf_counter()
        img = fr.pixels
        if self.p.perception == "window":
            x, y, w, h = fr.roi
            img = img[y : y + h, x : x + w]
            origin = (x, y)
        else:
            origin = (0, 0)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        gray = np.ascontiguousarray(self._scaled(gray))
        ph, pw = gray.shape[:2]
        if self.pipe is None or self.roi_size != (pw, ph):
            # 取景窗尺寸在一个会话里不变；变了（换视频）就从头来
            self.roi_size = (pw, ph)
            self.pipe = core.Pipeline(
                (0, 0, pw, ph),
                self.clk,
                gate=self.p.gate,
                selector_kw={"screenfix": self.p.screenfix, "stale": self.p.stale, "shift_chal": self.p.shift_chal},
            )
        k = self.clk.push(fr.t)
        t1 = time.perf_counter()
        r = self.gen(gray, fr.t)
        shift = r.bg_shift
        if shift is None:
            self.bg_fail += 1
            shift = np.zeros(2)
        t2 = time.perf_counter()
        s = self.s
        o = np.array(origin, float)
        st = self.pipe.step(k, r.dets, shift, o * s)
        t3 = time.perf_counter()

        def full(p):
            return None if p is None else [float(o[0] + p[0] / s), float(o[1] + p[1] / s)]

        box = None
        if st.tid is not None:
            tr = next((t for t in self.pipe.tk.tracks if t.id == st.tid and t.missed == 0), None)
            if tr is not None:
                d = min(r.dets, key=lambda d: (d[0] - tr.last_obs[0]) ** 2 + (d[1] - tr.last_obs[1]) ** 2, default=None)
                if d is not None:
                    box = [float(o[0] + d[3] / s), float(o[1] + d[4] / s), float(o[0] + d[5] / s), float(o[1] + d[6] / s)]
        cum = st.cum
        tracks = [
            [int(tid), *full((v[0] + cum[0], v[1] + cum[1]))] for tid, v in st.row[1].items() if v[5] == 0
        ]
        out = {
            "k": k,
            "index": fr.index,
            "frame": fr.frame_no,
            "t": fr.t,
            "skipped": fr.skipped,
            "origin": [int(origin[0]), int(origin[1])],
            "aim": None if st.aim is None else [float(st.aim[0] / s), float(st.aim[1] / s)],
            "phase": st.phase,
            "tid": None if st.tid is None else int(st.tid),
            "target": full(st.target),
            "box": box,
            "tracks": tracks,
            "n_dets": len(r.dets),
            "bg_ok": r.bg_shift is not None,
            "timing": {
                "prep_ms": (t1 - t0) * 1e3,
                "cand_ms": (t2 - t1) * 1e3,
                "core_ms": (t3 - t2) * 1e3,
            },
        }
        if self.p.checksum:
            out["crc32"] = zlib.crc32(fr.pixels.tobytes())
        return out


@dataclass
class Stats:
    frames: int = 0
    skipped: int = 0
    started: float = 0.0
    last_ms: dict = field(default_factory=dict)
    proc_ms: list = field(default_factory=list)


class Engine:
    """会话：一个后台线程从帧环按序取帧、处理、把结果交给订阅者。"""

    def __init__(self):
        self.params = Params()
        self.session = None
        self._thread = None
        self._stop = threading.Event()
        self._subs: list = []  # (loop, asyncio.Queue)
        self._lock = threading.Lock()
        self.stats = Stats()
        self.error = None
        self.reader = None

    # ── 订阅（WebSocket）──
    def subscribe(self, loop, q):
        with self._lock:
            self._subs.append((loop, q))

    def unsubscribe(self, q):
        with self._lock:
            self._subs = [s for s in self._subs if s[1] is not q]

    def _publish(self, msg: str):
        with self._lock:
            subs = list(self._subs)
        for loop, q in subs:
            loop.call_soon_threadsafe(q.put_nowait, msg)

    # ── 会话 ──
    def start(self, ring_name: str, session: str, params: dict | None = None):
        self.stop()
        if params:
            self.set_params(params)
        self.reader = ring.RingReader(ring_name)
        self.session = session
        self.error = None
        self.stats = Stats(started=time.perf_counter())
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(self.reader, session, Params(**asdict(self.params))), daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        self.session = None

    def set_params(self, d: dict):
        known = asdict(self.params)
        bad = set(d) - set(known)
        if bad:
            raise ValueError(f"未知参数 {sorted(bad)}")
        if "perception" in d and d["perception"] not in ("window", "full"):
            raise ValueError("perception 只能是 window / full")
        if "gate" in d and d["gate"] not in core.mot.GATES:
            raise ValueError(f"gate 只能是 {core.mot.GATES}")
        self.params = Params(**{**known, **d})

    def _run(self, reader: ring.RingReader, session: str, params: Params):
        proc = None
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                fr = reader.next(timeout=0.1)
                if fr is None:
                    if reader.closed and reader.next_index >= reader.write_count:
                        break
                    continue
                t1 = time.perf_counter()
                if proc is None:
                    proc = Processor(params, fr.width, fr.height)
                out = proc(fr)
                t2 = time.perf_counter()
                out["session"] = session
                out["timing"]["read_ms"] = fr.copy_ms
                out["timing"]["wait_ms"] = (t1 - t0) * 1e3 - fr.copy_ms
                out["timing"]["proc_ms"] = (t2 - t1) * 1e3
                self.stats.frames += 1
                self.stats.skipped += fr.skipped
                self.stats.last_ms = out["timing"]
                self.stats.proc_ms.append(out["timing"]["proc_ms"])
                if len(self.stats.proc_ms) > 600:
                    del self.stats.proc_ms[:300]
                self._publish(json.dumps(out))
        except Exception as e:  # noqa: BLE001 —— 报到状态里，不让线程静默死掉
            self.error = f"{type(e).__name__}: {e}"
            self._publish(json.dumps({"session": session, "error": self.error}))
        finally:
            self._publish(json.dumps({"session": session, "end": True, "frames": self.stats.frames}))

    def status(self) -> dict:
        st = self.stats
        el = time.perf_counter() - st.started if st.started else 0.0
        pm = st.proc_ms[-120:]
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "session": self.session,
            "frames": st.frames,
            "skipped": st.skipped,
            "torn": self.reader.torn if self.reader else 0,
            "fps": st.frames / el if el > 0 else 0.0,
            "proc_ms_median": float(np.median(pm)) if pm else None,
            "last_ms": st.last_ms,
            "error": self.error,
            "params": asdict(self.params),
        }


def create_app(engine: Engine | None = None):
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect  # noqa: PLC0415
    from pydantic import BaseModel  # noqa: PLC0415

    from contextlib import asynccontextmanager  # noqa: PLC0415

    eng = engine or Engine()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        eng.stop()

    app = FastAPI(title="motion-gimbal vision", lifespan=lifespan)
    app.state.engine = eng

    class StartReq(BaseModel):
        ring: str
        session: str = "default"
        params: dict | None = None

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/session/start")
    def start(req: StartReq):
        try:
            eng.start(req.ring, req.session, req.params)
        except (RuntimeError, ValueError, OSError, FileNotFoundError) as e:
            raise HTTPException(400, str(e)) from e
        return eng.status()

    @app.post("/session/stop")
    def stop():
        eng.stop()
        return eng.status()

    @app.get("/status")
    def status():
        return eng.status()

    @app.get("/params")
    def get_params():
        return asdict(eng.params)

    @app.put("/params")
    def put_params(d: dict):
        """参数在**下一次**会话开始时生效（会话中途换参数会让跟踪状态前后不一致）。"""
        try:
            eng.set_params(d)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e)) from e
        return asdict(eng.params)

    @app.websocket("/results")
    async def results(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue()
        eng.subscribe(asyncio.get_running_loop(), q)
        try:
            while True:
                await ws.send_text(await q.get())
        except WebSocketDisconnect:
            pass
        finally:
            eng.unsubscribe(q)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--cv-threads",
        type=int,
        default=int(os.environ.get("MG_CV_THREADS", "0")),
        help="OpenCV 线程数（0 = OpenCV 默认）；测量时设 1，避免与其他进程争用时数字失真",
    )
    args = ap.parse_args()
    if args.cv_threads > 0:
        cv2.setNumThreads(args.cv_threads)
    import uvicorn  # noqa: PLC0415

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning", ws="websockets")


if __name__ == "__main__":
    main()
