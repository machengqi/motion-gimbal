"""E2 / E2b：在 MoCA 上做**端到端跟拍**评测 —— 运动候选 → 跟踪 → 显著性选择 → 平滑瞄准 → 虚拟取景窗。

    uv run python scripts/eval_e2.py --split dev [--protocol e2|e2b] [--frac 0.5] [--methods ...] [--workers 16]

输出 `runs/<protocol>-<split>-frac<frac>-<时间>.json`（逐段指标 + 汇总）并打印汇总。
⚠️ 划分见 `vision/splits/moca_split.json`（`scripts/make_split.py`）：修复与参数只在 dev 上选，test 只在
最后各跑一次。

# 方法（同一套指标）

`--methods` 逗号分隔，默认 [`DEFAULT_METHODS`]：

| 键 | 回路 | 说明 |
|---|---|---|
| `center`        | —    | 对照 (a)：取景窗固定（E2：画面中心；E2b：画布上的初始中心） |
| `naive`         | 开环 | 对照 (b)：每帧对准运动量最大的候选；无候选不动 |
| `closed_naive`  | 闭环 | 对照 (b) 的闭环版本 |
| `core:<变体>`   | 开环 | `core.run(...)`，感知看全幅；变体见 [`VARIANTS`] |
| `closed:<变体>` | 闭环 | 逐帧在线：感知只看取景窗内的像素，瞄准结果决定下一帧取景窗 |

**开环**：运动候选在全幅上生成，瞄准结果只用来摆放取景窗，取景窗不影响感知（高估真实云台）。
**闭环**：每帧裁出当前取景窗交给候选生成（相当于装在云台上的相机），瞄准结果决定**下一帧**的取景窗；
评测用的是这一帧实际裁出的取景窗（含一帧延迟）。闭环逐帧在线执行，回填只用到当前帧为止的观测
（开环 `core.run` 的回填不前视，2026-10-14 已在全集上验证与逐帧在线逐帧相同）。

# 两种协议

- **`e2`**：取景窗在**素材画面**坐标里，宽高为全幅的 `--frac`，中心钳在取景窗不出界的范围内（与
  `src-tauri/src/gimbal.rs` 一致）；瞄准点为 `None` 时取景窗在画面里不动。⚠️ MoCA 是摄影师跟拍的素材，
  画面中心近似目标 ⇒ 固定中心近似 oracle。
- **`e2b`（背景稳定化）**：用 `gimbal_vision.stabilize`（SIFT + RANSAC，独立于被测方法、不看真值）逐帧
  累积出画布偏移 `o[k]`（画布 = 画面 + `o[k]`），抵消摄影师的跟拍。取景窗住在**画布**上：瞄准点为 `None`
  时它在画布里不动（即相对世界不动），目标真实地离开初始位置。约束只有一条：**取景窗中心钳在当前帧的
  素材范围内**（中心像素必须是真实拍到的；窗的其余部分可以伸出素材边界，伸出部分补黑、不参与感知）。
  所有方法共用同一份 `o[k]`。细节与局限见 docs/EXPERIMENTS.md 的 E2b 小节。

# 指标（只在有真值框、非复制帧上算）

- **time-in-frame**（主指标）：真值框中心落在取景窗内的帧占比。
- **完全在窗内**：真值框整个落在取景窗内的帧占比。
- **归一化中心误差**：取景窗中心到真值框中心的距离 ÷ 取景窗对角线。
- **首次锁定**：从第一个有真值的帧起，到真值中心**连续** `LOCK_SEC` 秒都在窗内的起点所经过的秒数。
- 另外逐帧记录 E1 口径的单帧候选命中（`e1hit`）。

坐标：全部在 640 宽（按 `--width`）缩放后的素材像素系里比较（E2b 的取景窗从画布换回当帧画面系）。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from gimbal_vision import core, stabilize
from gimbal_vision.candidates.motion import MotionCandidates
from gimbal_vision.core import controller, mot, saliency, timing
from gimbal_vision.datasets import moca

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_split import load_split  # noqa: E402

TIGHT = 4.0
"""与 E1 相同的单帧命中判据（候选面积 ≤ TIGHT × 真值面积）。"""
LOCK_SEC = 0.5
RUNS = Path(__file__).resolve().parents[2] / "runs"
CACHE = RUNS / "diag_cache"

LEGACY_R = 9.0
"""修复前的观测噪声 `R = diag(9, 9)`（消融基线用）。"""

_OFF = {"shift_chal": False, "screenfix": False, "stale": False}
VARIANTS = {
    # 消融：修复前的系统 + 逐项打开
    "base": {"gate": "uncertainty", "meas_var": LEGACY_R, "sel": dict(_OFF)},
    "R": {"gate": "uncertainty", "meas_var": None, "sel": dict(_OFF)},
    "chal": {"gate": "uncertainty", "meas_var": LEGACY_R, "sel": {**_OFF, "shift_chal": True}},
    "sf": {"gate": "uncertainty", "meas_var": LEGACY_R, "sel": {**_OFF, "screenfix": True}},
    "stale": {"gate": "uncertainty", "meas_var": LEGACY_R, "sel": {**_OFF, "stale": True}},
    "all": {"gate": "uncertainty", "meas_var": None, "sel": {"shift_chal": True, "screenfix": True, "stale": True}},
    # 留一法：全部打开，只关掉一项
    "no_R": {"gate": "uncertainty", "meas_var": LEGACY_R, "sel": {"shift_chal": True, "screenfix": True, "stale": True}},
    "no_chal": {"gate": "uncertainty", "meas_var": None, "sel": {"shift_chal": False, "screenfix": True, "stale": True}},
    "no_sf": {"gate": "uncertainty", "meas_var": None, "sel": {"shift_chal": True, "screenfix": False, "stale": True}},
    "no_stale": {"gate": "uncertainty", "meas_var": None, "sel": {"shift_chal": True, "screenfix": True, "stale": False}},
    # 修复后的系统 = 核心代码的默认值（`mot.MEAS_VAR` + `Selector` 的默认开关）
    "fixed": {"gate": "uncertainty", "meas_var": None, "sel": {}},
    "speed_base": {"gate": "speed", "meas_var": LEGACY_R, "sel": dict(_OFF)},
    "speed": {"gate": "speed", "meas_var": None, "sel": {}},
}
DEFAULT_METHODS = "center,naive,closed_naive,core:base,core:fixed,core:speed,closed:base,closed:fixed,closed:speed"


# ── 取景窗几何 ──────────────────────────────────────────────────
class Window:
    """取景窗。状态（中心）存在**画布**坐标里；`o` 是当帧画布偏移（画布 = 画面 + o）。

    `mode="e2"`：画布 ≡ 画面（`o` 恒为 0），中心钳在整窗不出界的范围内（与 `gimbal.rs` 一致）。
    `mode="e2b"`：中心钳在当帧素材范围 `[o, o + (W, H)]` 内，窗可伸出素材边界。
    """

    def __init__(self, full_w, full_h, frac, mode="e2"):
        self.W, self.H = full_w, full_h
        self.w, self.h = max(1, int(full_w * frac)), max(1, int(full_h * frac))
        self.mode = mode
        self.o = np.zeros(2)
        self.clamped = 0
        self.cx, self.cy = full_w / 2, full_h / 2
        self._clamp()

    def _clamp(self):
        if self.mode == "e2":
            lo = self.o + (self.w / 2, self.h / 2)
            hi = self.o + (self.W - self.w / 2, self.H - self.h / 2)
        else:
            lo, hi = self.o, self.o + (self.W, self.H)
        cx, cy = float(np.clip(self.cx, lo[0], hi[0])), float(np.clip(self.cy, lo[1], hi[1]))
        moved = abs(cx - self.cx) > 1e-9 or abs(cy - self.cy) > 1e-9
        self.cx, self.cy = cx, cy
        return bool(moved)

    def frame(self, o):
        """换到新的一帧（画布偏移 `o`），并按新的素材范围钳位。返回这一帧是否被钳。"""
        self.o = np.asarray(o, float)
        return self._clamp()

    def set_center(self, cx, cy):
        """按**当帧画面坐标**设定中心。"""
        self.cx, self.cy = cx + self.o[0], cy + self.o[1]
        return self._clamp()

    def rect(self):
        """当帧画面坐标下的 `(x, y, w, h)`（整数）。E2 下与 `gimbal.rs` 的 `rect` 一致。"""
        x = round(self.cx - self.o[0] - self.w / 2)
        y = round(self.cy - self.o[1] - self.h / 2)
        if self.mode == "e2":
            x, y = min(max(x, 0), self.W - self.w), min(max(y, 0), self.H - self.h)
        return int(x), int(y), self.w, self.h


def crop(img, x, y, w, h, erode):
    """从画面裁 `(x, y, w, h)`；伸出画面的部分补黑。返回 `(图, 有效像素掩码或None)`。"""
    H, W = img.shape[:2]
    if x >= 0 and y >= 0 and x + w <= W and y + h <= H:
        return np.ascontiguousarray(img[y : y + h, x : x + w]), None
    out = np.zeros((h, w), img.dtype)
    m = np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 > x0 and y1 > y0:
        out[y0 - y : y1 - y, x0 - x : x1 - x] = img[y0:y1, x0:x1]
        m[y0 - y : y1 - y, x0 - x : x1 - x] = 255
        if erode > 0:
            m = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8), borderType=cv2.BORDER_REPLICATE)
    return out, m


MASK_ERODE = 12
"""补黑区域外再让出的像素（LK 窗口半宽 10 + 余量），避免角点的邻域跨到黑边上。"""


def naive_pick(dets):
    """「运动量最大」的候选：先比 conf（残余速度中位数，封顶 1.0），并列再比面积（运动点多）。"""
    if not dets:
        return None
    d = max(dets, key=lambda d: (d[7], d[2]))
    return d[0], d[1]


class Causal:
    """逐帧在线执行 跟踪 → 显著性选择 → 平滑瞄准（与 `core.run` 同一组件，回填只用到当前帧）。"""

    def __init__(self, clk, roi, variant):
        v = VARIANTS[variant]
        self.clk = clk
        self.tk = mot.Tracker(roi, clk, gate=v["gate"], meas_var=v["meas_var"])
        self.sel = saliency.Selector(clk, roi, **v["sel"])
        self.aim = controller.Aim(clk)
        self.max_gap = clk.frames_in(timing.FILL_SEC)
        self.raw: dict = {}
        self.filled: dict = {}
        self.prev_cum = np.zeros(2)

    def step(self, k, dets, shift, origin=None):
        row = self.tk.step(k, dets, shift)
        for tid, v in row[1].items():
            if v[5] == 0:
                self.raw.setdefault(tid, []).append((k, v[0], v[1]))
                self.filled[tid] = mot.fill_gaps({tid: self.raw[tid]}, self.max_gap)[tid]
        cum = row[2]
        bg = cum - self.prev_cum
        self.prev_cum = cum
        _, target, _, fresh = self.sel.step(k, self.tk.history, self.filled, bg, origin)
        return self.aim.step(k, bg, cum, target, fresh)


def load_sequence(name, width):
    seq = moca.load()[name]
    dup = moca.duplicates(seq)
    imgs, ks = [], []
    s = None
    for k, path in enumerate(seq.frames):
        if dup[k]:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        s = width / img.shape[1]
        if s != 1.0:
            img = cv2.resize(img, (width, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        imgs.append(img)
        ks.append(k)
    return seq, dup, imgs, ks, s


def canvas_offsets(name, width, imgs):
    """E2b 的画布偏移（缓存到 `runs/diag_cache/stab-<name>-<width>.pkl`）。"""
    f = CACHE / f"stab-{name}-{width}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    o, inl, fail = stabilize.offsets(imgs)
    d = {"o": o, "inliers": inl, "fail": fail}
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(d))
    return d


def prepare(name: str, width: int, keep=None, stab=True) -> dict:
    """读一段、跑全幅候选（开环共用，与协议、取景窗无关）。

    `keep`（可选）：在**去复制帧后**的帧序列里保留哪些下标（升序）—— 帧率 / 抖动实验用
    （`scripts/eval_fps.py`）。时间戳仍是保留帧的真实时间戳；E2b 画布偏移在完整序列上算好
    （与被测方法无关的「尺子」不随帧率变），再取保留帧的那几行。`keep=None` 与原 E2 / E2b 逐位相同。
    """
    seq, dup, imgs, ks, s = load_sequence(name, width)
    st = canvas_offsets(name, width, imgs) if stab else None
    if keep is not None:
        keep = np.asarray(keep, int)
        imgs, ks = [imgs[i] for i in keep], [ks[i] for i in keep]
        if st is not None:
            inl = st["inliers"][keep]
            st = {"o": st["o"][keep], "inliers": inl, "fail": int((inl[1:] < stabilize.MIN_INLIERS).sum())}
    n = len(imgs)
    H, W = imgs[0].shape[:2]
    ts = np.array(ks, dtype=float) / moca.FPS
    clk = timing.Clock(n, ts, roi_w=W)

    gts, motions = [], []
    for k in ks:
        b = seq.box(k)
        gts.append(None if b is None else b * s)
        motions.append(seq.motion(k))

    gen = MotionCandidates(width)
    obs, shifts, bg_fail = [], [], 0
    for i, img in enumerate(imgs):
        r = gen(img, ts[i])
        obs.append(r.dets)
        if r.bg_shift is None:
            bg_fail += 1
            shifts.append(np.zeros(2))
        else:
            shifts.append(np.asarray(r.bg_shift, float))
    return dict(
        name=name, width=width, seq=seq, dup=dup, imgs=imgs, ks=ks, ts=ts, clk=clk, W=W, H=H, stab=st,
        gts=gts, motions=motions, obs=obs, shifts=shifts, bg_fail=bg_fail,
    )


def eval_sequence(name: str, width: int, frac: float, protocol: str, methods: list[str]) -> dict:
    return evaluate(prepare(name, width, stab=protocol == "e2b"), frac, protocol, methods)


def evaluate(ctx: dict, frac: float, protocol: str, methods: list[str], aux: dict | None = None) -> dict:
    """在 [`prepare`] 的结果上按一种协议、一个取景窗大小评测各方法。

    `aux`（可选 dict）：开环 `core:*` 方法的逐帧输出 `(阶段, 瞄准点, tid)` 写进 `aux[方法]`，供诊断用。
    """
    seq, dup, imgs, ks, ts, clk = ctx["seq"], ctx["dup"], ctx["imgs"], ctx["ks"], ctx["ts"], ctx["clk"]
    width, W, H = ctx["width"], ctx["W"], ctx["H"]
    gts, motions, obs, shifts, bg_fail = ctx["gts"], ctx["motions"], ctx["obs"], ctx["shifts"], ctx["bg_fail"]
    n = len(imgs)
    if protocol == "e2b":
        stab = ctx["stab"]
        offs = stab["o"]
    else:
        stab, offs = None, np.zeros((n, 2))

    rects: dict[str, list] = {}
    clamps: dict[str, int] = {}

    def follow(points):
        win, out, c = Window(W, H, frac, protocol), [], 0
        for i, p in enumerate(points):
            c += win.frame(offs[i])
            if p is not None:
                c += win.set_center(*p)
            out.append(win.rect())
        return out, c

    closed_bg_fail = {}
    for m in methods:
        kind, _, var = m.partition(":")
        if kind == "center":
            rects[m], clamps[m] = follow([None] * n)
        elif kind == "naive":
            rects[m], clamps[m] = follow([naive_pick(d) for d in obs])
        elif kind == "core":
            v = VARIANTS[var]
            out = core.run(
                obs, (0, 0, W, H), clock=clk, gate=v["gate"], shifts=shifts, meas_var=v["meas_var"], selector_kw=v["sel"]
            )
            rects[m], clamps[m] = follow([r[1] for r in out])
            if aux is not None:
                aux[m] = out
        elif kind in ("closed", "closed_naive"):
            win = Window(W, H, frac, protocol)
            g = MotionCandidates(width)  # 像素密度与全幅相同 ⇒ 阈值缩放沿用全幅宽度
            cw, ch = win.w, win.h
            pipe = None if kind == "closed_naive" else Causal(clk, (0, 0, cw, ch), var)
            rs, fails, c = [], 0, 0
            for i, img in enumerate(imgs):
                c += win.frame(offs[i])
                x, y, _, _ = win.rect()
                rs.append((x, y, cw, ch))
                im, mask = crop(img, x, y, cw, ch, MASK_ERODE)
                r = g(im, ts[i], mask)
                if pipe is None:
                    p = naive_pick(r.dets)
                else:
                    sh = r.bg_shift
                    if sh is None:
                        fails += 1
                        sh = np.zeros(2)
                    p = pipe.step(i, r.dets, sh, (x, y))
                if p is not None:
                    c += win.set_center(x + p[0], y + p[1])  # 取景窗坐标 → 画面；作用于下一帧
            rects[m], clamps[m] = rs, c
            closed_bg_fail[m] = fails
        else:
            raise ValueError(m)

    # ── 逐帧记录 ──
    rows = []
    for i in range(n):
        g = gts[i]
        if g is None:
            continue
        gx1, gy1, gw, gh = g
        gcx, gcy = gx1 + gw / 2, gy1 + gh / 2
        garea = max(gw * gh, 1.0)
        e1 = i > 0 and any(
            gx1 <= d[0] <= gx1 + gw and gy1 <= d[1] <= gy1 + gh and d[2] <= TIGHT * garea for d in obs[i]
        )
        rec = {"i": i, "k": ks[i], "t": float(ts[i]), "motion": motions[i], "e1hit": bool(e1)}
        if protocol == "e2b":
            # 目标在画布上离初始中心多远（÷ 取景窗对角线）—— 衡量「稳定化后目标离开了初始中心多少」
            rec["gt_drift"] = float(
                np.hypot(gcx + offs[i][0] - W / 2, gcy + offs[i][1] - H / 2) / np.hypot(W * frac, H * frac)
            )
        for m in methods:
            x, y, w, h = rects[m][i]
            inside = x <= gcx <= x + w and y <= gcy <= y + h
            full = x <= gx1 and y <= gy1 and gx1 + gw <= x + w and gy1 + gh <= y + h
            err = float(np.hypot(x + w / 2 - gcx, y + h / 2 - gcy) / np.hypot(w, h))
            rec[m] = (bool(inside), bool(full), err)
        rows.append(rec)

    lock = {}
    if rows:
        t0 = rows[0]["t"]
        for m in methods:
            lock[m] = None
            for a in range(len(rows)):
                if not rows[a][m][0]:
                    continue
                b = a
                while b + 1 < len(rows) and rows[b + 1][m][0] and rows[b + 1]["i"] == rows[b]["i"] + 1:
                    b += 1
                if rows[b]["t"] - rows[a]["t"] >= LOCK_SEC:
                    lock[m] = rows[a]["t"] - t0
                    break
    extra = {}
    if stab is not None:
        extra = {
            "stab_fail": int(stab["fail"]),
            "stab_inliers_median": float(np.median(stab["inliers"][1:])) if n > 1 else None,
            "canvas_excursion": float(np.max(np.hypot(*(offs - offs[0]).T)) / W),
        }
    return {
        "name": ctx["name"],
        "frames": len(seq.frames),
        "duplicates": int(dup.sum()),
        "size": [W, H],
        "bg_fail": bg_fail,
        "closed_bg_fail": closed_bg_fail,
        "clamped_frames": clamps,
        "lock": lock,
        "rows": rows,
        **extra,
    }


def q(v):
    return np.percentile(v, [25, 50, 75]).round(3).tolist() if len(v) else None


def summarize(results, methods):
    rows = [r for res in results for r in res["rows"]]

    def agg(rs):
        if not rs:
            return None
        out = {"frames": len(rs), "e1hit": float(np.mean([r["e1hit"] for r in rs]))}
        if "gt_drift" in rs[0]:
            d = np.array([r["gt_drift"] for r in rs])
            out["gt_drift_quartiles"] = q(d)
        for m in methods:
            out[m] = {
                "in_frame": float(np.mean([r[m][0] for r in rs])),
                "fully_in": float(np.mean([r[m][1] for r in rs])),
                "err_mean": float(np.mean([r[m][2] for r in rs])),
                "err_median": float(np.median([r[m][2] for r in rs])),
            }
        return out

    labels = {moca.MOTION.get(m, "unlabeled"): m for m in (0, 1, 2, None)}
    vids = [res for res in results if res["rows"]]
    per_video = {}
    ref = methods[0]
    for m in methods:
        tif = [float(np.mean([r[m][0] for r in res["rows"]])) for res in vids]
        ref_tif = [float(np.mean([r[ref][0] for r in res["rows"]])) for res in vids]
        locks = [res["lock"][m] for res in vids]
        locked = [x for x in locks if x is not None]
        per_video[m] = {
            "in_frame_mean": float(np.mean(tif)),
            "in_frame_quartiles": q(tif),
            "never_locked": int(sum(x is None for x in locks)),
            "lock_sec_quartiles": q(locked),
            f"vs_{ref}_better_worse": [
                int(sum(a - b > 0.01 for a, b in zip(tif, ref_tif, strict=True))),
                int(sum(b - a > 0.01 for a, b in zip(tif, ref_tif, strict=True))),
            ],
            "clamped_frames": int(sum(res["clamped_frames"][m] for res in results)),
        }
    split = {}
    for key, sel in (("e1_hit", True), ("e1_miss", False)):
        rs = [r for r in rows if r["e1hit"] == sel]
        split[key] = {"frames": len(rs), **{m: float(np.mean([r[m][0] for r in rs])) if rs else None for m in methods}}
    out = {
        "all_frames": agg(rows),
        "by_motion": {name: agg([r for r in rows if r["motion"] == m]) for name, m in labels.items()},
        "per_video": per_video,
        "by_e1_hit": split,
        "duplicates_removed": int(sum(r["duplicates"] for r in results)),
        "bg_fail_frames_open": int(sum(r["bg_fail"] for r in results)),
        "bg_fail_frames_closed": {
            m: int(sum(r["closed_bg_fail"].get(m, 0) for r in results)) for m in methods if m.startswith("closed:")
        },
        "videos_without_gt": [r["name"] for r in results if not r["rows"]],
    }
    if results and "stab_fail" in results[0]:
        out["stab_fail_frames"] = int(sum(r["stab_fail"] for r in results))
        out["canvas_excursion_quartiles"] = q([r["canvas_excursion"] for r in results])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test", "all"))
    ap.add_argument("--protocol", default="e2", choices=("e2", "e2b"))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--methods", default=DEFAULT_METHODS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    methods = args.methods.split(",")
    for m in methods:
        kind, _, var = m.partition(":")
        if kind not in ("center", "naive", "closed_naive", "core", "closed") or (var and var not in VARIANTS):
            raise SystemExit(f"未知方法 {m}")

    names = load_split(args.split)
    if args.limit:
        names = names[: args.limit]
    t0 = time.time()
    n = len(names)
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(
            ex.map(eval_sequence, names, [args.width] * n, [args.frac] * n, [args.protocol] * n, [methods] * n)
        )
    summary = summarize(results, methods)
    summary.update(
        {
            "protocol": args.protocol,
            "split": args.split,
            "width": args.width,
            "frac": args.frac,
            "videos": n,
            "methods": methods,
            "variants": {k: VARIANTS[k] for k in sorted({m.partition(":")[2] for m in methods} - {""})},
            "meas_var_calibrated": list(mot.MEAS_VAR),
            "lock_sec": LOCK_SEC,
            "wall_sec": round(time.time() - t0, 1),
        }
    )
    RUNS.mkdir(exist_ok=True)
    tag = f"-{args.tag}" if args.tag else ""
    out = RUNS / f"{args.protocol}-{args.split}-frac{args.frac:g}{tag}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    videos = [
        {
            "name": r["name"],
            "frames": r["frames"],
            "duplicates": r["duplicates"],
            "gt_frames": len(r["rows"]),
            "motion_labels": sorted({str(x["motion"]) for x in r["rows"]}),
            "e1hit": float(np.mean([x["e1hit"] for x in r["rows"]])) if r["rows"] else None,
            "in_frame": {m: float(np.mean([x[m][0] for x in r["rows"]])) if r["rows"] else None for m in methods},
            "lock_sec": r["lock"],
            "bg_fail": r["bg_fail"],
            "closed_bg_fail": r["closed_bg_fail"],
            "clamped_frames": r["clamped_frames"],
            **{k: r[k] for k in ("stab_fail", "stab_inliers_median", "canvas_excursion") if k in r},
        }
        for r in results
    ]
    out.write_text(json.dumps({"summary": summary, "videos": videos}, ensure_ascii=False, indent=2, default=lambda o: o.item()), encoding="utf-8")
    a = summary["all_frames"]
    bm = summary["by_motion"]
    print(f"{args.protocol} {args.split} frac={args.frac:g}  帧 {a['frames']}  用时 {summary['wall_sec']}s")
    print(f"{'方法':22s} 全部   loco   subtle still  完全在窗  逐段均值  优/劣(对{methods[0]})")
    for m in methods:
        cells = [100 * a[m]["in_frame"]] + [
            100 * bm[k][m]["in_frame"] if bm.get(k) else float("nan") for k in ("locomotion", "subtle_motion", "still")
        ]
        pv = summary["per_video"][m]
        print(
            f"{m:22s} " + " ".join(f"{c:5.1f}" for c in cells)
            + f"  {100 * a[m]['fully_in']:5.1f}    {100 * pv['in_frame_mean']:5.1f}   {pv[f'vs_{methods[0]}_better_worse']}"
        )
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
