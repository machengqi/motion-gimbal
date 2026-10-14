"""E2：在 MoCA 上做**端到端跟拍**评测 —— 运动候选 → 跟踪 → 显著性选择 → 平滑瞄准 → 虚拟取景窗。

    uv run python scripts/eval_e2.py [--frac 0.5] [--limit N] [--workers 6] [--debug NAME]

输出 `runs/e2-<时间>.json`（逐段指标 + 汇总）并打印汇总。

# 方法（同一套指标）

| 键 | 回路 | 说明 |
|---|---|---|
| `core`        | 开环 | `core.run(gate="uncertainty", shifts=外部背景位移)`，感知看全幅 |
| `core_speed`  | 开环 | 消融：`gate="speed"`（前期系统的门控） |
| `causal`      | 开环 | 同 `core`，但逐帧在线执行（见下「因果性」） |
| `center`      | —    | 对照 (a)：取景窗固定在画面中心 |
| `naive`       | 开环 | 对照 (b)：每帧对准运动量最大的候选；无候选不动 |
| `closed`        | 闭环 | 逐帧在线：感知只看取景窗内的像素，瞄准结果决定下一帧取景窗 |
| `closed_speed`  | 闭环 | 同上，`gate="speed"` |
| `closed_naive`  | 闭环 | 对照 (b) 的闭环版本 |

**开环**：运动候选在全幅上生成，瞄准结果只用来摆放取景窗，取景窗不影响感知。
这高估了真实云台的效果 —— 真实云台只能看到取景窗里的画面。
**闭环**：每帧从全幅裁出当前取景窗交给候选生成（相当于装在云台上的相机），
瞄准结果决定**下一帧**的取景窗；评测用的是这一帧实际裁出的取景窗（含一帧延迟）。

# 因果性

`core.run` 是离线口径：`mot.fill_gaps` 在整段历史上回填断档，第 k 帧的选择器
能看到「k 之后几帧（`timing.FILL_SEC`）的观测」插出来的点。`causal` / `closed*`
逐帧执行，回填只用到当前帧为止的观测 —— 与在线运行一致。`core` 与 `causal`
的差就是这点前视的影响。两者的时钟都用整段时间戳的中位间隔（同样是一点前视，
在线时由运行中的估计代替）。

# 取景窗

宽高为全幅的 `--frac`（默认 0.5，与 Rust 宿主 `WINDOW_FRAC` 一致），中心钳在
取景窗不出界的范围内（与 `src-tauri/src/gimbal.rs` 的 `set_center` / `rect` 一致）。
初始对准画面中心；瞄准点为 `None` 时取景窗不动。

# 指标（只在有真值框、非复制帧上算）

- **time-in-frame**（主指标）：真值框中心落在取景窗内的帧占比。
- **完全在窗内**：真值框整个落在取景窗内的帧占比。
- **归一化中心误差**：取景窗中心到真值框中心的距离 ÷ 取景窗对角线。
- **首次锁定**：从第一个有真值的帧起，到真值中心**连续** `LOCK_SEC` 秒都在窗内的
  起点所经过的时间（秒）；从未满足记为未锁定。
- 另外逐帧记录 E1 口径的单帧候选命中（`e1hit`），用于拆分「时间连续性补回了多少」。

坐标：全部在 640 宽（按 `--width`）缩放后的全幅像素系里比较；`core.run` 返回的瞄准点
已是当前帧画面系（选择器 = 背景系 + 累计背景位移，控制器每帧 `+= bg`）；闭环下
瞄准点在取景窗坐标系，加上该帧取景窗原点换回全幅。
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from gimbal_vision import core
from gimbal_vision.candidates.motion import MotionCandidates
from gimbal_vision.core import controller, mot, saliency, timing
from gimbal_vision.datasets import moca

TIGHT = 4.0
"""与 E1 相同的单帧命中判据（候选面积 ≤ TIGHT × 真值面积）。"""
LOCK_SEC = 0.5
RUNS = Path(__file__).resolve().parents[2] / "runs"

OPEN = ("center", "naive", "core", "core_speed", "causal")
CLOSED = ("closed", "closed_speed", "closed_naive")
METHODS = OPEN + CLOSED


# ── 取景窗几何（与 src-tauri/src/gimbal.rs 一致）────────────────────
class Window:
    def __init__(self, full_w, full_h, frac):
        self.W, self.H = full_w, full_h
        self.w, self.h = max(1, int(full_w * frac)), max(1, int(full_h * frac))
        self.set_center(full_w / 2, full_h / 2)

    def set_center(self, cx, cy):
        hw, hh = self.w / 2, self.h / 2
        self.cx = float(np.clip(cx, hw, self.W - hw))
        self.cy = float(np.clip(cy, hh, self.H - hh))

    def rect(self):
        x = min(max(round(self.cx - self.w / 2), 0), self.W - self.w)
        y = min(max(round(self.cy - self.h / 2), 0), self.H - self.h)
        return int(x), int(y), self.w, self.h


def naive_pick(dets):
    """「运动量最大」的候选：先比 conf（残余速度中位数，封顶 1.0），并列再比面积（运动点多）。"""
    if not dets:
        return None
    d = max(dets, key=lambda d: (d[7], d[2]))
    return d[0], d[1]


class Causal:
    """逐帧在线执行 跟踪 → 显著性选择 → 平滑瞄准（与 `core.run` 同一组件，回填只用到当前帧）。"""

    def __init__(self, clk, roi, gate):
        self.clk = clk
        self.tk = mot.Tracker(roi, clk, gate=gate)
        self.sel = saliency.Selector(clk, roi)
        self.aim = controller.Aim(clk)
        self.max_gap = clk.frames_in(timing.FILL_SEC)
        self.raw: dict = {}
        self.filled: dict = {}
        self.prev_cum = np.zeros(2)

    def step(self, k, dets, shift):
        row = self.tk.step(k, dets, shift)
        for tid, v in row[1].items():
            if v[5] == 0:
                self.raw.setdefault(tid, []).append((k, v[0], v[1]))
                self.filled[tid] = mot.fill_gaps({tid: self.raw[tid]}, self.max_gap)[tid]
        cum = row[2]
        bg = cum - self.prev_cum
        self.prev_cum = cum
        _, target, _, fresh = self.sel.step(k, self.tk.history, self.filled, bg)
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


def eval_sequence(name: str, width: int, frac: float, debug: bool = False) -> dict:
    seq, dup, imgs, ks, s = load_sequence(name, width)
    n = len(imgs)
    H, W = imgs[0].shape[:2]
    ts = np.array(ks, dtype=float) / moca.FPS
    clk = timing.Clock(n, ts, roi_w=W)

    # 真值（缩放到 640 宽）
    gts, motions = [], []
    for k in ks:
        b = seq.box(k)
        gts.append(None if b is None else b * s)
        motions.append(seq.motion(k))

    # ── 全幅候选（开环共用）──
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

    centers: dict[str, list] = {}  # 每帧取景窗矩形 (x, y, w, h)

    def follow(points):
        win, out = Window(W, H, frac), []
        for p in points:
            if p is not None:
                win.set_center(*p)
            out.append(win.rect())
        return out

    centers["center"] = follow([None] * n)
    centers["naive"] = follow([naive_pick(d) for d in obs])
    roi = (0, 0, W, H)
    run_unc = core.run(obs, roi, clock=clk, gate="uncertainty", shifts=shifts)
    run_spd = core.run(obs, roi, clock=clk, gate="speed", shifts=shifts)
    centers["core"] = follow([r[1] for r in run_unc])
    centers["core_speed"] = follow([r[1] for r in run_spd])
    cz = Causal(clk, roi, "uncertainty")
    causal_pts = [cz.step(i, obs[i], shifts[i]) for i in range(n)]
    centers["causal"] = follow(causal_pts)
    diff = [
        a[1] is not None and b is not None and float(np.hypot(a[1][0] - b[0], a[1][1] - b[1])) > 1e-6
        or (a[1] is None) != (b is None)
        for a, b in zip(run_unc, causal_pts, strict=True)
    ]

    # ── 闭环：感知只看取景窗 ──
    closed_bg_fail = {}
    for m in CLOSED:
        win = Window(W, H, frac)
        g = MotionCandidates(width)  # 像素密度与全幅相同 ⇒ 阈值缩放沿用全幅宽度
        cw, ch = win.w, win.h
        pipe = None if m == "closed_naive" else Causal(clk, (0, 0, cw, ch), "speed" if m == "closed_speed" else "uncertainty")
        rects, fails = [], 0
        for i, img in enumerate(imgs):
            x, y, _, _ = win.rect()
            rects.append((x, y, cw, ch))
            r = g(np.ascontiguousarray(img[y : y + ch, x : x + cw]), ts[i])
            if pipe is None:
                p = naive_pick(r.dets)
            else:
                sh = r.bg_shift
                if sh is None:
                    fails += 1
                    sh = np.zeros(2)
                p = pipe.step(i, r.dets, sh)
            if p is not None:
                win.set_center(x + p[0], y + p[1])  # 取景窗坐标 → 全幅；作用于下一帧
        centers[m] = rects
        closed_bg_fail[m] = fails

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
        rec = {
            "i": i, "k": ks[i], "t": float(ts[i]), "motion": motions[i], "e1hit": bool(e1),
            "phase": run_unc[i][0], "aimed": run_unc[i][1] is not None,
        }
        for m in METHODS:
            x, y, w, h = centers[m][i]
            inside = x <= gcx <= x + w and y <= gcy <= y + h
            full = x <= gx1 and y <= gy1 and gx1 + gw <= x + w and gy1 + gh <= y + h
            err = float(np.hypot(x + w / 2 - gcx, y + h / 2 - gcy) / np.hypot(w, h))
            rec[m] = (bool(inside), bool(full), err)
        rows.append(rec)
        if debug and i % 10 == 0:
            cr = run_unc[i]
            print(
                f"{name} k={ks[i]:4d} gt=({gcx:6.1f},{gcy:6.1f}) aim={cr[1] and tuple(round(v, 1) for v in cr[1])} "
                f"phase={cr[0]} core_win={centers['core'][i]} closed_win={centers['closed'][i]} "
                f"in(core,closed)={rec['core'][0]},{rec['closed'][0]}"
            )

    # 首次锁定
    lock = {}
    if rows:
        t0 = rows[0]["t"]
        for m in METHODS:
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
    return {
        "name": name,
        "frames": len(seq.frames),
        "duplicates": int(dup.sum()),
        "size": [W, H],
        "bg_fail": bg_fail,
        "closed_bg_fail": closed_bg_fail,
        "core_vs_causal_diff_frames": int(sum(diff)),
        "lock": lock,
        "rows": rows,
    }


def q(v):
    return np.percentile(v, [25, 50, 75]).round(3).tolist() if len(v) else None


def summarize(results):
    rows = [r for res in results for r in res["rows"]]

    def agg(rs):
        if not rs:
            return None
        out = {"frames": len(rs), "e1hit": float(np.mean([r["e1hit"] for r in rs]))}
        for m in METHODS:
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
    for m in METHODS:
        tif = [float(np.mean([r[m][0] for r in res["rows"]])) for res in vids]
        locks = [res["lock"][m] for res in vids]
        locked = [x for x in locks if x is not None]
        per_video[m] = {
            "in_frame_mean": float(np.mean(tif)),
            "in_frame_quartiles": q(tif),
            "never_locked": int(sum(x is None for x in locks)),
            "lock_sec_quartiles": q(locked),
        }
    # 拆分：E1 单帧命中 / 未命中的帧上，各方法的 time-in-frame
    split = {}
    for key, sel in (("e1_hit", True), ("e1_miss", False)):
        rs = [r for r in rows if r["e1hit"] == sel]
        split[key] = {"frames": len(rs), **{m: float(np.mean([r[m][0] for r in rs])) if rs else None for m in METHODS}}
    # `core` 的选择器阶段（wait 时 aimed=False 表示还没有过任何估计）
    phases = {}
    for ph in ("track", "jump", "wait"):
        for aimed in (True, False):
            rs = [r for r in rows if r["phase"] == ph and r["aimed"] == aimed]
            if rs:
                phases[f"{ph}{'' if aimed else '_noaim'}"] = {
                    "frames": len(rs), "core_in_frame": float(np.mean([r["core"][0] for r in rs])),
                    "center_in_frame": float(np.mean([r["center"][0] for r in rs])),
                    "e1hit": float(np.mean([r["e1hit"] for r in rs])),
                }
    return {
        "all_frames": agg(rows),
        "core_by_phase": phases,
        "by_motion": {name: agg([r for r in rows if r["motion"] == m]) for name, m in labels.items()},
        "per_video": per_video,
        "by_e1_hit": split,
        "duplicates_removed": int(sum(r["duplicates"] for r in results)),
        "bg_fail_frames_open": int(sum(r["bg_fail"] for r in results)),
        "bg_fail_frames_closed": {m: int(sum(r["closed_bg_fail"][m] for r in results)) for m in CLOSED},
        "core_vs_causal_diff_frames": int(sum(r["core_vs_causal_diff_frames"] for r in results)),
        "videos_without_gt": [r["name"] for r in results if not r["rows"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--debug", default="", help="只跑这一段并打印抽样轨迹")
    args = ap.parse_args()

    if args.debug:
        res = eval_sequence(args.debug, args.width, args.frac, debug=True)
        print(json.dumps(summarize([res]), ensure_ascii=False, indent=2))
        return

    names = sorted(moca.load())
    if args.limit:
        names = names[: args.limit]
    t0 = time.time()
    n = len(names)
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(eval_sequence, names, [args.width] * n, [args.frac] * n))
    summary = summarize(results)
    summary.update(
        {"width": args.width, "frac": args.frac, "videos": n, "lock_sec": LOCK_SEC, "wall_sec": round(time.time() - t0, 1)}
    )
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"e2-frac{args.frac:g}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    videos = [
        {
            "name": r["name"],
            "frames": r["frames"],
            "duplicates": r["duplicates"],
            "gt_frames": len(r["rows"]),
            "motion_labels": sorted({str(x["motion"]) for x in r["rows"]}),
            "e1hit": float(np.mean([x["e1hit"] for x in r["rows"]])) if r["rows"] else None,
            "in_frame": {m: float(np.mean([x[m][0] for x in r["rows"]])) if r["rows"] else None for m in METHODS},
            "lock_sec": r["lock"],
            "bg_fail": r["bg_fail"],
            "closed_bg_fail": r["closed_bg_fail"],
            "core_vs_causal_diff_frames": r["core_vs_causal_diff_frames"],
        }
        for r in results
    ]
    out.write_text(json.dumps({"summary": summary, "videos": videos}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
