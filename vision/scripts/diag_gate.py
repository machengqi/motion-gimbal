"""任务 2：门控反转（speed 优于 uncertainty）的测量。**只测量，不改核心默认参数。**

    uv run python scripts/diag_gate.py [--workers 6] [--limit N]

全部 141 段 MoCA，开环全幅候选（与 E2 同一份缓存）。三件事：

1. **框心抖动 vs R**：拿 `speed` 门控的关联结果（距离闸，对观测残差没有截断）
   取每条轨迹段的原始观测框心，在**帧间隔相等**的连续三帧上算二阶差分
   `d2 = z[k+1] − 2 z[k] + z[k−1]`。目标做匀速运动时 `var(d2) = 6 σ²`（σ² = 每轴观测噪声），
   加速度只会让它变大 ⇒ `σ̂² = var(d2) / 6` 是观测噪声的**上界**。另给 MAD 稳健估计
   （`(1.4826·MAD)² / 6`，抗少量关联错误）。分「全部轨迹」与「落在真值框内的轨迹」两组。

2. **NIS**：把 `speed` 门控下每条轨迹段的原始观测序列，原样重放进 uncertainty 模式的
   Kalman（`Track(dt_aware=True)`，同一组 Q / R），每个观测在更新前算马氏距离平方
   （= NIS）。模型自洽时 NIS ~ χ²(2)：均值 2，超过 `GATE_CHI2 = 9.21` 的占 1%。

3. **轨迹寿命与断轨**：两种门控各跑一遍跟踪器，记录每个轨迹段（一个 `Track` 对象从出生到
   被删）的观测数与时长；「断轨」= 轨迹段被删除；另计缝合次数。目标段 = 半数以上观测落在
   真值框内的轨迹段。

另：对照实验（临时把 `Track.R` 换成实测量级，只在本脚本的子类里）的端到端 time-in-frame。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import load  # noqa: E402
from eval_e2 import RUNS, Window  # noqa: E402

from gimbal_vision.core import controller, mot, saliency, timing  # noqa: E402
from gimbal_vision.datasets import moca  # noqa: E402

# 2026-10-20：核心默认值已改（R 标定、chal 平移、screenfix、stale）；本脚本固定用修复前的口径，
# 以保持 2026-10-14 诊断结论可复现。
LEGACY_SEL = {"shift_chal": False, "screenfix": False, "stale": False}
LEGACY_R = 9.0


def make_tracker(R=None):
    """带记录的 Tracker：只覆盖 `_born`（换成记录观测的 Track 子类），关联逻辑不动。"""

    class LTrack(mot.Track):

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.R = np.diag([LEGACY_R if R is None else R] * 2)
            self.log = []  # [(k, zx, zy)]

        def update(self, zx, zy, area):
            self.log.append((self.k_now, zx, zy))
            super().update(zx, zy, area)

    class LTracker(mot.Tracker):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.segments = []
            self.stitches = 0
            self.k_now = 0

        def _born(self, tid, d, k):
            t = LTrack(tid, d[0], d[1], d[2], dt_aware=self.gate == "uncertainty")
            t.last_k = k
            t.edge = self.edge_of(d)
            t.k_now = k
            t.log.append((k, d[0], d[1]))
            if tid < self.next_id:
                self.stitches += 1
            self.segments.append(t)
            return t

        def step(self, k, dets_all, bg_shift=None):
            for t in self.tracks:
                t.k_now = k
            return super().step(k, dets_all, bg_shift)

    return LTracker


def in_box(p, g):
    return g is not None and g[0] <= p[0] <= g[0] + g[2] and g[1] <= p[1] <= g[1] + g[3]


def run_tracker(d, gate, R=None):
    clk, W, H = d["clk"], d["W"], d["H"]
    tk = make_tracker(R)((0, 0, W, H), clk, gate=gate)
    for k, dets in enumerate(d["obs"]):
        # 新出生轨迹的 k_now 在 _born 里赋；存活轨迹在 step 开头赋
        tk.step(k, dets, d["shifts"][k])
    return tk


def end_to_end(d, gate, R=None, frac=0.5):
    """与 core.run 同一条流水线，只是 Tracker 换成本脚本的子类（R 可临时替换）。"""
    clk, W, H, n = d["clk"], d["W"], d["H"], len(d["obs"])
    tk = run_tracker(d, gate, R)
    hist = tk.history
    seq = mot.fill_gaps(mot.paths(hist), clk.frames_in(timing.FILL_SEC))
    sel, aim = saliency.Selector(clk, (0, 0, W, H), **LEGACY_SEL), controller.Aim(clk)
    prev, win, hits = np.zeros(2), Window(W, H, frac), []
    for k in range(n):
        cum = hist[k][2]
        bg = cum - prev
        prev = cum
        _, tgt, _, fresh = sel.step(k, hist, seq, bg)
        p = aim.step(k, bg, cum, tgt, fresh)
        if p is not None:
            win.set_center(*p)
        g = d["gts"][k]
        if g is not None:
            x, y, w, h = win.rect()
            hits.append(x <= g[0] + g[2] / 2 <= x + w and y <= g[1] + g[3] / 2 <= y + h)
    return hits


def analyse(name, r_alt=None):
    d = load(name)
    clk, gts, ts = d["clk"], d["gts"], d["ts"]
    out = {"name": name}
    # ── 1+2：speed 门控下的原始观测 ──
    tk = run_tracker(d, "speed")
    d2_all, d2_gt, nis_all, nis_gt = [], [], [], []
    for seg in tk.segments:
        log = seg.log
        on_gt = np.mean([in_box((x, y), gts[k]) for k, x, y in log]) > 0.5 if log else False
        # 二阶差分：连续三帧、帧号连续且两段 dt 相等
        for (k0, x0, y0), (k1, x1, y1), (k2, x2, y2) in zip(log, log[1:], log[2:], strict=False):
            if k1 == k0 + 1 and k2 == k1 + 1 and abs((ts[k2] - ts[k1]) - (ts[k1] - ts[k0])) < 1e-6:
                v = (x2 - 2 * x1 + x0, y2 - 2 * y1 + y0)
                d2_all.append(v)
                if on_gt:
                    d2_gt.append(v)
        # NIS 重放：uncertainty 模式的 Kalman，同一串观测
        if len(log) >= 2:
            t = mot.Track(-1, log[0][1], log[0][2], 0.0, dt_aware=True, meas_var=LEGACY_R)
            kk = log[0][0]
            for k, x, y in log[1:]:
                while kk < k:
                    kk += 1
                    t.predict(clk.dt(kk))
                m = t.mahal2(x, y)
                nis_all.append(m)
                if on_gt:
                    nis_gt.append(m)
                t.update(x, y, 0.0)
    out["d2_all"], out["d2_gt"] = d2_all, d2_gt
    out["nis_all"], out["nis_gt"] = nis_all, nis_gt

    # ── 3：两种门控的轨迹段寿命 ──
    for gate in ("uncertainty", "speed"):
        tk = run_tracker(d, gate)
        segs = []
        for seg in tk.segments:
            ks = [k for k, _, _ in seg.log]
            on_gt = np.mean([in_box((x, y), gts[k]) for k, x, y in seg.log]) > 0.5
            segs.append((len(ks), float(ts[ks[-1]] - ts[ks[0]]), bool(on_gt)))
        alive = len(tk.tracks)
        out[gate] = {
            "segments": segs,
            "deleted": len(tk.segments) - alive,
            "stitches": tk.stitches,
            "dets": int(sum(len(x) for x in d["obs"])),
            "duration": float(ts[-1] - ts[0]),
        }
    # ── 对照：端到端 time-in-frame（默认 R，以及临时 R）──
    out["tif"] = {
        "uncertainty": end_to_end(d, "uncertainty"),
        "speed": end_to_end(d, "speed"),
    }
    if r_alt:
        for r in r_alt:
            out["tif"][f"uncertainty_R{r:g}"] = end_to_end(d, "uncertainty", R=r)
    out["motions"] = [d["motions"][k] for k in range(len(gts)) if gts[k] is not None]
    return out


def sigma_est(v):
    a = np.asarray(v)
    if len(a) == 0:
        return None
    var6 = a.var(axis=0) / 6
    med = np.median(a, axis=0)
    mad = np.median(np.abs(a - med), axis=0) * 1.4826
    return {
        "n": len(a),
        "sigma2_var_xy": var6.round(1).tolist(),
        "sigma2_mad_xy": (mad**2 / 6).round(1).tolist(),
        "abs_d2_pctl_50_90_99": np.percentile(np.hypot(a[:, 0], a[:, 1]), [50, 90, 99]).round(1).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--r-alt", default="", help="逗号分隔的临时 R（px²），只在本脚本里替换")
    a = ap.parse_args()
    r_alt = [float(x) for x in a.r_alt.split(",") if x]
    names = sorted(moca.load())
    if a.limit:
        names = names[: a.limit]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(analyse, names, [r_alt] * len(names)))

    print(f"R = diag(9, 9) ⇒ σ = 3px/轴")
    for key, lab in (("d2_all", "全部轨迹"), ("d2_gt", "真值框内轨迹")):
        print(f"[1] 框心观测噪声估计（{lab}）:", sigma_est([v for r in res for v in r[key]]))
    for key, lab in (("nis_all", "全部"), ("nis_gt", "真值框内")):
        v = np.array([x for r in res for x in r[key]])
        print(f"[2] NIS（{lab}） n={len(v)} 均值 {v.mean():.1f} 中位 {np.median(v):.1f} "
              f">9.21 占 {100 * (v > timing.GATE_CHI2).mean():.1f}%（自洽时 1%）  分位 50/90/99 = "
              f"{np.percentile(v, [50, 90, 99]).round(1).tolist()}")
    for gate in ("uncertainty", "speed"):
        segs = [s for r in res for s in r[gate]["segments"]]
        gt_segs = [s for s in segs if s[2]]
        dur = sum(r[gate]["duration"] for r in res)
        n_obs = np.array([s[0] for s in segs])
        print(
            f"[3] {gate:11s} 轨迹段 {len(segs)}（每秒 {len(segs) / dur:.1f}），删除 {sum(r[gate]['deleted'] for r in res)}，"
            f"缝合 {sum(r[gate]['stitches'] for r in res)}；观测数 均值 {n_obs.mean():.2f} 中位 {np.median(n_obs):.0f}，"
            f"单观测段占 {100 * (n_obs == 1).mean():.1f}%；时长均值 {np.mean([s[1] for s in segs]):.3f}s"
        )
        g_obs = np.array([s[0] for s in gt_segs])
        print(
            f"    目标段 {len(gt_segs)}（每秒 {len(gt_segs) / dur:.2f}）；观测数 均值 {g_obs.mean():.2f} 中位 "
            f"{np.median(g_obs):.0f}；单观测段占 {100 * (g_obs == 1).mean():.1f}%；时长均值 "
            f"{np.mean([s[1] for s in gt_segs]):.3f}s；≥ 半个判据窗（{timing.WINDOW_SEC * 0.5:.2f}s）的段占 "
            f"{100 * np.mean([s[1] >= timing.WINDOW_SEC * 0.5 for s in gt_segs]):.1f}%"
        )
    motions = [m for r in res for m in r["motions"]]
    for key in res[0]["tif"]:
        h = np.array([x for r in res for x in r["tif"][key]])
        mm = np.array(motions, dtype=object)
        by = {lab: round(100 * h[mm == m].mean(), 1) for m, lab in moca.MOTION.items() if (mm == m).any()}
        print(f"[4] time-in-frame {key:22s} 全部 {100 * h.mean():.1f}%  {by}")
    out = RUNS / f"diag-gate-{time.strftime('%Y%m%d-%H%M%S')}.json"
    slim = [{k: v for k, v in r.items() if k not in ("d2_all", "d2_gt", "nis_all", "nis_gt")} for r in res]
    out.write_text(json.dumps(slim, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"写入 {out}  用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
