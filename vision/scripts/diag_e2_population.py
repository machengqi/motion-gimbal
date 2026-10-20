"""任务 1 的全集统计 + 反事实对照（只在脚本里包一层，**不改核心代码**）。

    uv run python scripts/diag_e2_population.py [--workers 6] [--limit N]

对每段视频（开环 `core` 口径，uncertainty 门控）：
- 首次锁定（选择器第一次给出 pos）的帧号、是否落在真值框内、当时轨迹有几次观测；
- 选择器 pos 落在真值框内 / 落在画面外的帧占比；
- 「画面系钉死」帧：选择器当前对象在画面系里几乎不动、而背景在动（台标 / 水印 / 黑边的签名）；
- time-in-frame（取景窗几何与 eval_e2 同一个 `Window`），先与 E2 的 `core` 数逐段对账。

反事实（包装 `saliency.Selector`，不复制其逻辑）：
- `delay`：在时间窗 `WINDOW_SEC` 填满之前不允许**首次**锁定（step 之后把 pos 撤回 None）；
- `screenfix`：喂给选择器的 `hist[k]` 里去掉「画面系散布 < STATIC_R、而背景系净位移大」的轨迹
  （排除名单在画面系的镜像）；
- `both`：两者同时；
- `stale`：pos 连续 `STATIC_SEC` 秒没有新证据（fresh）就撤回成 None（下一个候选直接接管，不走远跳印证）；
- `stale+sf`：`stale` + `screenfix`。
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


def in_box(p, g):
    return p is not None and g is not None and g[0] <= p[0] <= g[0] + g[2] and g[1] <= p[1] <= g[1] + g[3]


def screen_fixed(seq, hist, k, tid, lo, need, r):
    """画面系里离自身重心最远 < r，且背景系净位移 > 2r ⇒ 画面系钉死。"""
    pts = [p for p in seq.get(tid, []) if lo <= p[0] <= k]
    if len(pts) < need * 0.8 or pts[-1][0] < k - 1:
        return False
    b = np.array([[x, y] for _, x, y in pts])
    f = b + np.array([hist[kk][2] for kk, _, _ in pts])
    spread = float(np.max(np.hypot(*(f - f.mean(0)).T)))
    return spread < r and float(np.hypot(*(b[-1] - b[0]))) > 2 * r


def make_selector(delay: bool, screenfix: bool, stale: bool = False):
    class S(saliency.Selector):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.last_fresh = 0

        def step(self, k, hist, seq, bg):
            if stale and self.pos is not None and self.clk.t[k] - self.clk.t[self.last_fresh] > timing.STATIC_SEC:
                self.pos, self.cur, self.chal, self.streak = None, None, None, 0
            h = hist
            if screenfix:
                lo, need = self.clk.window(k, timing.STATIC_SEC)
                row = hist[k]
                keep = {t: v for t, v in row[1].items() if not screen_fixed(seq, hist, k, t, lo, need, self.static_r)}
                h = list(hist)
                h[k] = (row[0], keep, row[2])
            was_none = self.pos is None
            out = super().step(k, h, seq, bg)
            if out[3]:
                self.last_fresh = k
            if delay and was_none and out[1] is not None and k < self.clk.frames_in(timing.WINDOW_SEC):
                self.pos, self.cur, self.chal, self.streak = None, None, None, 0
                return "wait", None, None, False
            return out

    return S


def follow(points, W, H, frac):
    win, out = Window(W, H, frac), []
    for p in points:
        if p is not None:
            win.set_center(*p)
        out.append(win.rect())
    return out


def run_variant(d, hist, seq, cls, frac):
    clk, n, W, H = d["clk"], len(d["obs"]), d["W"], d["H"]
    sel, aim = cls(clk, (0, 0, W, H), **LEGACY_SEL), controller.Aim(clk)
    prev = np.zeros(2)
    out = []
    for k in range(n):
        cum = hist[k][2]
        bg = cum - prev
        prev = cum
        ph, tgt, tid, fresh = sel.step(k, hist, seq, bg)
        out.append((ph, tgt, tid, aim.step(k, bg, cum, tgt, fresh)))
    return out


def analyse(name, frac=0.5):
    d = load(name)
    clk, gts = d["clk"], d["gts"]
    n, W, H = len(d["obs"]), d["W"], d["H"]
    hist = mot.track(d["obs"], (0, 0, W, H), clock=clk, gate="uncertainty", shifts=d["shifts"], meas_var=LEGACY_R)
    seq = mot.fill_gaps(mot.paths(hist), clk.frames_in(timing.FILL_SEC))
    res = {"name": name, "n": n}
    for key, cls in (
        ("core", saliency.Selector),
        ("delay", make_selector(True, False)),
        ("screenfix", make_selector(False, True)),
        ("both", make_selector(True, True)),
        ("stale", make_selector(False, False, True)),
        ("stale+sf", make_selector(False, True, True)),
    ):
        rows = run_variant(d, hist, seq, cls, frac)
        rects = follow([r[3] for r in rows], W, H, frac)
        gi = [i for i in range(n) if gts[i] is not None]
        tif = []
        for i in gi:
            g = gts[i]
            x, y, w, h = rects[i]
            cx, cy = g[0] + g[2] / 2, g[1] + g[3] / 2
            tif.append(x <= cx <= x + w and y <= cy <= y + h)
        rec = {"in_frame": float(np.mean(tif)) if tif else None, "gt_frames": len(gi)}
        if key == "core":
            first = next((k for k in range(n) if rows[k][1] is not None), None)
            rec["first_k"] = first
            rec["first_on_gt"] = None if first is None else bool(in_box(rows[first][1], gts[first]))
            if first is not None:
                v = hist[first][1][rows[first][2]]
                rec["first_hits"] = int(v[4])
            sel_in = [in_box(rows[i][1], gts[i]) for i in gi if rows[i][1] is not None]
            rec["sel_on_gt"] = float(np.mean(sel_in)) if sel_in else None
            rec["sel_offscreen"] = float(np.mean([
                not (0 <= rows[i][1][0] <= W and 0 <= rows[i][1][1] <= H) for i in range(n) if rows[i][1] is not None
            ])) if any(r[1] is not None for r in rows) else None
            # 画面系钉死：当前对象（track 阶段）是否满足 screen_fixed
            sf = 0
            for k in range(n):
                if rows[k][0] == "track" and rows[k][2] is not None:
                    lo, need = clk.window(k, timing.STATIC_SEC)
                    sf += screen_fixed(seq, hist, k, rows[k][2], lo, need, saliency.STATIC_R * clk.s)
            rec["track_screen_fixed"] = sf
            rec["track_frames"] = sum(r[0] == "track" for r in rows)
        res[key] = rec
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--e2", default=str(RUNS / "e2-frac0.5-20261014-103010.json"))
    a = ap.parse_args()
    names = sorted(moca.load())
    if a.limit:
        names = names[: a.limit]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(analyse, names))
    vids = [r for r in res if r["core"]["gt_frames"]]
    # 与 E2 对账
    e2 = {v["name"]: v for v in json.loads(Path(a.e2).read_text(encoding="utf-8"))["videos"]}
    bad = [r["name"] for r in vids if abs(r["core"]["in_frame"] - e2[r["name"]]["in_frame"]["core"]) > 1e-9]
    print(f"对账：{len(vids)} 段，与 E2 core 不一致 {len(bad)} 段 {bad[:5]}")

    def frame_weighted(key):
        num = sum(r[key]["in_frame"] * r[key]["gt_frames"] for r in vids)
        return 100 * num / sum(r[key]["gt_frames"] for r in vids)

    for key in ("core", "delay", "screenfix", "both", "stale", "stale+sf"):
        per = [r[key]["in_frame"] for r in vids]
        print(f"{key:10s} 全帧 {frame_weighted(key):5.1f}%  逐段均值 {100 * np.mean(per):5.1f}%")
    on = [r for r in vids if r["core"]["first_on_gt"]]
    off = [r for r in vids if r["core"]["first_on_gt"] is False]
    print(f"首次锁定在目标上 {len(on)} 段（逐段 in-frame 均值 {100 * np.mean([r['core']['in_frame'] for r in on]):.1f}%），"
          f"不在目标上 {len(off)} 段（{100 * np.mean([r['core']['in_frame'] for r in off]):.1f}%），从未锁定 "
          f"{sum(r['core']['first_k'] is None for r in vids)} 段")
    fk = [r["core"]["first_k"] for r in vids if r["core"]["first_k"] is not None]
    print(f"首次锁定帧号分布: {dict(zip(*np.unique(fk, return_counts=True)))}")
    fh = [r["core"]["first_hits"] for r in vids if r["core"]["first_k"] is not None]
    print(f"首次锁定时该轨迹 hits 分布: {dict(zip(*np.unique(fh, return_counts=True)))}")
    sf = [(r["name"], r["core"]["track_screen_fixed"], r["core"]["track_frames"]) for r in vids if r["core"]["track_screen_fixed"] > 10]
    print(f"track 阶段对象是「画面系钉死」>10 帧的视频 {len(sf)} 段: {sorted(sf, key=lambda x: -x[1])[:15]}")
    tot_sf = sum(r["core"]["track_screen_fixed"] for r in vids)
    tot_tr = sum(r["core"]["track_frames"] for r in vids)
    print(f"全集 track 帧 {tot_tr}，其中对象画面系钉死 {tot_sf}（{100 * tot_sf / tot_tr:.1f}%）")
    off_scr = [r["core"]["sel_offscreen"] for r in vids if r["core"]["sel_offscreen"] is not None]
    print(f"选择器 pos 在画面外的帧占比：逐段均值 {100 * np.mean(off_scr):.1f}%，>50% 的视频 {sum(x > 0.5 for x in off_scr)} 段")
    # 逐段配对
    for key in ("delay", "screenfix", "both", "stale", "stale+sf"):
        dlt = [r[key]["in_frame"] - r["core"]["in_frame"] for r in vids]
        print(f"{key} 对 core：优 {sum(x > 0.01 for x in dlt)} / 劣 {sum(x < -0.01 for x in dlt)} 段")
    for nm in ("cuttlefish_3", "flatfish_2", "black_cat_1", "peacock_flounder_1"):
        r = next(x for x in vids if x["name"] == nm)
        print(nm, {k: round(100 * r[k]["in_frame"], 1) for k in ("core", "delay", "screenfix", "both", "stale", "stale+sf")},
              "首锁", r["core"]["first_k"], r["core"]["first_on_gt"])
    out = RUNS / f"diag-e2-population-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {out}  用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
