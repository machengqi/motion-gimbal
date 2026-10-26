"""**开环参数扫描**：同一份候选上换常数重跑 跟踪 → 选择 → 瞄准，一次给出 E2 / E2b 两种协议 × 两种窗口的
time-in-frame。只用于灵敏度与重标定（docs/EXPERIMENTS.md）；⚠️ 选值只在 dev 上做，test 只报告。

    uv run python scripts/sweep_e2.py --split dev --grid lead [--workers 8]
    uv run python scripts/sweep_e2.py --split dev --grid screen
    uv run python scripts/sweep_e2.py --split dev --grid scan:WINDOW_SEC
    uv run python scripts/sweep_e2.py --split test --grid final --final-json <dev 选出的配置.json>

与 `eval_e2.py` 的 `core:fixed` 是同一条路径：候选取 `runs/diag_cache/<name>-640.pkl`（与 `eval_e2` 的开环
候选同一份代码生成），`core.run` 的四步在这里展开（为了读出选择器的诊断量），取景窗几何直接用
`eval_e2.Window`。默认配置在 dev 上应逐位复现 `eval_e2 --methods core:fixed`（脚本输出里核对）。

# 配置

一个配置 = 若干覆盖项：

- `timing`：`core.timing` 的模块常数（运行时读，进程内临时改写、跑完恢复）。`STATIC_SEC` 与 `STALE_SEC`
  绑定（`timing.STALE_SEC` 的文档：同一个时间尺度，不单独调）。
- `sal`：`core.saliency` 的模块常数（`LEAD`、`HOLD`、`STATIC_R`、`BAN_R`、`DRIFT`）。
- `params`：`candidates.motion.Params` 的字段 —— 候选要重算（逐帧流式读图，结果缓存）。
- `warps`：`"lk"` = 把被测方法自己的逐帧单应交给 `Selector(warps=...)`（排除图用单应传播），
  `warp_disp` 同 `Selector`。

# 协议

`e2-0.5`、`e2-0.33`（素材画面），`e2b-0.5`、`e2b-0.33`（平移画布，= `eval_e2 --protocol e2b`），以及
`--canvas-h` 时的 `e2bh-0.5`、`e2bh-0.33`：画布换成 SIFT **单应**链 —— 无瞄准点时取景窗中心按
SIFT 逐帧单应传播（世界里的同一点），其余规则同 E2b（中心钳在当帧素材内）。平移画布下这条规则与
`Window(mode="e2b")` 等价（脚本自检）。固定点基线（`center`）同样在每种协议下给出。
"""

from __future__ import annotations

import lowprio  # noqa: F401,I001  — 必须在 numpy / cv2 之前

import argparse
import contextlib
import hashlib
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import load  # noqa: E402
from eval_e2 import CACHE, RUNS, Window  # noqa: E402
from make_split import load_split  # noqa: E402

from gimbal_vision.candidates.motion import MotionCandidates, Params  # noqa: E402
from gimbal_vision.core import controller, mot, saliency, timing  # noqa: E402
from gimbal_vision.datasets import moca  # noqa: E402

FRACS = (0.5, 0.33)
LABELS = {0: "locomotion", 1: "subtle_motion", 2: "still", None: "unlabeled"}

TIMING_KEYS = (
    "WINDOW_SEC", "STATIC_SEC", "CONFIRM_SEC", "FILL_SEC", "MISS_SEC", "STITCH_SEC", "VEL_SEC",
    "JUMP_PPS", "CONFIRM_R_PPS", "GATE_PPS", "STITCH_GATE_PPS", "AIM_MAX_PPS", "AIM_ACC_PPS2",
    "DECAY_PER_SEC", "GAIN_KEEP",
)
"""表4-2 的常数里会影响开环结果的那些。`CUT_PPS` 不在内：外部给了背景位移时镜头切换判据不执行。"""

SAL_KEYS = ("LEAD", "HOLD", "STATIC_R", "BAN_R")
PARAM_KEYS = ("max_pts", "quality", "min_dist", "fb_err", "ransac_px", "resid_pps", "cluster_r", "min_pts", "pad")


# ── 覆盖常数 ────────────────────────────────────────────────────
@contextlib.contextmanager
def patched(cfg):
    saved = []
    tim = dict(cfg.get("timing", {}))
    if "STATIC_SEC" in tim:
        tim["STALE_SEC"] = tim["STATIC_SEC"]
    for mod, over in ((timing, tim), (saliency, cfg.get("sal", {}))):
        for k, v in over.items():
            if not hasattr(mod, k):
                raise KeyError(k)
            saved.append((mod, k, getattr(mod, k)))
            setattr(mod, k, v)
    try:
        yield
    finally:
        for mod, k, v in reversed(saved):
            setattr(mod, k, v)


def params_key(over):
    return hashlib.md5(json.dumps(over, sort_keys=True).encode()).hexdigest()[:10]


def candidates_with(name, over):
    """按 `Params` 覆盖重算全幅候选（逐帧流式读图，与 `eval_e2.load_sequence` 同一预处理）。缓存。"""
    f = CACHE / f"obs-{name}-640-{params_key(over)}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    seq = moca.load()[name]
    dup = moca.duplicates(seq)
    gen = MotionCandidates(640, Params(**over))
    obs, shifts, Hs = [], [], []
    for k, path in enumerate(seq.frames):
        if dup[k]:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        s = 640 / img.shape[1]
        if s != 1.0:
            img = cv2.resize(img, (640, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        r = gen(img, k / moca.FPS)
        obs.append(r.dets)
        Hs.append(r.homography)
        shifts.append(np.zeros(2) if r.bg_shift is None else np.asarray(r.bg_shift, float))
    d = {"obs": obs, "shifts": shifts, "Hs": Hs, "params": over}
    f.write_bytes(pickle.dumps(d))
    return d


# ── 跑一个配置（= core.run 展开）──────────────────────────────
def run_config(d, obs, shifts, Hs, cfg, gts):
    clk = d["clk"]
    W, H = d["W"], d["H"]
    roi = (0, 0, W, H)
    sel_kw = dict(cfg.get("sel", {}))
    if cfg.get("warps") == "lk":
        sel_kw["warps"] = Hs
        sel_kw["warp_disp"] = bool(cfg.get("warp_disp", False))
    with patched(cfg):
        hist = mot.track(obs, roi, clock=clk, gate=cfg.get("gate", "uncertainty"), shifts=shifts,
                         meas_var=cfg.get("meas_var"))
        seq = mot.fill_gaps(mot.paths(hist), clk.frames_in(timing.FILL_SEC))
        sel, aim = saliency.Selector(clk, roi, **sel_kw), controller.Aim(clk)
        aims, prev_cum = [], np.zeros(2)
        diag = {"gt_frames": 0, "excl_on_target": 0, "cand_on_target": 0, "pick_on_target": 0, "pick_frames": 0}
        for k in range(len(obs)):
            cum = hist[k][2]
            bg = cum - prev_cum
            prev_cum = cum
            phase, target, tid, fresh = sel.step(k, hist, seq, bg)
            aims.append(aim.step(k, bg, cum, target, fresh))
            g = gts[k]
            if g is None:
                continue
            diag["gt_frames"] += 1

            def on_t(t, g=g, cum=cum, row=hist[k][1]):
                v = row[t]
                x, y = v[0] + cum[0], v[1] + cum[1]
                return g[0] <= x <= g[0] + g[2] and g[1] <= y <= g[1] + g[3]

            diag["excl_on_target"] += int(any(on_t(t) for t in sel.last_excluded))
            diag["cand_on_target"] += int(any(on_t(t) for t, v in hist[k][1].items() if v[5] == 0))
            if fresh and tid is not None:
                diag["pick_frames"] += 1
                diag["pick_on_target"] += int(on_t(tid))
        diag.update(sel.stats)
    return aims, diag


def follow_canvas(points, offs, W, H, frac):
    win, out = Window(W, H, frac, "e2b"), []
    for i, p in enumerate(points):
        win.frame(offs[i])
        if p is not None:
            win.set_center(*p)
        out.append(win.rect())
    return out


def follow_warp(points, warps, W, H, frac):
    """取景窗中心在**当帧画面**里传播：无瞄准点时按 `warps[k]`（k−1 → k）搬到下一帧，钳在素材内。"""
    w, h = max(1, int(W * frac)), max(1, int(H * frac))
    f = np.array([W / 2, H / 2])
    out = []
    for i, p in enumerate(points):
        if i > 0 and warps[i] is not None:
            q = np.asarray(warps[i], float) @ np.array([f[0], f[1], 1.0])
            f = q[:2] / q[2]
        f = np.clip(f, (0, 0), (W, H))
        if p is not None:
            f = np.clip(np.array(p, float), (0, 0), (W, H))
        out.append((int(round(f[0] - w / 2)), int(round(f[1] - h / 2)), w, h))
    return out


def follow_e2(points, W, H, frac):
    win, out = Window(W, H, frac, "e2"), []
    for p in points:
        if p is not None:
            win.set_center(*p)
        out.append(win.rect())
    return out


def protocols(canvas_h):
    ps = [f"e2-{f:g}" for f in FRACS] + [f"e2b-{f:g}" for f in FRACS]
    if canvas_h:
        ps += [f"e2bh-{f:g}" for f in FRACS]
    return ps


def eval_video(name, configs, canvas_h):
    d = load(name)
    W, H = d["W"], d["H"]
    n = len(d["ks"])
    gts = d["gts"]
    offs = pickle.loads((CACHE / f"stab-{name}-640.pkl").read_bytes())["o"]
    sift_H = pickle.loads((CACHE / f"stabHS-{name}-640.pkl").read_bytes())["H"] if canvas_h else None
    labels = [d["motions"][i] for i in range(n)]
    idx = [i for i in range(n) if gts[i] is not None]
    gc = {i: (gts[i][0] + gts[i][2] / 2, gts[i][1] + gts[i][3] / 2) for i in idx}

    def score(rects):
        ins = [rects[i][0] <= gc[i][0] <= rects[i][0] + rects[i][2] and rects[i][1] <= gc[i][1] <= rects[i][1] + rects[i][3]
               for i in idx]
        by = {}
        for i, v in zip(idx, ins, strict=True):
            lab = LABELS.get(labels[i], "unlabeled")
            a = by.setdefault(lab, [0, 0])
            a[0] += 1
            a[1] += int(v)
        return {"n": len(idx), "in": int(sum(ins)), "by": by}

    def all_protocols(points):
        res = {}
        for f in FRACS:
            res[f"e2-{f:g}"] = score(follow_e2(points, W, H, f))
            res[f"e2b-{f:g}"] = score(follow_canvas(points, offs, W, H, f))
            if canvas_h:
                res[f"e2bh-{f:g}"] = score(follow_warp(points, sift_H, W, H, f))
        return res

    out = {"name": name, "configs": {}}
    out["center"] = all_protocols([None] * n)
    # 自检：平移画布下「中心按逐帧平移传播」≡ Window(mode="e2b")
    t = [None] + [np.array([[1, 0, offs[i - 1][0] - offs[i][0]], [0, 1, offs[i - 1][1] - offs[i][1]], [0, 0, 1.0]])
                  for i in range(1, n)]
    a = follow_canvas([None] * n, offs, W, H, 0.5)
    b = follow_warp([None] * n, t, W, H, 0.5)
    out["warp_selfcheck_mismatch"] = int(sum(x != y for x, y in zip(a, b, strict=True)))
    for cname, cfg in configs.items():
        if cfg.get("params"):
            c = candidates_with(name, cfg["params"])
            obs, shifts, Hs = c["obs"], c["shifts"], c["Hs"]
        else:
            obs, shifts, Hs = d["obs"], d["shifts"], d["Hs"]
        aims, diag = run_config(d, obs, shifts, Hs, cfg, gts)
        out["configs"][cname] = {"res": all_protocols(aims), "diag": diag}
    return out


# ── 配置表 ──────────────────────────────────────────────────────
def init_values():
    v = {k: getattr(timing, k) for k in TIMING_KEYS}
    v.update({k: getattr(saliency, k) for k in SAL_KEYS})
    v.update({k: getattr(Params(), k) for k in PARAM_KEYS})
    return v


def override(key, val):
    if key in TIMING_KEYS:
        return {"timing": {key: val}}
    if key in SAL_KEYS:
        return {"sal": {key: val}}
    if key in PARAM_KEYS:
        return {"params": {key: val}}
    raise KeyError(key)


def scaled(key, factor):
    """常数按自然参数化缩放：衰减 / 增益按「每参考帧」的比例，整数参数取整。"""
    v = init_values()[key]
    if key == "DECAY_PER_SEC":  # 每参考帧保留 0.88 ⇒ 缩放「每参考帧损失」1 − 0.88
        per = 1 - (1 - v ** (1 / timing.REF_FPS)) * factor
        return float(np.clip(per, 0.01, 0.999)) ** timing.REF_FPS
    if key == "GAIN_KEEP":  # 每参考帧消掉 0.55 ⇒ 缩放「消掉的比例」
        return float(np.clip(1 - (1 - v) * factor, 0.01, 0.99))
    if key in ("max_pts", "min_pts"):
        return int(max(1, round(v * factor)))
    return v * factor


def build(grid, final_json=None):
    if grid == "default":
        return {"init": {}}
    if grid == "lead":
        out = {}
        for lead in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0):
            for hold in (1.0, 1.1, 1.25, 1.5, 2.0):
                out[f"LEAD={lead:g},HOLD={hold:g}"] = {"sal": {"LEAD": lead, "HOLD": hold}}
        return out
    if grid in ("screen", "screen_t", "screen_p"):
        keys = []
        if grid in ("screen", "screen_t"):
            keys += [k for k in TIMING_KEYS] + ["STATIC_R", "BAN_R"]
        if grid in ("screen", "screen_p"):
            keys += list(PARAM_KEYS)
        out = {"init": {}}
        for k in keys:
            for fct in (0.5, 2.0):
                out[f"{k}x{fct:g}"] = override(k, scaled(k, fct))
        return out
    if grid.startswith("scan:"):
        k = grid.split(":", 1)[1]
        out = {"init": {}}
        for fct in (0.35, 0.5, 0.71, 0.85, 1.18, 1.41, 2.0, 2.83):
            out[f"{k}x{fct:g}"] = override(k, scaled(k, fct))
        return out
    if grid.startswith("pscan:"):  # 候选参数的一维扫描（每个值都要重算候选 ⇒ 点更稀）
        k = grid.split(":", 1)[1]
        out = {"init": {}}
        for fct in (0.5, 0.71, 1.41, 2.0, 2.83):
            out[f"{k}x{fct:g}"] = override(k, scaled(k, fct))
        return out
    if grid == "bgmodel":
        return {
            "init": {},
            "excl_H": {"warps": "lk"},
            "excl_H+disp_H": {"warps": "lk", "warp_disp": True},
            "no_exclusion": {"sal": {"BAN_R": 0.0, "STATIC_R": 0.0}},
        }
    if grid == "final":
        cfgs = json.loads(Path(final_json).read_text(encoding="utf-8"))
        return {"init": {}, **cfgs}
    if grid.startswith("json:"):
        return json.loads(Path(grid[5:]).read_text(encoding="utf-8"))
    raise SystemExit(f"未知 grid {grid}")


def merge(base, cfg):
    """`base` 与 `cfg` 按小节合并（`cfg` 优先）。"""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in cfg.items():
        out[k] = {**out.get(k, {}), **v} if isinstance(v, dict) else v
    return out


# ── 汇总 ────────────────────────────────────────────────────────
def summarize(results, configs, canvas_h):
    ps = protocols(canvas_h)

    def agg(get):
        out = {}
        for p in ps:
            n = sum(get(r)[p]["n"] for r in results)
            k = sum(get(r)[p]["in"] for r in results)
            by = {}
            for r in results:
                for lab, (nn, kk) in get(r)[p]["by"].items():
                    a = by.setdefault(lab, [0, 0])
                    a[0] += nn
                    a[1] += kk
            vids = [get(r)[p]["in"] / get(r)[p]["n"] for r in results if get(r)[p]["n"]]
            out[p] = {
                "tif": round(100 * k / max(n, 1), 2),
                "frames": n,
                "by_motion": {lab: round(100 * kk / max(nn, 1), 2) for lab, (nn, kk) in sorted(by.items())},
                "per_video_mean": round(100 * float(np.mean(vids)), 2),
            }
        return out

    summ = {"center": agg(lambda r: r["center"])}
    for c in configs:
        s = agg(lambda r, c=c: r["configs"][c]["res"])
        # 逐段对 init 的优 / 劣（> 1 个百分点）
        if "init" in configs and c != "init":
            for p in ps:
                bw = [0, 0]
                for r in results:
                    a, b = r["configs"][c]["res"][p], r["configs"]["init"]["res"][p]
                    if not a["n"]:
                        continue
                    dlt = a["in"] / a["n"] - b["in"] / b["n"]
                    bw[0] += dlt > 0.01
                    bw[1] += dlt < -0.01
                s[p]["vs_init_better_worse"] = bw
        dg = {}
        for r in results:
            for k, v in r["configs"][c]["diag"].items():
                dg[k] = dg.get(k, 0) + v
        s["diag"] = dg
        s["objective_e2_mean"] = round((s["e2-0.5"]["tif"] + s["e2-0.33"]["tif"]) / 2, 2)
        summ[c] = s
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test", "all"))
    ap.add_argument("--grid", default="default")
    ap.add_argument("--final-json", default=None)
    ap.add_argument("--canvas-h", action="store_true")
    ap.add_argument("--base-json", default=None, help="合并进每个配置的覆盖项（坐标轮换时用：已采纳的值）；`init` 即「当前基准」")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    configs = build(a.grid, a.final_json)
    base = json.loads(Path(a.base_json).read_text(encoding="utf-8")) if a.base_json else {}
    if base:
        configs = {k: merge(base, v) for k, v in configs.items()}
    names = load_split(a.split)
    # 长视频先跑，负载均衡
    allseq = moca.load()
    names = sorted(names, key=lambda nm: -len(allseq[nm].frames))
    t0 = time.time()
    n = len(names)
    with ProcessPoolExecutor(a.workers, initializer=lowprio.init_worker) as ex:
        results = list(ex.map(eval_video, names, [configs] * n, [a.canvas_h] * n, chunksize=1))
    summ = summarize(results, configs, a.canvas_h)
    out = {
        "split": a.split,
        "grid": a.grid,
        "videos": n,
        "configs": configs,
        "base": base,
        "init_values": {k: (float(v) if not isinstance(v, int) else v) for k, v in init_values().items()},
        "warp_selfcheck_mismatch_frames": int(sum(r["warp_selfcheck_mismatch"] for r in results)),
        "summary": summ,
        "per_video": {
            r["name"]: {c: {p: round(x["in"] / x["n"], 4) if x["n"] else None for p, x in r["configs"][c]["res"].items()}
                        for c in configs} | {"center": {p: round(x["in"] / x["n"], 4) if x["n"] else None
                                                         for p, x in r["center"].items()}}
            for r in results
        },
        "wall_sec": round(time.time() - t0, 1),
    }
    RUNS.mkdir(exist_ok=True)
    tag = f"-{a.tag}" if a.tag else ""
    safe = a.grid.replace(":", "-").replace("/", "_").replace("\\", "_")[:40]
    f = RUNS / f"sweep-{a.split}-{safe}{tag}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    ps = protocols(a.canvas_h)
    print(f"{a.split} grid={a.grid} 段 {n} 用时 {out['wall_sec']}s  自检不一致帧 {out['warp_selfcheck_mismatch_frames']}")
    print(f"{'配置':28s} " + " ".join(f"{p:>9s}" for p in ps) + "   E2均值")
    for c in ["center", *configs]:
        s = summ[c]
        print(f"{c:28s} " + " ".join(f"{s[p]['tif']:9.2f}" for p in ps) + (f"   {s['objective_e2_mean']:.2f}" if c != "center" else ""))
    print(f"写入 {f}")


if __name__ == "__main__":
    main()
