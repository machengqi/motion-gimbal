"""**三项消融**（开环 `core fixed` 为基准，E2 / E2b × 取景窗 0.5 / 0.33；不调参）：

    uv run python scripts/eval_abl.py --split dev [--workers 6]

1. 排除名单挂位置（现行，`Selector(excl="pos")`）vs 挂轨迹 ID（`excl="id"`）vs 不用名单（`excl="none"`），
   另在跟踪器输出上**人为注入编号切换**（[`inject_id_switches`]），强度 `RATES`（每条轨迹每秒的切换率）。
2. 显著性判据：净位移（现行，`disp="net"`）vs 路程（`disp="path"`，窗口内逐帧位移长度之和）。
3. 控制器：限加速度（现行，`Aim(limit="acc")`）vs 只限速度（`limit="vel"`，同一速度上限、不限加速度）；
   平滑性指标见 [`kinematics`]。

与 `sweep_e2.py` 同一条路径：候选取 `runs/diag_cache/<name>-640.pkl`，取景窗几何用 `eval_e2.Window`；
默认配置 `init` 应逐位复现 `sweep_e2 --grid default`（dev E2 81.05 / 51.57，E2b 69.31 / 50.86）。
跟踪只跑一次（跟踪器不受选择器 / 控制器开关影响），各配置共用同一份 `hist`；注入编号切换时同一强度下
三种排除方式共用同一份重标号结果。
"""

from __future__ import annotations

import lowprio  # noqa: F401,I001  — 必须在 numpy / cv2 之前

import argparse
import json
import pickle
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import load  # noqa: E402
from eval_e2 import CACHE, RUNS, Window  # noqa: E402
from make_split import load_split  # noqa: E402
from sweep_e2 import FRACS, LABELS  # noqa: E402

from gimbal_vision.core import controller, mot, saliency, timing  # noqa: E402
from gimbal_vision.datasets import moca  # noqa: E402

SEED = 20261026
RATES = (0.5, 1.0, 2.0)
"""注入的编号切换率（每条轨迹每秒）。对照：`STATIC_SEC` = 1.47s，静止判据要攒约 0.8 × 1.47 ≈ 1.2s 观测；
λ = 0.5 / 1 / 2 时一个编号活过 1.2s 的概率约 0.55 / 0.31 / 0.09。"""

BASE = {
    "init": {},
    "excl_id": {"sel": {"excl": "id"}},
    "excl_none": {"sel": {"excl": "none"}},
    "disp_path": {"sel": {"disp": "path"}},
    "aim_vel": {"aim": {"limit": "vel"}},
}
KIN_CONFIGS = ("init", "aim_vel", "disp_path", "excl_id", "excl_none")


def configs():
    out = {k: dict(v, rate=0.0) for k, v in BASE.items()}
    for r in RATES:
        for e in ("pos", "id", "none"):
            out[f"sw{r:g}_{e}"] = {"sel": {"excl": e}, "rate": r}
    return out


def inject_id_switches(hist, clk, rate, seed):
    """在跟踪器输出上**随机重分配轨迹 ID**：位置、速度、观测状态都不变，只换标签。

    每帧对每条在场轨迹（按 tid 升序）抽一个均匀数；该轨迹此前出现过且 `u < 1 − exp(−rate · dt)` 时，从这一帧起
    换成一个全新的 ID（之后一直沿用，直到下一次切换）。⇒ 等价于轨迹在该帧被切成两段：下游按 tid 取历史
    （`mot.paths` / 回填 / 净位移 / 散布 / 挂编号的排除名单）都从零开始，而挂位置的排除名单只看坐标。
    跟踪器内部状态（Kalman、关联）不受影响 —— 这里只模拟「编号不稳定」，不模拟关联错误。
    `rate <= 0` 原样返回。返回 `(新 hist, 切换次数)`。
    """
    if rate <= 0:
        return hist, 0
    rng = np.random.default_rng(seed)
    alias: dict[int, int] = {}
    nxt = 1 + max((t for row in hist for t in row[1]), default=0)
    out, n_sw = [], 0
    for k, row, cum in hist:
        p = 1.0 - np.exp(-rate * clk.dt(k))
        new = {}
        for tid in sorted(row):
            u = rng.random()
            if tid not in alias:
                alias[tid] = tid
            elif u < p:
                alias[tid] = nxt
                nxt += 1
                n_sw += 1
            new[alias[tid]] = row[tid]
        out.append((k, new, cum))
    return out, n_sw


def run_selector(hist, clk, roi, cfg, gts):
    """选择 → 瞄准（与 `core.run` 同一段逻辑），附带排除名单的诊断量。"""
    seq = mot.fill_gaps(mot.paths(hist), clk.frames_in(timing.FILL_SEC))
    sel = saliency.Selector(clk, roi, **cfg.get("sel", {}))
    aim = controller.Aim(clk, **cfg.get("aim", {}))
    aims, prev_cum = [], np.zeros(2)
    dg = {"gt_frames": 0, "excl_tf_moving": 0, "excl_tf_gt": 0, "excl_tf_on_target": 0, "excl_frames_on_target": 0,
          "pick_frames": 0, "pick_on_target": 0, "ids": len({t for row in hist for t in row[1]}),
          "obs_track_sec": float(sum(clk.dt(row[0]) for row in hist for v in row[1].values() if v[5] == 0))}
    for k in range(len(hist)):
        cum = hist[k][2]
        bg = cum - prev_cum
        prev_cum = cum
        phase, target, tid, fresh = sel.step(k, hist, seq, bg)
        aims.append(aim.step(k, bg, cum, target, fresh))
        if sel.last_excluded:
            # 被排除、但近 STATIC_SEC 内背景系散布已超过静止半径的轨迹·帧（= 已经动起来了还被压着）
            lo_s, n_s = clk.window(k, timing.STATIC_SEC)
            for t in sel.last_excluded:
                sp = saliency.spread(seq, k, t, lo_s, n_s)
                dg["excl_tf_moving"] += int(sp is not None and sp >= sel.static_r)
        g = gts[k]
        if g is None:
            continue
        dg["gt_frames"] += 1
        row = hist[k][1]

        def on_t(t, g=g, cum=cum, row=row):
            v = row[t]
            x, y = v[0] + cum[0], v[1] + cum[1]
            return g[0] <= x <= g[0] + g[2] and g[1] <= y <= g[1] + g[3]

        ex = [on_t(t) for t in sel.last_excluded]
        dg["excl_tf_gt"] += len(ex)
        dg["excl_tf_on_target"] += int(sum(ex))
        dg["excl_frames_on_target"] += int(any(ex))
        if fresh and tid is not None:
            dg["pick_frames"] += 1
            dg["pick_on_target"] += int(on_t(tid))
    dg.update(sel.stats)
    return aims, dg


def follow(points, W, H, frac, mode, offs):
    """取景窗按瞄准点摆放。返回 `(逐帧矩形, 逐帧中心在当帧素材坐标里的浮点位置 N×2)`。"""
    win, rects, cs = Window(W, H, frac, mode), [], []
    for i, p in enumerate(points):
        if mode == "e2b":
            win.frame(offs[i])
        if p is not None:
            win.set_center(*p)
        rects.append(win.rect())
        cs.append((win.cx - win.o[0], win.cy - win.o[1]))
    return rects, np.array(cs, float)


def deriv(x, t):
    """逐帧差分 ÷ 真实时间戳差，新时间戳取中点。"""
    dt = np.diff(t)
    return np.diff(x, axis=0) / dt[:, None], (t[1:] + t[:-1]) / 2


def kinematics(c, t, i0):
    """位置轨迹 `c`（N×2，px）从第 `i0` 帧起的速度 / 加速度 / 加加速度的模长（px/s、px/s²、px/s³）。"""
    if i0 is None or len(c) - i0 < 4:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    x, tt = c[i0:], t[i0:]
    v, tv = deriv(x, tt)
    a, ta = deriv(v, tv)
    j, _ = deriv(a, ta)
    return np.hypot(*v.T), np.hypot(*a.T), np.hypot(*j.T)


def score(rects, gts, labels):
    n = k = 0
    by: dict = {}
    for i, g in enumerate(gts):
        if g is None:
            continue
        cx, cy = g[0] + g[2] / 2, g[1] + g[3] / 2
        x, y, w, h = rects[i]
        ins = x <= cx <= x + w and y <= cy <= y + h
        n += 1
        k += int(ins)
        a = by.setdefault(LABELS.get(labels[i], "unlabeled"), [0, 0])
        a[0] += 1
        a[1] += int(ins)
    return {"n": n, "in": k, "by": by}


def protocols():
    return [f"e2-{f:g}" for f in FRACS] + [f"e2b-{f:g}" for f in FRACS]


def eval_video(name, cfgs):
    d = load(name)
    W, H, clk, gts, t = d["W"], d["H"], d["clk"], d["gts"], d["ts"]
    n = len(d["ks"])
    roi = (0, 0, W, H)
    offs = pickle.loads((CACHE / f"stab-{name}-640.pkl").read_bytes())["o"]
    labels = [d["motions"][i] for i in range(n)]
    hist0 = mot.track(d["obs"], roi, clock=clk, shifts=d["shifts"])
    seed = (SEED + zlib.crc32(name.encode())) % (2**32)
    injected = {0.0: (hist0, 0)}
    for r in RATES:
        injected[r] = inject_id_switches(hist0, clk, r, seed)
    out = {"name": name, "frames": n, "configs": {}, "switches": {f"{r:g}": injected[r][1] for r in RATES}}
    for cname, cfg in cfgs.items():
        hist, _ = injected[cfg["rate"]]
        aims, dg = run_selector(hist, clk, roi, cfg, gts)
        i0 = next((i for i, p in enumerate(aims) if p is not None), None)
        res, kin = {}, {}
        for f in FRACS:
            for mode in ("e2", "e2b"):
                p = f"{mode}-{f:g}"
                rects, cs = follow(aims, W, H, f, mode, offs)
                res[p] = score(rects, gts, labels)
                if cname in KIN_CONFIGS:
                    kin[p] = kinematics(cs, t, i0)
        if cname in KIN_CONFIGS:
            ap = np.array([(np.nan, np.nan) if p is None else p for p in aims], float)
            kin["aim"] = kinematics(ap, t, i0)  # 瞄准点一旦有值就不再变回 None
            # 扣除背景位移：瞄准点 − 累计背景位移（背景系），只剩控制器自己的运动（`Aim` 的 pos += bg 被扣掉）
            cum = np.array([row[2] for row in hist])
            kin["aim_bg"] = kinematics(ap - cum, t, i0)
        out["configs"][cname] = {"res": res, "diag": dg, "kin": kin, "i0": i0}
    return out


# ── 汇总 ────────────────────────────────────────────────────────
PAIRS = [("excl_id", "init"), ("excl_none", "init"), ("disp_path", "init"), ("aim_vel", "init")]
for _r in RATES:
    PAIRS += [(f"sw{_r:g}_id", f"sw{_r:g}_pos"), (f"sw{_r:g}_none", f"sw{_r:g}_pos"), (f"sw{_r:g}_pos", "init")]


def qs(x):
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return [round(float(v), 1) for v in np.percentile(x, [50, 90, 99])]


def summarize(results, cfgs):
    ps = protocols()
    summ = {}
    for c in cfgs:
        s = {}
        for p in ps:
            nn = sum(r["configs"][c]["res"][p]["n"] for r in results)
            kk = sum(r["configs"][c]["res"][p]["in"] for r in results)
            by: dict = {}
            for r in results:
                for lab, (a, b) in r["configs"][c]["res"][p]["by"].items():
                    x = by.setdefault(lab, [0, 0])
                    x[0] += a
                    x[1] += b
            vids = [r["configs"][c]["res"][p]["in"] / r["configs"][c]["res"][p]["n"] for r in results
                    if r["configs"][c]["res"][p]["n"]]
            s[p] = {"tif": round(100 * kk / max(nn, 1), 2), "frames": nn,
                    "by_motion": {lab: round(100 * b / max(a, 1), 2) for lab, (a, b) in sorted(by.items())},
                    "per_video_mean": round(100 * float(np.mean(vids)), 2)}
        dg: dict = {}
        for r in results:
            for k, v in r["configs"][c]["diag"].items():
                dg[k] = dg.get(k, 0) + v
        s["diag"] = dg
        if c in KIN_CONFIGS:
            kin = {}
            a_max = timing.AIM_ACC_PPS2 * 640 / timing.REF_W
            v_max = timing.AIM_MAX_PPS * 640 / timing.REF_W
            for p in [*ps, "aim", "aim_bg"]:
                v = np.concatenate([r["configs"][c]["kin"][p][0] for r in results])
                a = np.concatenate([r["configs"][c]["kin"][p][1] for r in results])
                j = np.concatenate([r["configs"][c]["kin"][p][2] for r in results])
                a_f, v_f = a[np.isfinite(a)], v[np.isfinite(v)]
                kin[p] = {
                    "speed_pps": qs(v), "acc_pps2": qs(a), "jerk_pps3": qs(j),
                    "n_speed": int(len(v_f)), "n_jerk": int(np.isfinite(j).sum()),
                    "frac_acc_over_limit": round(float(np.mean(a_f > a_max * 1.001)), 4) if len(a_f) else None,
                    "frac_speed_at_limit": round(float(np.mean(v_f > v_max * 0.999)), 4) if len(v_f) else None,
                }
            s["kin"] = kin
        summ[c] = s
    pairs = {}
    for a, b in PAIRS:
        pr = {}
        for p in ps:
            bw = [0, 0]
            for r in results:
                x, y = r["configs"][a]["res"][p], r["configs"][b]["res"][p]
                if not x["n"]:
                    continue
                dl = x["in"] / x["n"] - y["in"] / y["n"]
                bw[0] += int(dl > 0.01)
                bw[1] += int(dl < -0.01)
            pr[p] = {"delta_tif": round(summ[a][p]["tif"] - summ[b][p]["tif"], 2), "better_worse": bw}
        pairs[f"{a}_vs_{b}"] = pr
    return summ, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.workers > 6:
        raise SystemExit("--workers ≤ 6")
    cfgs = configs()
    names = load_split(a.split)
    allseq = moca.load()
    names = sorted(names, key=lambda nm: -len(allseq[nm].frames))
    if a.limit:
        names = names[: a.limit]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=lowprio.init_worker) as ex:
        results = list(ex.map(eval_video, names, [cfgs] * len(names), chunksize=1))
    summ, pairs = summarize(results, cfgs)
    out = {
        "split": a.split, "videos": len(names), "configs": cfgs, "rates_per_sec": list(RATES), "seed": SEED,
        "summary": summ, "pairs": pairs,
        "switches_injected": {f"{r:g}": int(sum(x["switches"][f"{r:g}"] for x in results)) for r in RATES},
        "per_video": {
            r["name"]: {c: {p: round(x["in"] / x["n"], 4) if x["n"] else None for p, x in r["configs"][c]["res"].items()}
                        for c in cfgs}
            for r in results
        },
        "per_video_diag": {r["name"]: {c: r["configs"][c]["diag"] for c in cfgs} for r in results},
        "wall_sec": round(time.time() - t0, 1),
    }
    RUNS.mkdir(exist_ok=True)
    tag = f"-{a.tag}" if a.tag else ""
    f = RUNS / f"abl-{a.split}{tag}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    ps = protocols()
    print(f"{a.split} 段 {len(names)} 用时 {out['wall_sec']}s  注入切换 {out['switches_injected']}")
    print(f"{'配置':16s} " + " ".join(f"{p:>8s}" for p in ps) + "  新建/并入/排除轨迹帧/其中在真值框内")
    for c in cfgs:
        s = summ[c]
        dg = s["diag"]
        print(f"{c:16s} " + " ".join(f"{s[p]['tif']:8.2f}" for p in ps)
              + f"  {dg['mark_new']}/{dg['mark_merge']}/{dg['excl_tf_gt']}/{dg['excl_tf_on_target']}")
    for c in KIN_CONFIGS:
        k = summ[c]["kin"]
        print(f"{c:10s} kin e2-0.5 v{k['e2-0.5']['speed_pps']} a{k['e2-0.5']['acc_pps2']} j{k['e2-0.5']['jerk_pps3']}")
    print(f"写入 {f}")


if __name__ == "__main__":
    main()
