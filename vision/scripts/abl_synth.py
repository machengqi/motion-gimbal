"""**合成对照**（⛔ 合成数据，只用于说明机制，不作为 MoCA 结论）：排除名单挂位置 / 挂编号 / 不用名单，
在注入编号切换下的差别。

    uv run python scripts/abl_synth.py [--seeds 20]

场景见 [`scene`]：1041×684 画面，相机缓慢平移（0.3 px/帧），2 个同外观的静止干扰物（背景系固定点，整段留在
画面内），目标在画面里绕椭圆（220×130px）⇒ 背景系里持续运动；目标每 3 秒被遮挡 1.2 秒（删掉目标的检测）。
遮挡期间合格候选只剩静止干扰物、且不足 3 个 ⇒ 置信闸不生效，没有排除名单就会锁到干扰物上。
（先试过 `synthetic.make_scene` 的稠密网格：贴边截断让刚进画面的静止物体看起来在动，且新进画面、尚未攒够
静止判据的干扰物在遮挡期成了唯一候选 —— 这两个效应掩盖了要看的机制，所以换成这个最小场景。）编号切换按 `eval_abl.inject_id_switches` 注入，强度与 MoCA 实验相同。

另一组（`sway`）：`synthetic.make_scene` 的默认场景（每 5 个干扰物有 1 个绕固定点往复摆动，振幅 15px、1 Hz；
`sway_amp` 另试 30px）上比较显著性判据净位移 vs 路程（其余同默认）。

指标：瞄准点落在以目标真值为中心、0.33 倍画面宽高的框内的帧占比（目标遮挡帧也算，真值位置已知）；
瞄准点离某个干扰物比离目标近、且离该干扰物 < `LOCK_R` 的帧占比（「锁在干扰物上」）。
"""

from __future__ import annotations

import lowprio  # noqa: F401,I001

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_abl import RATES, SEED, inject_id_switches  # noqa: E402
from eval_e2 import RUNS  # noqa: E402

from gimbal_vision.core import controller, mot, saliency, timing  # noqa: E402

N, FPS = 450, 30.0
N_DIST = 2
LOCK_R = 130.0
OCC_EVERY, OCC_LEN = 3.0, 1.2
FRAC = 0.33


def scene(seed):
    """自造的检测级场景（不用 `make_scene`：它的贴边截断会让刚进画面的静止物体看起来在动，掩盖要看的机制）。

    `N_DIST` 个背景系固定点上的静止干扰物，相机匀速平移；目标在画面里绕椭圆（背景系里持续运动）。检测只在物体**完整落在画面内**时给出，框心加 1.5px 高斯噪声，5% 随机漏检；遮挡期间删掉目标的检测。
    返回 `(scene, obs, occ, dist)`，`dist[k]` 是第 k 帧画面系里的干扰物位置。
    """
    rng = np.random.default_rng(seed)
    W, H, size = 1041.0, 684.0, 60.0
    ts = np.arange(N) / FPS
    pan = np.array([-0.3, 0.0])
    cum = np.outer(np.arange(N), pan)
    # 少量静止干扰物（背景系固定点），整段都留在画面内 ⇒ 合格候选 < 3 个，置信闸不生效
    grid = np.column_stack([rng.uniform(350, 950, N_DIST), rng.uniform(120, 560, N_DIST)])
    ph = rng.uniform(0, 2 * np.pi)
    target = np.array([[W / 2 + 220 * np.cos(0.8 * t + ph), H / 2 + 130 * np.sin(1.1 * t + ph)] for t in ts])
    occ = (ts % OCC_EVERY) >= (OCC_EVERY - OCC_LEN)
    half = size / 2
    obs, dist = [], []
    for k in range(N):
        g = grid + cum[k]
        inside = (g[:, 0] > half) & (g[:, 0] < W - half) & (g[:, 1] > half) & (g[:, 1] < H - half)
        g = g[inside]
        # 与目标重叠的干扰物不出检测（同外观物体不互相穿过）
        g = g[np.hypot(*(g - target[k]).T) > size]
        dist.append(g)
        pts = list(g) + ([] if occ[k] else [target[k]])
        dets = []
        for p in pts:
            if rng.random() < 0.05:
                continue
            cx, cy = p[0] + rng.normal(0, 1.5), p[1] + rng.normal(0, 1.5)
            dets.append((cx, cy, size * size, cx - half, cy - half, cx + half, cy + half, 0.9))
        obs.append(dets)

    class S:
        pass

    sc = S()
    sc.ts, sc.target, sc.cum, sc.roi = ts, target, cum, (0, 0, int(W), int(H))
    return sc, obs, occ, dist


def run(sc, obs, rate, excl, seed):
    clk = timing.Clock(len(obs), ts=sc.ts, roi_w=sc.roi[2])
    shifts = [np.zeros(2)] + list(np.diff(sc.cum, axis=0))
    hist = mot.track(obs, sc.roi, clock=clk, shifts=shifts)
    hist, nsw = inject_id_switches(hist, clk, rate, seed)
    seq = mot.fill_gaps(mot.paths(hist), clk.frames_in(timing.FILL_SEC))
    sel, aim = saliency.Selector(clk, sc.roi, excl=excl), controller.Aim(clk)
    out, prev = [], np.zeros(2)
    st = {"excl_on_target": 0}
    for k in range(len(obs)):
        cum = hist[k][2]
        bg = cum - prev
        prev = cum
        _, target, _, fresh = sel.step(k, hist, seq, bg)
        out.append(aim.step(k, bg, cum, target, fresh))
        for t in sel.last_excluded:  # 被排除的轨迹落在目标真值 30px 内（= 压住了目标本身）
            v = hist[k][1][t]
            st["excl_on_target"] += int(np.hypot(v[0] + cum[0] - sc.target[k][0], v[1] + cum[1] - sc.target[k][1]) < 30)
    return out, nsw, {**sel.stats, **st}


def metrics(sc, dist, occ, aims):
    W, H = sc.roi[2], sc.roi[3]
    hw, hh = W * FRAC / 2, H * FRAC / 2
    inw, on_d, occ_in = [], [], []
    for k, p in enumerate(aims):
        t = sc.target[k]
        if p is None:
            inw.append(False)
            on_d.append(False)
            if occ[k]:
                occ_in.append(False)
            continue
        ok = abs(p[0] - t[0]) <= hw and abs(p[1] - t[1]) <= hh
        inw.append(ok)
        if occ[k]:
            occ_in.append(ok)
        ds = [np.hypot(d[0] - p[0], d[1] - p[1]) for d in dist[k]]
        dt = np.hypot(p[0] - t[0], p[1] - t[1])
        on_d.append(bool(ds) and min(ds) < dt and min(ds) < LOCK_R)
    return float(np.mean(inw)), float(np.mean(on_d)), float(np.mean(occ_in)) if occ_in else None


def sway(seeds):
    """净位移 vs 路程：`make_scene` 默认场景（含摆动干扰物）上的窗内占比与「锁在摆动干扰物上」占比。"""
    from gimbal_vision.synthetic import make_scene  # noqa: PLC0415

    out = {}
    for amp in (15.0, 30.0):
        for disp in ("net", "path"):
            ins = []
            for s in range(seeds):
                sc = make_scene(n=240, sway_amp=amp, seed=s)
                clk = timing.Clock(len(sc.obs), ts=sc.ts, roi_w=sc.roi[2])
                shifts = [np.zeros(2)] + list(np.diff(sc.cum, axis=0))
                hist = mot.track(sc.obs, sc.roi, clock=clk, shifts=shifts)
                seq = mot.fill_gaps(mot.paths(hist), clk.frames_in(timing.FILL_SEC))
                sel, aim = saliency.Selector(clk, sc.roi, disp=disp), controller.Aim(clk)
                prev, n_in = np.zeros(2), 0
                W, H = sc.roi[2], sc.roi[3]
                for k in range(len(sc.obs)):
                    cum = hist[k][2]
                    bg = cum - prev
                    prev = cum
                    _, target, _, fresh = sel.step(k, hist, seq, bg)
                    p = aim.step(k, bg, cum, target, fresh)
                    t = sc.target[k]
                    n_in += int(p is not None and abs(p[0] - t[0]) <= W * FRAC / 2 and abs(p[1] - t[1]) <= H * FRAC / 2)
                ins.append(n_in / len(sc.obs))
            out[f"amp{amp:g}/{disp}"] = {"in_window": round(100 * float(np.mean(ins)), 2),
                                         "per_seed_in": [round(v, 4) for v in ins]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    rows = {}
    for s in range(a.seeds):
        sc, obs, occ, dist = scene(s)
        for r in (0.0, *RATES):
            for e in ("pos", "id", "none"):
                aims, nsw, st = run(sc, obs, r, e, SEED + s)
                m = metrics(sc, dist, occ, aims)
                x = rows.setdefault(f"{r:g}/{e}", {"in": [], "on_distractor": [], "in_occluded": [], "switches": 0,
                                                    "mark_new": 0, "excluded": 0, "excl_on_target": 0})
                x["in"].append(m[0])
                x["on_distractor"].append(m[1])
                x["in_occluded"].append(m[2])
                x["switches"] += nsw
                x["mark_new"] += st["mark_new"]
                x["excluded"] += st["excluded"]
                x["excl_on_target"] += st["excl_on_target"]
    summ = {}
    for key, x in rows.items():
        summ[key] = {
            "in_window": round(100 * float(np.mean(x["in"])), 2),
            "in_window_occluded": round(100 * float(np.mean(x["in_occluded"])), 2),
            "on_distractor": round(100 * float(np.mean(x["on_distractor"])), 2),
            "switches": x["switches"], "mark_new": x["mark_new"], "excluded": x["excluded"],
            "excl_on_target": x["excl_on_target"],
            "per_seed_in": [round(v, 4) for v in x["in"]],
        }
    sw = sway(a.seeds)
    for key, v in sw.items():
        print(f"sway {key:12s} 窗内 {v['in_window']:.2f}%")
    out = {"synthetic": True, "sway_net_vs_path": sw, "seeds": a.seeds, "n": N, "fps": FPS, "n_distractors": N_DIST, "lock_r": LOCK_R, "occ_every": OCC_EVERY,
           "occ_len": OCC_LEN, "frac": FRAC, "rates_per_sec": list(RATES), "summary": summ,
           "wall_sec": round(time.time() - t0, 1)}
    RUNS.mkdir(exist_ok=True)
    f = RUNS / f"abl-synth-{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'强度/名单':12s} 窗内% 遮挡期窗内% 锁干扰物% 切换 新建点 排除轨迹帧 其中压住目标")
    for key, s in summ.items():
        print(f"{key:12s} {s['in_window']:6.2f} {s['in_window_occluded']:8.2f} {s['on_distractor']:8.2f} "
              f"{s['switches']:6d} {s['mark_new']:5d} {s['excluded']:6d} {s['excl_on_target']:6d}")
    print(f"写入 {f}")


if __name__ == "__main__":
    main()
