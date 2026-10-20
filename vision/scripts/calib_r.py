"""按实测检测噪声标定跟踪器的观测噪声 `mot.Track.R`。**只用 dev 集。**

    uv run python scripts/calib_r.py [--split dev] [--workers 6]

# 量什么

观测噪声 R 描述的是「检测给出的框心」相对「目标真实位置」的散布。这里直接拿真值量：

```text
样本   dev 集每段、每个**人工标注帧**（MoCA 每 5 帧标一次；插值帧不用，避免把插值误差算进来）上，
       与真值框匹配的运动候选 —— 匹配判据同 E1：候选中心落在真值框内，且面积 ≤ TIGHT × 真值面积
残差   e = 候选中心 − 真值框中心（640 宽画面系，px），每轴分开
去偏   每段减去该段残差的中位数（标注框与运动点云的系统偏差 —— 例如只有头在动 —— 不是逐帧噪声，
       Kalman 也不靠 R 吸收它）
估计   σ_axis = 1.4826 · median(|e − median(e)|)          （MAD，抗住少量错配）
       R = diag(σx², σy²)
```

⚠️ 「中心落在真值框内」对残差有截断（残差不会超过半个框宽），大目标上影响小、小目标上会把 σ 估低；
同时报告不去偏的结果与按框大小分组的结果，便于判断截断的影响。
⚠️ 结论是对**这一个检测器（`candidates.motion`）在这个数据集上**的标定；换检测器要重测。
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
from eval_e2 import RUNS, TIGHT  # noqa: E402
from make_split import load_split  # noqa: E402

from gimbal_vision.datasets import moca  # noqa: E402

MAD_K = 1.4826
"""正态分布下 MAD → σ 的换算系数。"""


def match(det, g):
    gx, gy, gw, gh = g
    return gx <= det[0] <= gx + gw and gy <= det[1] <= gy + gh and det[2] <= TIGHT * max(gw * gh, 1.0)


def residuals(name):
    """返回 `(name, 静态残差 [(ex, ey, √真值面积)], 帧间残差差分 [(dex, dey)])`。"""
    d = load(name)
    keys = moca.load()[name].keys
    stat, diff = [], []
    prev = None  # 上一帧（列表里相邻）唯一匹配候选的残差
    for i, k in enumerate(d["ks"]):
        g = d["gts"][i]
        if g is None or i == 0:
            prev = None
            continue
        cx, cy = g[0] + g[2] / 2, g[1] + g[3] / 2
        m = [(det[0] - cx, det[1] - cy) for det in d["obs"][i] if match(det, g)]
        if k in keys:
            stat += [(ex, ey, float(np.sqrt(max(g[2] * g[3], 1.0)))) for ex, ey in m]
        cur = m[0] if len(m) == 1 else None
        if cur is not None and prev is not None:
            diff.append((cur[0] - prev[0], cur[1] - prev[1]))
        prev = cur
    return name, stat, diff


def mad_sigma(v):
    v = np.asarray(v, float)
    return float(MAD_K * np.median(np.abs(v - np.median(v))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    names = load_split(a.split)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = {n: (st, df) for n, st, df in ex.map(residuals, names)}
    raw, deb, size, dif = [], [], [], []
    for _, (st, df) in sorted(res.items()):
        dif += df
        if not st:
            continue
        e = np.array(st)
        raw.append(e[:, :2])
        deb.append(e[:, :2] - np.median(e[:, :2], axis=0))
        size.append(e[:, 2])
    raw, deb, size = np.concatenate(raw), np.concatenate(deb), np.concatenate(size)
    dif = np.array(dif)

    def est(e):
        sx, sy = mad_sigma(e[:, 0]), mad_sigma(e[:, 1])
        return {"n": len(e), "sigma_xy": [round(sx, 2), round(sy, 2)], "R_xy": [round(sx * sx, 1), round(sy * sy, 1)]}

    edges = np.percentile(size, [0, 33.3, 66.7, 100])
    by_size = {}
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (size >= lo) & (size <= hi)
        by_size[f"sqrt_area_{lo:.0f}-{hi:.0f}px"] = est(deb[m])
    out = {
        "split": a.split,
        "videos": len(names),
        "videos_with_samples": sum(1 for st, _ in res.values() if st),
        "method": "keyframes only; E1 match (center in GT box, area <= TIGHT*GT); per-video median removed; sigma = 1.4826*MAD",
        "temporal": {
            "n": len(dif),
            "sigma_xy": [round(mad_sigma(dif[:, i]) / np.sqrt(2), 2) for i in (0, 1)],
            "R_xy": [round(mad_sigma(dif[:, i]) ** 2 / 2, 1) for i in (0, 1)],
            "note": "consecutive kept frames, exactly one GT-matched candidate in both; R = (1.4826*MAD(diff))^2 / 2",
        },
        "debiased": est(deb),
        "raw": est(raw),
        "debiased_by_gt_size": by_size,
        "abs_resid_pctl_50_90_99": np.percentile(np.hypot(deb[:, 0], deb[:, 1]), [50, 90, 99]).round(1).tolist(),
        "wall_sec": round(time.time() - t0, 1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    f = RUNS / f"calib-r-{a.split}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {f}")


if __name__ == "__main__":
    main()
