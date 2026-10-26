"""背景模型的累积漂移：**平移 vs 相似变换 vs 单应**，同一把尺子。**不看真值。**

    uv run python scripts/diag_drift.py --split test [--spans 24,48,96] [--workers 6]

口径与 `diag_e2b_stab.py` 相同（那里只量了平移模型在画面中心的漂移）：

- 每隔 `span/2` 帧取一个区间 `[k, k+span]`，对两端帧直接做 SIFT + RANSAC 单应（`stabilize.Registrar`），
  得到长基线参照 `H_direct`。只用内点 ≥ `MIN_DIRECT`、且 SIFT 平移累积位移 < `MAX_SPAN_W` 倍画面宽的区间
  （与 `diag_e2b_stab.py` 同一组区间）。
- 每个背景模型把第 k 帧的一个点逐帧传到第 k+span 帧，与 `H_direct` 给出的位置比较，差 = 区间内的累积
  漂移（外加长基线配准自身的误差，因此是上界）。
- 点：画面中心（= `diag_e2b_stab.py` 的口径），以及 3×3 网格（画面宽高的 20% / 50% / 80%）中参照位置仍落在
  第 k+span 帧画面内的点。⚠️ 参照本身是单应模型，网格点上会偏向单应类模型；画面中心处平移模型的参照
  就是单应把中心映到哪，与平移的定义一致。

模型（逐帧估计 → 逐帧累积）：

| 键 | 逐帧估计 | 累积 |
|---|---|---|
| `sift_T` | SIFT 单应把画面中心映到哪（= E2b 画布 `o[k]`） | 所有点加同一个平移和 |
| `sift_S` | SIFT 相似变换（4 自由度，`estimateAffinePartial2D`） | 矩阵链 |
| `sift_H` | SIFT 单应 | 矩阵链 |
| `lk_T`   | 被测方法 `candidates.motion` 的 `bg_shift`（LK + RANSAC 单应，画面中心）= 跟踪器 / 排除图的背景系 | 平移和 |
| `lk_H`   | 被测方法的单应 `homography` | 矩阵链 |

逐帧估计失败时：平移记 0、矩阵记单位阵（与 E2 / E2b 的处理一致）。
SIFT 的逐帧单应 / 相似变换缓存到 `runs/diag_cache/stabHS-<name>-640.pkl`（E2b 单应画布也用它）。
"""

from __future__ import annotations

import lowprio  # noqa: F401,I001  — 必须在 numpy / cv2 之前

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import load  # noqa: E402
from eval_e2 import CACHE, RUNS, canvas_offsets, load_sequence  # noqa: E402
from make_split import load_split  # noqa: E402

from gimbal_vision import stabilize  # noqa: E402

MIN_DIRECT = 30
MAX_SPAN_W = 0.5
GRID = (0.2, 0.5, 0.8)
MODELS = ("sift_T", "sift_S", "sift_H", "lk_T", "lk_H")


def sift_transforms(name, width, imgs):
    """SIFT 逐帧单应 / 相似变换（缓存）。"""
    f = CACHE / f"stabHS-{name}-{width}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    Hs, Ss, inl = stabilize.transforms(imgs)
    d = {"H": Hs, "S": Ss, "inliers": inl}
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(d))
    return d


def apply(M, p):
    q = M @ np.array([p[0], p[1], 1.0])
    return q[:2] / q[2]


def chain(mats, k0, k1):
    """第 k0 帧 → 第 k1 帧：`M[k1] ··· M[k0+1]`（`None` 当单位阵）。"""
    out = np.eye(3)
    for j in range(k0 + 1, k1 + 1):
        if mats[j] is not None:
            out = np.asarray(mats[j], float) @ out
    return out


def analyse(name, spans):
    _, _, imgs, _, _ = load_sequence(name, 640)
    H_img, W = imgs[0].shape[:2]
    st = canvas_offsets(name, 640, imgs)
    o = st["o"]
    hs = sift_transforms(name, 640, imgs)
    lk = load(name)
    lk_shift = np.array(lk["shifts"])  # 第 k 个：k−1 → k
    lk_H = lk["Hs"]
    n = len(imgs)
    c = np.array([W / 2, H_img / 2])
    # 自检：SIFT 单应把中心映到哪 ⇒ 应逐帧复现画布偏移 o
    t_sift = np.zeros((n, 2))
    for k in range(1, n):
        if hs["H"][k] is not None:
            t_sift[k] = apply(np.asarray(hs["H"][k], float), c) - c
    o_check = float(np.max(np.abs(stabilize.accumulate([None] + list(t_sift[1:])) - o)))

    reg = stabilize.Registrar()
    feats: dict[int, tuple] = {}

    def feat(k):
        if k not in feats:
            feats[k] = reg.features(imgs[k])
        return feats[k]

    grid = [np.array([gx * W, gy * H_img]) for gy in GRID for gx in GRID]
    out = {"name": name, "frames": n, "o_check_px": o_check, "sift_fail": int(sum(h is None for h in hs["H"][1:])),
           "sift_S_fail": int(sum(s is None for s in hs["S"][1:])),
           "lk_fail": int(sum(h is None for h in lk_H[1:])), "spans": {}}
    for span in spans:
        rec = {m: {"center": [], "grid": []} for m in MODELS}
        skipped = used = 0
        for k in range(0, n - span, max(1, span // 2)):
            Hd, nin = reg.homography(feat(k), feat(k + span))
            if Hd is None or nin < MIN_DIRECT or np.hypot(*(o[k] - o[k + span])) > MAX_SPAN_W * W:
                skipped += 1
                continue
            used += 1
            pred = {
                "sift_T": lambda p, a=t_sift[k + 1 : k + span + 1].sum(0): p + a,
                "lk_T": lambda p, a=lk_shift[k + 1 : k + span + 1].sum(0): p + a,
            }
            for key, mats in (("sift_S", hs["S"]), ("sift_H", hs["H"]), ("lk_H", lk_H)):
                M = chain(mats, k, k + span)
                pred[key] = lambda p, M=M: apply(M, p)
            ref_c = apply(Hd, c)
            for m in MODELS:
                rec[m]["center"].append(float(np.hypot(*(pred[m](c) - ref_c))))
            for g in grid:
                r = apply(Hd, g)
                if not (0 <= r[0] <= W and 0 <= r[1] <= H_img):
                    continue
                for m in MODELS:
                    rec[m]["grid"].append(float(np.hypot(*(pred[m](g) - r))))
        rec["used"], rec["skipped"] = used, skipped
        out["spans"][str(span)] = rec
        # 释放已不需要的特征
        feats.clear()
    return out


def pct(v):
    return np.percentile(v, [50, 90, 99]).round(1).tolist() if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--spans", default="24,48,96")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    spans = [int(x) for x in a.spans.split(",")]
    names = load_split(a.split)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=lowprio.init_worker) as ex:
        res = list(ex.map(analyse, names, [spans] * len(names)))
    summary = {}
    for span in spans:
        s = str(span)
        row = {"intervals": int(sum(r["spans"][s]["used"] for r in res)),
               "skipped": int(sum(r["spans"][s]["skipped"] for r in res))}
        for m in MODELS:
            cen = [x for r in res for x in r["spans"][s][m]["center"]]
            grd = [x for r in res for x in r["spans"][s][m]["grid"]]
            row[m] = {
                "center_p50_90_99": pct(cen),
                "grid_p50_90_99": pct(grd),
                "grid_points": len(grd),
                "center_gt50px": int(sum(x > 50 for x in cen)),
                "grid_gt20px_frac": round(float(np.mean(np.array(grd) > 20)), 4) if grd else None,
            }
        summary[s] = row
    out = {
        "split": a.split,
        "spans": spans,
        "videos": len(res),
        "min_direct_inliers": MIN_DIRECT,
        "max_span_w": MAX_SPAN_W,
        "grid": GRID,
        "o_check_max_px": max(r["o_check_px"] for r in res),
        "fail_frames": {k: int(sum(r[k] for r in res)) for k in ("sift_fail", "sift_S_fail", "lk_fail")},
        "summary": summary,
        "per_video": [
            {"name": r["name"], "frames": r["frames"],
             **{f"{m}_s48_center_max": (max(r["spans"]["48"][m]["center"]) if "48" in r["spans"] and r["spans"]["48"][m]["center"] else None) for m in MODELS},
             **{f"{m}_s48_grid_p90": (pct(r["spans"]["48"][m]["grid"]) or [None, None])[1] if "48" in r["spans"] else None for m in MODELS}}
            for r in res
        ],
        "wall_sec": round(time.time() - t0, 1),
    }
    f = RUNS / f"diag-drift-{a.split}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_video"}, ensure_ascii=False, indent=1))
    print(f"写入 {f}")


if __name__ == "__main__":
    main()
