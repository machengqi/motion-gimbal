"""E2b 稳定化的自检：画布偏移 `o[k]`（`gimbal_vision.stabilize`）的累积漂移有多大。**不看真值。**

    uv run python scripts/diag_e2b_stab.py --split dev [--span 48] [--workers 16]

三件事：

1. **长基线直接配准**：第 k 帧与第 k+span 帧之间直接做 SIFT + RANSAC（同一个 `Registrar`），得到这段
   区间的平移 `t_direct`；逐帧累积给出的是 `o[k] − o[k+span]`。两者之差 = 这段区间内逐帧累积的漂移
   （外加长基线配准自己的误差，所以是漂移的**上界**）。只用内点 ≥ `MIN_DIRECT`、且累积位移 < `MAX_SPAN_W`
   倍画面宽的区间（摇得更远时两帧几乎不重叠，直接配准本身不可信），k 每 `span/2` 取一次。
2. **与被测方法的背景估计对照**：`candidates.motion` 的逐帧 `bg_shift`（LK + RANSAC）与 SIFT 逐帧平移的差 ——
   两套独立估计的一致程度；另报整段累积后两者终点之差 ÷ 累积路程。
3. 画布偏移的范围：整段里素材在画布上走了多远（÷ 画面宽）。
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
from eval_e2 import RUNS, canvas_offsets, load_sequence  # noqa: E402
from make_split import load_split  # noqa: E402

from gimbal_vision import stabilize  # noqa: E402

MIN_DIRECT = 30
MAX_SPAN_W = 0.5


def analyse(name, span):
    _, _, imgs, _, _ = load_sequence(name, 640)
    st = canvas_offsets(name, 640, imgs)
    o = st["o"]
    H, W = imgs[0].shape[:2]
    reg = stabilize.Registrar()
    errs, skipped = [], 0
    for k in range(0, len(imgs) - span, max(1, span // 2)):
        t, n = reg.shift(reg.features(imgs[k]), reg.features(imgs[k + span]), (W, H))
        if t is None or n < MIN_DIRECT or np.hypot(*(o[k] - o[k + span])) > MAX_SPAN_W * W:
            skipped += 1
            continue
        errs.append(float(np.hypot(*((o[k] - o[k + span]) - t))))
    d = load(name)  # 被测方法的逐帧 bg_shift（与 E2 同一份缓存）
    lk = np.array(d["shifts"][1:])
    sift = -(np.diff(o, axis=0))  # t[k] = o[k−1] − o[k]
    ok = np.array([n >= stabilize.MIN_INLIERS for n in st["inliers"][1:]])
    dif = np.hypot(*(lk[ok] - sift[ok]).T) if ok.any() else np.array([])
    path = float(np.sum(np.hypot(*sift.T)))
    end = float(np.hypot(*(np.sum(lk, axis=0) - np.sum(sift, axis=0))))
    return {
        "name": name,
        "frames": len(imgs),
        "stab_fail": int(st["fail"]),
        "drift_px": errs,
        "drift_skipped": skipped,
        "lk_vs_sift_px": dif.tolist(),
        "path_px": path,
        "lk_vs_sift_end_px": end,
        "excursion_w": float(np.max(np.hypot(*(o - o[0]).T)) / W),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--span", type=int, default=48)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    names = load_split(a.split)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(analyse, names, [a.span] * len(names)))
    drift = np.array([x for r in res for x in r["drift_px"]])
    dif = np.array([x for r in res for x in r["lk_vs_sift_px"]])
    exc = np.array([r["excursion_w"] for r in res])
    out = {
        "split": a.split,
        "span_frames": a.span,
        "videos": len(res),
        "stab_fail_frames": int(sum(r["stab_fail"] for r in res)),
        "frames": int(sum(r["frames"] for r in res)),
        "drift_intervals": len(drift),
        "drift_skipped": int(sum(r["drift_skipped"] for r in res)),
        "drift_px_pctl_50_90_99": np.percentile(drift, [50, 90, 99]).round(1).tolist() if len(drift) else None,
        "lk_vs_sift_per_frame_px_pctl_50_90_99": np.percentile(dif, [50, 90, 99]).round(2).tolist(),
        "lk_vs_sift_end_over_path_quartiles_max": np.percentile(
            [r["lk_vs_sift_end_px"] / max(r["path_px"], 1.0) for r in res], [25, 50, 75, 100]
        ).round(3).tolist(),
        "excursion_over_width_quartiles_max": np.percentile(exc, [25, 50, 75, 100]).round(2).tolist(),
        "per_video": [{k: v for k, v in r.items() if k != "lk_vs_sift_px"} for r in res],
        "wall_sec": round(time.time() - t0, 1),
    }
    f = RUNS / f"diag-e2b-stab-{a.split}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_video"}, ensure_ascii=False, indent=1))
    print(f"写入 {f}")


if __name__ == "__main__":
    main()
