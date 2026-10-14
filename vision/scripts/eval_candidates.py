"""在 MoCA 上评测**运动候选生成**：伪装动物能不能只靠运动被框出来。

    uv run python scripts/eval_candidates.py [--width 640] [--limit N] [--workers 6]

输出 `runs/candidates-<时间>.json`（逐段指标 + 汇总）并打印汇总。

# 指标

- **命中**：某个候选框的中心落在真值框内，且面积 ≤ `TIGHT` × 真值框面积。
  看中心是因为下游跟踪器只用候选中心；限面积防止「盖住大半画面的框」蒙混过关。
- **IoU≥0.3 召回**：与真值框 IoU ≥ 0.3 的候选存在 —— 便于和其他工作对照。
- **误报**：中心不在真值框内的候选个数（每帧）。
- **耗时**：候选生成本身的单帧耗时（不含 JPEG 解码）。

复制帧先剔除（判据见 `datasets.moca.DUP_FRAC`），保留其余帧的真实时间戳。
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from gimbal_vision.candidates.motion import MotionCandidates
from gimbal_vision.datasets import moca

TIGHT = 4.0
RUNS = Path(__file__).resolve().parents[2] / "runs"


def iou(a, b):
    """a, b: (x1, y1, x2, y2)。"""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def eval_sequence(name: str, width: int) -> dict:
    seq = moca.load()[name]
    dup = moca.duplicates(seq)
    gen = None
    rows = []
    for k, path in enumerate(seq.frames):
        if dup[k]:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        s = width / img.shape[1]
        if s != 1.0:
            img = cv2.resize(img, (width, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        if gen is None:
            gen = MotionCandidates(width)
        t0 = time.perf_counter()
        res = gen(img, k / moca.FPS)
        ms = (time.perf_counter() - t0) * 1000
        box = seq.box(k)
        if box is None or k == 0:
            continue
        gx1, gy1, gw, gh = box * s
        g = (gx1, gy1, gx1 + gw, gy1 + gh)
        garea = max(gw * gh, 1.0)
        inside = [g[0] <= d[0] <= g[2] and g[1] <= d[1] <= g[3] for d in res.dets]
        hit = any(i and d[2] <= TIGHT * garea for i, d in zip(inside, res.dets, strict=True))
        hit_iou = any(iou((d[3], d[4], d[5], d[6]), g) >= 0.3 for d in res.dets)
        rows.append(
            {
                "k": k,
                "hit": bool(hit),
                "hit_iou": bool(hit_iou),
                "false": int(sum(not i for i in inside)),
                "n": len(res.dets),
                "ms": ms,
                "motion": seq.motion(k),
                "bg_ok": res.bg_shift is not None,
            }
        )
    return {"name": name, "frames": len(seq.frames), "duplicates": int(dup.sum()), "rows": rows}


def summarize(results: list[dict]) -> dict:
    rows = [r for res in results for r in res["rows"]]

    def agg(rs):
        if not rs:
            return None
        return {
            "frames": len(rs),
            "hit": float(np.mean([r["hit"] for r in rs])),
            "hit_iou30": float(np.mean([r["hit_iou"] for r in rs])),
            "false_per_frame": float(np.mean([r["false"] for r in rs])),
            "zero_candidate_frames": float(np.mean([r["n"] == 0 for r in rs])),
        }

    per_video = [np.mean([r["hit"] for r in res["rows"]]) for res in results if res["rows"]]
    ms = np.array([r["ms"] for r in rows])
    return {
        "all_frames": agg(rows),
        "per_video_mean_hit": float(np.mean(per_video)),
        "per_video_hit_quartiles": np.percentile(per_video, [25, 50, 75]).round(3).tolist(),
        "by_motion": {moca.MOTION.get(m, "unlabeled"): agg([r for r in rows if r["motion"] == m]) for m in (0, 1, 2, None)},
        "ms_per_frame": {"mean": float(ms.mean()), "p50": float(np.median(ms)), "p95": float(np.percentile(ms, 95))},
        "duplicates_removed": int(sum(r["duplicates"] for r in results)),
        "bg_estimate_failed_frames": int(sum(not r["bg_ok"] for r in rows)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    names = sorted(moca.load())
    if args.limit:
        names = names[: args.limit]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(eval_sequence, names, [args.width] * len(names)))
    summary = summarize(results)
    summary["width"] = args.width
    summary["videos"] = len(results)
    summary["wall_sec"] = round(time.time() - t0, 1)
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"candidates-{time.strftime('%Y%m%d-%H%M%S')}.json"
    per_video = [
        {"name": r["name"], "hit": float(np.mean([x["hit"] for x in r["rows"]])) if r["rows"] else None,
         "false_per_frame": float(np.mean([x["false"] for x in r["rows"]])) if r["rows"] else None,
         "frames": r["frames"], "duplicates": r["duplicates"]}
        for r in results
    ]
    out.write_text(json.dumps({"summary": summary, "videos": per_video}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
