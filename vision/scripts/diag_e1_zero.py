"""E1 单帧命中率为 0（或接近 0）的视频**逐段归因**：每条归因都对应一组可核查的统计量。

    uv run python scripts/diag_e1_zero.py [--workers 4] [--videos a,b,c]

对每段视频重跑一遍 `candidates.motion` 的四步（同一份代码路径与参数，逐帧核对候选与 `diag_cache` 一致），
额外记下中间量：

| 量 | 含义 |
|---|---|
| `gt_area_frac` / `gt_wh` | 真值框面积 ÷ 画面面积；真值框宽高（640 宽像素） |
| `motion_frac` | 有真值的帧里 locomotion / subtle / still / 无标签 的占比（沿用稀疏标签的推断，同 E1） |
| `pts_tracked` | 通过前后向校验的点对数 |
| `inlier_frac` | 单应 RANSAC 内点 ÷ 点对数 |
| `bg_pps` | 相机运动：单应把画面中心移动的速度（px/s） |
| `gt_pts` | 落在真值框内的点对数（目标上有没有可跟的纹理） |
| `gt_pts_share` | 真值框内的点占全部点的比例（目标占比大时单应可能把它的运动当成相机运动吸收） |
| `gt_inlier_frac` | 真值框内的点里被判为单应内点的比例 |
| `gt_resid_pps` | 真值框内点的残余速度中位数 / 90 分位（px/s），对照阈值 `resid_pps · s` |
| `gt_moving` | 真值框内超过阈值的运动点数 |
| `gt_resid_bgH_med` / `gt_moving_bgH_ge_min_pts_frac` | 只用真值框**外**的点拟合单应后，框内点的残余速度中位数 / 框内运动点 ≥ `min_pts` 的帧占比 —— 与上两行对比，看目标运动是否被全帧单应吸收 |
| `gt_center_resid_pps` | 真值框中心自身相对单应预测的速度（按真值标注算，衡量「目标相对背景动了多少」） |
| `cand_in_gt_too_big` | 有候选中心落在真值框内、但面积 > 4 倍真值面积（被 E1 面积判据拒掉）的帧占比 |
| `cand_any` | 有任何候选的帧占比；`cand_near` 最近候选中心到真值框的距离 ÷ 画面宽 |
| `gt_out_of_frame` | 真值框超出画面的帧占比；`gt_jump` 相邻帧真值中心跳变 > 0.2 倍画面宽的次数 |

另对全部 141 段算同样的逐段中位数，作为对照（`population`）。
"""

from __future__ import annotations

import lowprio  # noqa: F401,I001  — 必须在 numpy / cv2 之前

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import load  # noqa: E402
from eval_e2 import RUNS, TIGHT  # noqa: E402

from gimbal_vision.candidates.motion import MotionCandidates  # noqa: E402
from gimbal_vision.datasets import moca  # noqa: E402

TARGETS = [
    "crab", "crab_2", "devil_scorpionfish", "fossa", "polar_bear_3", "snow_leopard_10", "wolf",
    "eastern_screech_owl_0", "seal",
]
"""E1（`runs/candidates-20261014-055702.json`）命中率恰为 0 的 7 段 + 论文 6.7 节列出的两段近 0 段。"""


def internals(gen, prev, gray, dt):
    """`MotionCandidates.__call__` 的①②③步，返回中间量（与 motion.py 同一组调用与参数）。"""
    p = gen.p
    p0 = cv2.goodFeaturesToTrack(prev, maxCorners=p.max_pts, qualityLevel=p.quality, minDistance=p.min_dist, mask=None)
    if p0 is None or len(p0) < 8:
        return None
    lk = {"winSize": (21, 21), "maxLevel": 3}
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, **lk)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p1, None, **lk)
    fb = np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < p.fb_err)
    a, b = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
    if len(a) < 8:
        return None
    H, inl = cv2.findHomography(a, b, cv2.RANSAC, p.ransac_px)
    if H is None:
        return None
    pred = cv2.perspectiveTransform(a.reshape(-1, 1, 2), H).reshape(-1, 2)
    resid = np.linalg.norm(b - pred, axis=1) / dt
    return {"a": a, "b": b, "H": H, "inl": inl.ravel().astype(bool), "resid": resid, "n_p0": len(p0)}


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def p90(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.percentile(v, 90)) if v else None


def analyse(name):
    from eval_e2 import load_sequence  # noqa: PLC0415

    seq, dup, imgs, ks, s = load_sequence(name, 640)
    d = load(name)
    ts = np.array(ks, float) / moca.FPS
    H_img, W = imgs[0].shape[:2]
    gen = MotionCandidates(640)
    thr = gen.p.resid_pps * gen.s
    rows, mismatch = [], 0
    for i, img in enumerate(imgs):
        r = gen(img, ts[i])
        if [tuple(np.round(x, 6)) for x in r.dets] != [tuple(np.round(x, 6)) for x in d["obs"][i]]:
            mismatch += 1
        g = d["gts"][i]
        if i == 0 or g is None:
            continue
        dt = max(ts[i] - ts[i - 1], 1e-4)
        it = internals(gen, imgs[i - 1], img, dt)
        gx, gy, gw, gh = g
        garea = max(gw * gh, 1.0)
        row = {"i": i, "motion": d["motions"][i], "gt_area_frac": garea / (W * H_img), "gw": gw, "gh": gh,
               "gt_out": bool(gx < 0 or gy < 0 or gx + gw > W or gy + gh > H_img)}
        dets = r.dets
        inside = [gx <= q[0] <= gx + gw and gy <= q[1] <= gy + gh for q in dets]
        row["hit"] = any(f and q[2] <= TIGHT * garea for f, q in zip(inside, dets, strict=True))
        row["too_big"] = any(f and q[2] > TIGHT * garea for f, q in zip(inside, dets, strict=True))
        row["cand_any"] = bool(dets)
        if dets:
            dist = [np.hypot(max(gx - q[0], 0, q[0] - gx - gw), max(gy - q[1], 0, q[1] - gy - gh)) for q in dets]
            row["cand_near"] = float(min(dist)) / W
        if it is None:
            row["est_fail"] = True
            rows.append(row)
            continue
        b, resid, inl = it["b"], it["resid"], it["inl"]
        c = np.array([[[W / 2, H_img / 2]]])
        row["bg_pps"] = float(np.hypot(*(cv2.perspectiveTransform(c, it["H"]).reshape(2) - c.reshape(2)))) / dt
        row["pts"] = len(b)
        row["inlier_frac"] = float(inl.mean())
        m = (b[:, 0] >= gx) & (b[:, 0] <= gx + gw) & (b[:, 1] >= gy) & (b[:, 1] <= gy + gh)
        row["gt_pts"] = int(m.sum())
        row["gt_pts_share"] = float(m.mean())
        row["moving_all"] = int((resid > thr).sum())
        if m.any():
            row["gt_inlier_frac"] = float(inl[m].mean())
            row["gt_resid_med"] = float(np.median(resid[m]))
            row["gt_resid_p90"] = float(np.percentile(resid[m], 90))
            row["gt_moving"] = int((resid[m] > thr).sum())
        # 只用真值框外的点拟合单应（「纯背景」相机运动），看目标的运动是否被全帧单应吸收
        if (~m).sum() >= 8 and m.any():
            Hb, _ = cv2.findHomography(it["a"][~m], b[~m], cv2.RANSAC, gen.p.ransac_px)
            if Hb is not None:
                pb_ = cv2.perspectiveTransform(it["a"][m].reshape(-1, 1, 2), Hb).reshape(-1, 2)
                rb = np.linalg.norm(b[m] - pb_, axis=1) / dt
                row["gt_resid_bgH_med"] = float(np.median(rb))
                row["gt_moving_bgH"] = int((rb > thr).sum())
                row["bgH_vs_H_center_px"] = float(np.hypot(*(cv2.perspectiveTransform(np.array([[[W / 2, H_img / 2]]]), Hb)
                                                             - cv2.perspectiveTransform(np.array([[[W / 2, H_img / 2]]]), it["H"])).reshape(2)))
        # 真值中心自身相对背景的速度：上一帧真值中心按单应搬到这一帧，与这一帧真值中心之差
        g0 = d["gts"][i - 1]
        if g0 is not None:
            c0 = np.array([[[g0[0] + g0[2] / 2, g0[1] + g0[3] / 2]]])
            c1 = np.array([gx + gw / 2, gy + gh / 2])
            row["gt_center_resid_pps"] = float(np.hypot(*(cv2.perspectiveTransform(c0, it["H"]).reshape(2) - c1))) / dt
            row["gt_jump"] = bool(np.hypot(*(c1 - c0.reshape(2))) > 0.2 * W)
        rows.append(row)

    def frac(key):
        return float(np.mean([bool(r.get(key, False)) for r in rows])) if rows else None

    labs = [moca.MOTION.get(r["motion"], "unlabeled") for r in rows]
    # 真值中心速度按「标注帧之间」算更稳：插值帧上它是分段常数，取中位数即可
    out = {
        "name": name,
        "frames": len(seq.frames),
        "duplicates": int(dup.sum()),
        "gt_frames": len(rows),
        "cand_mismatch_frames": mismatch,
        "resid_threshold_pps": thr,
        "hit_frac": frac("hit"),
        "motion_frac": {k: round(labs.count(k) / max(len(labs), 1), 3) for k in ("locomotion", "subtle_motion", "still", "unlabeled")},
        "gt_area_frac_med": med([r["gt_area_frac"] for r in rows]),
        "gt_wh_med": [med([r["gw"] for r in rows]), med([r["gh"] for r in rows])],
        "pts_tracked_med": med([r.get("pts") for r in rows]),
        "inlier_frac_med": med([r.get("inlier_frac") for r in rows]),
        "bg_pps_med": med([r.get("bg_pps") for r in rows]),
        "bg_pps_p90": p90([r.get("bg_pps") for r in rows]),
        "gt_pts_med": med([r.get("gt_pts") for r in rows]),
        "gt_pts_zero_frac": float(np.mean([r.get("gt_pts", 0) == 0 for r in rows])) if rows else None,
        "gt_pts_share_med": med([r.get("gt_pts_share") for r in rows]),
        "gt_inlier_frac_med": med([r.get("gt_inlier_frac") for r in rows]),
        "gt_resid_med": med([r.get("gt_resid_med") for r in rows]),
        "gt_resid_p90_med": med([r.get("gt_resid_p90") for r in rows]),
        "gt_moving_med": med([r.get("gt_moving") for r in rows]),
        "gt_moving_ge_min_pts_frac": float(np.mean([r.get("gt_moving", 0) >= gen.p.min_pts for r in rows])) if rows else None,
        "moving_all_med": med([r.get("moving_all") for r in rows]),
        "gt_resid_bgH_med": med([r.get("gt_resid_bgH_med") for r in rows]),
        "gt_moving_bgH_ge_min_pts_frac": float(np.mean([r.get("gt_moving_bgH", 0) >= gen.p.min_pts for r in rows])) if rows else None,
        "bgH_vs_H_center_px_med": med([r.get("bgH_vs_H_center_px") for r in rows]),
        "gt_center_resid_pps_med": med([r.get("gt_center_resid_pps") for r in rows]),
        "gt_center_resid_pps_p90": p90([r.get("gt_center_resid_pps") for r in rows]),
        "cand_any_frac": frac("cand_any"),
        "cand_in_gt_too_big_frac": frac("too_big"),
        "cand_near_med": med([r.get("cand_near") for r in rows]),
        "est_fail_frames": int(sum(r.get("est_fail", False) for r in rows)),
        "gt_out_of_frame_frac": frac("gt_out"),
        "gt_jumps": int(sum(r.get("gt_jump", False) for r in rows)),
        "annotated_keyframes": len(seq.keys),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--videos", default="")
    a = ap.parse_args()
    names = a.videos.split(",") if a.videos else sorted(moca.load())
    allseq = moca.load()
    names = sorted(names, key=lambda nm: -len(allseq[nm].frames))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=lowprio.init_worker) as ex:
        res = list(ex.map(analyse, names, chunksize=1))
    by = {r["name"]: r for r in res}
    keys = [k for k, v in res[0].items() if isinstance(v, (int, float)) and k not in ("frames",)]
    pop = {}
    if len(res) > 20:
        rest = [r for r in res if r["name"] not in TARGETS and r["gt_frames"]]
        pop = {"videos": len(rest), **{k: med([r[k] for r in rest]) for k in keys}}
        pop["motion_frac_mean"] = {
            lab: round(float(np.mean([r["motion_frac"][lab] for r in rest])), 3)
            for lab in ("locomotion", "subtle_motion", "still", "unlabeled")
        }
    out = {
        "targets": {n: by[n] for n in TARGETS if n in by},
        "population_median_excluding_targets": pop,
        "all": res if len(res) > 20 else None,
        "wall_sec": round(time.time() - t0, 1),
    }
    f = RUNS / f"diag-e1-zero-{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for n in TARGETS:
        if n in by:
            print(json.dumps(by[n], ensure_ascii=False))
    if pop:
        print("population", json.dumps(pop, ensure_ascii=False))
    print(f"写入 {f}")


if __name__ == "__main__":
    main()
