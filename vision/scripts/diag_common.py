"""诊断脚本共用：加载一段 MoCA、跑全幅运动候选（结果缓存到 `runs/diag_cache/`）。

与 `eval_e2.eval_sequence` 的开环部分**同一条路径**（直接复用 `eval_e2.load_sequence`），
只多存了单应矩阵，供背景位移的对照用。
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_e2 import RUNS, load_sequence  # noqa: E402

from gimbal_vision.candidates.motion import MotionCandidates  # noqa: E402
from gimbal_vision.core import timing  # noqa: E402
from gimbal_vision.datasets import moca  # noqa: E402

CACHE = RUNS / "diag_cache"


def load(name: str, width: int = 640, keep_imgs: bool = False):
    """返回 dict：ks, ts, gts（中心+框，640 系）, motions, obs, shifts, Hs, clk, W, H。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}-{width}.pkl"
    imgs = None
    if f.exists() and not keep_imgs:
        d = pickle.loads(f.read_bytes())
    else:
        seq, dup, imgs, ks, s = load_sequence(name, width)
        ts = np.array(ks, dtype=float) / moca.FPS
        gen = MotionCandidates(width)
        obs, shifts, Hs, fails = [], [], [], 0
        for i, img in enumerate(imgs):
            r = gen(img, ts[i])
            obs.append(r.dets)
            Hs.append(r.homography)
            if r.bg_shift is None:
                fails += 1
                shifts.append(np.zeros(2))
            else:
                shifts.append(np.asarray(r.bg_shift, float))
        gts, motions = [], []
        for k in ks:
            b = seq.box(k)
            gts.append(None if b is None else b * s)
            motions.append(seq.motion(k))
        H, W = imgs[0].shape[:2]
        d = dict(name=name, ks=ks, ts=ts, gts=gts, motions=motions, obs=obs, shifts=shifts, Hs=Hs,
                 W=W, H=H, bg_fail=fails)
        f.write_bytes(pickle.dumps(d))
    d["clk"] = timing.Clock(len(d["ks"]), d["ts"], roi_w=d["W"])
    if keep_imgs:
        d["imgs"] = imgs
    return d


def gt_center(g):
    if g is None:
        return None
    return np.array([g[0] + g[2] / 2, g[1] + g[3] / 2])
