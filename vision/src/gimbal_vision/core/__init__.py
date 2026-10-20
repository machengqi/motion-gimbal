"""与具体应用无关的核心：时钟、多目标跟踪、运动显著性选择、平滑瞄准控制。"""

from __future__ import annotations

import numpy as np

from . import controller, mot, saliency, timing

__all__ = ["controller", "mot", "run", "saliency", "timing"]


def run(obs, roi, clock=None, gate="uncertainty", shifts=None, meas_var=None, selector_kw=None):
    """跑一整段：跟踪 → 显著性选择 → 平滑瞄准。

    `obs[k]` 是第 k 帧的检测列表（见 `mot.Detection`），`roi = (x, y, w, h)`。
    `shifts`（可选）是外部估计的逐帧背景位移。`meas_var` / `selector_kw` 传给跟踪器 / 选择器
    （只用于消融对照）。返回逐帧 `(阶段, 瞄准点, tid)`，瞄准点在画面系。
    """
    clk = clock or timing.Clock(len(obs), roi_w=roi[2])
    hist = mot.track(obs, roi, clock=clk, gate=gate, shifts=shifts, meas_var=meas_var)
    seq = mot.fill_gaps(mot.paths(hist), clk.frames_in(timing.FILL_SEC))
    selector, aim = saliency.Selector(clk, roi, **(selector_kw or {})), controller.Aim(clk)

    out, prev_cum = [], np.zeros(2)
    for k in range(len(obs)):
        cum = hist[k][2]
        bg = cum - prev_cum
        prev_cum = cum
        phase, target, tid, fresh = selector.step(k, hist, seq, bg)
        pos = aim.step(k, bg, cum, target, fresh)
        out.append((phase, pos, tid))
    return out
