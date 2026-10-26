"""与具体应用无关的核心：时钟、多目标跟踪、运动显著性选择、平滑瞄准控制。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import controller, mot, saliency, timing

__all__ = ["Pipeline", "Step", "controller", "mot", "run", "saliency", "timing"]


def run(obs, roi, clock=None, gate="uncertainty", shifts=None, meas_var=None, selector_kw=None, aim_kw=None):
    """跑一整段：跟踪 → 显著性选择 → 平滑瞄准。

    `obs[k]` 是第 k 帧的检测列表（见 `mot.Detection`），`roi = (x, y, w, h)`。
    `shifts`（可选）是外部估计的逐帧背景位移。`meas_var` / `selector_kw` 传给跟踪器 / 选择器
    （只用于消融对照），`aim_kw` 传给控制器（同）。返回逐帧 `(阶段, 瞄准点, tid)`，瞄准点在画面系。
    """
    clk = clock or timing.Clock(len(obs), roi_w=roi[2])
    hist = mot.track(obs, roi, clock=clk, gate=gate, shifts=shifts, meas_var=meas_var)
    seq = mot.fill_gaps(mot.paths(hist), clk.frames_in(timing.FILL_SEC))
    selector, aim = saliency.Selector(clk, roi, **(selector_kw or {})), controller.Aim(clk, **(aim_kw or {}))

    out, prev_cum = [], np.zeros(2)
    for k in range(len(obs)):
        cum = hist[k][2]
        bg = cum - prev_cum
        prev_cum = cum
        phase, target, tid, fresh = selector.step(k, hist, seq, bg)
        pos = aim.step(k, bg, cum, target, fresh)
        out.append((phase, pos, tid))
    return out


@dataclass
class Step:
    """[`Pipeline.step`] 的一帧输出。坐标都在**画面系**（这一帧送进来的画面）。"""

    k: int
    phase: str
    """`"wait"` / `"track"` / `"jump"`，见 `saliency.Selector.step`。"""
    aim: tuple[float, float] | None
    """平滑后的瞄准点 —— 交给云台的量。"""
    target: tuple[float, float] | None
    """选择器的估计位置（未平滑）。"""
    tid: int | None
    fresh: bool
    bg: np.ndarray
    """这一帧的背景位移。"""
    cum: np.ndarray
    """累计背景位移（背景系 + cum = 画面系）。"""
    row: tuple
    """跟踪器这一帧的原始输出 `(k, {tid: ...}, cum)`，见 `mot.track`。"""


class Pipeline:
    """**逐帧在线**的 跟踪 → 显著性选择 → 平滑瞄准，与 [`run`] 用同一组组件、同一组参数。

    ```python
    clk = timing.OnlineClock(roi_w=640)
    pipe = Pipeline(roi, clk)
    for dets, shift, t in stream:
        k = clk.push(t)
        s = pipe.step(k, dets, shift)   # s.aim 交给云台
    ```

    与 [`run`] 的唯一区别是断档回填：`run` 先跑完整段再回填，这里每来一个真实观测就补上它与上一次
    观测之间的空档 —— 回填只在两次真实观测之间插值，而选择器只评估这一帧有观测的轨迹，所以两者逐帧
    相同（同一个时钟下；见 tests/test_service.py）。时钟可以是离线 `timing.Clock`（复现实验），也可以是
    `timing.OnlineClock`（服务里用，帧到达时 `push` 时间戳）。
    """

    def __init__(self, roi, clock, gate="uncertainty", meas_var=None, selector_kw=None):
        self.clk = clock
        self.roi = roi
        self.tk = mot.Tracker(roi, clock, gate=gate, meas_var=meas_var)
        self.sel = saliency.Selector(clock, roi, **(selector_kw or {}))
        self.aimer = controller.Aim(clock)
        self.filled: dict[int, list] = {}
        """`tid -> [(帧号, 背景系x, 背景系y)]`，真实观测 + 两次观测之间的回填（即 `run` 里的 `seq`）。"""
        self._last: dict[int, tuple] = {}
        self.prev_cum = np.zeros(2)

    def _observe(self, tid, k, x, y):
        pts = self.filled.setdefault(tid, [])
        last = self._last.get(tid)
        if last is not None:
            k0, x0, y0 = last
            gap = k - k0
            if 1 < gap <= self.clk.frames_in(timing.FILL_SEC) + 1:
                for j in range(1, gap):
                    t = j / gap
                    pts.append((k0 + j, x0 + t * (x - x0), y0 + t * (y - y0)))
        pts.append((k, x, y))
        self._last[tid] = (k, x, y)

    def step(self, k, dets, shift=None, origin=None) -> Step:
        """走一帧。`dets` 见 `mot.Detection`；`shift` 是外部估计的背景位移（`None` = 跟踪器自己估）；
        `origin` 是这一帧画面左上角在原始素材里的坐标（取景窗模式下给，供画面坐标排除用）。"""
        row = self.tk.step(k, dets, shift)
        for tid, v in row[1].items():
            if v[5] == 0:
                self._observe(tid, k, v[0], v[1])
        cum = row[2]
        bg = cum - self.prev_cum
        self.prev_cum = cum
        phase, target, tid, fresh = self.sel.step(k, self.tk.history, self.filled, bg, origin)
        pos = self.aimer.step(k, bg, cum, target, fresh)
        return Step(k, phase, pos, target, tid, fresh, bg, cum, row)
