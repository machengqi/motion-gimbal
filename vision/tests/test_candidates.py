"""运动候选生成在伪装合成场景上的测试（实现验证，不是效果结论）。"""

from __future__ import annotations

import numpy as np
import pytest

from gimbal_vision import core
from gimbal_vision.candidates.motion import MotionCandidates
from gimbal_vision.core import timing
from gimbal_vision.synthetic import render_camouflage


@pytest.fixture(scope="module")
def video():
    return render_camouflage()


@pytest.fixture(scope="module")
def results(video):
    gen = MotionCandidates(video.frames[0].shape[1])
    return [gen(f, t) for f, t in zip(video.frames, video.ts, strict=True)]


def contains(det, p):
    return det[3] <= p[0] <= det[5] and det[4] <= p[1] <= det[6]


TIGHT = 9.0
"""「框住了目标」还要求框面积不超过目标外接方框的这么多倍。

⛔ 没有这一条，召回判据会被一个盖住整幅画面的大框骗过：关掉相机运动补偿时
全部特征点都在「动」，聚成一个大框，目标当然在框里 —— 召回反而是 100%。
"""


def finds(det, video, k):
    side = 2 * video.radius
    return contains(det, video.target[k]) and det[2] <= TIGHT * side * side


def test_camera_motion_comes_from_the_whole_frame(video, results):
    true = np.diff(video.cam, axis=0) * -1  # 相机右移 ⇒ 背景在画面里左移
    est = np.array([r.bg_shift for r in results[1:]])
    err = np.hypot(*(est - true).T)
    assert np.median(err) < 0.5, f"背景位移中位误差 {np.median(err):.2f}px"


def test_the_camouflaged_target_is_found_by_motion_alone(video, results):
    found = [any(finds(d, video, k) for d in r.dets) for k, r in enumerate(results) if k > 0]
    assert np.mean(found) > 0.8, f"候选召回 {np.mean(found):.1%}"


def test_the_static_background_yields_few_false_candidates(video, results):
    false = [
        sum(not finds(d, video, k) for d in r.dets) for k, r in enumerate(results) if k > 0
    ]
    assert np.mean(false) < 1.0, f"每帧误报 {np.mean(false):.2f}"


def test_end_to_end_with_motion_candidates(video, results):
    obs = [r.dets for r in results]
    shifts = [np.zeros(2) if r.bg_shift is None else r.bg_shift for r in results]
    h, w = video.frames[0].shape
    clk = timing.Clock(len(obs), ts=video.ts, roi_w=w)
    out = core.run(obs, (0, 0, w, h), clock=clk, shifts=shifts)
    hits = [
        pos is not None and np.hypot(*(np.array(pos) - video.target[k])) <= 2 * video.radius
        for k, (_, pos, _) in enumerate(out)
        if video.ts[k] >= 1.0
    ]
    assert np.mean(hits) > 0.8, f"端到端命中 {np.mean(hits):.1%}"
