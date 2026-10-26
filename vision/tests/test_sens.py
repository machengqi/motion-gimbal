"""背景模型对照（单应传播的排除图）与评测脚本组件的单元测试（实现验证，不是效果结论）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from gimbal_vision.core import controller, mot, saliency, timing
from gimbal_vision.synthetic import make_scene

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sweep_e2 import follow_canvas, follow_warp, merge, patched  # noqa: E402


def _run(scene, shifts, warps):
    clk = timing.Clock(len(scene.obs), ts=scene.ts, roi_w=scene.roi[2])
    hist = mot.track(scene.obs, scene.roi, clock=clk, shifts=shifts)
    seq = mot.fill_gaps(mot.paths(hist), clk.frames_in(timing.FILL_SEC))
    sel = saliency.Selector(clk, scene.roi, warps=warps)
    aim = controller.Aim(clk)
    out, prev = [], np.zeros(2)
    for k in range(len(scene.obs)):
        cum = hist[k][2]
        bg = cum - prev
        prev = cum
        phase, target, tid, fresh = sel.step(k, hist, seq, bg)
        out.append((phase, aim.step(k, bg, cum, target, fresh), tid, tuple(sel.last_excluded)))
    return out, sel.stats


def test_translation_warps_reproduce_the_default():
    """平移矩阵当单应传给排除图 ⇒ 与默认（背景系平移模型）逐帧一致，且排除图确实被用到。"""
    scene = make_scene(n=150, sway_every=1000)  # 几乎全是静止干扰物 ⇒ 排除图会标很多点
    shifts = [np.zeros(2)] + list(np.diff(scene.cum, axis=0))
    warps = [None] + [np.array([[1, 0, s[0]], [0, 1, s[1]], [0, 0, 1.0]]) for s in shifts[1:]]
    a, sa = _run(scene, shifts, None)
    b, sb = _run(scene, shifts, warps)
    assert sa["mark_new"] > 0 and sa["excluded"] > 0
    assert sa == sb
    for (pa, xa, ta, ea), (pb, xb, tb, eb) in zip(a, b, strict=True):
        assert pa == pb and ta == tb and ea == eb
        assert (xa is None) == (xb is None)
        if xa is not None:
            assert np.allclose(xa, xb, atol=1e-6)


def test_exclusion_points_follow_a_zoom_when_warped():
    """变焦时离中心远的静止点在画面里外移；单应传播的排除点跟着走，平移模型跟不上。"""
    ex = saliency.ExclusionMap(r=10.0)
    ex.mark(600.0, 300.0)
    c = np.array([320.0, 180.0])
    z = 1.05  # 每帧放大 5%
    Z = np.array([[z, 0, c[0] * (1 - z)], [0, z, c[1] * (1 - z)], [0, 0, 1.0]])
    p = np.array([600.0, 300.0])
    for _ in range(5):
        ex.warp(Z)
        p = c + z * (p - c)
    assert ex.hit(*p)
    assert np.hypot(*(p - [600.0, 300.0])) > 10.0, "平移模型（画面中心不动 ⇒ 位移 0）会把它放在原处"


def test_the_warp_follower_matches_the_e2b_window_for_translations():
    rng = np.random.default_rng(0)
    offs = np.cumsum(rng.normal(0, 8, (60, 2)), axis=0)
    offs[0] = 0
    pts = [None] * 20 + [tuple(rng.uniform(0, 300, 2)) for _ in range(10)] + [None] * 30
    t = [None] + [np.array([[1, 0, offs[i - 1][0] - offs[i][0]], [0, 1, offs[i - 1][1] - offs[i][1]], [0, 0, 1.0]])
                  for i in range(1, 60)]
    assert follow_canvas(pts, offs, 640, 360, 0.5) == follow_warp(pts, t, 640, 360, 0.5)


def test_patched_constants_are_restored_and_static_drags_stale():
    before = (timing.STATIC_SEC, timing.STALE_SEC, saliency.LEAD)
    with patched({"timing": {"STATIC_SEC": 1.0}, "sal": {"LEAD": 2.0}}):
        assert timing.STATIC_SEC == timing.STALE_SEC == 1.0 and saliency.LEAD == 2.0
    assert (timing.STATIC_SEC, timing.STALE_SEC, saliency.LEAD) == before
    assert merge({"timing": {"A": 1}}, {"timing": {"B": 2}, "sal": {"C": 3}}) == {
        "timing": {"A": 1, "B": 2}, "sal": {"C": 3}}
