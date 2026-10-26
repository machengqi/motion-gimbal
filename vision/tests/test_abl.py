"""三项消融开关（排除名单挂位置/编号/不用、净位移/路程、限加速度/限速度）与编号切换注入的单元测试。

只验证实现（默认行为逐位不变、开关确实生效），不是效果结论。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

from gimbal_vision import core
from gimbal_vision.core import controller, mot, saliency, timing
from gimbal_vision.synthetic import make_scene

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_abl import inject_id_switches, kinematics  # noqa: E402

GOLDEN = {
    # 加开关之前（thesis-int e02c4be）的 core.run 输出摘要：(默认场景, 几乎全是静止干扰物的场景)
    0: ("7991ead402476f9d", "ec1a21f40ff24773"),
    1: ("7843d96796ca6ec4", "52acb761db4b057a"),
    2: ("f1dcc69d91ff368c", "0ee70cc1f75388d8"),
}


def _digest(seed, selector_kw=None, aim_kw=None, **kw):
    sc = make_scene(seed=seed, **kw)
    clk = timing.Clock(len(sc.obs), ts=sc.ts, roi_w=sc.roi[2])
    shifts = [np.zeros(2)] + list(np.diff(sc.cum, axis=0))
    out = core.run(sc.obs, sc.roi, clock=clk, shifts=shifts, selector_kw=selector_kw, aim_kw=aim_kw)
    h = hashlib.sha256()
    for ph, pos, tid in out:
        h.update(repr((ph, None if pos is None else tuple(float(x) for x in pos), tid)).encode())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_defaults_are_bitwise_unchanged(seed):
    """默认参数、以及显式写出默认值，都与加开关之前的输出逐位相同。"""
    a, b = GOLDEN[seed]
    assert _digest(seed) == a
    assert _digest(seed, sway_every=1000, n=150) == b
    explicit = {"excl": "pos", "disp": "net"}
    assert _digest(seed, selector_kw=explicit, aim_kw={"limit": "acc"}) == a
    assert _digest(seed, selector_kw=explicit, aim_kw={"limit": "acc"}, sway_every=1000, n=150) == b


def test_switches_change_the_output():
    """三个开关确实生效（在合成场景上与默认输出不同）。"""
    a = GOLDEN[0][1]
    kw = dict(sway_every=1000, n=150)
    assert _digest(0, selector_kw={"excl": "none"}, **kw) != a
    assert _digest(0, selector_kw={"disp": "path"}) != GOLDEN[0][0]
    assert _digest(0, aim_kw={"limit": "vel"}) != GOLDEN[0][0]


def test_bad_modes_are_rejected():
    clk = timing.Clock(10)
    with pytest.raises(ValueError):
        saliency.Selector(clk, (0, 0, 640, 360), excl="tid")
    with pytest.raises(ValueError):
        saliency.Selector(clk, (0, 0, 640, 360), disp="len")
    with pytest.raises(ValueError):
        saliency.Selector(clk, (0, 0, 640, 360), warps=[None] * 10, excl="id")
    with pytest.raises(ValueError):
        controller.Aim(clk, limit="jerk")


def test_path_length_vs_net_displacement():
    """直线：路程 = 净位移；来回：路程大、净位移小；有效性条件相同。"""
    line = {1: [(k, 2.0 * k, 0.0) for k in range(20)]}
    assert saliency.path_len(line, 19, 1, 5, 15) == pytest.approx(saliency.net_disp(line, 19, 1, 5, 15))
    zig = {1: [(k, 5.0 * (k % 2), 0.0) for k in range(20)]}
    assert saliency.path_len(zig, 19, 1, 5, 15) == pytest.approx(70.0)
    assert saliency.net_disp(zig, 19, 1, 5, 15) == pytest.approx(0.0)  # 首尾都在 x=5
    short = {1: [(k, float(k), 0.0) for k in range(15, 20)]}
    assert saliency.path_len(short, 19, 1, 5, 15) is None
    assert saliency.net_disp(short, 19, 1, 5, 15) is None


def _select(sc, hist, clk, **kw):
    seq = mot.fill_gaps(mot.paths(hist), clk.frames_in(timing.FILL_SEC))
    sel = saliency.Selector(clk, sc.roi, **kw)
    prev = np.zeros(2)
    for k in range(len(hist)):
        cum = hist[k][2]
        sel.step(k, hist, seq, cum - prev)
        prev = cum
    return sel


def test_id_list_forgets_after_switches_position_list_does_not():
    """几乎全是静止干扰物：注入高频编号切换后，挂编号的名单排除量大降，挂位置的名单不降（相对无切换）。"""
    sc = make_scene(n=150, sway_every=1000)
    clk = timing.Clock(len(sc.obs), ts=sc.ts, roi_w=sc.roi[2])
    shifts = [np.zeros(2)] + list(np.diff(sc.cum, axis=0))
    hist = mot.track(sc.obs, sc.roi, clock=clk, shifts=shifts)
    sw, n = inject_id_switches(hist, clk, 2.0, 7)
    assert n > 0
    pos0, id0 = _select(sc, hist, clk), _select(sc, hist, clk, excl="id")
    pos1, id1 = _select(sc, sw, clk), _select(sc, sw, clk, excl="id")
    none = _select(sc, hist, clk, excl="none")
    assert none.stats["excluded"] == 0 and none.stats["mark_new"] == 0
    assert id0.stats["excluded"] > 0 and id0.banned_ids
    assert id1.stats["excluded"] < 0.6 * id0.stats["excluded"]
    assert pos1.stats["excluded"] > 0.8 * pos0.stats["excluded"]


def test_injection_only_relabels():
    sc = make_scene(n=90)
    clk = timing.Clock(len(sc.obs), ts=sc.ts, roi_w=sc.roi[2])
    hist = mot.track(sc.obs, sc.roi, clock=clk)
    same, n0 = inject_id_switches(hist, clk, 0.0, 1)
    assert same is hist and n0 == 0
    a, na = inject_id_switches(hist, clk, 1.0, 1)
    b, nb = inject_id_switches(hist, clk, 1.0, 1)
    c, nc = inject_id_switches(hist, clk, 4.0, 1)
    assert [(k, r) for k, r, _ in a] == [(k, r) for k, r, _ in b] and na == nb > 0 and nc > na
    for (k0, r0, c0), (k1, r1, c1) in zip(hist, c, strict=True):
        assert k0 == k1 and np.array_equal(c0, c1)
        assert sorted(r0.values()) == sorted(r1.values())


def test_velocity_only_controller_respects_speed_but_not_acceleration():
    clk = timing.Clock(40)
    acc, vel = controller.Aim(clk), controller.Aim(clk, limit="vel")
    pa, pv = [], []
    for k in range(40):
        tgt = (0.0, 0.0) if k < 5 else (300.0, 0.0)
        pa.append(acc.step(k, np.zeros(2), np.zeros(2), tgt, True))
        pv.append(vel.step(k, np.zeros(2), np.zeros(2), tgt, True))
    t = clk.t
    _, a_acc, _ = kinematics(np.array(pa), t, 0)
    _, a_vel, _ = kinematics(np.array(pv), t, 0)
    lim = timing.AIM_ACC_PPS2 * clk.s
    assert a_acc.max() <= lim * 1.001 + 1e-6
    assert a_vel.max() > 2 * lim
    vmax = timing.AIM_MAX_PPS * clk.s
    assert np.max(np.hypot(*np.diff(np.array(pv), axis=0).T)) / clk.med_dt <= vmax * 1.001
