"""核心链路在合成场景上的单元测试。

⚠️ 这些测试验证的是**实现正确**（背景补偿、排除法、时间归一化、控制平滑性），
不是方法的效果 —— 效果只能在真实数据集上评测（见 docs/PLAN.md）。
"""

from __future__ import annotations

import numpy as np
import pytest

from gimbal_vision import core
from gimbal_vision.core import controller, mot, saliency, timing
from gimbal_vision.synthetic import make_scene, subsample

HIT_R = 65.0
WARMUP_SEC = 1.0


def hit_ratio(out, scene, warmup_sec=WARMUP_SEC):
    """瞄准点落在目标真值 `HIT_R` 以内的帧数占比（跳过热身期）。"""
    hits = total = 0
    for k, (_, pos, _) in enumerate(out):
        if scene.ts[k] < warmup_sec:
            continue
        total += 1
        if pos is not None and np.hypot(*(np.array(pos) - scene.target[k])) <= HIT_R:
            hits += 1
    return hits / max(total, 1)


def clock_of(scene):
    return timing.Clock(len(scene.obs), ts=scene.ts, roi_w=scene.roi[2])


@pytest.fixture(scope="module")
def scene():
    return make_scene()


def test_background_motion_is_recovered_from_the_median_of_pairs(scene):
    hist = mot.track(scene.obs, scene.roi, clock=clock_of(scene))
    est = np.array([row[2] for row in hist])
    true = scene.cum - scene.cum[0]
    err = np.hypot(*(est[-1] - true[-1]))
    travelled = np.hypot(*true[-1])
    assert err < 0.05 * travelled, f"累计背景位移误差 {err:.1f}px / 走过 {travelled:.0f}px"


def test_the_independently_moving_target_is_followed(scene):
    out = core.run(scene.obs, scene.roi, clock=clock_of(scene))
    ratio = hit_ratio(out, scene)
    assert ratio > 0.9, f"命中率 {ratio:.1%}"


def test_without_background_compensation_the_wrong_object_is_chosen(scene):
    """对照组：把跟踪结果换回画面系（等于不做背景补偿），同一套判据应当失效。

    目标在画面系里几乎不动、干扰物一直在流动 ⇒ 不补偿就会挑中干扰物。
    这条保证上一条测试的高命中率确实来自背景补偿，而不是场景太简单。
    """
    clk = clock_of(scene)
    hist = mot.track(scene.obs, scene.roi, clock=clk)
    naive = [
        (k, {tid: (v[0] + cum[0], v[1] + cum[1], *v[2:]) for tid, v in d.items()}, np.zeros(2))
        for k, d, cum in hist
    ]
    seq = mot.fill_gaps(mot.paths(naive), clk.frames_in(timing.FILL_SEC))
    sel, aim = saliency.Selector(clk, scene.roi), controller.Aim(clk)
    out = []
    for k in range(len(naive)):
        phase, target, tid, fresh = sel.step(k, naive, seq, np.zeros(2))
        out.append((phase, aim.step(k, np.zeros(2), np.zeros(2), target, fresh), tid))
    assert hit_ratio(out, scene) < 0.5


def mean_hit(step, gate, seeds=range(5)):
    """多个随机种子上的平均命中率；`step` 是抽帧步长（2 = 15Hz）。"""
    rs = []
    for seed in seeds:
        low = subsample(make_scene(seed=seed), step)
        rs.append(hit_ratio(core.run(low.obs, low.roi, clock=clock_of(low), gate=gate), low))
    return float(np.mean(rs))


@pytest.mark.parametrize(("step", "floor"), [(2, 0.8), (3, 0.7)])
def test_uncertainty_gating_survives_a_lower_frame_rate(step, floor):
    ratio = mean_hit(step, "uncertainty")
    assert ratio > floor, f"{30 // step}Hz 命中率 {ratio:.1%}"


def test_the_speed_gate_breaks_at_a_lower_frame_rate():
    """对照组（docs/PLAN.md §6.1）：前期系统的「最大速度 × 帧间隔」闸在 15Hz 下失效。

    保证上一条的通过来自门控方式的改变，而不是场景在低帧率下本来就容易。
    """
    assert mean_hit(2, "speed") < 0.5


def test_both_gates_agree_at_the_reference_frame_rate(scene):
    """参考帧率下两种模型等价：改进只在帧率偏离参考值时起作用。"""
    clk = clock_of(scene)
    a = core.run(scene.obs, scene.roi, clock=clk, gate="uncertainty")
    b = core.run(scene.obs, scene.roi, clock=clk, gate="speed")
    assert hit_ratio(a, scene) == pytest.approx(hit_ratio(b, scene))


def test_the_aim_point_moves_with_bounded_acceleration(scene):
    """控制器自身的运动（扣掉估计的背景位移后）加速度不超过上限。

    ⚠️ 不能直接量画面系里的瞄准点：背景位移是逐帧**估计**出来的、带噪声，
    瞄准点每帧都叠加它 ⇒ 画面系里的加速度包含估计噪声，不是控制器的。
    """
    clk = clock_of(scene)
    out = core.run(scene.obs, scene.roi, clock=clk)
    cum = np.array([row[2] for row in mot.track(scene.obs, scene.roi, clock=clk)])
    own = np.array([np.array(p) - cum[k] for k, (_, p, _) in enumerate(out) if p is not None])
    vel = np.diff(own, axis=0) / clk.med_dt
    acc = np.hypot(*np.diff(vel, axis=0).T) / clk.med_dt
    limit = timing.AIM_ACC_PPS2 * clk.s
    assert acc.max() <= limit * 1.001, f"最大加速度 {acc.max():.0f} > 上限 {limit:.0f} px/s²"


def test_exclusion_is_keyed_to_positions_not_track_ids():
    ex = saliency.ExclusionMap(r=20.0)
    ex.mark(100.0, 100.0)
    assert ex.hit(110.0, 105.0), "同一位置附近的新轨迹（新 ID）也应被排除"
    assert not ex.hit(200.0, 100.0), "离开那个点就恢复资格"


def test_windows_are_cut_by_timestamps_not_frame_counts():
    ts = np.array([0.0, 0.1, 0.15, 0.2, 0.4, 0.45, 0.5])
    clk = timing.Clock(len(ts), ts=ts)
    lo, n = clk.window(6, 0.12)
    assert ts[lo] >= 0.5 - 0.12 and lo == 4 and n == 3


def test_the_gain_is_a_ratio_not_a_rate():
    """低帧率下单帧消掉的差距不能超过 100%（线性换算会过冲）。"""
    for hz in (30.0, 15.0, 8.0, 4.0):
        ts = np.arange(5) / hz
        g = timing.Clock(5, ts=ts).gain(timing.GAIN_KEEP, 2)
        assert 0.0 < g < 1.0, f"{hz}Hz 增益 {g}"


def test_frame_rate_guard_tolerates_float_error_at_the_boundary():
    """24 Hz 素材隔帧取出的 12 Hz 恰在下限上，不能因浮点误差（1/(2/24) = 11.999…）判为过低。"""
    ts = np.arange(60) * (2 / 24)
    clk = timing.Clock(len(ts), ts)
    assert clk.hz() < timing.HZ_MIN  # 浮点误差确实存在
    assert clk.ok()[0]
    slow = timing.Clock(40, np.arange(40) * (3 / 24))
    assert not slow.ok()[0]
