"""E2 修复与 E2b 协议组件的单元测试（实现验证，不是效果结论；效果见 docs/EXPERIMENTS.md）。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from gimbal_vision import stabilize
from gimbal_vision.candidates.motion import MotionCandidates
from gimbal_vision.core import mot, saliency, timing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calib_r import mad_sigma  # noqa: E402
from eval_e2 import Window, crop  # noqa: E402


def clock(n=200, w=640):
    return timing.Clock(n, np.arange(n) / 24.0, roi_w=w)


def texture(h, w, seed=0):
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (0, 0), 2.0)
    return cv2.normalize(t, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ── R：标定值与估计公式 ─────────────────────────────────────────
def test_tracks_use_the_calibrated_measurement_noise_unless_overridden():
    assert np.allclose(mot.Track(0, 1, 2, 3).R, np.diag(mot.MEAS_VAR))
    tk = mot.Tracker((0, 0, 640, 360), clock(), meas_var=9.0)
    tk.step(0, [(100.0, 100.0, 400.0, 90.0, 90.0, 110.0, 110.0, 1.0)], np.zeros(2))
    assert np.allclose(tk.tracks[0].R, np.diag([9.0, 9.0]))


def test_the_residual_difference_estimator_ignores_a_persistent_offset():
    """`calib_r` 的口径：残差的帧间差 ÷ √2。每段一个持续偏置（点云落在身体哪一块）不应进 R。"""
    rng = np.random.default_rng(1)
    sigma, diffs = 4.0, []
    for _ in range(200):
        e = rng.normal(0, 40) + rng.normal(0, sigma, 30)  # 段内偏置 σ=40，逐帧噪声 σ=4
        diffs += list(np.diff(e))
    est = mad_sigma(diffs) / np.sqrt(2)
    assert abs(est - sigma) < 0.3, est


# ── 印证中的远跳位置随背景平移 ──────────────────────────────────
@pytest.mark.parametrize("shift_chal", [True, False])
def test_the_challenger_moves_with_the_background_every_frame(shift_chal):
    sel = saliency.Selector(clock(), (0, 0, 640, 360), screenfix=False, stale=False, shift_chal=shift_chal)
    sel.pos, sel.chal, sel.streak = np.array([10.0, 10.0]), np.array([300.0, 200.0]), 2
    hist = [(0, {}, np.zeros(2))]  # 这一帧没有任何候选（wait）
    sel.step(0, hist, {}, np.array([5.0, -3.0]))
    assert np.allclose(sel.pos, [15.0, 7.0])
    want = [305.0, 197.0] if shift_chal else [300.0, 200.0]
    assert np.allclose(sel.chal, want), "chal 与 pos 同在画面系，必须每帧一起平移"


# ── 画面坐标排除 ─────────────────────────────────────────────────
def _scene(n, screen_fixed_obj, origin_step=0.0):
    """相机每帧右移 ⇒ 背景在画面里每帧 −4px。返回 hist, seq, origin。"""
    hist, seq, origin = [], {0: []}, {}
    for k in range(n):
        cum = np.array([-4.0 * k, 0.0])
        if screen_fixed_obj:
            bx, by = 300.0 - cum[0] - origin_step * k, 100.0  # 画面坐标 + 原点 恒定 → 素材上钉死
        else:
            bx, by = 300.0, 100.0  # 背景系不动 → 普通静止物体
        hist.append((k, {0: (bx, by, 0, 0, k, 0, 50.0)}, cum))
        seq[0].append((k, bx, by))
        origin[k] = np.array([origin_step * k, 0.0])
    return hist, seq, origin


def test_a_logo_fixed_on_screen_is_flagged_but_a_static_object_is_not():
    clk = clock()
    k = 60
    lo, need = clk.window(k, timing.STATIC_SEC)
    r = saliency.STATIC_R * clk.s
    hist, seq, org = _scene(k + 1, True)
    assert saliency.screen_fixed(seq, hist, org, k, 0, lo, need, r)
    hist, seq, org = _scene(k + 1, False)
    assert not saliency.screen_fixed(seq, hist, org, k, 0, lo, need, r)


def test_screen_coordinates_include_the_crop_origin():
    """闭环裁取景窗时，「画面」指原始素材：取景窗里不动、但取景窗本身在素材上移动的东西不算钉死。"""
    clk = clock()
    k = 60
    lo, need = clk.window(k, timing.STATIC_SEC)
    r = saliency.STATIC_R * clk.s
    hist, seq, _ = _scene(k + 1, True)  # 取景窗坐标里钉死
    moving_origin = {kk: np.array([3.0 * kk, 0.0]) for kk in range(k + 1)}
    assert not saliency.screen_fixed(seq, hist, moving_origin, k, 0, lo, need, r)
    hist, seq, org = _scene(k + 1, True, origin_step=3.0)  # 素材坐标里钉死
    assert saliency.screen_fixed(seq, hist, org, k, 0, lo, need, r)


# ── 过期撤回 ─────────────────────────────────────────────────────
@pytest.mark.parametrize("stale", [True, False])
def test_an_estimate_without_new_evidence_is_withdrawn(stale):
    clk = clock()
    sel = saliency.Selector(clk, (0, 0, 640, 360), screenfix=False, stale=stale)
    sel.pos, sel.last_fresh = np.array([100.0, 100.0]), 0
    k_ok = int(np.floor(timing.STALE_SEC * 24))  # 还没过期
    hist = [(kk, {}, np.zeros(2)) for kk in range(k_ok + 3)]
    assert sel.step(k_ok, hist, {}, np.zeros(2))[1] is not None
    out = sel.step(k_ok + 2, hist, {}, np.zeros(2))
    assert (out[1] is None) == stale


# ── 候选生成的有效像素掩码 ──────────────────────────────────────
def test_a_moving_black_border_is_not_motion_when_masked():
    """E2b 闭环：世界静止、素材边界在取景窗里移动。不给掩码，边界会被当成运动。"""
    world = texture(240, 320)
    gen_raw, gen_mask = MotionCandidates(320), MotionCandidates(320)
    n_raw = n_mask = 0
    for k in range(12):
        edge = 160 + 4 * k
        img = world.copy()
        img[:, edge:] = 0
        m = np.zeros_like(img)
        m[:, : edge - 12] = 255
        n_raw += len(gen_raw(img, k / 24.0).dets)
        n_mask += len(gen_mask(img, k / 24.0, m).dets)
    assert n_raw > 0, "对照：不加掩码时黑边应产生候选（否则这条测试没有检验力）"
    assert n_mask == 0


def test_crop_pads_outside_the_frame_and_marks_valid_pixels():
    img = np.full((100, 200), 7, np.uint8)
    im, m = crop(img, 150, -10, 80, 40, erode=0)
    assert im.shape == (40, 80) and m.shape == (40, 80)
    assert (im[10:, :50] == 7).all() and (im[:10] == 0).all() and (im[:, 50:] == 0).all()
    assert (m[10:, :50] == 255).all() and m[:10].sum() == 0
    im, m = crop(img, 10, 10, 50, 50, erode=12)
    assert m is None


# ── 稳定化 ───────────────────────────────────────────────────────
def test_accumulate_follows_the_sign_convention():
    """静止点在画面里 `p → p + t`，画布坐标 `p + o` 不变 ⇒ `o[k] = o[k−1] − t[k]`。"""
    o = stabilize.accumulate([None, np.array([2.0, 1.0]), None, np.array([-1.0, 0.0])])
    assert np.allclose(o, [[0, 0], [-2, -1], [-2, -1], [-1, -1]])


def test_the_stabilizer_recovers_a_pan():
    world = texture(400, 700, seed=3)
    pan = np.array([5.0, 2.0])  # 相机每帧右下移 ⇒ 背景在画面里左上移
    imgs = [world[int(2 * k) : int(2 * k) + 240, int(5 * k) : int(5 * k) + 320].copy() for k in range(10)]
    o, inl, fail = stabilize.offsets(imgs)
    assert fail == 0
    assert np.allclose(o[-1], 9 * pan, atol=1.0), o[-1]


# ── E2b 取景窗 ───────────────────────────────────────────────────
def test_the_e2b_window_stays_put_on_the_canvas_until_the_footage_leaves_it():
    win = Window(640, 360, 0.5, "e2b")
    assert win.rect() == (160, 90, 320, 180)
    assert not win.frame(np.array([100.0, 0.0]))  # 素材右移 100：窗在画布不动 ⇒ 画面里左移 100
    assert win.rect() == (60, 90, 320, 180)
    assert win.frame(np.array([-400.0, 0.0]))  # 素材覆盖画布 [-400, 240]，窗中心 320 已出界 ⇒ 被钳
    assert (win.cx, win.cy) == (240.0, 180.0)
    x, y, w, h = win.rect()
    assert x + w / 2 == pytest.approx(640, abs=1)  # 中心在画面右边界（素材范围内的最近点）
    win2 = Window(640, 360, 0.5, "e2")
    win2.set_center(0, 0)
    assert win2.rect() == (0, 0, 320, 180)
