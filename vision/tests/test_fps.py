"""帧率 / 抖动实验的取帧规则（`scripts/eval_fps.py`）的单元测试 —— 实现验证，效果见 docs/EXPERIMENTS.md。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from gimbal_vision.core import timing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_fps import CONDITIONS, gap_source, pick_frames, select  # noqa: E402


def dups(n, every=7):
    d = np.zeros(n, bool)
    d[every::every] = True
    return d


def test_the_24hz_condition_keeps_exactly_the_non_duplicate_frames():
    d = dups(200)
    assert select("x", d, "r24") == np.flatnonzero(~d).tolist()


def test_uniform_downsampling_replaces_a_duplicate_with_its_nearest_real_neighbour():
    d = np.zeros(20, bool)
    d[6] = True  # 步长 2 的目标帧 6 是复制帧 ⇒ 取 5（等距取较早的）
    assert pick_frames(d, lambda: 2) == [0, 2, 4, 5, 8, 10, 12, 14, 16, 18]
    assert pick_frames(np.zeros(13, bool), lambda: 4) == [0, 4, 8, 12]


def test_selected_frames_are_increasing_and_never_duplicates():
    d = dups(500, 5)
    for c in CONDITIONS:
        k = select("seq", d, c)
        assert all(b > a for a, b in zip(k, k[1:], strict=False))
        assert not d[k].any()


def test_jitter_conditions_share_the_12hz_mean_rate_and_are_seeded():
    d = np.zeros(20000, bool)
    for c in ("j1", "j2", "j3"):
        k = select("seq", d, c)
        assert abs(np.mean(np.diff(k)) - 2.0) < 0.05, c
        assert k == select("seq", d, c)  # 同名同条件 ⇒ 同一串帧
        assert k != select("other", d, c)


def test_jitter_levels_are_ordered_by_the_clock_jitter_measure():
    d = np.zeros(5000, bool)
    jit = {}
    for c in ("r12", "j1", "j2", "j3"):
        k = np.array(select("seq", d, c))
        jit[c] = timing.Clock(len(k), k / 24.0).jitter()
    assert abs(jit["r12"] - 1.0) < 1e-9
    assert jit["r12"] < jit["j1"] < jit["j2"] < jit["j3"], jit


def test_the_geometric_gap_source_is_random_frame_dropping():
    rng = np.random.default_rng(0)
    g = gap_source({"kind": "geometric", "p": 0.5}, rng)
    x = np.array([g() for _ in range(20000)])
    assert x.min() == 1
    assert abs((x == 1).mean() - 0.5) < 0.02
