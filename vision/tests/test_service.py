"""服务化：逐帧在线接口与离线 `core.run` 逐帧一致；服务经帧环取帧、经 WebSocket 回传的结果与离线一致。"""

from __future__ import annotations

import sys
import uuid

import cv2
import numpy as np
import pytest

from gimbal_vision import core, ring
from gimbal_vision.candidates.motion import MotionCandidates
from gimbal_vision.core import timing
from gimbal_vision.synthetic import make_scene, render_camouflage


def same(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol, rtol=0))


def online(obs, roi, clock, shifts=None, **kw):
    pipe = core.Pipeline(roi, clock, **kw)
    return [pipe.step(k, d, None if shifts is None else shifts[k]) for k, d in enumerate(obs)]


@pytest.mark.parametrize("gate", ["uncertainty", "speed"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_online_pipeline_matches_the_offline_run_frame_by_frame(gate, seed):
    """同一个时钟（含不均匀帧间隔）下，`Pipeline.step` 逐帧输出与 `core.run` 完全相同。"""
    sc = make_scene(n=150, seed=seed)
    rng = np.random.default_rng(seed)
    ts = np.cumsum(rng.uniform(0.6, 1.6, len(sc.obs)) / 30.0)  # 抖动的帧间隔
    clk = timing.Clock(len(sc.obs), ts, roi_w=sc.roi[2])
    ref = core.run(sc.obs, sc.roi, clock=clk, gate=gate)
    got = online(sc.obs, sc.roi, timing.Clock(len(sc.obs), ts, roi_w=sc.roi[2]), gate=gate)
    assert len(got) == len(ref)
    for k, (r, g) in enumerate(zip(ref, got, strict=True)):
        assert r[0] == g.phase and r[2] == g.tid, f"第 {k} 帧"
        assert same(r[1], g.aim), f"第 {k} 帧 瞄准点 {r[1]} != {g.aim}"


def test_external_background_shifts_are_honoured_the_same_way():
    sc = make_scene(n=120, seed=3)
    shifts = [np.array([-5.0, -1.5])] * len(sc.obs)
    clk = timing.Clock(len(sc.obs), sc.ts, roi_w=sc.roi[2])
    ref = core.run(sc.obs, sc.roi, clock=clk, shifts=shifts)
    got = online(sc.obs, sc.roi, timing.Clock(len(sc.obs), sc.ts, roi_w=sc.roi[2]), shifts=shifts)
    assert all(same(r[1], g.aim) and r[2] == g.tid for r, g in zip(ref, got, strict=True))


def test_with_a_growing_online_clock_uniform_timestamps_give_identical_results():
    """服务里用的是 `OnlineClock`（帧到达时 push 时间戳，不看未来）；帧间隔均匀时与离线逐帧相同。"""
    sc = make_scene(n=150, seed=4, fps=24.0)
    ref = core.run(sc.obs, sc.roi, clock=timing.Clock(len(sc.obs), sc.ts, roi_w=sc.roi[2]))
    clk = timing.OnlineClock(roi_w=sc.roi[2])
    pipe = core.Pipeline(sc.roi, clk)
    for k, d in enumerate(sc.obs):
        assert clk.push(sc.ts[k]) == k
        g = pipe.step(k, d)
        assert same(ref[k][1], g.aim) and ref[k][2] == g.tid, f"第 {k} 帧"
    assert clk.hz() == pytest.approx(24.0)


def test_the_online_clock_follows_the_frame_rate_causally():
    clk = timing.OnlineClock(roi_w=640)
    for k in range(20):
        clk.push(k / 30)
    assert clk.frames_in(1.0) == 30
    for k in range(20, 600):
        clk.push(20 / 30 + (k - 20) / 10)  # 掉到 10Hz
    assert clk.frames_in(1.0) == 10 and clk.ok()[0] is False
    with pytest.raises(ValueError):
        clk.push(0.0)


# ── 服务：帧环 → 处理 → WebSocket ─────────────────────────────────
def ring_name():
    base = f"mg-svc-{uuid.uuid4().hex[:12]}"
    return f"Local\\{base}" if sys.platform == "win32" else base


@pytest.fixture(scope="module")
def clip():
    return render_camouflage(n=90)


def offline_reference(clip):
    """离线口径：同样的灰度帧 → MotionCandidates → core.run（时钟用整段时间戳）。"""
    h, w = clip.frames[0].shape
    gen = MotionCandidates(w)
    obs, shifts = [], []
    for img, t in zip(clip.frames, clip.ts, strict=True):
        r = gen(img, t)
        obs.append(r.dets)
        shifts.append(np.zeros(2) if r.bg_shift is None else r.bg_shift)
    return core.run(obs, (0, 0, w, h), clock=timing.Clock(len(obs), clip.ts, roi_w=w), shifts=shifts)


def test_the_service_reads_the_ring_and_streams_the_same_aims_as_offline(clip):
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from gimbal_vision.service import create_app  # noqa: PLC0415

    h, w = clip.frames[0].shape
    name = ring_name()
    wr = ring.RingWriter(name, slots=4, capacity=w * h * 3, mode=ring.BLOCK)
    app = create_app()
    rgb = [cv2.cvtColor(f, cv2.COLOR_GRAY2RGB) for f in clip.frames]
    with TestClient(app) as c, c.websocket_connect("/results") as ws:
        r = c.post(
            "/session/start",
            json={"ring": name, "session": "t1", "params": {"perception": "full", "proc_width": 0, "checksum": True}},
        )
        assert r.status_code == 200, r.text
        got = []
        for k, img in enumerate(rgb):
            assert wr.write(k, float(clip.ts[k]), img, roi=(0, 0, w, h), timeout=5) == k
            msg = ws.receive_json()  # 闭环口径：一帧一结果
            assert msg["session"] == "t1" and msg["index"] == k and msg["frame"] == k
            got.append(msg)
        st = c.get("/status").json()
        assert st["frames"] == len(rgb) and st["error"] is None
        c.post("/session/stop")
    wr.close()

    import zlib  # noqa: PLC0415

    ref = offline_reference(clip)
    for k, (r, g) in enumerate(zip(ref, got, strict=True)):
        assert g["crc32"] == zlib.crc32(rgb[k].tobytes()), "帧环传过去的像素必须逐字节一致"
        assert g["t"] == pytest.approx(clip.ts[k])
        assert r[0] == g["phase"] and r[2] == g["tid"], f"第 {k} 帧"
        assert same(r[1], g["aim"], 1e-6), f"第 {k} 帧 {r[1]} != {g['aim']}"
    # 目标确实被跟住（实现验证，不是效果结论）：后半段瞄准点离真值不远
    err = [np.hypot(*(np.array(g["aim"]) - clip.target[k])) for k, g in enumerate(got) if k > 45 and g["aim"]]
    assert err and np.median(err) < 3 * clip.radius


def test_bad_requests_are_rejected():
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from gimbal_vision.service import create_app  # noqa: PLC0415

    with TestClient(create_app()) as c:
        assert c.get("/health").json() == {"ok": True}
        assert c.post("/session/start", json={"ring": ring_name()}).status_code == 400  # 没有这个帧环
        assert c.put("/params", json={"nope": 1}).status_code == 400
        assert c.put("/params", json={"perception": "full"}).json()["perception"] == "full"
