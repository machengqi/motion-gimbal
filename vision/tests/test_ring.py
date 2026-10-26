"""共享内存帧环：读写正确性、半帧、覆盖、背压，以及跨进程并发下不读到撕裂的帧。

写者在这里用 Python 实现（`ring.RingWriter`，只供测试）；宿主的 Rust 写者另有 `src-tauri/tests/ring.rs`，
跨语言的联调在 `src-tauri/tests/ipc.rs`。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import uuid

import numpy as np
import pytest

from gimbal_vision import ring


def name():
    base = f"mg-test-{uuid.uuid4().hex[:12]}"
    return f"Local\\{base}" if sys.platform == "win32" else base


def frame(v, w=64, h=48):
    return np.full((h, w, 3), v % 251, np.uint8)


def test_a_frame_round_trips_with_its_header():
    n = name()
    wr = ring.RingWriter(n, slots=3, capacity=64 * 48 * 3)
    rd = ring.RingReader(n)
    img = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    wr.write(7, 0.25, img, roi=(1, 2, 30, 20))
    f = rd.next(timeout=1)
    assert (f.index, f.frame_no, f.t, f.width, f.height, f.fmt, f.roi) == (0, 7, 0.25, 64, 48, ring.RGB24, (1, 2, 30, 20))
    assert np.array_equal(f.pixels, img)
    assert rd.read_count == 1
    rd.close()
    wr.close()


def test_a_half_written_frame_is_never_returned():
    """写者把 seq 置为奇数、只写了一半像素时，读者必须看到「还没写完」，而不是半帧。"""
    n = name()
    wr = ring.RingWriter(n, slots=2, capacity=64 * 48 * 3)
    rd = ring.RingReader(n)
    wr.write(0, 0.0, frame(1))
    assert isinstance(rd.next(timeout=1), ring.Frame)
    wr.begin(1)
    wr.fill(1, 1, 0.1, frame(2), upto=64 * 48 * 3 // 2)
    assert rd.try_read(1) == "pending"
    assert rd.next(timeout=0.05) is None
    wr.fill(1, 1, 0.1, frame(2))
    wr.commit(1)
    f = rd.next(timeout=1)
    assert f.frame_no == 1 and np.all(f.pixels == 2)
    rd.close()
    wr.close()


def test_a_slot_rewritten_during_the_copy_is_rejected():
    """拷贝前后 seq 不一致 ⇒ 丢弃（模拟：读完槽头后写者开始改写同一个槽）。"""
    n = name()
    wr = ring.RingWriter(n, slots=2, capacity=64 * 48 * 3, mode=ring.OVERWRITE)
    rd = ring.RingReader(n)
    wr.write(0, 0.0, frame(1))
    orig = rd.seqs[0]

    class Flip:
        """第一次读返回旧 seq，之后返回「正在写第 2 帧」的 seq —— 正是拷贝期间被套圈的情形。"""

        def __init__(self):
            self.n = 0

        def __getitem__(self, i):
            self.n += 1
            return np.uint64(2) if self.n == 1 else np.uint64(5)

    rd.seqs[0] = Flip()
    assert rd.try_read(0) == "torn"
    assert rd.torn == 1
    rd.seqs[0] = orig
    rd.close()
    wr.close()


def test_overwrite_mode_skips_to_the_newest_frame_and_counts_the_gap():
    n = name()
    wr = ring.RingWriter(n, slots=3, capacity=64 * 48 * 3, mode=ring.OVERWRITE)
    rd = ring.RingReader(n)
    for k in range(10):
        assert wr.write(k, k / 30, frame(k)) == k  # 实时源从不等待
    assert rd.try_read(0) == "overwritten"
    f = rd.next(timeout=1)
    assert f.frame_no == 9 and f.skipped == 9 and np.all(f.pixels == 9)
    assert f.t == pytest.approx(9 / 30)  # 时间戳是真实的 ⇒ 跳过的帧表现为帧间隔变大
    rd.close()
    wr.close()


def test_block_mode_waits_when_the_ring_is_full():
    n = name()
    wr = ring.RingWriter(n, slots=2, capacity=64 * 48 * 3, mode=ring.BLOCK)
    rd = ring.RingReader(n)
    assert wr.write(0, 0.0, frame(0)) == 0
    assert wr.write(1, 0.1, frame(1)) == 1
    assert wr.write(2, 0.2, frame(2), timeout=0.05) is None  # 环满：回放时掉速，不覆盖
    assert rd.next(timeout=1).frame_no == 0
    assert wr.write(2, 0.2, frame(2), timeout=0.05) == 2
    assert [rd.next(timeout=1).frame_no for _ in range(2)] == [1, 2]
    rd.close()
    wr.close()


WRITER = textwrap.dedent(
    """
    import sys, numpy as np
    from gimbal_vision import ring
    name, n = sys.argv[1], int(sys.argv[2])
    w, h = 512, 512
    wr = ring.RingWriter(name, slots=2, capacity=w * h * 3, mode=ring.OVERWRITE)
    print("ready", flush=True)
    sys.stdin.readline()
    for k in range(n):
        wr.write(k, k / 1000, np.full((h, w, 3), k % 251, np.uint8), roi=(k % 7, k % 5, 3, 3))
    wr.close_stream()
    print("done", flush=True)
    sys.stdin.readline()
    """
)


def test_concurrent_writer_in_another_process_never_yields_a_torn_frame():
    """写者在另一个进程里全速覆盖写（只有 2 个槽），读者逐帧校验：像素全等于 `帧号 % 251`、槽头自洽。"""
    n = name()
    p = subprocess.Popen(
        [sys.executable, "-c", WRITER, n, "3000"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        assert p.stdout.readline().strip() == "ready"
        rd = ring.RingReader(n)
        p.stdin.write("go\n")
        p.stdin.flush()
        got, last = 0, -1
        while True:
            f = rd.next(timeout=10)
            if f is None:
                break
            got += 1
            v = f.frame_no % 251
            assert f.frame_no > last
            assert f.pixels.min() == v and f.pixels.max() == v, f"第 {f.frame_no} 帧像素不一致（半帧）"
            assert f.roi == (f.frame_no % 7, f.frame_no % 5, 3, 3)
            assert f.t == pytest.approx(f.frame_no / 1000)
            last = f.frame_no
        assert got > 0 and last == 2999
        print(f"读到 {got} 帧，跳过 {rd.skipped}，序列锁拒绝 {rd.torn} 次")
        rd.close()
    finally:
        p.stdin.write("bye\n")
        p.stdin.flush()
        p.wait(timeout=10)
