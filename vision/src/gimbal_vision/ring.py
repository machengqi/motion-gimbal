"""**共享内存帧环** —— Rust 宿主写帧、视觉服务读帧（数据面）。布局与 `src-tauri/src/ring.rs` 逐字节一致。

```text
总头 128 B   magic "MGFR" | 版本 | 槽数 N | 总头长 | 槽头长 | 保留 | 每槽像素容量
             | 已发布帧数 write_count | 已消费帧数 read_count | 模式 | 已关闭
槽 i         槽头 64 B：写入序号 seq | 帧号 | 时间戳 | 宽 | 高 | 行跨度 | 像素格式 | 像素字节数
                        | 取景窗 x, y, w, h
             像素        RGB24 / GRAY8，按行紧排
```

第 `i` 帧（从 0 数的**写入序号**）放在槽 `i % N`。

# 序列锁：读者绝不会拿到半帧

写者：`seq ← 2i+1`（奇数 = 正在写）→ 写槽头与像素 → `seq ← 2i+2`（偶数 = 写完）→ `write_count ← i+1`。
读者：读 `seq`，不是 `2i+2` 就不读（奇数 = 还在写，更大 = 已被覆盖）→ 拷出槽头与像素 → 再读 `seq`，
前后不等 ⇒ 拷的过程中被改写了，丢弃。`seq` 同时编码了写入序号，读者据此区分「还没写到」与「已被覆盖」。

⚠️ Python 没有内存屏障原语；这里依赖 x86-64 的 TSO（读读不重排、写写不重排）与 8 字节对齐访问的原子性。
写者一侧（Rust）用 `AtomicU64` + Release/Acquire，不依赖平台。

# 两种模式（总头的 `mode`）

- `BLOCK`（视频回放）：环满（`write_count − read_count == N`）时写者等待 ⇒ 掉速而不丢帧，与播放循环的
  「掉速而不是丢帧」一致。
- `OVERWRITE`（实时源）：写者从不等待，覆盖最旧的槽；读者落后超过一圈时**跳到最新一帧**，被跳过的帧以
  「帧间隔变大」的形式（时间戳是真实采集时间）对算法可见。

# 平台

Windows：命名共享内存（`CreateFileMapping` 的页面文件映射，`Local\\` 命名空间），Python 侧用
`mmap.mmap(-1, size, tagname=name)` 打开同名对象。其他平台：`/dev/shm/<name>` 文件映射。
"""

from __future__ import annotations

import mmap
import os
import struct
import sys
import time
from dataclasses import dataclass

import numpy as np

MAGIC = 0x5246474D  # b"MGFR" 小端
VERSION = 1
HEADER = 128
SLOT_HEADER = 64
ALIGN = 64

BLOCK, OVERWRITE = 0, 1
RGB24, GRAY8 = 1, 2
CHANNELS = {RGB24: 3, GRAY8: 1}

# 总头字段偏移
_OFF_SLOTS, _OFF_CAP, _OFF_WRITE, _OFF_READ, _OFF_MODE, _OFF_CLOSED = 8, 24, 32, 40, 48, 52
# 槽头：seq u64 | frame u64 | t f64 | w h stride fmt len rx ry rw rh u32×9 | 保留 u32
_SLOT = struct.Struct("<QQdIIIIIIIIII")
assert _SLOT.size == SLOT_HEADER


def slot_stride(capacity: int) -> int:
    return (SLOT_HEADER + capacity + ALIGN - 1) // ALIGN * ALIGN


def total_size(slots: int, capacity: int) -> int:
    return HEADER + slots * slot_stride(capacity)


@dataclass
class Frame:
    index: int
    """写入序号（第几个写进环的帧）。"""
    frame_no: int
    t: float
    """时间戳（秒）：回放 = 视频时间戳；实时源 = 采集时间。"""
    width: int
    height: int
    fmt: int
    roi: tuple[int, int, int, int]
    """写入这一帧时云台取景窗在全幅里的矩形。"""
    pixels: np.ndarray
    """`(h, w, 3)` 或 `(h, w)` uint8，**拷贝**，不引用共享内存。"""
    skipped: int = 0
    """读到这一帧之前被覆盖跳过了多少帧（只在 OVERWRITE 模式下非零）。"""
    copy_ms: float = 0.0
    """从共享内存拷出这一帧（含序列锁校验）花了多久。"""


def _map(name: str, size: int, create: bool):
    if sys.platform == "win32":
        return mmap.mmap(-1, size, tagname=name, access=mmap.ACCESS_WRITE)
    path = f"/dev/shm/{name}"
    fd = os.open(path, os.O_RDWR | (os.O_CREAT if create else 0), 0o600)
    try:
        if create:
            os.ftruncate(fd, size)
        return mmap.mmap(fd, size)
    finally:
        os.close(fd)


class _Ring:
    def __init__(self, mm, slots, capacity):
        self.mm = mm
        self.slots = slots
        self.capacity = capacity
        self.stride = slot_stride(capacity)
        self.u64 = np.frombuffer(mm, dtype=np.uint64, count=HEADER // 8)
        self.u32 = np.frombuffer(mm, dtype=np.uint32, count=HEADER // 4)
        self.seqs = [np.frombuffer(mm, dtype=np.uint64, count=1, offset=self._slot(i)) for i in range(slots)]

    def _slot(self, i):
        return HEADER + (i % self.slots) * self.stride

    @property
    def write_count(self) -> int:
        return int(self.u64[_OFF_WRITE // 8])

    @property
    def read_count(self) -> int:
        return int(self.u64[_OFF_READ // 8])

    @property
    def mode(self) -> int:
        return int(self.u32[_OFF_MODE // 4])

    @property
    def closed(self) -> bool:
        return bool(self.u32[_OFF_CLOSED // 4])

    def close(self):
        # numpy 视图引用着 mmap 的缓冲区，先放掉才能关
        self.u64 = self.u32 = None
        self.seqs = []
        try:
            self.mm.close()
        except BufferError:
            pass


class RingReader(_Ring):
    """视觉服务一侧。打开宿主创建的帧环，按序取帧。"""

    def __init__(self, name: str):
        head = _map(name, HEADER, create=False)
        try:
            magic, ver, slots, hsz, ssz = struct.unpack_from("<IIIII", head, 0)
            (cap,) = struct.unpack_from("<Q", head, _OFF_CAP)
        finally:
            head.close()
        if magic != MAGIC:
            raise RuntimeError(f"共享内存 {name!r} 不是帧环（magic {magic:#x}）—— 宿主是否已创建？")
        if ver != VERSION or hsz != HEADER or ssz != SLOT_HEADER:
            raise RuntimeError(f"帧环版本不符：版本 {ver}、总头 {hsz}、槽头 {ssz}")
        super().__init__(_map(name, total_size(slots, cap), create=False), slots, cap)
        self.name = name
        self.next_index = 0
        self.torn = 0
        """拷贝过程中被改写（序列锁前后不一致）而丢弃的次数。"""
        self.skipped = 0
        self._skip = 0

    def try_read(self, i: int) -> Frame | str:
        """读写入序号为 `i` 的帧。返回帧，或 `"pending"`（还没写完）/ `"overwritten"` / `"torn"`。"""
        c0 = time.perf_counter()
        base = self._slot(i)
        want = 2 * i + 2
        s1 = int(self.seqs[i % self.slots][0])
        if s1 < want:
            return "pending"
        if s1 > want:
            return "overwritten"
        _, frame_no, t, w, h, stride, fmt, nbytes, rx, ry, rw, rh, _ = _SLOT.unpack_from(self.mm, base)
        if fmt not in CHANNELS or nbytes > self.capacity or stride * h != nbytes:
            return "torn"
        src = np.frombuffer(self.mm, dtype=np.uint8, count=nbytes, offset=base + SLOT_HEADER)
        pix = src.copy()
        del src
        s2 = int(self.seqs[i % self.slots][0])
        if s2 != s1:
            self.torn += 1
            return "torn"
        c = CHANNELS[fmt]
        pix = pix.reshape(h, stride)[:, : w * c]
        pix = pix.reshape(h, w, c) if c > 1 else pix
        return Frame(i, frame_no, t, w, h, fmt, (rx, ry, rw, rh), pix, copy_ms=(time.perf_counter() - c0) * 1e3)

    def release(self, i: int):
        """告诉写者第 `i` 帧已经用完（BLOCK 模式下写者据此判断环满）。"""
        self.u64[_OFF_READ // 8] = np.uint64(i + 1)

    def next(self, timeout: float | None = None, poll: float = 0.0005) -> Frame | None:
        """按序取下一帧；OVERWRITE 模式下落后时跳到最新一帧。超时或写者已关闭且读完返回 `None`。

        拿到帧即 `release`（像素已拷出，槽可以复用）。"""
        deadline = None if timeout is None else time.perf_counter() + timeout
        while True:
            w = self.write_count
            i = self.next_index
            if i < w:
                if w - i >= self.slots and self.mode == OVERWRITE:  # 槽 i 可能已被（或正在被）覆盖
                    self._skip += w - 1 - i
                    i = self.next_index = w - 1
                r = self.try_read(i)
                if isinstance(r, Frame):
                    r.skipped, self._skip = self._skip, 0
                    self.skipped += r.skipped
                    self.next_index = i + 1
                    self.release(i)
                    return r
                if r == "overwritten":
                    # 读的同时又被套圈：再看一次最新的 write_count
                    j = max(self.next_index, self.write_count - 1)
                    self._skip += j - self.next_index
                    self.next_index = j
                continue
            if self.closed:
                return None
            if deadline is not None and time.perf_counter() > deadline:
                return None
            time.sleep(poll)


class RingWriter(_Ring):
    """写者的 Python 实现 —— **只用于测试与基准**；宿主用的是 `ring.rs`。"""

    def __init__(self, name: str, slots: int, capacity: int, mode: int = BLOCK):
        super().__init__(_map(name, total_size(slots, capacity), create=True), slots, capacity)
        self.name = name
        mm = self.mm
        mm[:HEADER] = bytes(HEADER)
        struct.pack_into("<IIIII", mm, 0, MAGIC, VERSION, slots, HEADER, SLOT_HEADER)
        struct.pack_into("<Q", mm, _OFF_CAP, capacity)
        self.u32[_OFF_MODE // 4] = mode
        for s in self.seqs:
            s[0] = 0

    def begin(self, i: int):
        self.seqs[i % self.slots][0] = np.uint64(2 * i + 1)

    def fill(self, i, frame_no, t, pixels: np.ndarray, fmt=RGB24, roi=None, upto=None):
        """写槽头与像素；`upto` 只写前若干字节（测试「半帧」用）。"""
        h, w = pixels.shape[:2]
        c = CHANNELS[fmt]
        nbytes = w * h * c
        base = self._slot(i)
        rx, ry, rw, rh = roi or (0, 0, w, h)
        struct.pack_into("<QdIIIIIIIIII", self.mm, base + 8, frame_no, t, w, h, w * c, fmt, nbytes, rx, ry, rw, rh, 0)
        flat = np.ascontiguousarray(pixels).reshape(-1)
        dst = np.frombuffer(self.mm, dtype=np.uint8, count=nbytes, offset=base + SLOT_HEADER)
        n = nbytes if upto is None else upto
        dst[:n] = flat[:n]
        del dst

    def commit(self, i: int):
        self.seqs[i % self.slots][0] = np.uint64(2 * i + 2)
        self.u64[_OFF_WRITE // 8] = np.uint64(i + 1)

    def write(self, frame_no, t, pixels, fmt=RGB24, roi=None, timeout=None) -> int | None:
        i = self.write_count
        if self.mode == BLOCK:
            deadline = None if timeout is None else time.perf_counter() + timeout
            while i - self.read_count >= self.slots:
                if deadline is not None and time.perf_counter() > deadline:
                    return None
                time.sleep(0.0002)
        self.begin(i)
        self.fill(i, frame_no, t, pixels, fmt, roi)
        self.commit(i)
        return i

    def close_stream(self):
        self.u32[_OFF_CLOSED // 4] = 1
