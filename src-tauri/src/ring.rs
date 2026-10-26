//! **共享内存帧环**（数据面）—— 宿主写帧，Python 视觉服务读帧。
//!
//! 布局与 `vision/src/gimbal_vision/ring.py` 逐字节一致（小端）：
//!
//! ```text
//! 总头 128 B  0 magic "MGFR"  4 版本  8 槽数 N  12 总头长  16 槽头长  20 保留
//!             24 每槽像素容量 u64  32 已发布帧数 write_count u64  40 已消费帧数 read_count u64
//!             48 模式 u32（0 回放等待 / 1 实时覆盖）  52 已关闭 u32
//! 槽 i % N    槽头 64 B：0 写入序号 seq u64  8 帧号 u64  16 时间戳 f64（秒）
//!                       24 宽  28 高  32 行跨度  36 像素格式  40 像素字节数
//!                       44 取景窗 x  48 y  52 w  56 h  60 保留（均 u32）
//!             像素      紧跟槽头；槽按 64 B 对齐
//! ```
//!
//! # 序列锁
//!
//! 第 `i` 帧（写入序号）：`seq ← 2i+1`（奇数 = 正在写）→ 写槽头与像素 → `seq ← 2i+2` →
//! `write_count ← i+1`。读者拷贝前后各读一次 `seq`，必须都等于 `2i+2`，否则丢弃 —— 读者永远拿不到半帧，
//! 也能分清「还没写到」（`seq < 2i+2`）与「已被覆盖」（`seq > 2i+2`）。
//!
//! # 模式
//!
//! - [`Mode::Block`]（视频回放）：环满时写者等待 —— 与播放循环「掉速而不是丢帧」一致；
//! - [`Mode::Overwrite`]（实时源）：写者从不等待，覆盖最旧的槽；落后的读者跳到最新一帧，时间戳保留真实值，
//!   跳过的帧以「帧间隔变大」的形式对算法可见。
//!
//! Windows 上是命名共享内存（页面文件映射，`Local\` 命名空间）；其他平台是 `/dev/shm/<name>` 的文件映射。

use std::sync::atomic::{fence, AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};

pub const MAGIC: u32 = 0x5246_474D; // b"MGFR"
pub const VERSION: u32 = 1;
pub const HEADER: usize = 128;
pub const SLOT_HEADER: usize = 64;
const ALIGN: usize = 64;

const OFF_SLOTS: usize = 8;
const OFF_CAP: usize = 24;
const OFF_WRITE: usize = 32;
const OFF_READ: usize = 40;
const OFF_MODE: usize = 48;
const OFF_CLOSED: usize = 52;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Block = 0,
    Overwrite = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb24 = 1,
    Gray8 = 2,
}

impl PixelFormat {
    pub fn channels(self) -> u32 {
        match self {
            PixelFormat::Rgb24 => 3,
            PixelFormat::Gray8 => 1,
        }
    }
}

/// 一帧的元数据（写进槽头）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameMeta {
    pub frame_no: u64,
    /// 秒。回放 = 视频时间戳；实时源 = 采集时间。
    pub t: f64,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    /// 写这一帧时云台取景窗在全幅里的矩形 `(x, y, w, h)`。
    pub roi: (u32, u32, u32, u32),
}

pub fn slot_stride(capacity: usize) -> usize {
    (SLOT_HEADER + capacity).div_ceil(ALIGN) * ALIGN
}

pub fn total_size(slots: usize, capacity: usize) -> usize {
    HEADER + slots * slot_stride(capacity)
}

// ── 平台：一块可读写的命名共享内存 ────────────────────────────────
#[cfg(windows)]
mod shm {
    use anyhow::{bail, Result};
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_ALREADY_EXISTS, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, MapViewOfFile, OpenFileMappingW, UnmapViewOfFile, FILE_MAP_ALL_ACCESS,
        MEMORY_MAPPED_VIEW_ADDRESS, PAGE_READWRITE,
    };

    pub struct Shm {
        handle: HANDLE,
        ptr: *mut u8,
        len: usize,
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    impl Shm {
        pub fn create(name: &str, len: usize) -> Result<Self> {
            let w = wide(name);
            // SAFETY: 参数都是合法值；返回的句柄在 Drop 里关闭
            let handle = unsafe {
                CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE,
                    (len as u64 >> 32) as u32,
                    len as u32,
                    w.as_ptr(),
                )
            };
            if handle.is_null() {
                bail!("CreateFileMappingW({name}) 失败：{}", unsafe { GetLastError() });
            }
            if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
                unsafe { CloseHandle(handle) };
                bail!("共享内存 {name} 已存在");
            }
            Self::map(handle, len, name)
        }

        pub fn open(name: &str, len: usize) -> Result<Self> {
            let w = wide(name);
            let handle = unsafe { OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, w.as_ptr()) };
            if handle.is_null() {
                bail!("OpenFileMappingW({name}) 失败：{}", unsafe { GetLastError() });
            }
            Self::map(handle, len, name)
        }

        fn map(handle: HANDLE, len: usize, name: &str) -> Result<Self> {
            let view = unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, len) };
            if view.Value.is_null() {
                let e = unsafe { GetLastError() };
                unsafe { CloseHandle(handle) };
                bail!("MapViewOfFile({name}) 失败：{e}");
            }
            Ok(Self { handle, ptr: view.Value as *mut u8, len })
        }

        pub fn ptr(&self) -> *mut u8 {
            self.ptr
        }

        pub fn len(&self) -> usize {
            self.len
        }
    }

    impl Drop for Shm {
        fn drop(&mut self) {
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.ptr as _ });
                CloseHandle(self.handle);
            }
        }
    }
}

#[cfg(unix)]
mod shm {
    use anyhow::{Context, Result};
    use memmap2::MmapMut;

    pub struct Shm {
        map: MmapMut,
        path: Option<std::path::PathBuf>,
    }

    impl Shm {
        pub fn create(name: &str, len: usize) -> Result<Self> {
            let path = std::path::PathBuf::from(format!("/dev/shm/{name}"));
            let f = std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path)
                .with_context(|| format!("创建 {path:?}"))?;
            f.set_len(len as u64)?;
            let map = unsafe { MmapMut::map_mut(&f)? };
            Ok(Self { map, path: Some(path) })
        }

        pub fn open(name: &str, len: usize) -> Result<Self> {
            let f = std::fs::OpenOptions::new().read(true).write(true).open(format!("/dev/shm/{name}"))?;
            let map = unsafe { MmapMut::map_mut(&f)? };
            let _ = len;
            Ok(Self { map, path: None })
        }

        pub fn ptr(&self) -> *mut u8 {
            self.map.as_ptr() as *mut u8
        }

        pub fn len(&self) -> usize {
            self.map.len()
        }
    }

    impl Drop for Shm {
        fn drop(&mut self) {
            if let Some(p) = &self.path {
                let _ = std::fs::remove_file(p);
            }
        }
    }
}

// ── 帧环 ─────────────────────────────────────────────────────────
struct Mapped {
    shm: shm::Shm,
    slots: usize,
    capacity: usize,
    stride: usize,
}

// SAFETY: 共享内存里的并发访问全部经由原子量与序列锁协调
unsafe impl Send for Mapped {}
unsafe impl Sync for Mapped {}

impl Mapped {
    fn base(&self) -> *mut u8 {
        self.shm.ptr()
    }

    fn u64_at(&self, off: usize) -> &AtomicU64 {
        debug_assert!(off % 8 == 0 && off + 8 <= self.shm.len());
        // SAFETY: 偏移 8 字节对齐且在映射范围内；映射比 self 活得久
        unsafe { &*(self.base().add(off) as *const AtomicU64) }
    }

    fn u32_at(&self, off: usize) -> &AtomicU32 {
        unsafe { &*(self.base().add(off) as *const AtomicU32) }
    }

    fn slot(&self, i: u64) -> usize {
        HEADER + (i as usize % self.slots) * self.stride
    }

    fn seq(&self, i: u64) -> &AtomicU64 {
        self.u64_at(self.slot(i))
    }

    fn write_count(&self) -> u64 {
        self.u64_at(OFF_WRITE).load(Ordering::Acquire)
    }

    fn read_count(&self) -> u64 {
        self.u64_at(OFF_READ).load(Ordering::Acquire)
    }
}

/// 写一帧的结果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WriteOutcome {
    /// 写好了，值是写入序号；`waited` 是环满等待的时长。
    Written { index: u64, waited: Duration },
    /// 等待期间被要求停止（只在 Block 模式出现）。
    Stopped,
}

/// 宿主一侧：创建帧环并写帧。Drop 时置「已关闭」并释放共享内存。
pub struct RingWriter {
    m: Mapped,
    pub name: String,
    mode: Mode,
}

impl RingWriter {
    /// 新建帧环：`slots` 个槽，每槽可放 `capacity` 字节像素。
    pub fn create(name: &str, slots: usize, capacity: usize, mode: Mode) -> Result<Self> {
        if slots < 2 {
            bail!("帧环至少要 2 个槽");
        }
        let len = total_size(slots, capacity);
        let shm = shm::Shm::create(name, len)?;
        let m = Mapped { shm, slots, capacity, stride: slot_stride(capacity) };
        // SAFETY: 新建的映射只有自己在用
        unsafe {
            std::ptr::write_bytes(m.base(), 0, HEADER);
            let b = m.base();
            for (off, v) in [(0usize, MAGIC), (4, VERSION), (OFF_SLOTS, slots as u32), (12, HEADER as u32), (16, SLOT_HEADER as u32)] {
                std::ptr::write_unaligned(b.add(off) as *mut u32, v.to_le());
            }
            std::ptr::write_unaligned(b.add(OFF_CAP) as *mut u64, (capacity as u64).to_le());
            for i in 0..slots {
                std::ptr::write_bytes(b.add(HEADER + i * m.stride), 0, SLOT_HEADER);
            }
        }
        m.u32_at(OFF_MODE).store(mode as u32, Ordering::Release);
        Ok(Self { m, name: name.to_string(), mode })
    }

    /// 本进程内唯一的名字。Windows 下放在 `Local\` 命名空间（当前登录会话内可见）。
    pub fn unique_name(tag: &str) -> String {
        use std::sync::atomic::AtomicUsize;
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let base = format!("motion-gimbal-{tag}-{}-{n}", std::process::id());
        if cfg!(windows) { format!("Local\\{base}") } else { base }
    }

    pub fn slots(&self) -> usize {
        self.m.slots
    }

    pub fn capacity(&self) -> usize {
        self.m.capacity
    }

    pub fn write_count(&self) -> u64 {
        self.m.write_count()
    }

    pub fn read_count(&self) -> u64 {
        self.m.read_count()
    }

    /// 写一帧。Block 模式下环满就等（每 `poll` 查一次 `stop`），Overwrite 模式从不等。
    pub fn write(&self, meta: &FrameMeta, pixels: &[u8], stop: &AtomicBool) -> Result<WriteOutcome> {
        let c = meta.format.channels() as usize;
        let nbytes = meta.width as usize * meta.height as usize * c;
        if pixels.len() != nbytes {
            bail!("像素 {} 字节，按 {}×{}×{c} 应为 {nbytes}", pixels.len(), meta.width, meta.height);
        }
        if nbytes > self.m.capacity {
            bail!("一帧 {nbytes} 字节超过槽容量 {}", self.m.capacity);
        }
        let i = self.m.write_count();
        let t0 = Instant::now();
        if self.mode == Mode::Block {
            let mut spins = 0u32;
            while i - self.m.read_count() >= self.m.slots as u64 {
                if stop.load(Ordering::Relaxed) {
                    return Ok(WriteOutcome::Stopped);
                }
                spins += 1;
                if spins < 64 {
                    std::hint::spin_loop();
                } else {
                    std::thread::sleep(Duration::from_micros(200));
                }
            }
        }
        let waited = t0.elapsed();
        self.write_at(i, meta, pixels, None);
        Ok(WriteOutcome::Written { index: i, waited })
    }

    /// 写第 `i` 帧；`upto = Some(n)` 只写前 n 字节像素且**不提交**（测试半帧）。
    #[doc(hidden)]
    pub fn write_at(&self, i: u64, meta: &FrameMeta, pixels: &[u8], upto: Option<usize>) {
        let seq = self.m.seq(i);
        seq.store(2 * i + 1, Ordering::Relaxed);
        fence(Ordering::Release); // 「正在写」先于任何数据写入可见
        let base = self.m.slot(i);
        let (rx, ry, rw, rh) = meta.roi;
        let c = meta.format.channels();
        // SAFETY: 槽在映射范围内；读者靠序列锁丢弃并发改写时的不一致拷贝
        unsafe {
            let p = self.m.base().add(base);
            std::ptr::write_unaligned(p.add(8) as *mut u64, meta.frame_no.to_le());
            std::ptr::write_unaligned(p.add(16) as *mut u64, meta.t.to_bits().to_le());
            let u32s = [meta.width, meta.height, meta.width * c, meta.format as u32, pixels.len() as u32, rx, ry, rw, rh, 0];
            for (j, v) in u32s.iter().enumerate() {
                std::ptr::write_unaligned(p.add(24 + 4 * j) as *mut u32, v.to_le());
            }
            let n = upto.unwrap_or(pixels.len());
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), p.add(SLOT_HEADER), n);
        }
        if upto.is_some() {
            return;
        }
        seq.store(2 * i + 2, Ordering::Release);
        self.m.u64_at(OFF_WRITE).store(i + 1, Ordering::Release);
    }

    /// 告诉读者不会再有新帧了。
    pub fn close_stream(&self) {
        self.m.u32_at(OFF_CLOSED).store(1, Ordering::Release);
    }
}

impl Drop for RingWriter {
    fn drop(&mut self) {
        self.close_stream();
    }
}

/// 读一帧的结果。
#[derive(Debug, Clone, PartialEq)]
pub enum ReadOutcome {
    Frame { index: u64, meta: FrameMeta },
    /// 还没写完（或还没写到）。
    Pending,
    /// 已被后来的帧覆盖。
    Overwritten,
    /// 拷贝过程中被改写（序列锁前后不一致），已丢弃。
    Torn,
}

/// 读者的 Rust 实现 —— 宿主不用它（读者是 Python 服务），只用于测试与基准。
pub struct RingReader {
    m: Mapped,
    pub next_index: u64,
    pub torn: u64,
    pub skipped: u64,
}

impl RingReader {
    pub fn open(name: &str) -> Result<Self> {
        let head = shm::Shm::open(name, HEADER)?;
        let rd = |off: usize| unsafe { std::ptr::read_unaligned(head.ptr().add(off) as *const u32) };
        if rd(0) != MAGIC || rd(4) != VERSION {
            bail!("{name} 不是帧环");
        }
        let slots = rd(OFF_SLOTS) as usize;
        let capacity = unsafe { std::ptr::read_unaligned(head.ptr().add(OFF_CAP) as *const u64) } as usize;
        drop(head);
        let shm = shm::Shm::open(name, total_size(slots, capacity))?;
        Ok(Self { m: Mapped { shm, slots, capacity, stride: slot_stride(capacity) }, next_index: 0, torn: 0, skipped: 0 })
    }

    pub fn write_count(&self) -> u64 {
        self.m.write_count()
    }

    pub fn closed(&self) -> bool {
        self.m.u32_at(OFF_CLOSED).load(Ordering::Acquire) != 0
    }

    pub fn mode(&self) -> Mode {
        if self.m.u32_at(OFF_MODE).load(Ordering::Acquire) == 1 { Mode::Overwrite } else { Mode::Block }
    }

    /// 读第 `i` 帧到 `buf`（会被改成像素字节数）。
    pub fn try_read(&mut self, i: u64, buf: &mut Vec<u8>) -> ReadOutcome {
        let want = 2 * i + 2;
        let seq = self.m.seq(i);
        let s1 = seq.load(Ordering::Acquire);
        if s1 < want {
            return ReadOutcome::Pending;
        }
        if s1 > want {
            return ReadOutcome::Overwritten;
        }
        let base = self.m.slot(i);
        let meta;
        unsafe {
            let p = self.m.base().add(base);
            let r32 = |j: usize| u32::from_le(std::ptr::read_volatile(p.add(24 + 4 * j) as *const u32));
            let frame_no = u64::from_le(std::ptr::read_volatile(p.add(8) as *const u64));
            let t = f64::from_bits(u64::from_le(std::ptr::read_volatile(p.add(16) as *const u64)));
            let (w, h, fmt, n) = (r32(0), r32(1), r32(3), r32(4) as usize);
            if n > self.m.capacity {
                self.torn += 1;
                return ReadOutcome::Torn;
            }
            buf.resize(n, 0);
            std::ptr::copy_nonoverlapping(p.add(SLOT_HEADER), buf.as_mut_ptr(), n);
            let format = if fmt == 2 { PixelFormat::Gray8 } else { PixelFormat::Rgb24 };
            meta = FrameMeta { frame_no, t, width: w, height: h, format, roi: (r32(5), r32(6), r32(7), r32(8)) };
        }
        fence(Ordering::Acquire); // 拷贝完成先于第二次读 seq
        if seq.load(Ordering::Relaxed) != s1 {
            self.torn += 1;
            return ReadOutcome::Torn;
        }
        ReadOutcome::Frame { index: i, meta }
    }

    pub fn release(&self, i: u64) {
        self.m.u64_at(OFF_READ).store(i + 1, Ordering::Release);
    }

    /// 按序取下一帧（Overwrite 模式下落后则跳到最新一帧）；`timeout` 内没有新帧返回 `None`。
    pub fn next(&mut self, buf: &mut Vec<u8>, timeout: Duration) -> Option<(u64, FrameMeta)> {
        let t0 = Instant::now();
        loop {
            let w = self.m.write_count();
            let mut i = self.next_index;
            if i < w {
                if w - i >= self.m.slots as u64 && self.mode() == Mode::Overwrite {
                    self.skipped += w - 1 - i;
                    i = w - 1;
                    self.next_index = i;
                }
                match self.try_read(i, buf) {
                    ReadOutcome::Frame { index, meta } => {
                        self.next_index = index + 1;
                        self.release(index);
                        return Some((index, meta));
                    }
                    ReadOutcome::Overwritten => {
                        let j = self.next_index.max(self.m.write_count().saturating_sub(1));
                        self.skipped += j - self.next_index;
                        self.next_index = j;
                    }
                    _ => {}
                }
                continue;
            }
            if self.closed() || t0.elapsed() > timeout {
                return None;
            }
            std::hint::spin_loop();
            std::thread::yield_now();
        }
    }
}
