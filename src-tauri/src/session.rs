//! **播放会话** —— 帧源 → 虚拟云台 →（帧环 → 视觉服务 → 结果驱动云台）→ 回调。
//!
//! 与界面无关：Tauri 宿主在回调里推预览，联调测试与基准在回调里记指标，走的是同一份循环。
//!
//! ```text
//! 每帧：读帧 ─▶ 按时间戳等到点 ─▶ 取当前取景窗矩形 ─▶ 全幅帧 + 矩形写进帧环 ─▶ 回调
//!                                                     │
//!         结果线程（vision::ResultStream）：收到结果 ─▶ VirtualGimbal::aim_in ─▶ 交给下一次回调
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};

use crate::gimbal::{Rect, VirtualGimbal};
use crate::ring::{FrameMeta, Mode, PixelFormat, RingWriter, WriteOutcome};
use crate::video::{FrameReader, VideoInfo};
use crate::vision::{Applied, ResultStream, VisionService, WriteTimes};

#[derive(Debug, Clone)]
pub struct Options {
    /// 按时间戳节拍播放（假 = 尽快，只受帧环背压限制）。
    pub paced: bool,
    /// 每写一帧就等它的结果回来再读下一帧 —— 确定性的闭环（等价于离线闭环评测的「一帧延迟」）。
    pub lockstep: bool,
    pub ring_slots: usize,
    pub ring_mode: Mode,
    /// 下发给视觉服务的参数（`service.Params` 的字段）。
    pub params: Value,
    /// 最多播放多少帧（0 = 不限）。
    pub max_frames: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self { paced: true, lockstep: false, ring_slots: 4, ring_mode: Mode::Block, params: json!({}), max_frames: 0 }
    }
}

/// 写帧的耗时。
#[derive(Debug, Clone, Copy)]
pub struct WriteInfo {
    pub index: u64,
    /// 拷贝进共享内存（不含环满等待）。
    pub copy: Duration,
    /// 环满等待。
    pub waited: Duration,
}

/// 收尾回调的 `index`：没有新帧，只带回在途的结果。
pub const TAIL: u64 = u64::MAX;

/// 每帧一次的回调参数。
pub struct Tick<'a> {
    pub index: u64,
    pub t: f64,
    pub frame: &'a [u8],
    pub info: &'a VideoInfo,
    /// 这一帧裁取景窗、写帧环时用的矩形。
    pub rect: Rect,
    pub write: Option<WriteInfo>,
    /// 上一次回调以来收到的结果（按到达顺序）。
    pub applied: Vec<Applied>,
}

/// 跑一段视频。`vision` 为 `None` 时是纯手动模式（不建帧环、不连服务），与接入前的行为相同。
pub fn run(
    info: &VideoInfo,
    gimbal: &Arc<Mutex<Option<VirtualGimbal>>>,
    stop: &AtomicBool,
    auto: &Arc<AtomicBool>,
    opts: &Options,
    vision: Option<&VisionService>,
    on_tick: &mut dyn FnMut(Tick) -> Result<()>,
) -> Result<String> {
    let mut reader = FrameReader::open(info)?;
    let mut frame = vec![0u8; reader.frame_len()];

    // ── 接入视觉服务：建帧环 → 先连结果流 → 再开会话（先连后开，第一条结果不会丢）──
    let written: WriteTimes = Arc::new(Mutex::new(Default::default()));
    let link = match vision {
        None => None,
        Some(svc) => {
            let ring = RingWriter::create(&RingWriter::unique_name("ring"), opts.ring_slots, frame.len(), opts.ring_mode)?;
            let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
            let session = format!("s{}-{nanos}", std::process::id());
            let stream = ResultStream::connect(svc.port, &session, Arc::clone(gimbal), Arc::clone(auto), Arc::clone(&written))?;
            svc.post("/session/start", &json!({ "ring": ring.name, "session": session, "params": opts.params }))?;
            Some((ring, stream, svc))
        }
    };

    let started = Instant::now();
    let mut reason = "已停止".to_string();
    let mut pending: Vec<Applied> = Vec::new();
    let mut n = 0u64;
    while !stop.load(Ordering::SeqCst) {
        if opts.max_frames > 0 && n >= opts.max_frames {
            reason = "播放完毕".into();
            break;
        }
        if !reader.next(&mut frame)? {
            reason = "播放完毕".into();
            break;
        }
        n += 1;
        let t = reader.timestamp();
        // 按时间戳节拍播放：处理比实时快就等，慢就不等（掉速而不是丢帧）
        if opts.paced {
            if let Some(wait) = Duration::from_secs_f64(t).checked_sub(started.elapsed()) {
                std::thread::sleep(wait);
            }
        }
        let rect = {
            let g = gimbal.lock().unwrap();
            g.as_ref().ok_or_else(|| anyhow!("云台未初始化"))?.rect()
        };

        let mut write = None;
        if let Some((ring, stream, _)) = &link {
            let meta = FrameMeta {
                frame_no: reader.index - 1,
                t,
                width: info.width,
                height: info.height,
                format: PixelFormat::Rgb24,
                roi: (rect.x, rect.y, rect.w, rect.h),
            };
            let idx = ring.write_count();
            written.lock().unwrap().insert(idx, Instant::now());
            let t0 = Instant::now();
            match ring.write(&meta, &frame, stop)? {
                WriteOutcome::Stopped => break,
                WriteOutcome::Written { index, waited } => {
                    let done = Instant::now();
                    written.lock().unwrap().insert(index, done);
                    write = Some(WriteInfo { index, copy: (done - t0).saturating_sub(waited), waited });
                }
            }
            if opts.lockstep {
                let deadline = Instant::now() + Duration::from_secs(30);
                loop {
                    let left = deadline.saturating_duration_since(Instant::now());
                    let a = stream.rx.recv_timeout(left).map_err(|_| anyhow!("等第 {idx} 帧的结果超时"))?;
                    let hit = a.result.index == idx;
                    check(&a)?;
                    pending.push(a);
                    if hit {
                        break;
                    }
                }
            } else {
                while let Ok(a) = stream.rx.try_recv() {
                    check(&a)?;
                    pending.push(a);
                }
            }
        }
        on_tick(Tick {
            index: reader.index - 1,
            t,
            frame: &frame,
            info,
            rect,
            write,
            applied: std::mem::take(&mut pending),
        })?;
    }

    // ── 收尾：告诉服务没有新帧了，收完在途的结果 ──
    if let Some((ring, stream, svc)) = link {
        ring.close_stream();
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut last = Vec::new();
        let mut got_end = false;
        while Instant::now() < deadline {
            match stream.rx.recv_timeout(Duration::from_millis(200)) {
                Ok(a) => {
                    let end = a.result.end;
                    if !end {
                        check(&a)?;
                        last.push(a);
                    }
                    if end {
                        got_end = true;
                        break;
                    }
                }
                Err(_) => {
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                }
            }
        }
        let _ = svc.post("/session/stop", &json!({}));
        if !last.is_empty() || got_end {
            on_tick(Tick {
                index: TAIL,
                t: f64::NAN,
                frame: &frame,
                info,
                rect: gimbal.lock().unwrap().as_ref().map(|g| g.rect()).unwrap_or(Rect { x: 0, y: 0, w: 0, h: 0 }),
                write: None,
                applied: last,
            })?;
        }
        drop(stream);
        drop(ring);
    }
    Ok(reason)
}

fn check(a: &Applied) -> Result<()> {
    if let Some(e) = &a.result.error {
        bail!("视觉服务出错：{e}");
    }
    Ok(())
}
