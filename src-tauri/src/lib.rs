//! motion-gimbal 宿主：视频帧源 → 虚拟云台取景 → （视觉服务）→ 前端预览。
//!
//! 当前阶段：读视频、按标称帧率播放、手动操纵虚拟云台、把全景与取景窗推给前端。
//! 下一阶段在播放循环里接入 Python 视觉服务，由跟踪结果驱动云台（docs/PLAN.md）。

pub mod gimbal;
pub mod video;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use base64::Engine;
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use gimbal::{downscale_rgb, Rect, VirtualGimbal};
use video::{FrameReader, VideoInfo};

/// 取景窗占全幅的比例（宽、高同比）。
const WINDOW_FRAC: f64 = 0.5;
/// 预览图的最大宽度（只影响界面，不进算法）。
const PREVIEW_FULL_W: u32 = 640;
const PREVIEW_WIN_W: u32 = 480;
const PREVIEW_JPEG_Q: u8 = 75;

#[derive(Default)]
struct AppState {
    info: Mutex<Option<VideoInfo>>,
    gimbal: Arc<Mutex<Option<VirtualGimbal>>>,
    session: Mutex<Option<Session>>,
}

struct Session {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Preview {
    index: u64,
    t: f64,
    /// 实际推送帧率（最近一秒）。
    fps: f64,
    full: String,
    window: String,
    rect: Rect,
    full_size: (u32, u32),
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Ended {
    reason: String,
}

fn jpeg_b64(rgb: &[u8], w: u32, h: u32) -> anyhow::Result<String> {
    let mut buf = Vec::new();
    let enc = jpeg_encoder::Encoder::new(&mut buf, PREVIEW_JPEG_Q);
    enc.encode(rgb, w as u16, h as u16, jpeg_encoder::ColorType::Rgb)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(buf))
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

#[tauri::command]
fn open_video(path: String, state: State<'_, AppState>) -> Result<VideoInfo, String> {
    stop_session(&state);
    let info = video::probe(&path).map_err(err)?;
    let win = (
        (info.width as f64 * WINDOW_FRAC) as u32,
        (info.height as f64 * WINDOW_FRAC) as u32,
    );
    *state.gimbal.lock().unwrap() = Some(VirtualGimbal::new((info.width, info.height), win));
    *state.info.lock().unwrap() = Some(info.clone());
    Ok(info)
}

/// 手动操纵云台：把取景窗中心设到全幅坐标 `(cx, cy)`。
#[tauri::command]
fn set_gimbal(cx: f64, cy: f64, state: State<'_, AppState>) -> Result<Rect, String> {
    let mut g = state.gimbal.lock().unwrap();
    let g = g.as_mut().ok_or("先打开一段视频")?;
    g.set_center(cx, cy);
    Ok(g.rect())
}

#[tauri::command]
fn start(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    stop_session(&state);
    let info = state.info.lock().unwrap().clone().ok_or("先打开一段视频")?;
    let gimbal = Arc::clone(&state.gimbal);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_flag = Arc::clone(&stop);
    let handle = std::thread::spawn(move || {
        let reason = match play(&app, &info, &gimbal, &stop_flag) {
            Ok(r) => r,
            Err(e) => format!("出错：{e:#}"),
        };
        let _ = app.emit("ended", Ended { reason });
    });
    *state.session.lock().unwrap() = Some(Session { stop, handle });
    Ok(())
}

#[tauri::command]
fn stop(state: State<'_, AppState>) {
    stop_session(&state);
}

fn stop_session(state: &AppState) {
    if let Some(s) = state.session.lock().unwrap().take() {
        s.stop.store(true, Ordering::SeqCst);
        let _ = s.handle.join();
    }
}

/// 播放循环：按标称时间戳节拍读帧、裁取景窗、推预览。
fn play(
    app: &AppHandle,
    info: &VideoInfo,
    gimbal: &Mutex<Option<VirtualGimbal>>,
    stop: &AtomicBool,
) -> anyhow::Result<String> {
    let mut reader = FrameReader::open(info)?;
    let mut frame = vec![0u8; reader.frame_len()];
    let started = Instant::now();
    let mut window_start = Instant::now();
    let mut window_frames = 0u32;
    let mut fps = 0.0;
    while !stop.load(Ordering::SeqCst) {
        if !reader.next(&mut frame)? {
            return Ok("播放完毕".into());
        }
        let t = reader.timestamp();
        // 按时间戳节拍播放：处理比实时快就等，慢就不等（掉速而不是丢帧）
        let due = Duration::from_secs_f64(t);
        if let Some(wait) = due.checked_sub(started.elapsed()) {
            std::thread::sleep(wait);
        }

        let (rect, win) = {
            let g = gimbal.lock().unwrap();
            let g = g.as_ref().ok_or_else(|| anyhow::anyhow!("云台未初始化"))?;
            (g.rect(), g.crop_rgb(&frame))
        };
        let (full_small, fw, fh) = downscale_rgb(&frame, info.width, info.height, PREVIEW_FULL_W);
        let (win_small, ww, wh) = downscale_rgb(&win, rect.w, rect.h, PREVIEW_WIN_W);

        window_frames += 1;
        let el = window_start.elapsed().as_secs_f64();
        if el >= 1.0 {
            fps = window_frames as f64 / el;
            window_frames = 0;
            window_start = Instant::now();
        }
        app.emit(
            "preview",
            Preview {
                index: reader.index - 1,
                t,
                fps,
                full: jpeg_b64(&full_small, fw, fh)?,
                window: jpeg_b64(&win_small, ww, wh)?,
                rect,
                full_size: (info.width, info.height),
            },
        )?;
    }
    Ok("已停止".into())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![open_video, set_gimbal, start, stop])
        .run(tauri::generate_context!())
        .expect("启动 Tauri 应用失败");
}
