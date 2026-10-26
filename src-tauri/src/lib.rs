//! motion-gimbal 宿主：视频帧源 → 虚拟云台取景 →（帧环 → Python 视觉服务 → 瞄准点）→ 前端预览。
//!
//! 两种模式：
//! - **手动**：云台只受前端点击控制（`set_gimbal`），不连视觉服务；
//! - **自动跟拍**：开始播放时拉起/连接视觉服务，帧经共享内存帧环送过去，结果线程收到瞄准点即调
//!   [`VirtualGimbal::aim_in`]。播放中可随时切回「只显示不驱动」，点击仍可手动改云台。

pub mod gimbal;
pub mod ring;
pub mod session;
pub mod video;
pub mod vision;

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use base64::Engine;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use gimbal::{downscale_rgb, Rect, VirtualGimbal};
use video::VideoInfo;
use vision::VisionService;

/// 取景窗占全幅的比例（宽、高同比）。
const WINDOW_FRAC: f64 = 0.5;
/// 预览图的最大宽度（只影响界面，不进算法）。
const PREVIEW_FULL_W: u32 = 640;
const PREVIEW_WIN_W: u32 = 480;
const PREVIEW_JPEG_Q: u8 = 75;
/// 视觉服务端口（`MG_VISION_PORT` 可改）。
const VISION_PORT: u16 = 8765;

#[derive(Default)]
struct AppState {
    info: Mutex<Option<VideoInfo>>,
    gimbal: Arc<Mutex<Option<VirtualGimbal>>>,
    session: Mutex<Option<Session>>,
    /// 自动跟拍：结果是否驱动云台。
    auto: Arc<AtomicBool>,
    vision: Arc<Mutex<Option<VisionService>>>,
}

struct Session {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

/// 叠加层：全部在全幅坐标里。
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Overlay {
    k: u64,
    frame: u64,
    /// 瞄准点（全幅）= 结果里的 origin + aim。
    aim: Option<(f64, f64)>,
    target: Option<(f64, f64)>,
    bbox: Option<(f64, f64, f64, f64)>,
    tracks: Vec<(i64, f64, f64)>,
    phase: String,
    tid: Option<i64>,
    n_dets: u64,
    /// 帧写入帧环 → 云台动作（ms）。
    latency_ms: f64,
    /// 服务端处理一帧（ms）。
    proc_ms: f64,
    drove: bool,
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
    auto: bool,
    tracking: bool,
    /// 上一次推送以来收到的结果（通常 0 或 1 条）。
    overlays: Vec<Overlay>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Ended {
    reason: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AutoplayConfig {
    path: Option<String>,
    auto: bool,
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

/// `MG_METRICS_LOG` 指向的 JSON Lines 文件（测量用；不设就不记）。
fn metrics_log(v: serde_json::Value) {
    if let Some(p) = std::env::var_os("MG_METRICS_LOG") {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            let _ = writeln!(f, "{v}");
        }
    }
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

/// 自动跟拍开关。播放前打开 ⇒ 开始时接入视觉服务；播放中切换 ⇒ 只改「结果是否驱动云台」。
#[tauri::command]
fn set_auto(on: bool, state: State<'_, AppState>) {
    state.auto.store(on, Ordering::SeqCst);
}

#[tauri::command]
fn autoplay_config() -> AutoplayConfig {
    AutoplayConfig {
        path: std::env::var("MG_AUTOPLAY").ok().filter(|s| !s.is_empty()),
        auto: std::env::var("MG_AUTOPLAY_AUTO").map(|v| v == "1").unwrap_or(false),
    }
}

/// 前端每秒报一次它实际收到并显示的帧率（只写测量日志）。
#[tauri::command]
fn ui_stats(recv_fps: f64, index: u64) {
    metrics_log(serde_json::json!({ "kind": "ui", "recvFps": recv_fps, "index": index }));
}

#[tauri::command]
fn start(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    stop_session(&state);
    let info = state.info.lock().unwrap().clone().ok_or("先打开一段视频")?;
    let gimbal = Arc::clone(&state.gimbal);
    let auto = Arc::clone(&state.auto);
    let vision = Arc::clone(&state.vision);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_flag = Arc::clone(&stop);
    let tracking = auto.load(Ordering::SeqCst);
    let handle = std::thread::spawn(move || {
        let reason = match play(&app, &info, &gimbal, &auto, &vision, tracking, &stop_flag) {
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

/// 播放：可选接入视觉服务，每帧把全景与取景窗（缩小、JPEG、Base64）连同叠加层推给前端。
fn play(
    app: &AppHandle,
    info: &VideoInfo,
    gimbal: &Arc<Mutex<Option<VirtualGimbal>>>,
    auto: &Arc<AtomicBool>,
    vision: &Mutex<Option<VisionService>>,
    tracking: bool,
    stop: &AtomicBool,
) -> anyhow::Result<String> {
    let mut svc_guard = vision.lock().unwrap();
    if tracking {
        let port = std::env::var("MG_VISION_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(VISION_PORT);
        if svc_guard.as_ref().map(|s| !s.healthy()).unwrap_or(true) {
            let _ = app.emit("status", "正在启动视觉服务…");
            *svc_guard = Some(VisionService::ensure(port, true, Duration::from_secs(60))?);
            let _ = app.emit("status", "");
        }
    }
    let svc = if tracking { svc_guard.as_ref() } else { None };
    let opts = session::Options {
        paced: std::env::var("MG_UNPACED").map(|v| v != "1").unwrap_or(true),
        ..Default::default()
    };

    let mut window_start = Instant::now();
    let mut window_frames = 0u32;
    let mut encode_ms = 0.0;
    let mut fps = 0.0;
    let mut tick = |tk: session::Tick| -> anyhow::Result<()> {
        let overlays: Vec<Overlay> = tk
            .applied
            .iter()
            .filter(|a| !a.result.end)
            .map(|a| {
                let r = &a.result;
                Overlay {
                    k: r.k,
                    frame: r.frame,
                    aim: r.aim.map(|(x, y)| (r.origin.0 + x, r.origin.1 + y)),
                    target: r.target,
                    bbox: r.bbox,
                    tracks: r.tracks.clone(),
                    phase: r.phase.clone(),
                    tid: r.tid,
                    n_dets: r.n_dets,
                    latency_ms: a.latency.as_secs_f64() * 1e3,
                    proc_ms: r.timing.get("proc_ms").copied().unwrap_or(f64::NAN),
                    drove: a.drove,
                }
            })
            .collect();
        if tk.index == session::TAIL {
            return Ok(());
        }
        let e0 = Instant::now();
        // 推送用这一帧实际裁取景窗的矩形（结果线程可能已经把云台移走了）
        let win = gimbal::crop_rgb(tk.frame, info.width, tk.rect);
        let (full_small, fw, fh) = downscale_rgb(tk.frame, info.width, info.height, PREVIEW_FULL_W);
        let (win_small, ww, wh) = downscale_rgb(&win, tk.rect.w, tk.rect.h, PREVIEW_WIN_W);
        let full = jpeg_b64(&full_small, fw, fh)?;
        let window = jpeg_b64(&win_small, ww, wh)?;
        encode_ms += e0.elapsed().as_secs_f64() * 1e3;

        window_frames += 1;
        let el = window_start.elapsed().as_secs_f64();
        if el >= 1.0 {
            fps = window_frames as f64 / el;
            metrics_log(serde_json::json!({
                "kind": "push", "pushFps": fps, "index": tk.index, "w": info.width, "h": info.height,
                "encodeMs": encode_ms / window_frames as f64, "tracking": tracking,
                "paced": opts.paced,
            }));
            window_frames = 0;
            encode_ms = 0.0;
            window_start = Instant::now();
        }
        app.emit(
            "preview",
            Preview {
                index: tk.index,
                t: tk.t,
                fps,
                full,
                window,
                rect: tk.rect,
                full_size: (info.width, info.height),
                auto: auto.load(Ordering::Relaxed),
                tracking,
                overlays,
            },
        )?;
        Ok(())
    };
    session::run(info, gimbal, stop, auto, &opts, svc, &mut tick)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![open_video, set_gimbal, set_auto, start, stop, autoplay_config, ui_stats])
        .build(tauri::generate_context!())
        .expect("启动 Tauri 应用失败")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                let state = app.state::<AppState>();
                stop_session(&state);
                // 拉起的 Python 进程随宿主退出
                state.vision.lock().unwrap().take();
            }
        });
}
