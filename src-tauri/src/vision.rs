//! **视觉服务客户端** —— 控制面（HTTP）+ 结果流（WebSocket）+ 拉起 Python 进程。
//!
//! ```text
//! 宿主 ──帧环（共享内存）──▶ Python 服务
//!      ◀──WS /results（逐帧 JSON）──
//!      ──HTTP /session/start|stop, /status, /params──▶
//! ```
//!
//! 结果线程收到一帧结果就**立即**调 [`VirtualGimbal::aim_in`]（不等播放循环下一拍），
//! 所以「写帧 → 云台动作」的延迟里没有播放节拍的等待。

use std::collections::HashMap;
use std::io::ErrorKind;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::gimbal::{Rect, VirtualGimbal};

/// 服务回传的一帧结果（字段含义见 `vision/src/gimbal_vision/service.py`）。
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct VisionResult {
    pub session: String,
    pub k: u64,
    /// 帧环写入序号。
    pub index: u64,
    pub frame: u64,
    pub t: f64,
    pub skipped: u64,
    /// 送去感知的那块画面在全幅里的左上角。
    pub origin: (f64, f64),
    /// 瞄准点，`origin` 那块画面里的坐标。
    pub aim: Option<(f64, f64)>,
    pub phase: String,
    pub tid: Option<i64>,
    /// 以下全幅坐标，只给界面用。
    pub target: Option<(f64, f64)>,
    #[serde(rename = "box")]
    pub bbox: Option<(f64, f64, f64, f64)>,
    pub tracks: Vec<(i64, f64, f64)>,
    pub n_dets: u64,
    pub bg_ok: bool,
    pub timing: HashMap<String, f64>,
    pub crc32: Option<u32>,
    pub error: Option<String>,
    pub end: bool,
}

/// 一条已经处理过（可能已驱动云台）的结果。
#[derive(Debug, Clone)]
pub struct Applied {
    pub result: VisionResult,
    /// 帧写进帧环 → 收到结果并完成 `aim_in` 的时间。
    pub latency: Duration,
    /// 是否驱动了云台（手动模式下只显示不驱动）。
    pub drove: bool,
    pub rect_after: Option<Rect>,
}

// ── 进程与控制面 ─────────────────────────────────────────────────
pub struct VisionService {
    pub port: u16,
    child: Option<Child>,
    agent: ureq::Agent,
}

fn vision_dir() -> PathBuf {
    std::env::var_os("MG_VISION_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("vision"))
}

impl VisionService {
    fn agent() -> ureq::Agent {
        ureq::AgentBuilder::new().timeout(Duration::from_secs(10)).build()
    }

    /// 连接 `port` 上已在运行的服务；没有就在 `vision/` 下拉起一个，等它就绪（最多 `wait`）。
    pub fn ensure(port: u16, spawn: bool, wait: Duration) -> Result<Self> {
        let mut s = Self { port, child: None, agent: Self::agent() };
        if s.healthy() {
            return Ok(s);
        }
        if !spawn {
            bail!("127.0.0.1:{port} 上没有视觉服务");
        }
        let dir = vision_dir();
        let py = if cfg!(windows) { dir.join(".venv/Scripts/python.exe") } else { dir.join(".venv/bin/python") };
        // 直接起 venv 里的解释器：经 `uv run` 中转时，结束 uv 不一定能带走子进程
        let mut cmd = if py.exists() {
            let mut c = Command::new(py);
            c.args(["-m", "gimbal_vision.service", "--port", &port.to_string()]);
            c
        } else {
            let mut c = Command::new("uv");
            c.args(["run", "--extra", "service", "python", "-m", "gimbal_vision.service", "--port", &port.to_string()]);
            c
        };
        let log = std::env::temp_dir().join(format!("motion-gimbal-vision-{port}.log"));
        let f = std::fs::File::create(&log)?;
        let child = cmd
            .current_dir(&dir)
            .stdin(Stdio::null())
            .stdout(f.try_clone()?)
            .stderr(f)
            .spawn()
            .with_context(|| format!("在 {dir:?} 拉起视觉服务失败"))?;
        s.child = Some(child);
        let t0 = Instant::now();
        while t0.elapsed() < wait {
            if s.healthy() {
                return Ok(s);
            }
            if let Some(c) = s.child.as_mut() {
                if let Ok(Some(st)) = c.try_wait() {
                    bail!("视觉服务退出了（{st}），日志 {log:?}");
                }
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        bail!("视觉服务 {wait:?} 内没有就绪，日志 {log:?}")
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }

    pub fn healthy(&self) -> bool {
        ureq::AgentBuilder::new()
            .timeout(Duration::from_millis(500))
            .build()
            .get(&self.url("/health"))
            .call()
            .is_ok()
    }

    pub fn get(&self, path: &str) -> Result<Value> {
        Ok(self.agent.get(&self.url(path)).call().map_err(http_err)?.into_json()?)
    }

    pub fn post(&self, path: &str, body: &Value) -> Result<Value> {
        Ok(self.agent.post(&self.url(path)).send_json(body.clone()).map_err(http_err)?.into_json()?)
    }

    pub fn put(&self, path: &str, body: &Value) -> Result<Value> {
        Ok(self.agent.put(&self.url(path)).send_json(body.clone()).map_err(http_err)?.into_json()?)
    }
}

fn http_err(e: ureq::Error) -> anyhow::Error {
    match e {
        ureq::Error::Status(code, r) => anyhow!("HTTP {code}：{}", r.into_string().unwrap_or_default()),
        e => anyhow!(e),
    }
}

impl Drop for VisionService {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

// ── 结果流 ───────────────────────────────────────────────────────
/// 写帧时刻表：写入序号 → 写完的时刻（结果线程据此算延迟）。
pub type WriteTimes = Arc<Mutex<HashMap<u64, Instant>>>;

/// 一条结果流：后台线程读 WS，收到即驱动云台，再把 [`Applied`] 交给播放循环。
pub struct ResultStream {
    pub rx: Receiver<Applied>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl ResultStream {
    /// `auto` 为真时用结果驱动云台；为假时只回传（界面照样叠加）。
    pub fn connect(
        port: u16,
        session: &str,
        gimbal: Arc<Mutex<Option<VirtualGimbal>>>,
        auto: Arc<AtomicBool>,
        written: WriteTimes,
    ) -> Result<Self> {
        let tcp = TcpStream::connect(("127.0.0.1", port)).context("连接视觉服务的结果流")?;
        tcp.set_nodelay(true)?;
        let (mut ws, _) = tungstenite::client(format!("ws://127.0.0.1:{port}/results"), tcp)
            .map_err(|e| anyhow!("WebSocket 握手失败：{e}"))?;
        ws.get_ref().set_read_timeout(Some(Duration::from_millis(100)))?;
        let (tx, rx): (Sender<Applied>, Receiver<Applied>) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = Arc::clone(&stop);
        let session = session.to_string();
        let handle = std::thread::spawn(move || {
            while !stop2.load(Ordering::Relaxed) {
                let msg = match ws.read() {
                    Ok(m) => m,
                    Err(tungstenite::Error::Io(e)) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => continue,
                    Err(_) => break,
                };
                let text = match msg {
                    tungstenite::Message::Text(t) => t,
                    tungstenite::Message::Close(_) => break,
                    _ => continue,
                };
                let Ok(r) = serde_json::from_str::<VisionResult>(&text) else { continue };
                if r.session != session {
                    continue;
                }
                let mut drove = false;
                let mut rect_after = None;
                if let (Some((ax, ay)), true) = (r.aim, auto.load(Ordering::Relaxed)) {
                    if let Some(g) = gimbal.lock().unwrap().as_mut() {
                        g.aim_in(r.origin.0, r.origin.1, ax, ay);
                        rect_after = Some(g.rect());
                        drove = true;
                    }
                }
                let now = Instant::now();
                let latency = written
                    .lock()
                    .unwrap()
                    .remove(&r.index)
                    .map(|t| now.saturating_duration_since(t))
                    .unwrap_or_default();
                if tx.send(Applied { result: r, latency, drove, rect_after }).is_err() {
                    break;
                }
            }
            let _ = ws.close(None);
        });
        Ok(Self { rx, stop, handle: Some(handle) })
    }
}

impl Drop for ResultStream {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
