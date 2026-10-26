//! **跨语言链路实测**：帧环单帧读写、控制面往返、写帧→云台动作的端到端延迟、各分辨率下的处理帧率。
//!
//! ```text
//! cargo run --release --example bench_ipc -- --media ../runs/ipc-media --out ../runs/ipc-bench-<时间>.json
//! ```
//!
//! `--media` 下要有 `viper_{640x360,1280x720,1920x1080}_15.mp4`（MoCA 帧拼成的 15 fps 视频）。
//! 视觉服务由本程序在空闲端口拉起，`OMP_NUM_THREADS=1`、`MG_CV_THREADS=1`（单线程，避免争用失真）。
//! 所有数字都是这次运行实测的；CPU 占用每 2 秒采样一次一并记录。

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use motion_gimbal_lib::gimbal::VirtualGimbal;
use motion_gimbal_lib::ring::{FrameMeta, Mode, PixelFormat, RingReader, RingWriter};
use motion_gimbal_lib::session::{self, Options, TAIL};
use motion_gimbal_lib::video::probe;
use motion_gimbal_lib::vision::VisionService;
use serde_json::{json, Value};

fn pct(v: &[f64]) -> Value {
    let mut v: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return Value::Null;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| v[((f * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
    json!({
        "n": v.len(), "mean": v.iter().sum::<f64>() / v.len() as f64,
        "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "min": v[0], "max": v[v.len() - 1],
    })
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn powershell(cmd: &str) -> String {
    Command::new("powershell")
        .args(["-NoProfile", "-Command", cmd])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// 后台每 2 秒采一次整机 CPU 占用（%）。
struct CpuSampler {
    stop: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<f64>>>,
    h: Option<std::thread::JoinHandle<()>>,
}

impl CpuSampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(Mutex::new(Vec::new()));
        let (s2, v2) = (Arc::clone(&stop), Arc::clone(&samples));
        let h = std::thread::spawn(move || {
            while !s2.load(Ordering::Relaxed) {
                if let Ok(x) = powershell("(Get-CimInstance Win32_Processor).LoadPercentage").parse::<f64>() {
                    v2.lock().unwrap().push(x);
                }
                for _ in 0..20 {
                    if s2.load(Ordering::Relaxed) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        });
        Self { stop, samples, h: Some(h) }
    }

    fn finish(mut self) -> Value {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.h.take() {
            let _ = h.join();
        }
        let v = self.samples.lock().unwrap().clone();
        json!({ "samples": v, "summary": pct(&v) })
    }
}

/// 帧环单帧写 / 读（Rust 写者 + Rust 读者，同进程；Python 读者的拷贝耗时见链路结果里的 `read_ms`）。
fn ring_micro(w: u32, h: u32, iters: usize) -> Value {
    let name = RingWriter::unique_name("bench");
    let len = (w * h * 3) as usize;
    let wr = RingWriter::create(&name, 4, len, Mode::Block).unwrap();
    let mut rd = RingReader::open(&name).unwrap();
    let px: Vec<u8> = (0..len).map(|i| (i % 253) as u8).collect();
    let stop = AtomicBool::new(false);
    let mut buf = Vec::with_capacity(len);
    let (mut wt, mut rt) = (Vec::new(), Vec::new());
    for k in 0..iters as u64 {
        let meta = FrameMeta { frame_no: k, t: k as f64 / 15.0, width: w, height: h, format: PixelFormat::Rgb24, roi: (0, 0, w / 2, h / 2) };
        let t0 = Instant::now();
        wr.write(&meta, &px, &stop).unwrap();
        let t1 = Instant::now();
        rd.next(&mut buf, Duration::from_secs(1)).unwrap();
        let t2 = Instant::now();
        wt.push(ms(t1 - t0));
        rt.push(ms(t2 - t1));
    }
    assert_eq!(buf, px);
    json!({ "width": w, "height": h, "bytes": len, "iters": iters, "write_ms": pct(&wt), "read_ms": pct(&rt) })
}

fn rtt(svc: &VisionService, path: &str, n: usize) -> Value {
    let mut v = Vec::new();
    for _ in 0..n {
        let t0 = Instant::now();
        svc.get(path).unwrap();
        v.push(ms(t0.elapsed()));
    }
    pct(&v)
}

struct RunOut {
    json: Value,
}

/// 跑一段视频：`paced` = 按 15 fps 时间戳节拍（实时口径）；否则尽快（最大吞吐，受帧环背压）。
fn chain(svc: &VisionService, path: &str, proc_width: u32, paced: bool, probe_rtt: bool) -> RunOut {
    let info = probe(path).unwrap();
    let full = (info.width, info.height);
    let gimbal = Arc::new(Mutex::new(Some(VirtualGimbal::new(full, (full.0 / 2, full.1 / 2)))));
    let auto = Arc::new(AtomicBool::new(true));
    let stop = AtomicBool::new(false);
    let opts = Options { paced, lockstep: false, params: json!({ "proc_width": proc_width }), ..Default::default() };

    // 运行期间每 100 ms 测一次控制面往返（GET /status）
    let rtt_stop = Arc::new(AtomicBool::new(false));
    let rtt_h = probe_rtt.then(|| {
        let port = svc.port;
        let s = Arc::clone(&rtt_stop);
        std::thread::spawn(move || {
            let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(5)).build();
            let url = format!("http://127.0.0.1:{port}/status");
            let mut v = Vec::new();
            while !s.load(Ordering::Relaxed) {
                let t0 = Instant::now();
                if agent.get(&url).call().is_ok() {
                    v.push(ms(t0.elapsed()));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            v
        })
    });

    let cpu = CpuSampler::start();
    let (mut lat, mut proc_ms, mut read_ms, mut cand, mut core_ms, mut prep) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    let (mut wcopy, mut wwait) = (vec![], vec![]);
    let mut arrivals: Vec<Instant> = Vec::new();
    // 节拍滞后：这一帧实际写进帧环的时刻比它的时间戳晚了多少（相对第一帧对齐）
    let mut lag = Vec::new();
    let mut first: Option<(Instant, f64)> = None;
    let (mut frames, mut aims, mut skipped) = (0u64, 0u64, 0u64);
    let t0 = Instant::now();
    let reason = session::run(&info, &gimbal, &stop, &auto, &opts, Some(svc), &mut |tk| {
        if tk.index != TAIL {
            frames += 1;
            let now = Instant::now();
            let (i0, t0v) = *first.get_or_insert((now, tk.t));
            lag.push(ms(now - i0) - (tk.t - t0v) * 1e3);
        }
        if let Some(w) = tk.write {
            wcopy.push(ms(w.copy));
            wwait.push(ms(w.waited));
        }
        for a in &tk.applied {
            let r = &a.result;
            if r.end {
                continue;
            }
            arrivals.push(Instant::now());
            lat.push(ms(a.latency));
            let g = |k: &str| r.timing.get(k).copied().unwrap_or(f64::NAN);
            proc_ms.push(g("proc_ms"));
            read_ms.push(g("read_ms"));
            cand.push(g("cand_ms"));
            core_ms.push(g("core_ms"));
            prep.push(g("prep_ms"));
            aims += r.aim.is_some() as u64;
            skipped += r.skipped;
        }
        Ok(())
    })
    .unwrap();
    let wall = t0.elapsed().as_secs_f64();
    // 第一帧本身晚到（ffmpeg 起进程、会话建立），以「最早的一帧」为零点：之后每帧比节拍晚多少
    let lag0 = lag.iter().copied().fold(f64::INFINITY, f64::min);
    let lag: Vec<f64> = lag.iter().map(|x| x - lag0).collect();
    rtt_stop.store(true, Ordering::Relaxed);
    let rtt_v = rtt_h.map(|h| h.join().unwrap());
    let duration = frames as f64 / info.fps;
    // 处理帧率：结果数 ÷ 第一条到最后一条结果之间的时长
    let proc_fps = if arrivals.len() > 1 {
        (arrivals.len() - 1) as f64 / (arrivals[arrivals.len() - 1] - arrivals[0]).as_secs_f64()
    } else {
        f64::NAN
    };
    let p50 = |v: &Vec<f64>| pct(v)["p50"].as_f64().unwrap_or(f64::NAN);
    let interval_ms = 1e3 / info.fps;
    let json = json!({
        "video": path, "width": info.width, "height": info.height, "source_fps": info.fps,
        "proc_width": if proc_width == 0 { Value::from("native") } else { Value::from(proc_width) },
        "paced": paced, "reason": reason, "frames": frames, "results": lat.len(), "aims": aims, "skipped": skipped,
        "wall_s": wall, "video_duration_s": duration,
        "realtime_factor": duration / wall,
        "throughput_fps": frames as f64 / wall,
        "result_fps": proc_fps,
        "latency_ms": pct(&lat),
        "service_proc_ms": pct(&proc_ms),
        "service_read_ms": pct(&read_ms),
        "service_prep_ms": pct(&prep),
        "service_cand_ms": pct(&cand),
        "service_core_ms": pct(&core_ms),
        "ring_write_copy_ms": pct(&wcopy),
        "ring_write_wait_ms": pct(&wwait),
        "pacing_lag_ms": pct(&lag),
        "control_rtt_under_load_ms": rtt_v.map(|v| pct(&v)).unwrap_or(Value::Null),
        "keeps_15hz": paced.then(|| {
            let lag_p99 = pct(&lag)["p99"].as_f64().unwrap_or(f64::INFINITY);
            lat.len() as u64 == frames && lag_p99 < interval_ms && p50(&proc_ms) < interval_ms
        }),
        "cpu_load_percent": cpu.finish(),
    });
    eprintln!(
        "{}x{} pw={} paced={} 帧 {} 结果 {} 墙钟 {:.1}s 实时比 {:.3} 吞吐 {:.1}fps 结果 {:.1}fps 延迟p50 {:.1}ms 处理p50 {:.1}ms",
        info.width, info.height, proc_width, paced, frames, lat.len(), wall, duration / wall, frames as f64 / wall, proc_fps,
        p50(&lat), p50(&proc_ms)
    );
    RunOut { json }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let media = arg("--media").unwrap_or_else(|| "../runs/ipc-media".into());
    let out = arg("--out").expect("--out <json>");
    let only = arg("--only"); // 例如 1280x720，只跑一种分辨率（调试用）

    // 视觉服务单线程运行（由本进程拉起，环境变量会被继承）
    for (k, v) in [("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("MG_CV_THREADS", "1")] {
        std::env::set_var(k, v);
    }
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let svc = VisionService::ensure(port, true, Duration::from_secs(90)).expect("拉起视觉服务");

    let machine = json!({
        "cpu": powershell("(Get-CimInstance Win32_Processor).Name"),
        "cores": powershell("(Get-CimInstance Win32_Processor).NumberOfCores"),
        "logical": powershell("(Get-CimInstance Win32_Processor).NumberOfLogicalProcessors"),
        "ram_bytes": powershell("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"),
        "os": powershell("(Get-CimInstance Win32_OperatingSystem).Caption + ' ' + (Get-CimInstance Win32_OperatingSystem).Version"),
        "build": if cfg!(debug_assertions) { "debug" } else { "release" },
        "service_threads": "OMP_NUM_THREADS=1, cv2.setNumThreads(1)",
    });
    eprintln!("{machine}");

    let cpu_idle = CpuSampler::start();
    let micro = vec![ring_micro(1280, 720, 300), ring_micro(1920, 1080, 300)];
    let control_idle = json!({ "GET /status": rtt(&svc, "/status", 500), "GET /health": rtt(&svc, "/health", 500) });
    let cpu_idle = cpu_idle.finish();
    eprintln!("帧环 {micro:?}\n控制面 {control_idle}");

    let mut runs = Vec::new();
    for res in ["640x360", "1280x720", "1920x1080"] {
        if only.as_deref().is_some_and(|o| o != res) {
            continue;
        }
        let path = format!("{media}/viper_{res}_15.mp4");
        for pw in [640u32, 0] {
            if res == "640x360" && pw == 0 {
                continue; // 640 宽时两者相同
            }
            runs.push(chain(&svc, &path, pw, true, res == "1280x720" && pw == 640).json);
            runs.push(chain(&svc, &path, pw, false, false).json);
        }
    }

    let doc = json!({
        "kind": "ipc-bench",
        "date": powershell("Get-Date -Format o"),
        "machine": machine,
        "method": {
            "ring_micro": "同进程 Rust 写者 + Rust 读者，Block 模式 4 槽；写 = seq 置奇→拷像素→seq 置偶→发布；读 = 序列锁校验 + 拷出到 Vec；各 300 次",
            "control_rtt": "ureq 阻塞客户端对 FastAPI/uvicorn 的 GET 往返（127.0.0.1，长连接）",
            "latency": "宿主写帧环完成的时刻 → 结果线程收到该帧结果并完成 VirtualGimbal::aim_in 的时刻（同一进程 Instant）",
            "paced": "按视频时间戳（15 fps）节拍写帧；pacing_lag_ms = 每帧实际写入时刻相对其时间戳的滞后（以最不滞后的一帧为零点）；keeps_15hz = 结果数==帧数 且 滞后 p99 < 一个帧间隔（66.7 ms）且 服务处理中位数 < 帧间隔；realtime_factor = 视频时长 / 墙钟（墙钟含会话建立与收尾）",
            "unpaced": "不等节拍、尽快写（Block 模式环满等待），throughput_fps 为整条链（解码+写环+处理+回传）的吞吐",
            "proc_width": "640 = 帧先缩放到 640 宽等效再处理（算法标定口径）；native = 原分辨率处理",
            "perception": "window（闭环：只看取景窗，取景窗 = 全幅的 1/2 × 1/2）",
        },
        "cpu_load_percent_during_micro": cpu_idle,
        "ring_micro": micro,
        "control_rtt_idle_ms": control_idle,
        "runs": runs,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    eprintln!("写入 {out}");
}
