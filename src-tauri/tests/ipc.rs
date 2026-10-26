//! **Rust–Python 联调**：拉起真实的 Python 视觉服务，宿主播放循环把帧写进共享内存帧环，
//! 校验结果经 WebSocket 回传、像素逐字节一致、瞄准点按「那一帧的取景窗」驱动了虚拟云台。
//!
//! 需要 `vision/.venv`（`cd vision && uv sync`）和 PATH 里的 ffmpeg。

use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use motion_gimbal_lib::gimbal::VirtualGimbal;
use motion_gimbal_lib::session::{self, Options, TAIL};
use motion_gimbal_lib::video::probe;
use motion_gimbal_lib::vision::{Applied, VisionService};
use serde_json::json;

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, e) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *e = c;
    }
    let mut c = !0u32;
    for &b in data {
        c = table[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    !c
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn clip(dir: &std::path::Path, frames: u32) -> String {
    let path = dir.join("pan.mp4");
    // 有平移、有独立运动的测试图样（testsrc2 的色块与计数器在动）
    let ok = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i", &format!("testsrc2=size=480x270:rate=15,trim=end_frame={frames}")])
        .args(["-pix_fmt", "yuv420p"])
        .arg(&path)
        .status()
        .expect("需要 ffmpeg 在 PATH 里")
        .success();
    assert!(ok);
    path.to_string_lossy().into_owned()
}

struct Fixture {
    svc: VisionService,
    dir: std::path::PathBuf,
}

fn fixture(tag: &str) -> Fixture {
    let dir = std::env::temp_dir().join(format!("mg-ipc-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let svc = VisionService::ensure(free_port(), true, Duration::from_secs(90)).expect("拉起视觉服务");
    Fixture { svc, dir }
}

#[test]
fn frames_go_through_the_ring_and_results_drive_the_gimbal_in_lockstep() {
    let fx = fixture("lock");
    let n = 30;
    let info = probe(&clip(&fx.dir, n)).unwrap();
    let full = (info.width, info.height);
    let win = (full.0 / 2, full.1 / 2);
    let gimbal = Arc::new(Mutex::new(Some(VirtualGimbal::new(full, win))));
    let auto = Arc::new(AtomicBool::new(true));
    let stop = AtomicBool::new(false);
    let opts = Options { paced: false, lockstep: true, params: json!({ "checksum": true }), ..Default::default() };

    // 参照云台：只用结果按同样的规则推一遍，宿主每帧实际用的矩形必须与它一致
    let mut reference = VirtualGimbal::new(full, win);
    let mut results: Vec<Applied> = Vec::new();
    let mut ticks = 0u64;
    let reason = session::run(&info, &gimbal, &stop, &auto, &opts, Some(&fx.svc), &mut |tk| {
        if tk.index == TAIL {
            assert!(tk.applied.is_empty(), "lockstep 下不该有在途结果");
            return Ok(());
        }
        assert_eq!(tk.rect, reference.rect(), "第 {} 帧的取景窗应由上一帧的结果决定", tk.index);
        let w = tk.write.expect("自动模式下每帧都写帧环");
        assert_eq!(w.index, ticks);
        assert_eq!(tk.applied.len(), 1, "lockstep：一帧一结果");
        let a = &tk.applied[0];
        let r = &a.result;
        assert_eq!((r.index, r.frame), (w.index, tk.index));
        assert!((r.t - tk.t).abs() < 1e-9, "时间戳随帧传递");
        assert_eq!(r.crc32, Some(crc32(tk.frame)), "帧环传过去的像素必须逐字节一致");
        assert_eq!(r.origin, (tk.rect.x as f64, tk.rect.y as f64), "感知的是这一帧的取景窗");
        if let Some((x, y)) = r.aim {
            assert!(a.drove);
            reference.aim_in(r.origin.0, r.origin.1, x, y);
            assert_eq!(a.rect_after, Some(reference.rect()));
        }
        assert!(a.latency > Duration::ZERO);
        results.push(a.clone());
        ticks += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(reason, "播放完毕");
    assert_eq!(ticks, n as u64);
    assert!(results.iter().any(|a| a.result.aim.is_some()), "testsrc2 里有运动，应当至少有一次瞄准");

    // 控制面
    let st = fx.svc.get("/status").unwrap();
    assert_eq!(st["frames"], n);
    assert_eq!(st["running"], false);
    assert!(fx.svc.put("/params", &json!({ "perception": "nope" })).is_err());
    let _ = std::fs::remove_dir_all(&fx.dir);
}

#[test]
fn paced_playback_collects_every_result_and_manual_mode_does_not_move_the_gimbal() {
    let fx = fixture("paced");
    let n = 20;
    let info = probe(&clip(&fx.dir, n)).unwrap();
    let full = (info.width, info.height);
    let g0 = VirtualGimbal::new(full, (full.0 / 2, full.1 / 2));
    let gimbal = Arc::new(Mutex::new(Some(g0.clone())));
    // 结果照常回传，但「自动跟拍」关着 ⇒ 不驱动云台（手动模式保留）
    let auto = Arc::new(AtomicBool::new(false));
    let stop = AtomicBool::new(false);
    let opts = Options { paced: true, ..Default::default() };
    let mut got = 0;
    session::run(&info, &gimbal, &stop, &auto, &opts, Some(&fx.svc), &mut |tk| {
        for a in &tk.applied {
            assert!(!a.drove && a.rect_after.is_none());
            got += 1;
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(got, n, "每帧一条结果，收尾时收齐在途的");
    assert_eq!(gimbal.lock().unwrap().as_ref().unwrap().rect(), g0.rect());

    // 不接视觉服务的纯手动模式：不写帧环，没有结果
    let mut frames = 0;
    session::run(&info, &gimbal, &stop, &auto, &Options { paced: false, ..Default::default() }, None, &mut |tk| {
        assert!(tk.write.is_none() && tk.applied.is_empty());
        frames += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(frames, n);
    let _ = std::fs::remove_dir_all(&fx.dir);
}
