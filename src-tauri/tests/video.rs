//! 视频帧源：用 ffmpeg 现场生成一段测试视频，验证探测与逐帧读取。

use std::process::Command;

use motion_gimbal_lib::video::{probe, FrameReader};

fn make_clip(dir: &std::path::Path) -> String {
    let path = dir.join("testsrc.mp4");
    let ok = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=1"])
        .args(["-pix_fmt", "yuv420p"])
        .arg(&path)
        .status()
        .expect("需要 ffmpeg 在 PATH 里")
        .success();
    assert!(ok, "ffmpeg 生成测试视频失败");
    path.to_string_lossy().into_owned()
}

#[test]
fn a_clip_is_probed_and_read_frame_by_frame() {
    let dir = std::env::temp_dir().join(format!("mg-video-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = make_clip(&dir);

    let info = probe(&path).unwrap();
    assert_eq!((info.width, info.height), (320, 240));
    assert!((info.fps - 24.0).abs() < 1e-6, "帧率 {}", info.fps);

    let mut reader = FrameReader::open(&info).unwrap();
    let mut buf = vec![0u8; reader.frame_len()];
    let mut n = 0;
    let mut last_t = -1.0;
    while reader.next(&mut buf).unwrap() {
        let t = reader.timestamp();
        assert!(t > last_t, "时间戳必须递增");
        last_t = t;
        n += 1;
    }
    assert_eq!(n, 24, "1 秒 24fps 应读到 24 帧");
    assert!(buf.iter().any(|&b| b != 0), "读到的不该是全黑");
    drop(reader);
    let _ = std::fs::remove_dir_all(&dir);
}
