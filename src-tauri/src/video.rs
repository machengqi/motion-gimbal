//! **视频帧源** —— 交给 ffmpeg 解码，逐帧读 RGB24 原始像素。
//!
//! 为什么不链接解码库：ffmpeg 命令行在各平台都现成，按管道读原始帧最简单、
//! 也最容易换成摄像头输入（`-f dshow` / `-f v4l2`）。代价是多一次进程间拷贝，
//! 对 720p 来说可以忽略。

use std::io::Read;
use std::process::{Child, ChildStdout, Command, Stdio};

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoInfo {
    pub path: String,
    pub width: u32,
    pub height: u32,
    /// 标称帧率（`r_frame_rate`）。⚠️ 只是标称值：时间戳以逐帧读到的为准。
    pub fps: f64,
    pub frames: Option<u64>,
}

/// 用 ffprobe 读视频的尺寸与帧率。
pub fn probe(path: &str) -> Result<VideoInfo> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            path,
        ])
        .output()
        .context("启动 ffprobe 失败 —— 需要 ffmpeg 在 PATH 里")?;
    if !out.status.success() {
        bail!("ffprobe 读不了 {path}：{}", String::from_utf8_lossy(&out.stderr));
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    let s = v["streams"].get(0).ok_or_else(|| anyhow!("{path} 里没有视频流"))?;
    let num = |k: &str| s[k].as_u64().ok_or_else(|| anyhow!("ffprobe 没给 {k}"));
    let rate = s["r_frame_rate"].as_str().unwrap_or("0/1");
    let fps = match rate.split_once('/') {
        Some((a, b)) => a.parse::<f64>()? / b.parse::<f64>()?.max(1.0),
        None => rate.parse()?,
    };
    Ok(VideoInfo {
        path: path.to_string(),
        width: num("width")? as u32,
        height: num("height")? as u32,
        fps,
        frames: s["nb_frames"].as_str().and_then(|n| n.parse().ok()),
    })
}

/// 逐帧读 RGB24。帧号从 0 开始；时间戳 = 帧号 / 标称帧率。
pub struct FrameReader {
    child: Child,
    stdout: ChildStdout,
    pub info: VideoInfo,
    pub index: u64,
}

impl FrameReader {
    pub fn open(info: &VideoInfo) -> Result<Self> {
        let mut child = Command::new("ffmpeg")
            .args(["-v", "error", "-i", &info.path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .stdin(Stdio::null())
            .spawn()
            .context("启动 ffmpeg 失败 —— 需要 ffmpeg 在 PATH 里")?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("拿不到 ffmpeg 的输出管道"))?;
        Ok(Self { child, stdout, info: info.clone(), index: 0 })
    }

    pub fn frame_len(&self) -> usize {
        (self.info.width * self.info.height * 3) as usize
    }

    /// 读下一帧到 `buf`（长度必须是 `frame_len`）。到结尾返回 `false`。
    pub fn next(&mut self, buf: &mut [u8]) -> Result<bool> {
        match self.stdout.read_exact(buf) {
            Ok(()) => {
                self.index += 1;
                Ok(true)
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// 这一帧（刚读到的）的标称时间戳（秒）。
    pub fn timestamp(&self) -> f64 {
        (self.index.saturating_sub(1)) as f64 / self.info.fps.max(1e-6)
    }
}

impl Drop for FrameReader {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
