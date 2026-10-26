//! **虚拟云台** —— 在全幅画面里移动一个取景窗。
//!
//! 取景窗就是「装在云台上的相机」看到的画面：跟踪算法只看取景窗，云台转动 =
//! 取景窗在全幅里平移。同一段视频、同一份标注下可完全复现，便于离线实验。
//! 换成真实云台时，只需把 [`VirtualGimbal::aim`] 换成发给云台的转动指令。

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, Clone)]
pub struct VirtualGimbal {
    full: (u32, u32),
    win: (u32, u32),
    /// 取景窗中心（全幅坐标）。
    center: (f64, f64),
}

impl VirtualGimbal {
    /// `win` 会被钳到不超过全幅。初始对准画面中心。
    pub fn new(full: (u32, u32), win: (u32, u32)) -> Self {
        let win = (win.0.min(full.0).max(1), win.1.min(full.1).max(1));
        let mut g = Self { full, win, center: (0.0, 0.0) };
        g.set_center(full.0 as f64 / 2.0, full.1 as f64 / 2.0);
        g
    }

    pub fn center(&self) -> (f64, f64) {
        self.center
    }

    pub fn window_size(&self) -> (u32, u32) {
        self.win
    }

    /// 把中心设到 `(cx, cy)`（全幅坐标），钳在取景窗不出界的范围内。
    pub fn set_center(&mut self, cx: f64, cy: f64) {
        let hw = self.win.0 as f64 / 2.0;
        let hh = self.win.1 as f64 / 2.0;
        self.center = (
            cx.clamp(hw, self.full.0 as f64 - hw),
            cy.clamp(hh, self.full.1 as f64 - hh),
        );
    }

    /// **对准取景窗里的一点**（取景窗坐标）：中心移到那个点所在的全幅位置。
    ///
    /// 平滑性由上游控制器（Python `core.controller`）负责，这里只做几何与边界。
    pub fn aim(&mut self, x_in_window: f64, y_in_window: f64) {
        let r = self.rect();
        self.aim_in(r.x as f64, r.y as f64, x_in_window, y_in_window);
    }

    /// **对准某一帧取景窗里的一点**：`(ox, oy)` 是**那一帧**取景窗的左上角（全幅坐标）。
    ///
    /// 视觉服务的结果总是晚于帧到达：算出结果时云台可能已经动过（上一条结果、或手动操作）。
    /// 按结果所对应那一帧的取景窗换算，而不是按当前取景窗 —— 否则延迟期间的移动会被重复叠加。
    /// 与 [`aim`](Self::aim) 的关系：`aim(x, y) == aim_in(当前左上角, x, y)`。
    pub fn aim_in(&mut self, ox: f64, oy: f64, x: f64, y: f64) {
        self.set_center(ox + x, oy + y);
    }

    /// 取景窗在全幅里的整数像素矩形。
    pub fn rect(&self) -> Rect {
        let x = (self.center.0 - self.win.0 as f64 / 2.0).round().max(0.0) as u32;
        let y = (self.center.1 - self.win.1 as f64 / 2.0).round().max(0.0) as u32;
        Rect {
            x: x.min(self.full.0 - self.win.0),
            y: y.min(self.full.1 - self.win.1),
            w: self.win.0,
            h: self.win.1,
        }
    }

    /// 从 RGB24 全幅帧裁出取景窗。
    pub fn crop_rgb(&self, frame: &[u8]) -> Vec<u8> {
        crop_rgb(frame, self.full.0, self.rect())
    }
}

/// 从宽为 `full_w` 的 RGB24 帧裁出矩形 `r`。
pub fn crop_rgb(frame: &[u8], full_w: u32, r: Rect) -> Vec<u8> {
    let stride = full_w as usize * 3;
    let mut out = Vec::with_capacity((r.w * r.h * 3) as usize);
    for row in r.y..r.y + r.h {
        let start = row as usize * stride + r.x as usize * 3;
        out.extend_from_slice(&frame[start..start + r.w as usize * 3]);
    }
    out
}

/// 最近邻缩放 RGB24（预览用，不进算法）。
pub fn downscale_rgb(src: &[u8], w: u32, h: u32, max_w: u32) -> (Vec<u8>, u32, u32) {
    if w <= max_w {
        return (src.to_vec(), w, h);
    }
    let nw = max_w;
    let nh = ((h as u64 * nw as u64) / w as u64).max(1) as u32;
    let mut out = Vec::with_capacity((nw * nh * 3) as usize);
    for y in 0..nh {
        let sy = (y as u64 * h as u64 / nh as u64) as usize;
        for x in 0..nw {
            let sx = (x as u64 * w as u64 / nw as u64) as usize;
            let i = (sy * w as usize + sx) * 3;
            out.extend_from_slice(&src[i..i + 3]);
        }
    }
    (out, nw, nh)
}
