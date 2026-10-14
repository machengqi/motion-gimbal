//! 虚拟云台的几何：钳边界、对准、裁剪。

use motion_gimbal_lib::gimbal::{downscale_rgb, VirtualGimbal};

#[test]
fn the_window_never_leaves_the_frame() {
    let mut g = VirtualGimbal::new((1280, 720), (640, 360));
    g.set_center(-1000.0, 99999.0);
    let r = g.rect();
    assert_eq!((r.x, r.y), (0, 360));
    g.set_center(99999.0, -5.0);
    let r = g.rect();
    assert_eq!((r.x, r.y), (640, 0));
}

#[test]
fn aiming_moves_the_centre_onto_the_point_in_the_window() {
    let mut g = VirtualGimbal::new((1280, 720), (640, 360));
    // 初始中心 (640, 360)，窗口左上角 (320, 180)
    g.aim(420.0, 230.0); // 窗口坐标 → 全幅 (740, 410)
    assert_eq!(g.center(), (740.0, 410.0));
}

#[test]
fn cropping_takes_exactly_the_window_pixels() {
    let (w, h) = (8u32, 6u32);
    // 每个像素的 R 通道 = 列号，G = 行号
    let frame: Vec<u8> = (0..h).flat_map(|y| (0..w).flat_map(move |x| [x as u8, y as u8, 0])).collect();
    let mut g = VirtualGimbal::new((w, h), (4, 2));
    g.set_center(5.0, 4.0); // 窗口左上角 (3, 3)
    let crop = g.crop_rgb(&frame);
    assert_eq!(crop.len(), 4 * 2 * 3);
    assert_eq!(&crop[0..3], &[3, 3, 0]);
    assert_eq!(&crop[crop.len() - 3..], &[6, 4, 0]);
}

#[test]
fn downscaling_keeps_the_aspect_ratio() {
    let src = vec![0u8; 1280 * 720 * 3];
    let (out, w, h) = downscale_rgb(&src, 1280, 720, 640);
    assert_eq!((w, h), (640, 360));
    assert_eq!(out.len(), 640 * 360 * 3);
}
