//! 共享内存帧环（Rust 写者 + Rust 读者）：往返、半帧、覆盖、背压、并发下不读到撕裂帧。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use motion_gimbal_lib::ring::{FrameMeta, Mode, PixelFormat, ReadOutcome, RingReader, RingWriter, WriteOutcome};

fn meta(k: u64, w: u32, h: u32) -> FrameMeta {
    FrameMeta { frame_no: k, t: k as f64 / 30.0, width: w, height: h, format: PixelFormat::Rgb24, roi: (k as u32 % 7, 1, 2, 3) }
}

fn filled(k: u64, w: u32, h: u32) -> Vec<u8> {
    vec![(k % 251) as u8; (w * h * 3) as usize]
}

#[test]
fn a_frame_round_trips_with_its_header() {
    let name = RingWriter::unique_name("t-rt");
    let wr = RingWriter::create(&name, 3, 16 * 8 * 3, Mode::Block).unwrap();
    let mut rd = RingReader::open(&name).unwrap();
    let px: Vec<u8> = (0..16 * 8 * 3).map(|i| (i * 7 % 256) as u8).collect();
    let stop = AtomicBool::new(false);
    assert!(matches!(wr.write(&meta(5, 16, 8), &px, &stop).unwrap(), WriteOutcome::Written { index: 0, .. }));
    let mut buf = Vec::new();
    let (i, m) = rd.next(&mut buf, Duration::from_secs(1)).unwrap();
    assert_eq!((i, m), (0, meta(5, 16, 8)));
    assert_eq!(buf, px);
    assert_eq!(wr.read_count(), 1);
}

#[test]
fn a_half_written_frame_is_never_returned() {
    let name = RingWriter::unique_name("t-half");
    let (w, h) = (32, 16);
    let wr = RingWriter::create(&name, 2, (w * h * 3) as usize, Mode::Block).unwrap();
    let mut rd = RingReader::open(&name).unwrap();
    let mut buf = Vec::new();
    // 写者写到一半（seq 为奇数）：读者看到的是「还没写完」
    wr.write_at(0, &meta(0, w, h), &filled(9, w, h), Some((w * h * 3 / 2) as usize));
    assert_eq!(rd.try_read(0, &mut buf), ReadOutcome::Pending);
    assert!(rd.next(&mut buf, Duration::from_millis(20)).is_none());
    // 写完提交后才能读到，而且是完整的一帧
    wr.write_at(0, &meta(0, w, h), &filled(9, w, h), None);
    let (_, m) = rd.next(&mut buf, Duration::from_secs(1)).unwrap();
    assert_eq!(m.frame_no, 0);
    assert!(buf.iter().all(|&b| b == 9));
}

#[test]
fn overwrite_mode_never_waits_and_the_reader_jumps_to_the_newest_frame() {
    let name = RingWriter::unique_name("t-ow");
    let (w, h) = (8, 8);
    let wr = RingWriter::create(&name, 3, (w * h * 3) as usize, Mode::Overwrite).unwrap();
    let mut rd = RingReader::open(&name).unwrap();
    let stop = AtomicBool::new(false);
    for k in 0..10 {
        assert!(matches!(wr.write(&meta(k, w, h), &filled(k, w, h), &stop).unwrap(), WriteOutcome::Written { index, .. } if index == k));
    }
    let mut buf = Vec::new();
    assert_eq!(rd.try_read(0, &mut buf), ReadOutcome::Overwritten);
    let (i, m) = rd.next(&mut buf, Duration::from_secs(1)).unwrap();
    assert_eq!((i, m.frame_no, rd.skipped), (9, 9, 9));
    assert!((m.t - 9.0 / 30.0).abs() < 1e-12, "保留真实时间戳");
}

#[test]
fn block_mode_waits_when_full_and_stops_on_request() {
    let name = RingWriter::unique_name("t-block");
    let (w, h) = (8, 8);
    let wr = Arc::new(RingWriter::create(&name, 2, (w * h * 3) as usize, Mode::Block).unwrap());
    let mut rd = RingReader::open(&name).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    for k in 0..2 {
        wr.write(&meta(k, w, h), &filled(k, w, h), &stop).unwrap();
    }
    // 环满：第三帧要等，直到读者消费一帧
    let (wr2, stop2) = (Arc::clone(&wr), Arc::clone(&stop));
    let t = std::thread::spawn(move || wr2.write(&meta(2, w, h), &filled(2, w, h), &stop2).unwrap());
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(wr.write_count(), 2, "环满时写者不得覆盖未读的帧");
    let mut buf = Vec::new();
    assert_eq!(rd.next(&mut buf, Duration::from_secs(1)).unwrap().1.frame_no, 0);
    match t.join().unwrap() {
        WriteOutcome::Written { index, waited } => assert!(index == 2 && waited >= Duration::from_millis(40)),
        o => panic!("{o:?}"),
    }
    // 再次环满时要求停止 ⇒ 写者放弃等待
    let (wr3, stop3) = (Arc::clone(&wr), Arc::clone(&stop));
    let t = std::thread::spawn(move || wr3.write(&meta(3, w, h), &filled(3, w, h), &stop3).unwrap());
    std::thread::sleep(Duration::from_millis(20));
    stop.store(true, Ordering::SeqCst);
    assert_eq!(t.join().unwrap(), WriteOutcome::Stopped);
}

#[test]
fn a_concurrent_reader_never_sees_a_torn_frame() {
    // 写者线程全速覆盖写（2 个槽、每帧 768 KB），读者线程逐帧校验像素全等于 `帧号 % 251`
    let name = RingWriter::unique_name("t-race");
    let (w, h) = (512u32, 512u32);
    let wr = RingWriter::create(&name, 2, (w * h * 3) as usize, Mode::Overwrite).unwrap();
    let mut rd = RingReader::open(&name).unwrap();
    let n = 2000u64;
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let (mut got, mut last) = (0u64, None);
        while let Some((_, m)) = rd.next(&mut buf, Duration::from_secs(5)) {
            let v = (m.frame_no % 251) as u8;
            assert!(buf.iter().all(|&b| b == v), "第 {} 帧是半帧", m.frame_no);
            assert_eq!(m.roi.0, m.frame_no as u32 % 7);
            assert!(last.map_or(true, |l| m.frame_no > l));
            last = Some(m.frame_no);
            got += 1;
        }
        (got, last, rd.torn, rd.skipped)
    });
    let stop = AtomicBool::new(false);
    for k in 0..n {
        wr.write(&meta(k, w, h), &filled(k, w, h), &stop).unwrap();
    }
    wr.close_stream();
    let (got, last, torn, skipped) = reader.join().unwrap();
    eprintln!("读到 {got} 帧，跳过 {skipped}，序列锁拒绝 {torn} 次");
    assert_eq!(last, Some(n - 1));
    assert!(got > 0);
}
