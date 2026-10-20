"""任务 3：反常读数 3 —— 单应矩阵中心平移的累计值 vs 逐帧相位相关累加值，谁更准。

    uv run python scripts/diag_bgshift.py [arabian_horn_viper] [--span 10]

每帧（640 宽、去复制帧，与 E2 同一份图像）算几种背景位移：

| 键 | 做法 |
|---|---|
| `H`        | `MotionCandidates` 的输出（全部可靠点对拟合单应，画面中心的映射）—— 即 E2 用的 |
| `PC`       | `cv2.phaseCorrelate`（上一帧 → 这一帧，Hanning 窗） |
| `PC_raw`   | 同上，不加窗 |
| `Hm`       | 与 `H` 同一组 LK 点对，但**剔除落在真值框（外扩 20px）内的点**后再拟合单应，画面中心的映射 |
| `Hm_gt`    | `Hm` 的单应矩阵在**目标所在位置**（真值框中心）的位移 |
| `PCm`      | 相位相关，但窗函数在真值框（外扩 20px）内置零 —— 检验 PC 是否被目标带走 |
| `TM`       | 与 LK / 单应 / 相位相关都无关的第三种测法：真值框外取 48px 方块，NCC 模板匹配（±40px），
               峰值 ≥ 0.7 的方块位移取中位数 |

判据（不依赖任何一种逐帧估计）：
1. **长基线直接配准**：第 k 帧与第 k+span 帧之间直接做 LK + RANSAC 单应（剔除两帧真值框内的点），
   画面中心的位移 = 「真」位移的独立估计；与各方法在同一区间的逐帧累加值比。
   ⚠️ 长基线配准也会错（重叠少、纹理弱），所以同时报告它的内点数，并且只用内点 ≥ 30 的区间。
2. **目标在背景系里的轨迹**：真值框中心 − 累计背景位移。
3. 每帧特征点有多少落在目标上、RANSAC 内点有多少落在目标上（单应被目标「带走」的直接证据）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import CACHE, load  # noqa: E402

from gimbal_vision.candidates.motion import Params  # noqa: E402

P = Params()
LK = {"winSize": (21, 21), "maxLevel": 3}


def lk_pairs(a, b):
    """与 MotionCandidates 同样的取点 + 前后向校验。返回 (p0, p1)。"""
    p0 = cv2.goodFeaturesToTrack(a, maxCorners=P.max_pts, qualityLevel=P.quality, minDistance=P.min_dist)
    if p0 is None or len(p0) < 8:
        return None, None
    p1, s1, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, **LK)
    pb, s2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, **LK)
    fb = np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1)
    ok = (s1.ravel() == 1) & (s2.ravel() == 1) & (fb < P.fb_err)
    return p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]


def inbox(pts, g, pad=20.0):
    if g is None:
        return np.zeros(len(pts), bool)
    return (pts[:, 0] >= g[0] - pad) & (pts[:, 0] <= g[0] + g[2] + pad) & (pts[:, 1] >= g[1] - pad) & (pts[:, 1] <= g[1] + g[3] + pad)


def map_pt(H, p):
    return cv2.perspectiveTransform(np.array([[p]], np.float64), H).reshape(2) - np.asarray(p, float)


def template_shift(a, b, g, size=48, stride=56, rng=40, thr=0.7):
    h, w = a.shape
    out = []
    for y in range(rng, h - size - rng, stride):
        for x in range(rng, w - size - rng, stride):
            box = np.array([[x, y], [x + size, y + size], [x, y + size], [x + size, y]], float)
            if inbox(box, g).any():
                continue
            t = a[y : y + size, x : x + size]
            if t.std() < 4:
                continue
            s = b[y - rng : y + size + rng, x - rng : x + size + rng]
            r = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx >= thr:
                out.append((loc[0] - rng, loc[1] - rng))
    return (np.median(np.array(out, float), 0).tolist(), len(out)) if out else ([np.nan, np.nan], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="arabian_horn_viper")
    ap.add_argument("--span", type=int, default=10)
    a = ap.parse_args()
    d = load(a.name, keep_imgs=True)
    imgs, gts, n, W, H = d["imgs"], d["gts"], len(d["imgs"]), d["W"], d["H"]
    c = (W / 2, H / 2)
    han = cv2.createHanningWindow((W, H), cv2.CV_64F)
    rows = []
    for k in range(n):
        r = {"k": k, "H": np.asarray(d["shifts"][k], float).tolist()}
        if k == 0:
            r.update(PC=[0, 0], PC_raw=[0, 0], Hm=[0, 0], Hm_gt=[0, 0], PCm=[0, 0], TM=[0, 0], tm_n=0, n_pts=0, n_on_gt=0, inl=0, inl_on_gt=0, pc_resp=1.0)
            rows.append(r)
            continue
        prev, cur = imgs[k - 1].astype(np.float64), imgs[k].astype(np.float64)
        (sx, sy), resp = cv2.phaseCorrelate(prev, cur, han)
        (rx, ry), _ = cv2.phaseCorrelate(prev, cur)
        r["PC"], r["PC_raw"], r["pc_resp"] = [sx, sy], [rx, ry], float(resp)
        g_prev = gts[k - 1] if gts[k - 1] is not None else gts[k]
        wm = han.copy()
        if g_prev is not None:
            x0, y0 = max(int(g_prev[0] - 20), 0), max(int(g_prev[1] - 20), 0)
            wm[y0 : int(g_prev[1] + g_prev[3] + 20), x0 : int(g_prev[0] + g_prev[2] + 20)] = 0
        (mx_, my_), _ = cv2.phaseCorrelate(prev, cur, wm)
        r["PCm"] = [mx_, my_]
        r["TM"], r["tm_n"] = template_shift(imgs[k - 1], imgs[k], g_prev)
        p0, p1 = lk_pairs(imgs[k - 1], imgs[k])
        on = inbox(p0, g_prev)
        r["n_pts"], r["n_on_gt"] = len(p0), int(on.sum())
        Hf, mask = cv2.findHomography(p0, p1, cv2.RANSAC, P.ransac_px)
        m = mask.ravel().astype(bool)
        r["inl"], r["inl_on_gt"] = int(m.sum()), int((m & on).sum())
        # 原 H 的内点里，目标上的点与背景上的点各自的位移中位数
        r["inl_disp_gt"] = np.median(p1[m & on] - p0[m & on], 0).tolist() if (m & on).any() else None
        r["inl_disp_bg"] = np.median(p1[m & ~on] - p0[m & ~on], 0).tolist() if (m & ~on).any() else None
        if (~on).sum() >= 8:
            Hm, _ = cv2.findHomography(p0[~on], p1[~on], cv2.RANSAC, P.ransac_px)
            r["Hm"] = map_pt(Hm, c).tolist()
            gc = (g_prev[0] + g_prev[2] / 2, g_prev[1] + g_prev[3] / 2) if g_prev is not None else c
            r["Hm_gt"] = map_pt(Hm, gc).tolist()
        else:
            r["Hm"] = r["Hm_gt"] = [np.nan, np.nan]
        rows.append(r)

    keys = ("H", "PC", "PC_raw", "Hm", "Hm_gt", "PCm", "TM")
    per = {kk: np.array([r[kk] for r in rows], float) for kk in keys}
    cum = {kk: np.nancumsum(v, axis=0) for kk, v in per.items()}
    print(f"{a.name}: {n} 帧，{W}x{H}")
    print("累计背景位移（x, y）：")
    for k in list(range(0, n, 30)) + [n - 1]:
        g = gts[k]
        gc = None if g is None else (g[0] + g[2] / 2, g[1] + g[3] / 2)
        line = "  ".join(f"{kk}=({cum[kk][k][0]:7.1f},{cum[kk][k][1]:7.1f})" for kk in keys)
        r = rows[k]
        print(f" k={k:3d} {line}  点 {r['n_pts']} 目标上 {r['n_on_gt']}  内点 {r['inl']} 目标上 {r['inl_on_gt']}  gt={gc and tuple(round(v) for v in gc)}")

    # 每帧逐项：原 H 内点落在目标上的比例
    fr = np.array([r["inl_on_gt"] / max(r["inl"], 1) for r in rows[1:]])
    fp = np.array([r["n_on_gt"] / max(r["n_pts"], 1) for r in rows[1:]])
    for lo, hi in ((1, 120), (120, n)):
        print(f"帧 {lo}-{hi - 1}: 特征点在目标上占 {100 * fp[lo - 1:hi - 1].mean():.0f}%，H 的 RANSAC 内点在目标上占 "
              f"{100 * fr[lo - 1:hi - 1].mean():.0f}%；|H−Hm| 逐帧均值 "
              f"{np.nanmean(np.hypot(*(per['H'][lo:hi] - per['Hm'][lo:hi]).T)):.2f}px，|PC−Hm| {np.nanmean(np.hypot(*(per['PC'][lo:hi] - per['Hm'][lo:hi]).T)):.2f}px")

    # ── 判据 1：长基线 ──
    print(f"\n长基线直接配准（span={a.span} 帧，剔除真值框内的点；只用内点 ≥ 30 的区间）")
    errs = {kk: [] for kk in keys}
    segs = []
    for k in range(0, n - a.span, a.span // 2 or 1):
        j = k + a.span
        p0, p1 = lk_pairs(imgs[k], imgs[j])
        if p0 is None:
            continue
        ok = ~inbox(p0, gts[k]) & ~inbox(p1, gts[j])
        if ok.sum() < 30:
            continue
        Hd, mask = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 3.0)
        if Hd is None or mask.sum() < 30:
            continue
        truth = map_pt(Hd, c)
        rec = {"k": k, "truth": truth.tolist(), "inl": int(mask.sum())}
        for kk in keys:
            est = cum[kk][j] - cum[kk][k]
            rec[kk] = est.tolist()
            errs[kk].append((k, float(np.hypot(*(est - truth))), est[1] - truth[1]))
        segs.append(rec)
    for kk in keys:
        e = np.array([x[1] for x in errs[kk]])
        dy = np.array([x[2] for x in errs[kk]])
        early = np.array([x[1] for x in errs[kk] if x[0] < 120])
        late = np.array([x[1] for x in errs[kk] if x[0] >= 120])
        print(f"  {kk:7s} 区间数 {len(e)}  |误差| 中位 {np.median(e):.1f}px 均值 {e.mean():.1f}px  "
              f"(k<120: 中位 {np.median(early):.1f}；k≥120: 中位 {np.median(late):.1f})  y 误差均值 {dy.mean():+.1f}px")
    tr = np.array([s["truth"] for s in segs])
    print(f"  长基线「真」位移 y 的逐区间累加（每 span 一个不重叠区间）≈ {tr[::2, 1].sum():.0f}px，x ≈ {tr[::2, 0].sum():.0f}px "
          f"（覆盖帧 {segs[0]['k']}–{segs[::2][-1]['k'] + a.span}）")
    for kk in keys:
        tot = sum(np.array(s[kk]) for s in segs[::2])
        print(f"     同样区间上 {kk:7s} 累加 = ({tot[0]:.0f}, {tot[1]:.0f})")

    # ── 判据 2：目标在背景系里的轨迹 ──
    print("\n目标（真值框中心）在背景系里的位置 = 画面位置 − 累计背景位移：")
    for k in list(range(0, n, 30)) + [n - 1]:
        g = gts[k]
        if g is None:
            continue
        gc = np.array([g[0] + g[2] / 2, g[1] + g[3] / 2])
        print(f" k={k:3d} " + "  ".join(f"{kk}=({(gc - cum[kk][k])[0]:7.1f},{(gc - cum[kk][k])[1]:7.1f})" for kk in keys))
    ts = d["ts"]
    for kk in keys:
        traj = np.array([np.array([g[0] + g[2] / 2, g[1] + g[3] / 2]) - cum[kk][k] for k, g in enumerate(gts) if g is not None])
        sp = np.hypot(*np.diff(traj, axis=0).T) / np.diff(ts[[k for k, g in enumerate(gts) if g is not None]])
        path = np.hypot(*np.diff(traj, axis=0).T).sum()
        print(f"  {kk:7s} 背景系净位移 {np.hypot(*(traj[-1] - traj[0])):.0f}px  路程 {path:.0f}px  速度中位 {np.median(sp):.0f}px/s")

    out = CACHE / f"{a.name}-bgshift.json"
    out.write_text(json.dumps({"rows": rows, "long": segs}, default=float), encoding="utf-8")
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
