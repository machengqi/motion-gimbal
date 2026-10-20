"""**运动候选生成** —— 不靠外观，靠「扣掉相机运动后还在动」。

伪装目标最难被外观检测器框到，而它一动就暴露。这个模块每帧：

```text
① 特征点        上一帧取角点（Shi-Tomasi），LK 光流跟到这一帧，前后向校验去掉不可靠的
② 相机运动      全部可靠点对拟合单应矩阵（RANSAC）—— 点绝大多数落在背景上
③ 残余运动      每个点的实际位移 − 单应预测的位移，换算成 px/秒（按真实 dt）
④ 聚团成框      残余运动超过阈值的点按距离聚团，点数够的团输出一个候选框
```

同时输出这一帧的**背景平移**（单应矩阵把画面中心映到哪），交给跟踪器作为外部
背景位移 —— 候选只框运动区域，跟踪器的「候选位移中位数」前提不成立。

⚠️ 风吹的草叶、景深视差也会产生残余运动，它们同样会成为候选：这是有意的，
筛掉它们是后面排除名单与净位移判据的工作。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..core import timing


@dataclass
class Params:
    max_pts: int = 2000
    """每帧最多取多少个角点。"""
    quality: float = 0.003
    """角点质量阈值（相对最强角点）。伪装场景纹理弱，阈值要低。"""
    min_dist: float = 6.0
    """角点最小间距（px）。"""
    fb_err: float = 1.0
    """前后向光流误差上限（px）—— 超过即视为不可靠。"""
    ransac_px: float = 1.5
    """单应 RANSAC 的重投影阈值（px）。"""
    resid_pps: float = 2.0 * timing.REF_FPS
    """残余运动阈值（px/秒 @`timing.REF_W`）。按真实 dt 换算，与帧率无关。"""
    cluster_r: float = 24.0
    """聚团半径（px @`timing.REF_W`）。"""
    min_pts: int = 4
    """一个候选至少由多少个运动点组成。"""
    pad: float = 6.0
    """候选框外扩（px）。"""


@dataclass
class FrameResult:
    dets: list[tuple]
    """候选 `(cx, cy, area, x1, y1, x2, y2, conf)`，当前帧画面系。"""
    bg_shift: np.ndarray | None
    """这一帧的背景平移（画面中心的位移，px）；估计失败为 `None`。"""
    homography: np.ndarray | None
    n_tracked: int
    n_moving: int


class MotionCandidates:
    """**逐帧喂灰度图 + 时间戳，逐帧出候选与背景位移。**"""

    def __init__(self, width, params: Params | None = None):
        self.p = params or Params()
        self.s = width / timing.REF_W
        self.prev = None
        self.prev_t = None
        self.prev_mask = None

    def reset(self):
        self.prev = self.prev_t = self.prev_mask = None

    def __call__(self, gray: np.ndarray, t: float, mask: np.ndarray | None = None) -> FrameResult:
        """`mask`（可选，uint8，非零 = 有效像素）：画面里只有一部分是真实像素时（例如取景窗伸出了
        素材边界、其余补黑），角点只在上一帧的有效区取，跟到这一帧后落在无效区的点丢掉 ——
        否则有效区的边界随相机移动，会被当成运动。"""
        p = self.p
        prev, prev_t, prev_mask = self.prev, self.prev_t, self.prev_mask
        self.prev, self.prev_t, self.prev_mask = gray, t, mask
        if prev is None:
            return FrameResult([], np.zeros(2), np.eye(3), 0, 0)
        dt = max(t - prev_t, 1e-4)

        p0 = cv2.goodFeaturesToTrack(
            prev, maxCorners=p.max_pts, qualityLevel=p.quality, minDistance=p.min_dist, mask=prev_mask
        )
        if p0 is None or len(p0) < 8:
            return FrameResult([], None, None, 0, 0)
        lk = {"winSize": (21, 21), "maxLevel": 3}
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, **lk)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p1, None, **lk)
        fb = np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < p.fb_err)
        if mask is not None:
            q = np.round(p1.reshape(-1, 2)).astype(int)
            hh, ww = mask.shape[:2]
            inside = (q[:, 0] >= 0) & (q[:, 0] < ww) & (q[:, 1] >= 0) & (q[:, 1] < hh)
            ok &= inside
            ok[inside] &= mask[q[inside, 1], q[inside, 0]] > 0
        a, b = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
        if len(a) < 8:
            return FrameResult([], None, None, int(len(a)), 0)

        H, _ = cv2.findHomography(a, b, cv2.RANSAC, p.ransac_px)
        if H is None:
            return FrameResult([], None, None, int(len(a)), 0)
        pred = cv2.perspectiveTransform(a.reshape(-1, 1, 2), H).reshape(-1, 2)
        resid = np.linalg.norm(b - pred, axis=1) / dt

        h, w = gray.shape[:2]
        c = np.array([[[w / 2, h / 2]]], dtype=np.float64)
        bg = cv2.perspectiveTransform(c, H).reshape(2) - c.reshape(2)

        moving = b[resid > p.resid_pps * self.s]
        speed = resid[resid > p.resid_pps * self.s]
        dets = self._cluster(moving, speed, w, h)
        return FrameResult(dets, bg, H, int(len(a)), int(len(moving)))

    def _cluster(self, pts, speed, w, h):
        p = self.p
        if len(pts) < p.min_pts:
            return []
        tree = cKDTree(pts)
        pairs = tree.query_pairs(p.cluster_r * self.s, output_type="ndarray")
        n = len(pts)
        if len(pairs):
            from scipy.sparse import coo_matrix  # noqa: PLC0415

            g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
            _, labels = connected_components(g, directed=False)
        else:
            labels = np.arange(n)
        out = []
        for lab in np.unique(labels):
            idx = labels == lab
            if idx.sum() < p.min_pts:
                continue
            q = pts[idx]
            pad = p.pad * self.s
            x1, y1 = max(q[:, 0].min() - pad, 0.0), max(q[:, 1].min() - pad, 0.0)
            x2, y2 = min(q[:, 0].max() + pad, w), min(q[:, 1].max() + pad, h)
            conf = float(min(1.0, np.median(speed[idx]) / (4 * p.resid_pps * self.s)))
            out.append(
                (
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    float((x2 - x1) * (y2 - y1)),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    conf,
                )
            )
        return out
