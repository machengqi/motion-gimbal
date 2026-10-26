"""**评测用的背景稳定化** —— 把手持跟拍的素材换算到一块「画布」（世界坐标）上。

E2b 协议（`vision/scripts/eval_e2.py --protocol e2b`）用它抵消摄影师的跟拍：画布坐标 = 素材画面坐标
+ 第 k 帧的偏移 `o[k]`。画布里静止的背景点在任何一帧都落在同一个画布坐标上，目标相对背景的运动
在画布里才真实可见。

# 为什么不用被测方法自己的背景估计

被测方法（`candidates.motion`）用 Shi-Tomasi 角点 + LK 光流 + RANSAC 单应。评测的参照系若取自被测方法，
方法的背景估计误差会同时出现在「方法的输出」和「评测的尺子」上而互相抵消。这里用一套**独立**的估计：
SIFT 特征 + 描述子匹配（比值检验）+ RANSAC 单应，不看真值、不看任何方法的输出，所有方法共用同一份结果。

# 只取平移

每对相邻帧拟合单应，取它把画面中心映到哪 —— 与 `candidates.motion` 的 `bg_shift` 同一口径，但来自独立
特征。画布只建模平移（相当于云台摇移）；变焦、旋转、视差不建模，它们在画布上表现为缓慢的形变与漂移。
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_INLIERS = 15
"""单应内点少于此数视为配准失败（记 0 平移，并计数）。"""

RATIO = 0.75
"""Lowe 比值检验阈值。"""

RANSAC_PX = 3.0
"""RANSAC 重投影阈值（px）。"""


class Registrar:
    """SIFT 配准器。实例可复用（内部只缓存检测器与匹配器）。"""

    def __init__(self, n_features=1500):
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def features(self, gray):
        return self.sift.detectAndCompute(gray, None)

    def matches(self, fa, fb):
        """两帧特征 → 比值检验后的对应点 `(pa, pb, 匹配数)`；点不够时 `pa = pb = None`。"""
        (ka, da), (kb, db) = fa, fb
        if da is None or db is None or len(ka) < MIN_INLIERS or len(kb) < MIN_INLIERS:
            return None, None, 0
        good = [m for m, n in (p for p in self.bf.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < RATIO * n.distance]
        if len(good) < MIN_INLIERS:
            return None, None, len(good)
        pa = np.float32([ka[m.queryIdx].pt for m in good])
        pb = np.float32([kb[m.trainIdx].pt for m in good])
        return pa, pb, len(good)

    def homography(self, fa, fb):
        """两帧特征 → `(单应 H（a → b）, 内点数)`；失败返回 `(None, 内点数)`。"""
        pa, pb, n = self.matches(fa, fb)
        if pa is None:
            return None, n
        H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, RANSAC_PX)
        n_in = int(inl.sum()) if inl is not None else 0
        if H is None or n_in < MIN_INLIERS:
            return None, n_in
        return H, n_in

    def similarity(self, fa, fb):
        """两帧特征 → `(相似变换（4 自由度，3×3 形式，a → b）, 内点数)`；失败返回 `(None, 内点数)`。

        只用于背景模型的对照（`scripts/diag_drift.py`），画布本身仍取单应把画面中心映到哪。
        """
        pa, pb, n = self.matches(fa, fb)
        if pa is None:
            return None, n
        A, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
        n_in = int(inl.sum()) if inl is not None else 0
        if A is None or n_in < MIN_INLIERS:
            return None, n_in
        return np.vstack([A, [0.0, 0.0, 1.0]]), n_in

    def shift(self, fa, fb, size):
        """两帧特征 → `(画面中心的平移 (dx, dy), 内点数)`；失败返回 `(None, 内点数)`。

        平移的含义：a 帧里画面中心处的静止背景点，在 b 帧里移到了 `中心 + (dx, dy)`。
        """
        H, n_in = self.homography(fa, fb)
        if H is None:
            return None, n_in
        w, h = size
        c = np.array([[[w / 2, h / 2]]], dtype=np.float64)
        return cv2.perspectiveTransform(c, H).reshape(2) - c.reshape(2), n_in


def accumulate(shifts):
    """逐帧平移 → 画布偏移 `o`（`o[0] = 0`；画布坐标 = 画面坐标 + `o[k]`）。

    第 k 帧的平移 `t[k]`（k−1 → k）：静止点在画面里 `p → p + t[k]`，在画布里不动 ⇒ `o[k] = o[k−1] − t[k]`。
    `shifts[0]` 忽略；`None` 当 0。
    """
    o = np.zeros((len(shifts), 2))
    for k in range(1, len(shifts)):
        t = shifts[k]
        o[k] = o[k - 1] - (np.zeros(2) if t is None else np.asarray(t, float))
    return o


def transforms(imgs):
    """一段灰度帧 → 逐帧 `(单应列表, 相似变换列表, 内点数)`（第 k 个：第 k−1 帧 → 第 k 帧；第 0 个与失败帧为 `None`）。

    与 [`offsets`] 用同一套特征与匹配；`offsets` 的平移 = 单应把画面中心映到哪。
    """
    reg = Registrar()
    prev = reg.features(imgs[0])
    Hs, Ss, inl = [None], [None], [0]
    for k in range(1, len(imgs)):
        cur = reg.features(imgs[k])  # 逐帧滚动，不同时持有整段的描述子（长视频会占上 GB 内存）
        H, n = reg.homography(prev, cur)
        S, _ = reg.similarity(prev, cur)
        prev = cur
        Hs.append(H)
        Ss.append(S)
        inl.append(n)
    return Hs, Ss, np.array(inl)


def offsets(imgs):
    """一段灰度帧 → `(画布偏移 o, 每帧内点数, 失败帧数)`。"""
    reg = Registrar()
    h, w = imgs[0].shape[:2]
    feats = [reg.features(g) for g in imgs]
    shifts, inl, fail = [None], [0], 0
    for k in range(1, len(imgs)):
        t, n = reg.shift(feats[k - 1], feats[k], (w, h))
        fail += t is None
        shifts.append(t)
        inl.append(n)
    return accumulate(shifts), np.array(inl), fail
