"""**运动显著性目标选择** —— 在一群外观相近的物体里，找出相对背景独立运动的那一个。

# 方法：排除法 + 显著性判据 + 等待

```text
① 排除名单   背景系里「一直没动过」的位置永久标为干扰点，之后不再考虑
② 显著性     剩下的候选里，取时间窗内背景系**净位移**最大的
③ 置信闸     没有候选明显突出时不更新估计，跟着背景滑动，等目标再次运动
```

# ⭐ 排除名单挂在「背景系里的位置」上，不挂轨迹 ID

检测断续、遮挡、外观变化都会让轨迹 ID 频繁切换；挂在 ID 上，名单会跟着丢。
而静止干扰物在背景系里是**固定点** ⇒ 挂在点上，轨迹死了重生、ID 换了都认得。

顺带的性质：目标停下不动时，它所在的点会被暂时压住（本来就该等）；
**一旦它离开那个点，就自动重新获得资格**。标记在位置上，不在物体上。

# ⭐ 净位移，不是路程

来回摆动的东西（风吹的草叶、晃动的枝条）路程大、净位移小；
持续移动的目标净位移大。判据用净位移，摆动类干扰天然被压低。

# 锁定一条轨迹一路跟下去是不行的

轨迹可能从目标身上平滑地漂到相邻物体上，而全程 `missed == 0`、不报错。
⇒ 判据每帧都重新评估；锁定只提供滞后（`HOLD`），不提供豁免。
"""

from __future__ import annotations

import numpy as np

from . import timing

STATIC_R = 18.0
"""时间窗 `timing.STATIC_SEC` 内离自身重心最远不超过此值（@`timing.REF_W`）⇒ 判静止。

用**散布**而不是首尾净位移：来回晃的东西首尾能回到原点，但它不是静止的。
"""

BAN_R = 26.0
"""离已标记干扰点多近算「落在干扰点上」（@`timing.REF_W`）。"""

DRIFT = 0.15
"""干扰点随重新确认缓慢更新的系数 —— 抵消背景估计的累积漂移。"""

LEAD = 1.6
"""**置信闸**：赢家的净位移须达到候选中位数的此倍数，才允许更新估计。"""

HOLD = 1.0
"""换选择的滞后倍数。`1.0` = 不滞后（排除法已拿掉了噪声候选）。"""


class ExclusionMap:
    """**已确认的干扰点** —— 背景系里的一组点，只增不减。"""

    def __init__(self, r):
        self.r = r
        self.pts: list[list[float]] = []  # [[x, y, 确认次数]]

    def hit(self, x, y):
        r2 = self.r * self.r
        return any((p[0] - x) ** 2 + (p[1] - y) ** 2 <= r2 for p in self.pts)

    def mark(self, x, y):
        r2 = self.r * self.r
        for p in self.pts:
            if (p[0] - x) ** 2 + (p[1] - y) ** 2 <= r2:
                p[0] += DRIFT * (x - p[0])
                p[1] += DRIFT * (y - p[1])
                p[2] += 1
                return
        self.pts.append([x, y, 1])


def net_disp(seq, k, tid, lo, need):
    """背景系里一个时间窗内的**净位移**。历史不够返回 `None`。"""
    pts = [p for p in seq.get(tid, []) if lo <= p[0] <= k]
    if not pts or pts[-1][0] < k - 1 or len(pts) < need * 0.6 or pts[0][0] > lo + 2:
        return None
    a = np.array([[x, y] for _, x, y in pts])
    return float(np.hypot(*(a[-1] - a[0])))


def spread(seq, k, tid, lo, need):
    """背景系里一个时间窗内**离自身重心最远**多少。观测不够返回 `None`。"""
    pts = [p for p in seq.get(tid, []) if lo <= p[0] <= k]
    if len(pts) < need * 0.8 or pts[-1][0] < k - 1:
        return None
    a = np.array([[x, y] for _, x, y in pts])
    return float(np.max(np.hypot(*(a - a.mean(0)).T)))


class Selector:
    """**逐帧喂跟踪结果，逐帧出「目标在哪」。** 状态全在这里，可在线使用。"""

    def __init__(self, clock, roi):
        self.clk = clock
        self.roi = roi
        # 这两个只与画面缩放有关，而一段序列内画面尺寸不变 ⇒ 可以算死
        self.static_r = STATIC_R * clock.s
        self.excluded = ExclusionMap(BAN_R * clock.s)
        self.pos = None
        self.cur = None
        self.chal = None
        self.streak = 0

    # ⛔ 必须是 property：在线时钟随帧更新，算死会停在第一帧的估计上。
    @property
    def n_confirm(self):
        return max(2, self.clk.frames_in(timing.CONFIRM_SEC))

    def step(self, k, hist, seq, bg):
        """走一帧。返回 `(阶段, 画面系位置或None, tid或None, 是否新证据)`。

        阶段：`"wait"` 等待 / `"track"` 跟踪 / `"jump"` 印证后跳到新位置。
        """
        clk = self.clk
        cum = hist[k][2]
        # 不确定时估计值跟着背景滑动，不外推自身速度
        if self.pos is not None:
            self.pos = self.pos + bg

        # ① 排除名单：近 STATIC_SEC 秒几乎没动过的位置，标为干扰点
        lo_s, n_s = clk.window(k, timing.STATIC_SEC)
        lo_b, n_b = clk.window(k, timing.WINDOW_SEC)
        for tid, v in hist[k][1].items():
            if v[5] != 0:
                continue
            sp = spread(seq, k, tid, lo_s, n_s)
            if sp is not None and sp < self.static_r:
                self.excluded.mark(v[0], v[1])

        # ② 排除法 + 显著性
        cand = {}
        for tid, v in hist[k][1].items():
            if v[5] != 0 or self.excluded.hit(v[0], v[1]):
                continue
            d = net_disp(seq, k, tid, lo_b, n_b)
            if d is not None:
                cand[tid] = d

        # ③ 置信闸
        if len(cand) >= 3:
            med = float(np.median(list(cand.values())))
            if max(cand.values()) < LEAD * max(med, 1e-6):
                cand = {}

        phase, fresh = "wait", False
        if cand:
            best = max(cand, key=cand.get)
            keep = self.cur in cand and cand[best] <= HOLD * cand[self.cur]
            pick = self.cur if keep else best
            v = hist[k][1][pick]
            p = np.array([v[0] + cum[0], v[1] + cum[1]])
            if self.pos is None or float(np.hypot(*(p - self.pos))) <= clk.speed(timing.JUMP_PPS, k):
                self.pos, self.cur, self.chal, self.streak = p, pick, None, 0
                phase, fresh = "track", True
            else:
                # 远跳要连续印证；印证按**位置**比，不按 tid（tid 会频繁切换）
                if self.chal is not None:
                    self.chal = self.chal + bg
                    near = float(np.hypot(*(p - self.chal))) <= clk.speed(timing.CONFIRM_R_PPS, k)
                    self.streak = self.streak + 1 if near else 1
                else:
                    self.streak = 1
                self.chal = p
                if self.streak >= self.n_confirm:
                    self.pos, self.cur, self.chal, self.streak = p, pick, None, 0
                    phase, fresh = "jump", True
        return phase, (None if self.pos is None else tuple(self.pos)), self.cur, fresh
