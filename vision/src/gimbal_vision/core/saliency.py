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

# 两条可选规则（默认值见 `Selector`，依据见 docs/EXPERIMENTS.md 的 E2 修复消融）

- **画面坐标排除**（`screenfix`）：台标、水印、黑边钉在**画面**上，背景系里却随相机一起「运动」，
  净位移大 ⇒ 会被当成最显著的目标。判据是排除名单在画面系的镜像：时间窗 `STATIC_SEC` 内画面系
  散布 < `STATIC_R`、而背景系净位移 > 2·`STATIC_R` 的轨迹，这一帧不参与选择。逐帧判、不入名单 ——
  目标被跟拍得纹丝不动时也满足这个签名，一旦它在画面里动起来就立即恢复资格。
- **过期撤回**（`stale`）：估计位置连续 `timing.STALE_SEC` 没有新证据就撤回成「无估计」，下一个
  候选直接接管，不再走远跳印证。防止首锁锁错之后，回跳印证一直凑不齐而永远停在错处。

# 消融开关（只用于 docs/EXPERIMENTS.md 的消融对照；默认值即现行实现，逐位不变）

- `excl="pos"`（默认）排除名单挂在背景系位置上；`"id"` 挂在轨迹 ID 上（被判静止的 tid 永久排除，
  ID 一换就失效，也不会波及路过该位置的其他轨迹）；`"none"` 不用排除名单（`screenfix` 不受影响）。
- `disp="net"`（默认）显著性用窗口内净位移；`"path"` 换成路程（窗口内逐帧位移长度之和），有效性条件不变。

# 可选：排除图用单应传播（`warps`）

默认的背景系是**平移**模型（画面坐标 − 累计背景位移）：相机旋转、变焦、景深视差时，离画面中心越远的
静止物体在背景系里越会「漂」。给 `Selector(warps=...)`（逐帧单应 `H[k]`：第 k−1 帧 → 第 k 帧，例如
`candidates.motion` 的 `homography`）时，排除图改为存在**当帧画面坐标**里、每帧用 `H[k]` 传播，
「静止」散布也在当帧画面系里算（窗口内的历史位置用窗口内的单应链换到当帧）。`warp_disp=True` 时净位移
判据同样用单应链。平移矩阵当 `warps` 时与默认行为一致（测试 `test_translation_warps_reproduce_the_default`）。
只用于背景模型的对照实验（docs/EXPERIMENTS.md）。

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
        """标记一个干扰点。返回 `True` = 新建了一个点，`False` = 并入已有点。"""
        r2 = self.r * self.r
        for p in self.pts:
            if (p[0] - x) ** 2 + (p[1] - y) ** 2 <= r2:
                p[0] += DRIFT * (x - p[0])
                p[1] += DRIFT * (y - p[1])
                p[2] += 1
                return False
        self.pts.append([x, y, 1])
        return True

    def warp(self, H):
        """把所有点按 3×3 矩阵 `H` 搬到下一帧的画面坐标（只在 `Selector(warps=...)` 时用）。"""
        if not self.pts:
            return
        a = np.array([[p[0], p[1], 1.0] for p in self.pts]) @ np.asarray(H, float).T
        a = a[:, :2] / a[:, 2:3]
        for p, q in zip(self.pts, a, strict=True):
            p[0], p[1] = float(q[0]), float(q[1])


def net_disp(seq, k, tid, lo, need):
    """背景系里一个时间窗内的**净位移**。历史不够返回 `None`。"""
    pts = [p for p in seq.get(tid, []) if lo <= p[0] <= k]
    if not pts or pts[-1][0] < k - 1 or len(pts) < need * 0.6 or pts[0][0] > lo + 2:
        return None
    a = np.array([[x, y] for _, x, y in pts])
    return float(np.hypot(*(a[-1] - a[0])))


def path_len(seq, k, tid, lo, need):
    """背景系里一个时间窗内的**路程**（逐帧位移长度之和；消融用，见 `Selector(disp="path")`）。

    有效性条件与 [`net_disp`] 完全相同，只换统计量。
    """
    pts = [p for p in seq.get(tid, []) if lo <= p[0] <= k]
    if not pts or pts[-1][0] < k - 1 or len(pts) < need * 0.6 or pts[0][0] > lo + 2:
        return None
    a = np.array([[x, y] for _, x, y in pts])
    return float(np.sum(np.hypot(*np.diff(a, axis=0).T))) if len(a) > 1 else 0.0


EXCL_MODES = ("pos", "id", "none")
DISP_MODES = ("net", "path")


def spread(seq, k, tid, lo, need):
    """背景系里一个时间窗内**离自身重心最远**多少。观测不够返回 `None`。"""
    pts = [p for p in seq.get(tid, []) if lo <= p[0] <= k]
    if len(pts) < need * 0.8 or pts[-1][0] < k - 1:
        return None
    a = np.array([[x, y] for _, x, y in pts])
    return float(np.max(np.hypot(*(a - a.mean(0)).T)))


def screen_fixed(seq, hist, origin, k, tid, lo, need, r):
    """`tid` 在时间窗 `[lo, k]` 内是否「钉在画面上」：画面系散布 < `r` 且背景系净位移 > `2r`。

    画面系 = 背景系 + 当帧累计背景位移 + 当帧画面原点（`origin[帧号]`；裁剪取景窗时是取景窗在全幅里的
    左上角，让「画面」始终指原始素材的像素坐标）。观测不够返回 `False`。
    """
    pts = [p for p in seq.get(tid, []) if lo <= p[0] <= k]
    if len(pts) < need * 0.8 or pts[-1][0] < k - 1:
        return False
    b = np.array([[x, y] for _, x, y in pts])
    f = b + np.array([hist[kk][2] + origin.get(kk, 0.0) for kk, _, _ in pts])
    sp = float(np.max(np.hypot(*(f - f.mean(0)).T)))
    return sp < r and float(np.hypot(*(b[-1] - b[0]))) > 2 * r


class Selector:
    """**逐帧喂跟踪结果，逐帧出「目标在哪」。** 状态全在这里，可在线使用。

    `screenfix` / `stale` 见模块文档；`shift_chal=False` 复现修复前的行为（印证中的远跳位置只在
    下一次远跳时才随背景平移），只用于消融对照。`excl` / `disp` 见模块文档的「消融开关」。
    """

    def __init__(
        self, clock, roi, screenfix=True, stale=True, shift_chal=True, warps=None, warp_disp=False,
        excl="pos", disp="net",
    ):
        if excl not in EXCL_MODES or disp not in DISP_MODES:
            raise ValueError((excl, disp))
        if warps is not None and (excl != "pos" or disp != "net"):
            raise ValueError("单应传播只与默认的 excl / disp 组合")
        self.excl_mode = excl
        self.disp_mode = disp
        self.banned_ids: set = set()
        """`excl="id"` 时被排除的 tid。"""
        self.clk = clock
        self.warps = warps
        """逐帧单应（第 k−1 帧 → 第 k 帧，`None` 当单位阵）；给了就用单应传播排除图，见模块文档。"""
        self.warp_disp = warp_disp and warps is not None
        self._chain: dict[int, np.ndarray] = {}  # τ → 第 τ 帧画面 → 第 k 帧画面 的单应链
        self.stats = {"mark_new": 0, "mark_merge": 0, "excluded": 0, "gate_checked": 0, "gate_fired": 0}
        self.last_excluded: list = []
        """这一帧被排除图拿掉的 tid（诊断用）。"""
        self.roi = roi
        self.screenfix = screenfix
        self.stale = stale
        self.shift_chal = shift_chal
        # 这两个只与画面缩放有关，而一段序列内画面尺寸不变 ⇒ 可以算死
        self.static_r = STATIC_R * clock.s
        self.excluded = ExclusionMap(BAN_R * clock.s)
        self.pos = None
        self.cur = None
        self.chal = None
        self.streak = 0
        self.last_fresh = None
        self.origin: dict[int, np.ndarray] = {}

    # ⛔ 必须是 property：在线时钟随帧更新，算死会停在第一帧的估计上。
    @property
    def n_confirm(self):
        return max(2, self.clk.frames_in(timing.CONFIRM_SEC))

    def _warped(self, seq, hist, k, tid, lo):
        """`tid` 在 `[lo, k]` 内的观测，换到第 k 帧画面坐标（单应链）。返回 `(帧号列表, N×2 数组)`。"""
        pts = [p for p in seq.get(tid, []) if lo <= p[0] <= k]
        if not pts:
            return [], np.zeros((0, 2))
        a = np.empty((len(pts), 2))
        for i, (kk, x, y) in enumerate(pts):
            c = hist[kk][2]
            q = self._chain[kk] @ np.array([x + c[0], y + c[1], 1.0])
            a[i] = q[:2] / q[2]
        return [p[0] for p in pts], a

    def _spread_w(self, seq, hist, k, tid, lo, need):
        ks, a = self._warped(seq, hist, k, tid, lo)
        if len(ks) < need * 0.8 or ks[-1] < k - 1:
            return None
        return float(np.max(np.hypot(*(a - a.mean(0)).T)))

    def _net_disp_w(self, seq, hist, k, tid, lo, need):
        ks, a = self._warped(seq, hist, k, tid, lo)
        if not ks or ks[-1] < k - 1 or len(ks) < need * 0.6 or ks[0] > lo + 2:
            return None
        return float(np.hypot(*(a[-1] - a[0])))

    def step(self, k, hist, seq, bg, origin=None):
        """走一帧。返回 `(阶段, 画面系位置或None, tid或None, 是否新证据)`。

        `origin`：这一帧画面左上角在原始素材里的坐标（只给 `screenfix` 用），默认 `(0, 0)`。
        阶段：`"wait"` 等待 / `"track"` 跟踪 / `"jump"` 印证后跳到新位置。
        """
        clk = self.clk
        cum = hist[k][2]
        self.origin[k] = np.zeros(2) if origin is None else np.asarray(origin, float)
        # 不确定时估计值跟着背景滑动，不外推自身速度
        if self.pos is not None:
            self.pos = self.pos + bg
        # ⛔ 印证中的远跳位置与 pos 同在画面系，必须每帧一起平移；只在远跳分支里平移的话，
        #    中间隔着没有候选的帧时它会停在旧画面坐标上，与新观测比较时差出这几帧的背景位移。
        if self.shift_chal and self.chal is not None:
            self.chal = self.chal + bg
        if (
            self.stale
            and self.pos is not None
            and self.last_fresh is not None
            and clk.t[k] - clk.t[self.last_fresh] > timing.STALE_SEC
        ):
            self.pos, self.cur, self.chal, self.streak = None, None, None, 0

        # ① 排除名单：近 STATIC_SEC 秒几乎没动过的位置，标为干扰点
        lo_s, n_s = clk.window(k, timing.STATIC_SEC)
        lo_b, n_b = clk.window(k, timing.WINDOW_SEC)
        warped = self.warps is not None
        if warped:
            # 排除图与窗口内的历史都换到这一帧的画面坐标（单应传播）
            Hk = self.warps[k] if k > 0 and self.warps[k] is not None else np.eye(3)
            Hk = np.asarray(Hk, float)
            lo = min(lo_s, lo_b)
            self._chain = {kk: Hk @ M for kk, M in self._chain.items() if kk >= lo}
            self._chain[k] = np.eye(3)
            if k > 0:
                self.excluded.warp(Hk)

        def loc(v):  # 排除图里比较用的坐标：默认背景系；单应模式下当帧画面系
            return (v[0] + cum[0], v[1] + cum[1]) if warped else (v[0], v[1])

        for tid, v in hist[k][1].items():
            if v[5] != 0 or self.excl_mode == "none":
                continue
            if warped:
                sp = self._spread_w(seq, hist, k, tid, lo_s, n_s)
            else:
                sp = spread(seq, k, tid, lo_s, n_s)
            if sp is not None and sp < self.static_r:
                if self.excl_mode == "id":
                    new = tid not in self.banned_ids
                    self.banned_ids.add(tid)
                else:
                    new = self.excluded.mark(*loc(v))
                self.stats["mark_new" if new else "mark_merge"] += 1

        # ② 排除法 + 显著性
        cand = {}
        self.last_excluded = []
        for tid, v in hist[k][1].items():
            if v[5] != 0:
                continue
            if self.excl_mode == "id":
                hit = tid in self.banned_ids
            elif self.excl_mode == "none":
                hit = False
            else:
                hit = self.excluded.hit(*loc(v))
            if hit:
                self.last_excluded.append(tid)
                self.stats["excluded"] += 1
                continue
            if self.screenfix and screen_fixed(seq, hist, self.origin, k, tid, lo_s, n_s, self.static_r):
                continue
            if self.warp_disp:
                d = self._net_disp_w(seq, hist, k, tid, lo_b, n_b)
            elif self.disp_mode == "path":
                d = path_len(seq, k, tid, lo_b, n_b)
            else:
                d = net_disp(seq, k, tid, lo_b, n_b)
            if d is not None:
                cand[tid] = d

        # ③ 置信闸
        if len(cand) >= 3:
            self.stats["gate_checked"] += 1
            med = float(np.median(list(cand.values())))
            if max(cand.values()) < LEAD * max(med, 1e-6):
                self.stats["gate_fired"] += 1
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
                    if not self.shift_chal:
                        self.chal = self.chal + bg
                    near = float(np.hypot(*(p - self.chal))) <= clk.speed(timing.CONFIRM_R_PPS, k)
                    self.streak = self.streak + 1 if near else 1
                else:
                    self.streak = 1
                self.chal = p
                if self.streak >= self.n_confirm:
                    self.pos, self.cur, self.chal, self.streak = p, pick, None, 0
                    phase, fresh = "jump", True
        if fresh:
            self.last_fresh = k
        return phase, (None if self.pos is None else tuple(self.pos)), self.cur, fresh
