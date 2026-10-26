"""**时间单位的常数表 + 时钟** —— 整条链不假设固定帧率，也不假设帧间隔均匀。

# 为什么参数要按「时间」而不是「帧」定义

「窗口 14 帧」在 30fps 是 0.47 秒，在 10fps 是 1.4 秒 —— 含义完全不同。
把按帧调好的常数原样搬到更低的帧率，跟踪会整体失效；而真实采集链路的
帧率既会变、帧间隔也**不均匀**（编码、传输、推理的耗时都在抖）。

⇒ 所有参数都用物理单位定义，运行时按**真实时间戳**换算到这一帧：

```text
 时间类    秒        窗口长度、寿命门槛、印证时长      ÷ 帧长
 速度类    px/秒     关联闸、跳变闸、控制器限速        × 帧长
 加速度类  px/秒²    控制器的加速度上限                × 帧长²
 绝对空间  px        物体半径、名单半径、贴边边距      不变
 无量纲    —         领先倍数、面积比、滞后            不变
```

⚠️ 衰减按「每秒」定义，不是「每帧」。
⚠️ 所有 px 量还要再乘「画面缩放」`Clock.s = roi_w / REF_W` —— **先按时间、再按画面宽**。

# 初值从哪来

下面的数值是在 `REF_FPS`、`REF_W` 的参考条件下标定的初值，**须在本项目的
数据上重新标定**（见 `docs/PLAN.md` 的实验章节）。注释里只写每个参数的含义
和换算类别，不写来自其他数据的实测数字。
"""

from __future__ import annotations

import numpy as np

REF_FPS = 30.0
"""常数标定时的参考帧率。⚠️ 只用于换算，不是对运行帧率的假设。"""

REF_W = 1041.0
"""常数标定时的参考画面宽度（像素）。实际阈值 = 阈值 × 画面宽 / REF_W。"""

# ── 时间类（秒）────────────────────────────────────────────────
WINDOW_SEC = 14 / REF_FPS
"""运动显著性判据的时间窗。"""

STATIC_SEC = 44 / REF_FPS
"""判「一个位置上的东西一直没动」要看多久。"""

CONFIRM_SEC = 7 / REF_FPS
"""估计值要跳到远处时，需要连续印证多久。"""

FILL_SEC = 6 / REF_FPS
"""轨迹断档回填的最长时长（只在两次真实观测之间插值）。"""

MISS_SEC = 3 / REF_FPS
"""轨迹连续丢失多久就删除。"""

STITCH_SEC = 16 / REF_FPS
"""删除后的轨迹在多长时间内还允许被缝合回来。"""

VEL_SEC = 6 / REF_FPS
"""控制器估计目标速度时回看多久。"""

STALE_SEC = STATIC_SEC
"""估计位置连续这么久没有新证据，就撤回（见 `saliency.Selector` 的 `stale`）。

取与 `STATIC_SEC` 相同的值，不单独调：「一个位置上的东西多久不动算静止」与「一个估计多久没被
印证算过期」是同一个时间尺度 —— 目标停下超过这个时长，它所在的点本来就会被排除名单压住。
"""

# ── 速度类（px/秒 @REF_W）──────────────────────────────────────
JUMP_PPS = 90 * REF_FPS
"""估计值一步允许移动的上限；超过就要先印证。"""

CONFIRM_R_PPS = 45 * REF_FPS
"""印证时「算同一个地方」的半径。"""

GATE_PPS = 90 * REF_FPS
"""数据关联的距离闸。"""

STITCH_GATE_PPS = 70 * REF_FPS
"""缝合的距离闸。"""

CUT_PPS = 25 * REF_FPS
"""判「镜头切换」的背景位移阈值。"""

AIM_MAX_PPS = 120 * REF_FPS
"""瞄准点速度的硬上限。"""

GAIN_KEEP = 1.0 - 0.55
"""趋近增益，存成「**每个参考帧剩下多少差距**」。

⛔ 它是**无量纲比例**，不是速率，不能乘 `REF_FPS` 做线性换算：
「每参考帧消掉 55%」⇒ 每参考帧剩 0.45 ⇒ `dt` 秒后剩 `0.45^(dt·REF_FPS)`。
线性换算在低帧率下会让单帧消掉超过 100% 的差距，即过冲。
"""

# ── 跟踪器的不确定性模型（「uncertainty」门控模式）────────────
ACC_PSD = 0.35 * REF_FPS**3
"""过程噪声：加速度白噪声的功率谱密度（px²/s³）。

速度方差每秒增长 `ACC_PSD`；在参考帧率下单帧增长量等于原按帧模型的 0.35 (px/帧)²。
"""

POS_PSD = 2.0 * REF_FPS
"""过程噪声：位置随机游走的功率谱密度（px²/s），吸收恒速模型之外的抖动。"""

INIT_VEL_VAR = 100.0 * REF_FPS**2
"""新轨迹的初始速度方差（(px/s)²）—— 速度未知。"""

GATE_CHI2 = 9.21
"""马氏距离门控阈值：二维 χ² 分布的 99% 分位点。

⭐ 闸宽由**预测的不确定性**决定：稳定轨迹的预测很准 ⇒ 闸窄，邻居进不来；
新轨迹或久未观测的轨迹不确定性大 ⇒ 闸自动放宽。与帧率的关系来自过程噪声
随 `dt` 的增长，而不是「最大速度 × 帧间隔」。见 docs/PLAN.md §6.1。
"""

# ── 加速度类（px/秒² @REF_W）──────────────────────────────────
AIM_ACC_PPS2 = 20 * REF_FPS * REF_FPS
"""瞄准点的加速度上限 —— 真正约束运动平滑性的那一条。"""

# ── 衰减（每秒）───────────────────────────────────────────────
DECAY_PER_SEC = 0.88**REF_FPS
"""目标短暂不可见期间，外推速度的衰减。"""

# ── 帧率护栏 ─────────────────────────────────────────────────
HZ_GOOD = 15.0
HZ_MIN = 12.0
"""帧率低于 `HZ_MIN` 时不尝试跟踪：低帧率下一个时间窗里的信息量不够，
指错的代价高于不指。具体阈值须在本项目数据上用帧率鲁棒性实验重新确定。"""
HZ_TOL = 1e-6
"""帧率比较的容差：24 Hz 素材隔帧取出的 12 Hz，`1 / (2/24)` 算出来是 11.999…，不能因此判为低于下限。"""


class Clock:
    """一段序列的时间轴。**所有单位换算都从这里出。**

    `ts` 是每帧的时间戳（秒）。给 `None` 就按 `REF_FPS` 均匀假设 ——
    ⚠️ 那只用于离线测试；在线运行必须传真实时间戳。
    """

    def __init__(self, n, ts=None, ref_w=REF_W, roi_w=None):
        self.n = n
        if ts is None:
            ts = np.arange(n, dtype=float) / REF_FPS
        self.t = np.asarray(ts, dtype=float)
        if len(self.t) != n:
            raise ValueError(f"时间戳 {len(self.t)} 个，帧 {n} 个 —— 必须一一对应")
        # 第 0 帧没有「上一帧」，用中位间隔占位，避免出现 0 或负数
        d = np.diff(self.t)
        self.med_dt = float(np.median(d)) if len(d) else 1.0 / REF_FPS
        self.dts = np.concatenate([[self.med_dt], d])
        self.s = 1.0 if roi_w is None else roi_w / ref_w

    def dt(self, k):
        """第 `k` 帧和上一帧之间隔了多久（秒）。重复帧可能为 0，下限钳到 1e-4。"""
        return max(float(self.dts[k]), 1e-4)

    def hz(self):
        """实际帧率（用中位间隔，抗住个别卡顿）。"""
        return 1.0 / self.med_dt

    def jitter(self):
        """帧间隔抖动：`p90 / 中位`。1.0 = 完全均匀。"""
        return float(np.percentile(self.dts[1:], 90) / self.med_dt) if self.n > 2 else 1.0

    def frames_in(self, seconds):
        """`seconds` 大约相当于多少帧（至少 1）。"""
        return max(1, int(round(seconds / self.med_dt)))

    def window(self, k, seconds):
        """以第 `k` 帧为右端、往回 `seconds` 秒的窗口 → `(起始帧号, 应有帧数)`。

        ⚠️ **按时间戳切，不按帧号切** —— 帧间隔不均匀时两者不等价。
        """
        t0 = self.t[k] - seconds
        lo = int(np.searchsorted(self.t, t0, side="left"))
        return lo, max(1, k - lo + 1)

    # ── 换算 ──────────────────────────────────────────────────
    def px(self, absolute):
        """绝对空间量：只按画面缩放。"""
        return absolute * self.s

    def speed(self, pps, k):
        """速度量 → 第 `k` 帧的位移上限（含画面缩放）。"""
        return pps * self.s * self.dt(k)

    def speed_raw(self, pps, k):
        """同上但**不乘画面缩放**。

        关联闸要覆盖两样东西：**运动**（随画面缩放）+ **检测噪声**（几个像素，
        不随画面缩小）。纯比例缩放在小画面上会覆盖不足，所以关联类的闸用它。
        """
        return pps * self.dt(k)

    def accel(self, pps2, k):
        """加速度量 → 第 `k` 帧的速度变化上限。"""
        return pps2 * self.s * self.dt(k) * self.dt(k)

    def decay(self, per_sec, k):
        """每秒衰减 → 这一帧的衰减。"""
        return per_sec ** self.dt(k)

    def gain(self, keep, k):
        """无量纲的「每参考帧剩下多少差距」→ 这一帧消掉多少。见 [`GAIN_KEEP`]。"""
        return 1.0 - keep ** (self.dt(k) * REF_FPS)

    def ok(self):
        """这条时间轴能不能跑。返回 `(能不能, 说明)`。"""
        hz = self.hz()
        if hz < HZ_MIN - HZ_TOL:
            return False, f"帧率只有 {hz:.1f}Hz（下限 {HZ_MIN:.0f}Hz）—— 不尝试跟踪"
        if hz < HZ_GOOD - HZ_TOL:
            return True, f"帧率 {hz:.1f}Hz 偏低（正常线 {HZ_GOOD:.0f}Hz），精度会下降"
        if self.jitter() > 1.6:
            return True, f"帧间隔抖动 {self.jitter():.2f}（1.0=均匀），精度会下降"
        return True, f"帧率 {hz:.1f}Hz、抖动 {self.jitter():.2f}"


class OnlineClock(Clock):
    """**在线时钟** —— 帧一边到达一边 `push(t)`，时间轴随之增长。

    与 [`Clock`] 的区别只在「中位帧间隔」怎么来：离线版用整段的中位数（看到了未来），这里用
    **到当前帧为止**最近 `window` 个间隔的中位数（因果）。帧间隔均匀时两者在第 1 帧之后逐位相同，
    这是 `core.Pipeline` 与离线 `core.run` 结果一致的前提（见 tests/test_service.py）。

    ⚠️ 第 0 帧还没有间隔，中位间隔先用 `1 / fps_hint` 占位；第 0 帧上没有任何量用到它
    （新轨迹不预测、瞄准点首帧直接取估计值）。
    """

    def __init__(self, ref_w=REF_W, roi_w=None, fps_hint=REF_FPS, window=512):
        self.n = 0
        self._t = np.empty(256)
        self._dts = np.empty(256)
        self.window_n = int(window)
        self.med_dt = 1.0 / fps_hint
        self.s = 1.0 if roi_w is None else roi_w / ref_w

    @property
    def t(self):
        return self._t[: self.n]

    @property
    def dts(self):
        return self._dts[: self.n]

    def push(self, ts):
        """追加一帧的时间戳（秒），返回它的帧号。时间戳必须不减（相等可以：dt 会钳到 1e-4）。"""
        ts = float(ts)
        if self.n and ts < self._t[self.n - 1]:
            raise ValueError(f"时间戳倒退：{ts} < {self._t[self.n - 1]}")
        if self.n == len(self._t):
            self._t = np.concatenate([self._t, np.empty(len(self._t))])
            self._dts = np.concatenate([self._dts, np.empty(len(self._dts))])
        k = self.n
        self._t[k] = ts
        self.n += 1
        if k:
            self._dts[k] = ts - self._t[k - 1]
            lo = max(1, self.n - self.window_n)
            self.med_dt = float(np.median(self._dts[lo : self.n]))
            if self.med_dt <= 0:
                self.med_dt = 1e-4
        else:
            self._dts[k] = self.med_dt
        return k
