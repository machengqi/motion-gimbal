"""**瞄准点控制器** —— 把估计器的跳变变成物理上合理的平滑运动。

瞄准点就是云台对准的位置（虚拟云台下 = 取景窗中心）。

# 分层：估计器该跳，控制器该平滑

| | 该不该跳 | 理由 |
|---|---|---|
| 估计器 | **该跳** | 新证据到达时更新本来就是突变的，强行平滑等于丢信息 |
| 控制器（这里） | **必须平滑** | 真实云台有转速与加速度上限；画面急跳既不可执行也不可看 |

⇒ 评价指标是「**目标落在瞄准范围内的时间占比**」，不是最后一帧的误差：
「早一点大致对」比「晚一点精确对」更有价值。

# 三个机制

```text
① 跟背景         pos += bg              —— 不确定时的基础行为
② 短暂不可见外推  + vel（带衰减）         —— 让运动保持连续
③ 限加速度趋近    朝估计值去，加速度有上限
```

⚠️ ② 必须带衰减：无衰减的等速外推会单调发散。
⚠️ 限的是**加速度**而不是速度：恒定限速卡住的是「快速追赶」，而平滑性
   约束的本质是加速度（minimum-jerk 一族的运动模型）。
"""

from __future__ import annotations

import numpy as np

from . import timing


class Aim:
    """一条平滑的瞄准点轨迹。**逐帧喂，逐帧出** —— 可在线使用。"""

    def __init__(self, clock):
        self.clk = clock
        self.pos = None  # 瞄准点（画面系）
        self.cvel = np.zeros(2)  # ⚠️ px/秒，不是「每帧位移」—— 帧长会变
        self.vel = np.zeros(2)  # 目标相对背景的速度（外推用），px/秒
        self.trail: list[tuple[int, np.ndarray]] = []  # 最近的估计位置（背景系）

    def step(self, k, bg, cum, target, fresh):
        """走一帧。

        `bg`     这一帧背景移动了多少（画面系）
        `cum`    累计背景位移（把 `target` 换算到背景系以估速度）
        `target` 估计器给出的位置，没有就 `None`
        `fresh`  这一帧的 `target` 是不是新证据
        """
        clk = self.clk
        # ① 先跟着背景走
        if self.pos is not None:
            self.pos = self.pos + bg

        if fresh and target is not None:
            self.trail.append((k, np.array(target, dtype=float) - cum))
            self.trail = self.trail[-max(2, clk.frames_in(timing.VEL_SEC)) :]
            # 速度只用真实观测之间的位移算，除以真实秒数而不是帧数
            if len(self.trail) >= 2 and self.trail[-1][0] > self.trail[0][0]:
                dt = clk.t[self.trail[-1][0]] - clk.t[self.trail[0][0]]
                self.vel = (self.trail[-1][1] - self.trail[0][1]) / max(dt, 1e-4)
        else:
            self.vel = self.vel * clk.decay(timing.DECAY_PER_SEC, k)

        if self.pos is None:
            self.pos = None if target is None else np.array(target, dtype=float)
            return None if self.pos is None else tuple(self.pos)

        # ② 期望速度（px/秒）。增益是无量纲比例，见 `timing.GAIN_KEEP`
        dt = clk.dt(k)
        if target is None:
            want = self.vel
        else:
            want = (np.array(target, dtype=float) - self.pos) * (clk.gain(timing.GAIN_KEEP, k) / dt)

        # ③ 限加速度，再限速度
        dv_max = timing.AIM_ACC_PPS2 * clk.s * dt
        dv = want - self.cvel
        n = float(np.hypot(*dv))
        if n > dv_max:
            dv = dv / n * dv_max
        self.cvel = self.cvel + dv
        v_max = timing.AIM_MAX_PPS * clk.s
        n = float(np.hypot(*self.cvel))
        if n > v_max:
            self.cvel = self.cvel / n * v_max
        self.pos = self.pos + self.cvel * dt
        return tuple(self.pos)
