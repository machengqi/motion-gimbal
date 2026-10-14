"""**多目标跟踪** —— 恒速 Kalman + 匈牙利关联，外加「相机自运动（背景运动）」估计。

这个模块只知道「一堆会动的框」，不知道框里是什么。

# 三个坐标系，别混

```text
画面      检测器输出的框所在的坐标系（取景区域 ROI）
背景系    画面坐标减掉累计背景位移 —— 「谁在相对背景运动」在这个系里判
全幅      相机原始整帧（虚拟云台下取景窗只是全幅里的一块）
```

# 背景运动 = 所有可靠配对位移的**中位数**

画面里多数物体是静止的（随相机一起「流过」画面）⇒ 它们的共同位移就是背景
位移；中位数天然抗住少数自己在动的物体和个别关联错误。

⚠️ 当前只建模**纯平移**。相机旋转、变焦、景深视差需要单应模型
（见 `docs/PLAN.md` 的局限与扩展）。
"""

from __future__ import annotations

import numpy as np

from . import timing

EDGE_MARGIN = 40.0
"""算背景位移时，离画面边缘近于此值（@`timing.REF_W`）的框不参与。

框被画面边界截断时只有一部分可见，框心走得比物体慢 ⇒ 贴边的框会把背景
速度系统性地估低，而漂移量与背景走过的总路程成正比。边距按画面宽缩放。
"""

STITCH_AREA = 0.8
"""缝合时允许的面积比变化（`|log(新/旧)|` 上限）。物体渐隐时框会明显缩小。"""

CUT_KEEP = 0.35
"""帧间配对率低于这个比例就判「镜头切换」，重置累计背景位移。"""


GATES = ("uncertainty", "speed")
"""关联门控的两种模式。

`uncertainty`（默认）  按真实 `dt` 建模的 Kalman + 马氏距离门控，再叠加物理上限
`speed`                前期系统的原设计：按帧 Kalman + 「最大速度 × 帧间隔」距离闸。
                       保留用于消融对照；低帧率下闸宽超过物体间距会抢配邻居（§6.1）
"""


class Track:
    """一条轨迹的恒速 Kalman，在**画面坐标系**里跑。

    为什么不在背景系里跑：背景在动时，多数物体在画面里的运动是一致的，
    关联反而最容易；「偏离共同运动」在报告时减掉累计背景位移即可得到。

    两种模型（见 [`GATES`]）：
    - `dt_aware=True`：速度单位 px/秒，状态转移与过程噪声按真实 `dt` 计算；
    - `dt_aware=False`：前期系统的按帧模型，速度单位 px/帧，噪声每帧固定。
    """

    Q = np.diag([2.0, 2.0, 0.35, 0.35])
    """按帧模型的过程噪声（每帧）。"""
    R = np.diag([9.0, 9.0])
    """观测噪声（px²）。量级应与检测框心的抖动相符，换检测器后要重新实测。"""
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)

    def __init__(self, tid, x, y, area, dt_aware=True):
        self.id = tid
        self.dt_aware = dt_aware
        self.x = np.array([x, y, 0.0, 0.0])
        vel_var = timing.INIT_VEL_VAR if dt_aware else 100.0
        self.P = np.diag([4.0, 4.0, vel_var, vel_var])
        self.area = area
        self.age = self.missed = self.hits = 0
        self.last_obs = (x, y)
        self.last_k = None
        self.edge = 0.0
        """上次观测时框离画面边缘多少像素。

        存距离不存布尔：背景估计、候选筛选、重捕获三处用不同阈值。
        ⛔ 构造时的 0 只是占位，**新建轨迹的三处都必须立刻赋真值**
        （新生轨迹不走 `update()`，漏赋会让依赖它的分支永远不执行且不报错）。
        """

    def predict(self, dt=1.0):
        """`dt` 是距上一帧的秒数；按帧模型忽略它（每帧算一步）。"""
        if self.dt_aware:
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
            # 连续白噪声加速度模型（每轴）+ 位置随机游走
            q, qp = timing.ACC_PSD, timing.POS_PSD
            pp, pv, vv = qp * dt + q * dt**3 / 3, q * dt**2 / 2, q * dt
            Q = np.array([[pp, 0, pv, 0], [0, pp, 0, pv], [pv, 0, vv, 0], [0, pv, 0, vv]])
        else:
            F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
            Q = self.Q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += 1
        self.missed += 1

    def mahal2(self, zx, zy):
        """观测 `(zx, zy)` 相对当前预测的马氏距离平方。"""
        y = np.array([zx, zy]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(y @ np.linalg.solve(S, y))

    def update(self, zx, zy, area):
        H = self.H
        y = np.array([zx, zy]) - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.area = area
        self.missed = 0
        self.hits += 1
        self.last_obs = (zx, zy)

    @property
    def pos(self):
        return self.x[0], self.x[1]


def hungarian(cost):
    """最优指派（匈牙利算法）。返回 `{行: 列}`。"""
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    r, c = linear_sum_assignment(cost)
    return dict(zip(r.tolist(), c.tolist(), strict=True))


Detection = tuple[float, float, float, float, float, float, float, float]
"""一个检测：`(cx, cy, area, x1, y1, x2, y2, conf)`，画面坐标系。"""


class Tracker:
    """**逐帧喂、逐帧出** —— 在线和离线走同一份代码。

    ```python
    tk = Tracker(roi, clock)
    for k, dets in enumerate(obs):
        row = tk.step(k, dets)   # (帧号, {tid: (...)}, 累计背景位移)
    ```
    """

    def __init__(self, roi, clock=None, stitch=True, min_conf=0.0, n_hint=1, gate="uncertainty"):
        if gate not in GATES:
            raise ValueError(f"gate 只能是 {GATES}，收到 {gate!r}")
        self.gate = gate
        self.clk = clock or timing.Clock(n_hint, roi_w=roi[2])
        self.stitch = stitch
        self.min_conf = min_conf
        self.rw, self.rh = roi[2], roi[3]
        self.margin = EDGE_MARGIN * self.rw / timing.REF_W
        self.tracks, self.next_id, self.cum = [], 0, np.zeros(2)
        self.last_shift = np.zeros(2)
        self.morgue, self.history = [], []

    # ⛔ 这两个必须是 property，不能在 `__init__` 里算死：在线用法里时钟随帧
    #    更新（帧率估计越跑越准），算死就停在第一帧的估计上，而且是静默的。
    @property
    def max_miss(self):
        return self.clk.frames_in(timing.MISS_SEC)

    @property
    def stitch_gap(self):
        return self.clk.frames_in(timing.STITCH_SEC)

    def edge_of(self, d):
        """框离画面边缘最近多少像素（负数 = 已经超出）。"""
        return float(min(d[3], d[4], self.rw - d[5], self.rh - d[6]))

    def _born(self, tid, d, k):
        t = Track(tid, d[0], d[1], d[2], dt_aware=self.gate == "uncertainty")
        t.last_k = k
        t.edge = self.edge_of(d)
        return t

    def step(self, k, dets_all):
        clk, edge_of, margin = self.clk, self.edge_of, self.margin
        max_miss, stitch_gap = self.max_miss, self.stitch_gap
        tracks, next_id, cum = self.tracks, self.next_id, self.cum
        last_shift, morgue = self.last_shift, self.morgue
        dets = [d for d in dets_all if d[7] >= self.min_conf]
        for t in tracks:
            t.predict(clk.dt(k))

        pairs, shifts = {}, []
        if tracks and dets:
            # 物理上限：两种模式都有。uncertainty 模式下真正起作用的是马氏距离闸
            gate = clk.speed_raw(timing.GATE_PPS, k)
            by_uncertainty = self.gate == "uncertainty"
            cost = np.full((len(tracks), len(dets)), 1e4)
            for i, t in enumerate(tracks):
                # ⛔ 不拿「上次观测 + 背景已滚过的距离」当预测位置：那个先验对
                #    唯一关心的目标恰好是错的（它不随背景走），会把它的检测让给
                #    邻居。关联质量和判别力不是同一个目标。
                tx, ty = t.pos
                for j, d in enumerate(dets):
                    dist = float(np.hypot(tx - d[0], ty - d[1]))
                    if dist < gate and (
                        not by_uncertainty or t.mahal2(d[0], d[1]) <= timing.GATE_CHI2
                    ):
                        ar = abs(np.log(max(d[2], 1) / max(t.area, 1)))
                        cost[i, j] = dist + 40.0 * ar
            pairs = {i: j for i, j in hungarian(cost).items() if cost[i, j] < 1e4}
            for i, j in pairs.items():
                # ⛔ 只用「上一帧刚观测过、且两帧都没贴边」的配对算背景位移：
                #    断档后重新配上的轨迹会把多帧位移当一帧算；贴边见 EDGE_MARGIN。
                if (
                    tracks[i].last_k == k - 1
                    and edge_of(dets[j]) >= margin
                    and tracks[i].edge >= margin
                ):
                    shifts.append(
                        (dets[j][0] - tracks[i].last_obs[0], dets[j][1] - tracks[i].last_obs[1])
                    )

        # ⛔ 镜头切换要重置：签名是「配对率很低」或「位移大到不像物理量」。
        #    不重置的话累计背景位移会被永久污染，之后所有背景系坐标都是错的。
        cut = bool(shifts) and (
            float(np.hypot(*np.median(np.array(shifts), axis=0)))
            > clk.speed_raw(timing.CUT_PPS, k)
            or len(pairs) < CUT_KEEP * max(len(dets), 1)
        )
        if cut:
            tracks, pairs, shifts = [], {}, []
            for dd in dets:
                tracks.append(self._born(next_id, dd, k))
                next_id += 1

        if shifts:
            last_shift = np.median(np.array(shifts), axis=0)
            cum = cum + last_shift
        elif cut:
            last_shift = np.zeros(2)

        used = set()
        for i, j in pairs.items():
            tracks[i].update(dets[j][0], dets[j][1], dets[j][2])
            tracks[i].last_k = k
            tracks[i].edge = edge_of(dets[j])
            used.add(j)

        # ⭐ 新建轨迹之前，先看能不能缝回一条刚删掉的。
        #    与「把 max_miss 调大」不同：调大会让滑行中的轨迹参与关联竞争、
        #    从活轨迹手里抢检测；缝合只发生在一个检测本来就要新建轨迹时。
        for j, d in enumerate(dets):
            if j in used:
                continue
            best = None
            bd = clk.speed_raw(timing.STITCH_GATE_PPS, k) if self.stitch else -1.0
            for m in morgue:
                gap = k - m["last_k"]
                if gap < 1 or gap > stitch_gap:
                    continue
                pxy = (
                    m["last_obs"][0] + last_shift[0] * gap,
                    m["last_obs"][1] + last_shift[1] * gap,
                )
                dist = float(np.hypot(pxy[0] - d[0], pxy[1] - d[1]))
                if dist < bd and abs(np.log(max(d[2], 1) / max(m["area"], 1))) < STITCH_AREA:
                    best, bd = m, dist
            if best is not None:
                morgue.remove(best)
                t = self._born(best["id"], d, k)
                t.hits = best["hits"]
                tracks.append(t)
            else:
                tracks.append(self._born(next_id, d, k))
                next_id += 1

        for t in [t for t in tracks if t.missed > max_miss]:
            morgue.append(
                {"id": t.id, "last_obs": t.last_obs, "last_k": t.last_k, "area": t.area, "hits": t.hits}
            )
        morgue = [m for m in morgue if k - m["last_k"] <= stitch_gap]
        tracks = [t for t in tracks if t.missed <= max_miss]

        self.tracks, self.next_id, self.cum = tracks, next_id, cum
        self.last_shift, self.morgue = last_shift, morgue
        row = (
            k,
            {
                t.id: (t.pos[0] - cum[0], t.pos[1] - cum[1], t.x[2], t.x[3], t.hits, t.missed, t.edge)
                for t in tracks
            },
            cum.copy(),
        )
        self.history.append(row)
        return row


def track(obs, roi, clock=None, stitch=True, min_conf=0.0, gate="uncertainty"):
    """离线口径：把 [`Tracker`] 跑一遍。**这里不许有第二份逻辑。**

    `obs[k]` 是第 k 帧的检测列表（见 [`Detection`]）。返回逐帧
    `(帧号, {tid: (x, y, vx, vy, hits, missed, edge)}, 累计背景位移)`，
    其中 `(x, y)` 在**背景系**；`(vx, vy)` 的单位随门控模式不同（px/秒 或 px/帧）。
    """
    clk = clock or timing.Clock(len(obs), roi_w=roi[2])
    tk = Tracker(roi, clk, stitch=stitch, min_conf=min_conf, gate=gate)
    for k, dets in enumerate(obs):
        tk.step(k, dets)
    return tk.history


def paths(history):
    """`tid -> [(帧号, 背景系x, 背景系y)]`，**只收真实观测**（滑行中的不算）。"""
    seq = {}
    for row in history:
        for tid, v in row[1].items():
            if v[5] == 0:
                seq.setdefault(tid, []).append((row[0], v[0], v[1]))
    return seq


def fill_gaps(seq, max_gap):
    """**断档回填**：断了又接上的轨迹，中间几帧用两端真实观测线性插值补出来。

    思路来自 OC-SORT 的 Observation-Centric Re-Update。作用：物体被短暂遮挡后
    重新出现时，历史立刻连续，不必重新攒满一个判据窗口。

    ⚠️ **只在两次真实观测之间插值，绝不外推**（外推会单调发散）。
    缝错了就是凭空捏造的运动，所以 `max_gap` 要小。
    """
    if max_gap <= 0:
        return seq
    out = {}
    for tid, pts in seq.items():
        if len(pts) < 2:
            out[tid] = pts
            continue
        filled = [pts[0]]
        for (k0, x0, y0), (k1, x1, y1) in zip(pts, pts[1:], strict=False):
            gap = k1 - k0
            if 1 < gap <= max_gap + 1:
                for j in range(1, gap):
                    t = j / gap
                    filled.append((k0 + j, x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
            filled.append((k1, x1, y1))
        out[tid] = filled
    return out
