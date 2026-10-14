"""**合成检测序列** —— 只用于单元测试与调试，⛔ 不作为论文实验数据。

场景：相机匀速平移（画面里背景整体流动），画面中铺满外观相同的物体：

```text
静止干扰物   固定在背景系的网格点上            ⇒ 画面里随背景一起流动
摆动干扰物   绕背景系固定点往复摆动（风吹草叶） ⇒ 路程大、净位移小
目标         在画面里只做小幅绕圈（相机大致在跟它）⇒ 背景系里持续远离
```

⭐ 这个设计专门检验背景补偿：目标在**画面系**里几乎不动，而干扰物在画面系里
一直在流动 ⇒ 不做背景补偿、按画面系位移挑「动得最多的」会挑错。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scene:
    obs: list[list[tuple]]
    """逐帧检测 `(cx, cy, area, x1, y1, x2, y2, conf)`，画面系。"""
    ts: np.ndarray
    """逐帧时间戳（秒）。"""
    target: np.ndarray
    """逐帧目标真值位置（画面系），形状 `(n, 2)`。"""
    cum: np.ndarray
    """逐帧累计背景位移真值（画面系），形状 `(n, 2)`。"""
    roi: tuple[int, int, int, int] = field(default=(0, 0, 1041, 684))


def make_scene(
    n=180,
    fps=30.0,
    pan=(-5.0, -1.5),
    spacing=160.0,
    size=60.0,
    sway_every=5,
    sway_amp=15.0,
    noise=1.5,
    miss=0.05,
    seed=0,
    roi=(0, 0, 1041, 684),
):
    """生成一段合成序列。`pan` 是背景在画面里的每帧位移（px/帧）。"""
    rng = np.random.default_rng(seed)
    w, h = roi[2], roi[3]
    ts = np.arange(n) / fps
    cum = np.outer(np.arange(n), np.array(pan, dtype=float))

    # 背景系网格要覆盖整段相机走过的范围
    span = np.abs(cum).max(axis=0) + np.array([w, h]) + 2 * spacing
    xs = np.arange(-span[0], span[0], spacing)
    ys = np.arange(-span[1], span[1], spacing)
    grid = np.array([(x, y) for x in xs for y in ys])
    grid = grid + rng.uniform(-spacing * 0.3, spacing * 0.3, size=grid.shape)
    sway = np.zeros(len(grid), dtype=bool)
    sway[::sway_every] = True
    phase = rng.uniform(0, 2 * np.pi, size=len(grid))

    center = np.array([w * 0.5, h * 0.5])
    target = np.array([center + np.array([90 * np.cos(t * 0.8), 60 * np.sin(t * 1.1)]) for t in ts])

    obs = []
    half = size / 2
    for k in range(n):
        dets = []
        sw = sway[:, None] * sway_amp * np.stack(
            [np.sin(2 * np.pi * ts[k] + phase), np.cos(2 * np.pi * ts[k] + phase) * 0.5], axis=1
        )
        pts = list(grid + sw + cum[k]) + [target[k]]
        for p in pts:
            if rng.random() < miss:
                continue
            x1, y1, x2, y2 = p[0] - half, p[1] - half, p[0] + half, p[1] + half
            # 被画面边界截断：只剩可见部分，框心随之偏移
            cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
            if cx2 - cx1 < size * 0.3 or cy2 - cy1 < size * 0.3:
                continue
            cx = (cx1 + cx2) / 2 + rng.normal(0, noise)
            cy = (cy1 + cy2) / 2 + rng.normal(0, noise)
            area = (cx2 - cx1) * (cy2 - cy1)
            dets.append((cx, cy, area, cx1, cy1, cx2, cy2, 0.9))
        obs.append(dets)
    return Scene(obs=obs, ts=ts, target=target, cum=cum, roi=roi)


def subsample(scene: Scene, step: int) -> Scene:
    """按 `step` 抽帧，模拟低帧率；时间戳保留真实值。"""
    idx = np.arange(0, len(scene.obs), step)
    return Scene(
        obs=[scene.obs[i] for i in idx],
        ts=scene.ts[idx],
        target=scene.target[idx],
        cum=scene.cum[idx],
        roi=scene.roi,
    )
