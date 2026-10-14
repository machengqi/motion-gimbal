# motion-gimbal

移动相机下的运动显著性目标选择与平滑跟拍：在外观难以区分的场景里，找出相对背景独立
运动的目标（例如伪装的动物），排除静止与摆动类干扰，并驱动云台持续跟拍。

Rust（实时核心 / Tauri 宿主）+ Python（视觉算法）异构架构。

- 规划：[docs/PLAN.md](docs/PLAN.md)
- 文献调研：[docs/literature.md](docs/literature.md)

## Python 核心

```bash
cd vision
uv sync
uv run pytest
```

| 模块 | 作用 |
|---|---|
| `core/timing.py` | 时钟；所有参数按物理时间定义，按真实时间戳换算 |
| `core/mot.py` | Kalman + 匈牙利多目标跟踪；背景运动 = 可靠配对位移的中位数 |
| `core/saliency.py` | 排除名单（挂在背景系位置上）+ 净位移显著性 + 置信闸 |
| `core/controller.py` | 限加速度的平滑瞄准点控制 |
| `synthetic.py` | 合成场景，仅用于单元测试 |
