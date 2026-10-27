# motion-gimbal

移动相机下的运动显著性目标选择与平滑跟拍：在外观难以区分的场景里，找出相对背景独立
运动的目标（例如伪装的动物），排除静止与摆动类干扰，并驱动云台持续跟拍。

Rust（实时宿主 / Tauri）+ Python（视觉算法）异构架构，两侧以共享内存帧环传帧、
HTTP 控制会话、WebSocket 回传逐帧结果。

本仓库对应毕业论文《基于 Rust 与 Python 异构架构的实时智能任务系统设计与实现》，
论文中的实现与实验结果对应 `thesis-int` 分支。

## 系统结构

```text
┌──────────────── Rust 宿主（src-tauri） ────────────────┐        ┌────────── Python 视觉服务（vision） ──────────┐
│ video.rs  ffmpeg 解码 → RGB24 帧，按时间戳节拍播放        │        │ ring.py      读帧环（序列锁校验）              │
│ gimbal.rs 虚拟云台：在全幅画面里移动取景窗               │ 帧环   │ candidates   运动候选：扣除相机运动后仍在动的区域 │
│ ring.rs   共享内存帧环写者 ─────────────────────────────┼──────▶│ core.mot     Kalman + 匈牙利多目标跟踪         │
│ vision.rs 拉起服务 / HTTP 控制 / WebSocket 收结果 ◀──────┼───────│ core.saliency 运动显著性目标选择               │
│ session.rs 播放会话：帧源 → 云台 → 帧环 → 结果驱动云台   │ 结果流 │ core.controller 限加速度的平滑瞄准             │
│ lib.rs    Tauri 命令与事件 → 前端预览                    │        │ service.py   FastAPI 服务（HTTP + WebSocket）  │
└──────────────────────────┬─────────────────────────────┘        └───────────────────────────────────────────────┘
                           │ Tauri 事件
                  ┌────────▼────────┐
                  │ app/App.vue 前端 │ 全景视图、取景窗视图、延迟与帧率曲线
                  └─────────────────┘
```

## 目录结构

```text
motion-gimbal/
├── README.md
├── docs/
│   ├── PLAN.md                 设计规划
│   ├── literature.md           文献调研
│   └── EXPERIMENTS.md          实验记录（命令、设置、结果，按时间顺序）
│
├── src-tauri/                  Rust 宿主（Tauri 2）
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   ├── src/
│   │   ├── main.rs             程序入口
│   │   ├── lib.rs              Tauri 命令（打开视频、开始/停止、设置云台、自动跟拍开关）与事件推送
│   │   ├── video.rs            视频帧源：ffmpeg 管道读取 RGB24 原始帧与时间戳
│   │   ├── gimbal.rs           虚拟云台：取景窗定位、钳边界、裁剪与缩放
│   │   ├── session.rs          播放会话：帧源 → 云台 →（帧环 → 视觉服务 → 结果驱动云台）
│   │   ├── ring.rs             共享内存帧环写者（等待 / 覆盖两种模式）
│   │   └── vision.rs           视觉服务客户端：拉起 Python 进程、HTTP 控制面、WebSocket 结果流
│   ├── examples/
│   │   └── bench_ipc.rs        跨语言链路实测（帧环读写、控制面往返、端到端延迟、吞吐）
│   └── tests/
│       ├── gimbal.rs           虚拟云台几何
│       ├── video.rs            视频探测与逐帧读取
│       ├── ring.rs             帧环往返、半帧拒读、覆盖、背压、并发
│       └── ipc.rs              Rust–Python 联调：经帧环与 WebSocket 的逐帧闭环
│
├── app/                        前端（Vue 3 + TypeScript）
│   ├── App.vue                 控制栏、全景视图、取景窗视图、指标曲线
│   └── main.ts
├── index.html / package.json / vite.config.ts / tsconfig.json
│
└── vision/                     Python 视觉服务（uv 管理）
    ├── pyproject.toml
    ├── splits/moca_split.json  MoCA 开发集 / 测试集划分（按视频，固定种子）
    ├── src/gimbal_vision/
    │   ├── core/
    │   │   ├── timing.py       时钟：参数按物理时间定义，按真实时间戳换算；帧率与抖动估计
    │   │   ├── mot.py          多目标跟踪（恒速 Kalman + 匈牙利关联）与背景运动估计
    │   │   ├── saliency.py     目标选择：排除名单 + 净位移显著性 + 置信闸
    │   │   ├── controller.py   瞄准点控制：限速度与加速度的平滑运动
    │   │   └── __init__.py     离线 run() 与逐帧 Pipeline
    │   ├── candidates/
    │   │   └── motion.py       运动候选生成：特征点光流 + 单应估计相机运动，残差聚类成框
    │   ├── datasets/moca.py    MoCA 数据集加载器
    │   ├── stabilize.py        评测用的背景稳定化（E2b 协议的画布坐标）
    │   ├── ring.py             共享内存帧环读者（布局与 ring.rs 逐字节一致）
    │   ├── service.py          视觉服务（FastAPI：/health、/session、/status、/params、WS /results）
    │   └── synthetic.py        合成场景，仅用于单元测试
    ├── scripts/                评测与诊断脚本（见下表）
    └── tests/                  单元测试（见下文）
```

### 评测与诊断脚本（`vision/scripts/`）

| 脚本 | 用途 | 论文对应 |
|---|---|---|
| `make_split.py` | 生成 MoCA 开发集 / 测试集划分 | 6.1 |
| `eval_candidates.py` | E1：运动候选的单帧命中率 | 6.3 |
| `eval_e2.py` | E2 / E2b：端到端跟拍评测（开环、闭环） | 6.4 |
| `eval_fps.py` | 帧率与帧间隔抖动实验 | 6.5 |
| `calib_r.py` | 跟踪器观测噪声标定 | 6.6 |
| `sweep_e2.py` | 参数灵敏度扫描与重标定 | 4.7 |
| `eval_abl.py` | 消融：排除名单、净位移 / 路程、限加速度 / 只限速度 | 4.5、4.6 |
| `abl_synth.py` | 消融的合成场景对照（只说明机制，不作为数据集结论） | 4.5 |
| `diag_drift.py` | 背景模型累积漂移：平移 / 相似 / 单应 | 2.3 |
| `diag_e1_zero.py` | E1 零命中视频的逐段归因 | 6.7 |
| `diag_e2b_stab.py` | E2b 背景稳定化自检 | 6.4 |
| `diag_gate.py`、`diag_e2_population.py`、`diag_e2_failures.py`、`diag_bgshift.py` | E2 失败案例与门控行为诊断 | 6.6、6.7 |
| `diag_common.py`、`lowprio.py` | 脚本公用：候选缓存；单线程、低优先级运行 | — |

## 运行

依赖：Rust（stable）、Node.js、[uv](https://docs.astral.sh/uv/)、PATH 中的 ffmpeg。

```bash
# Python 视觉服务
cd vision
uv sync --extra service
uv run pytest                     # 65 个用例

# Rust 宿主
cd src-tauri
cargo test                        # 13 个用例（ipc 联调需要上面的 vision/.venv）

# 桌面程序（前端 + 宿主，自动拉起视觉服务）
npm install
npm run tauri dev
```

### 数据集

实验使用公开数据集 MoCA（Moving Camouflaged Animals，Lamdouar 等，ACCV 2020，CC BY 4.0），
数据不入库。下载 <https://thor.robots.ox.ac.uk/datasets/MoCA/MoCA.zip> 并解压到仓库根目录的
`data/moca/`。

### 复现实验

在 `vision/` 下运行，结果写入仓库根目录的 `runs/`：

```bash
uv run python scripts/eval_candidates.py --workers 6
uv run python scripts/eval_e2.py --split test --protocol e2b --frac 0.5 --workers 6
uv run python scripts/eval_fps.py --split test --workers 4
uv run python scripts/eval_abl.py --split test --workers 6
```

参数只在开发集上选定，测试集只报告。各实验的完整命令、设置与结果见
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 测试

| 层次 | 内容 |
|---|---|
| Rust | 虚拟云台几何、视频帧源、帧环（半帧拒读、覆盖跳帧、背压、并发）、Rust–Python 联调（逐帧校验像素 CRC、帧号、时间戳与取景窗原点） |
| Python | 背景运动恢复、目标选择与门控、加速度有界、排除名单、时间窗按时间戳切分、运动候选、帧环读写、`Pipeline` 与离线 `run` 逐帧一致、服务端到端结果与离线一致，以及帧率、灵敏度与消融实验的开关 |

单元测试使用合成场景验证**实现正确**，方法效果在 MoCA 上评测。
