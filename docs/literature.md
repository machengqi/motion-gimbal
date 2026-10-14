# 文献调研：移动相机下的伪装动物「运动显著目标」实时跟拍

调研日期：2026-10-14

## 证据等级说明

- **[A]**：亲自打开读过论文原文（PDF 全文 / arXiv 页 / 项目页 / 代码文件）。
- **[B]**：来自二手资料（搜索摘要、第三方汇总表、综述转述）。
- **[C]**：本报告的推断。
- 「我在 X、Y 里没找到」只表示在所列来源里没检索到，**不等于不存在**。

本次实际打开过的原始材料：MoCA 项目页、MoCA zip（HTTP HEAD）、Betrayed by Motion / SLT-Net(MoCA-Mask) / CAD(ECCV16) / Motion Grouping / FlowSAM / BoT-SORT / Making and Breaking of Camouflage 七篇 PDF 全文（pdftotext 检索），SLT-Net 项目页与 Google Drive 文件夹，CAMotion 项目页与 arXiv HTML，MSVCOD arXiv HTML，COTD arXiv 页 / GitHub / IEEE DataPort 页，ByteTrack `byte_tracker.py` 与 BoT-SORT `gmc.py` 源码页，GitHub API 许可证字段，arXiv API 元数据（标题、作者、comment 字段）。

---

## A. 数据集核实（最重要）

### A.1 总表

| 数据集 | 是否存在 | 规模 | 分辨率 / 帧率 | 标注 | 许可证 | 下载可用性（2026-10-14 实测） | 适合单目标跟拍评测？ |
|---|---|---|---|---|---|---|---|
| **MoCA** | 是 [A] | 141 段，37,250 帧，67 类动物，约 26 分钟 [A] | 主要 720×1280，统一重采样到 24fps [A] | 每 5 帧一个紧致框 + 运动类型标签（locomotion / deformation / still），其余帧线性插值 [A] | CC BY 4.0（项目页声明）[A] | zip 可下（HTTP 200，9,917,063,824 字节）[A] | **最合适**，见 A.2 |
| **MoCA-Mask** | 是 [A] | 87 段，22,939 帧；训练 71 段 / 19,313 帧，测试 16 段 / 3,626 帧 [B，数字由搜索摘要给出；87 段和 22,939 帧经原文核实 A] | 继承 MoCA（720×1280 为主）[A] | 每 5 帧一张像素级掩码 + 框；另提供光流伪标签版本 [A] | **项目页和 README 都没写许可证** [A]；SLT-Net 仓库 GitHub API 没有 license 字段 [A] | Google Drive 文件夹可访问：`MoCA-Mask.zip` 1.2 GB、`MoCA-Mask-Pseudo.zip` 6.17 GB [A]；也有百度盘链接（未核实） | 可用作第二套，但段数少、测试仅 16 段 |
| **CAD**（Camouflaged Animal Dataset） | 是 [A] | 9 段，共 839 帧（按原文附表逐段相加：218+31+104+84+105+61+76+80+80）[A]；181 张掩码 [B，来自 CAMotion 对比表] | 原文附录未写分辨率 [A] | 每 5 帧一张手工像素级掩码 [A] | 我在原文和搜索结果里都没找到许可证 | **原下载页已失效**：`user.tu-berlin.de/pbideau/...` 301 跳转到 `pia-bideau.com/motionSegmentation/index.html`，后者 404；论文给的代码地址 `vis-www.cs.umass.edu/motionSegmentation/` 连接失败 [A]。可用的镜像**未核实** | 太小（9 段、单段 31~218 帧），只适合作定性 / 补充，不适合做主评测 [C] |
| **MSVCOD** | 是（arXiv 预印本）[A] | 162 段，9,486 帧，6fps；4 类目标（动物 / 人 / 医学 / 车辆）[A] | 未查到 | 像素掩码 + 框 + 实例 / 类别标签，每秒 6 帧标注 [A] | 未查到 | 论文写「将会公开」[A]；是否已发布**未核实** | 帧率太低（6fps）且含非动物，不适合测帧率鲁棒性 [C] |
| **CAMotion** | 是（arXiv 2604.08287，2026-04，在审）[A] | 474 段（训练 359 / 测试 115），约 15 万帧，30,028 帧有标注，151 个物种 [A] | 原文未给统一分辨率 [A] | 每 5 帧一张像素掩码 + 框 + 边缘标注；8 种属性标签 [A] | 项目页写「仅限非商业研究」[A] | 项目页 `camotion.focuslab.net.cn` 可访问（HTTP 200），提供 Google Drive / 百度盘 [A]；文件本身未下载核实 | 规模最大、明确提到「frequent camera pose variations」[A]；可作第二基准。是否每段只有一个目标需按属性表里「multiple objects」筛 [C] |
| **COTD**（Camouflaged Object Tracking Dataset） | 是（arXiv 2408.13877）[A] | 200 段，约 8 万帧，平均 406.55 帧 / 段；20 类（含 person）[A] | 270×480 ~ 2488×1152 [A] | **逐帧框**（`groundtruth_rect.txt`，单目标跟踪 SOT 格式）[A] | arXiv 页的 CC BY-NC-ND 4.0 是**论文**的许可证，不是数据集的；数据集许可证我没找到 [A] | GitHub 只给百度盘（提取码 qpeg）[A]；IEEE DataPort 页写「需订阅」且「文件尚未上传」[A] | **最贴近「单目标跟拍」的标注格式**，但它是 SOT 设定（第一帧给框），且原文没说明相机是否运动 [A]，需要自己筛选移动相机片段 [C] |
| **BIRDSAI** | 是（WACV 2020）[B] | 未核实 | 热红外，固定翼无人机，有大幅相机转动 [B] | 未核实 | 未核实 | 未核实 | 是移动相机 + 动物，但模态是热红外，与「伪装」设定不同 [C] |

### A.2 MoCA 细节（主推）

- 论文：Hala Lamdouar, Charig Yang, Weidi Xie, Andrew Zisserman. *Betrayed by Motion: Camouflaged Object Discovery via Motion Segmentation*. ACCV 2020. arXiv:2011.11630。[A]
- 项目页：https://www.robots.ox.ac.uk/~vgg/data/MoCA/ （可访问）[A]
- 下载：https://thor.robots.ox.ac.uk/datasets/MoCA/MoCA.zip （约 10 GB，实测 HTTP 200）[A]
- 原文写明：「每段视频是一段连续镜头，**只有一只伪装动物**」，长 1~79 秒；视频来自 YouTube [A]。
- 标注：每 5 帧一个紧致框 + 运动标签，其余帧线性插值 [A]。项目页没说明标注文件格式（CSV / JSON）[A]，需下载后确认。
- **⚠ 对第 5 点实验很关键的一条坑** [A]：Motion Grouping（Yang et al., ICCV 2021）附录 B 写道，MoCA 原作者「把所有视频重采样到 24fps，即使原视频帧率更低，**导致部分相邻帧完全相同**」。他们因此每 3 帧取 1 帧（每段至多 100 帧）。
  - 推论 [C]：MoCA 的「24fps 时间戳」对一部分视频不是真实采样时刻，重复帧等价于一种**非均匀帧间隔**。做帧率鲁棒性实验时，必须先检测重复帧（相邻帧差为 0）并恢复真实时间戳，否则会把数据集自带的时间伪影算进「帧率抖动」的结论里。这条反过来也可以作为一个现成的「真实非均匀间隔」测试条件。
- 常用子集：**MoCA filtered**，88 段 / 4,803 帧——去掉「主要无 locomotion」的段落，裁掉台标和黑边，低分辨率的段落放大到 720×1280，每 3 帧取 1 帧，框插值成逐帧 [A，Motion Grouping 原文]。FlowSAM（2024）沿用此子集，评测指标是检测成功率 SR（预测掩码外接框与 GT 框的 IoU 过阈值的比例）[A]。
- 适合单目标跟拍的理由 [C]：单目标 + 移动相机 + 逐帧（插值）框 + **运动类型标签**。其中 still / deformation 标签正好可以分层评估「目标停下时跟随背景滑动、等它再动」这一行为，这是其他数据集没有的。
- 不足 [C]：没有标注「静态的相似干扰物」，第 4 点（排除名单）的收益只能通过整体指标间接体现；如果要直接量化，需要自己补标少量干扰物位置，或合成干扰物。

### A.3 MoCA-Mask 细节

- 论文：Xuelian Cheng, Huan Xiong, Deng-Ping Fan, Yiran Zhong, Mehrtash Harandi, Tom Drummond, Zongyuan Ge. *Implicit Motion Handling for Video Camouflaged Object Detection*. CVPR 2022. arXiv:2203.07363。[A]
- 项目页：https://xueliancheng.github.io/SLT-Net-project ；代码：https://github.com/XuelianCheng/SLT-Net （GitHub API 无 license 字段）[A]
- 原文写明是在 MoCA 基础上「重组」得到：MoCA 有些动物并不伪装，而且只有框，所以重标了掩码 [A]。
- 用途 [C]：如果论文想同时报告「分割 / 定位精度」，这是社区标准基准（Making and Breaking of Camouflage, ICCV 2023 等都在上面报分 [A]）。对「跟拍时间占比」来说，用它的框就足够。

### A.4 CAD 细节

- 论文：Pia Bideau, Erik Learned-Miller. *It's Moving! A Probabilistic Model for Causal Motion Segmentation in Moving Camera Videos*. ECCV 2016. arXiv:1604.00136。PDF 可访问：https://people.cs.umass.edu/~elm/papers/ECCV16_motseg.pdf [A]
- 9 段视频取自 YouTube：chameleon、frog、glow-worm beetle、snail、scorpion1–4、stickinsect；附录给了原 YouTube 链接和起始时间戳 [A]。所以理论上可以按附录自行重建视频，但 GT 掩码仍需原始包 [C]。
- 原文说明「全部为移动相机视频」[A]。

### A.5 其他候选（未深入核实）

- **LaSOT / GOT-10k** 等通用 SOT 基准里有大量动物类别、移动相机、逐帧框，但不以伪装为主；我本次没有打开核实，列为可选的「非伪装对照」[C]。
- **DAVIS-2016、SegTrackv2、FBMS-59** 是运动分割的通用基准，Motion Grouping / FlowSAM 都在上面报分 [A]，但不是伪装场景。

---

## B. 相关工作

### B.1 基于运动的伪装目标发现 / 运动分割

| 标题 | 作者 | 会议 / 期刊 · 年 | 链接 | 要点 | 等级 |
|---|---|---|---|---|---|
| It's Moving! A Probabilistic Model for Causal Motion Segmentation in Moving Camera Videos | P. Bideau, E. Learned-Miller | ECCV 2016 | arXiv:1604.00136 | 用光流角度 + 幅值的似然估计相机旋转 + 平移，因果式（只用过去帧）；提出 CAD 数据集 | A |
| Betrayed by Motion: Camouflaged Object Discovery via Motion Segmentation | H. Lamdouar, C. Yang, W. Xie, A. Zisserman | ACCV 2020 | arXiv:2011.11630 | 可微配准模块（从光流对应点回归单应 H + 内点权重，相当于学出来的 RANSAC）→ 对齐帧差 → 带记忆的分割；提出 MoCA | A |
| Self-supervised Video Object Segmentation by Motion Grouping | C. Yang, H. Lamdouar, E. Lu, A. Zisserman, W. Xie | ICCV 2021（CVPR21 RVSU 最佳论文） | arXiv:2104.07658；代码 Charigyang/motiongrouping（BSD-2-Clause） | 只用光流，slot-attention 式前景 / 背景分组，自监督；定义了 MoCA filtered 子集和 SR 指标 | A |
| Implicit Motion Handling for Video Camouflaged Object Detection (SLT-Net) | X. Cheng et al. | CVPR 2022 | arXiv:2203.07363 | 隐式运动（相关性）+ 长时 transformer；提出 MoCA-Mask | A |
| Segmenting Moving Objects via an Object-Centric Layered Representation (OCLR) | J. Xie, W. Xie, A. Zisserman | NeurIPS 2022 | arXiv:2207.02206；代码 Jyxarthur/OCLR_model（MIT） | 多目标、分层、合成数据训练 | A（元数据与许可证） |
| The Making and Breaking of Camouflage | H. Lamdouar, W. Xie, A. Zisserman | ICCV 2023（据 arXiv 2309.03899；会议名来自我的记忆，未在页面上核实） | arXiv:2309.03899 | 伪装程度打分；用 RGB + RAFT 光流的 transformer 在 MoCA-Mask 上达到 SOTA | A（正文）/ 会议名 C |
| Moving Object Segmentation: All You Need Is SAM (and Flow) (FlowSAM) | J. Xie, C. Yang, W. Xie, A. Zisserman | ACCV 2024（来自我的记忆；arXiv comment 字段只写了项目页） | arXiv:2404.12389；代码 Jyxarthur/flowsam（Apache-2.0） | SAM + 光流，在 MoCA filtered 上报 SR | A（正文与许可证）/ 会议名 C |
| SAM-PM: Enhancing Video Camouflaged Object Detection using Spatio-Temporal Attention | M. N. Meeran, G. Adethya T, B. P. Mantha | arXiv 2024（据我的记忆是 CVPR 2024 Workshop，未核实） | arXiv:2406.05802 | SAM + 时空注意力，VCOD | A（元数据） |
| CamSAM2: Segment Anything Accurately in Camouflaged Videos | Y. Zhou, Y. Li, Y. Fu, L. Benini, E. Konukoglu, G. Sun | arXiv 2025 | arXiv:2503.19730 | SAM2 适配伪装视频 | A（元数据） |
| A Survey of Camouflaged Object Detection and Beyond | F. Xiao et al. | arXiv 2024 | arXiv:2408.14562 | COD / VCOD 综述 | A（元数据） |
| CAMotion: A High-Quality Benchmark for Camouflaged Moving Object Detection in the Wild | S. Yao, H. Sun, R. Yu, X. Jiang, W. Ren, X. Cao | arXiv 2026（在审） | arXiv:2604.08287 | 新基准，见 A.1 | A |
| Camouflaged Object Tracking: A Benchmark (COTD, HiPTrack-MLS) | X. Guo, P. Zhong, H. Zhang, D. Huang, H. Shao, Q. Zhao, S. Li | arXiv 2024（v4 2025-05） | arXiv:2408.13877；代码 openat25/HIPTrack-MLS（Apache-2.0） | **伪装目标的单目标跟踪**；评测 20 个跟踪器，结论是它们在伪装目标上明显退化 | A |

小结 [C]：这条线几乎全部是**离线、逐像素、依赖光流**（RAFT / PWC-Net）的分割模型，评测指标是 J / SR / Sα 等分割或定位指标；**没有一篇以「实时、轻量、检测框级别、闭环取景」为目标**。这是本工作可以占的位置。

### B.2 多目标跟踪强基线

| 标题 | 作者 | 会议 · 年 | 代码 / 许可证（GitHub API 实测） | 与本工作的关系 | 等级 |
|---|---|---|---|---|---|
| Simple Online and Realtime Tracking (SORT) | A. Bewley, Z. Ge, L. Ott, F. Ramos, B. Upcroft | ICIP 2016 | abewley/sort，**GPL-3.0** | 恒速 Kalman + 匈牙利，本系统的骨架 | A |
| Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT) | N. Wojke, A. Bewley, D. Paulus | arXiv 2017（据我的记忆收录于 ICIP 2017，未核实） | nwojke/deep_sort，**GPL-3.0** | 外观 ReID；伪装场景下外观无区分力 [C] | A（元数据） |
| ByteTrack: Multi-Object Tracking by Associating Every Detection Box | Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, X. Wang | ECCV 2022（会议名来自我的记忆） | FoundationVision/ByteTrack，**MIT** | 低分框二次关联；**`buffer_size = int(frame_rate / 30.0 * args.track_buffer)`**——丢失缓冲按帧率缩放（见 C.2） | A（源码） |
| Observation-Centric SORT (OC-SORT) | J. Cao, J. Pang, X. Weng, R. Khirodkar, K. Kitani | CVPR 2023 | noahcao/OC_SORT，**MIT** | ORU（Observation-centric Re-Update）：重新关联时用虚拟轨迹回填空档——本系统第 1 点直接借鉴 | A |
| BoT-SORT: Robust Associations Multi-Pedestrian Tracking | N. Aharon, R. Orfaig, B.-Z. Bobrovsky | arXiv 2022 | NirAharon/BoT-SORT，**MIT** | **相机运动补偿 CMC**：OpenCV 视频稳定模块的 GMC，关键点 + 稀疏光流 + `cv2.estimateAffinePartial2D(..., cv2.RANSAC)` 估计仿射，用来校正 Kalman 预测；也支持 ORB / SIFT / ECC | A（正文 + `gmc.py`） |
| StrongSORT: Make DeepSORT Great Again | Y. Du et al. | IEEE TMM 2023 | dyhBUPT/StrongSORT，**GPL-3.0** | ECC 相机补偿 + 外观 | A（元数据） |
| Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking | M. Yang et al. | AAAI 2024 | 未查许可证 | 弱线索（置信度、高度）辅助关联 | A（元数据） |
| BoxMOT（工具库，集成多个跟踪器） | M. Broström | — | mikel-brostrom/boxmot，**AGPL-3.0** | 一键跑 ByteTrack / OC-SORT / BoT-SORT 等；**AGPL 有传染性**，只适合评测用，不宜并入要分发的代码 [C] | A（许可证） |

### B.3 相机自运动补偿 / 移动相机运动检测

| 标题 | 作者 | 出处 · 年 | 要点 | 等级 |
|---|---|---|---|---|
| Motion Tracking with an Active Camera | D. Murray, A. Basu | IEEE TPAMI 16(5), 1994 | 云台相机上做实时运动检测：用已知云台角把图像映射对齐，再套用静止相机的运动检测，形态学滤波吸收补偿误差。**「云台 + 背景补偿 + 运动目标」的经典起点** | B |
| A Unified Approach to Moving Object Detection in 2D and 3D Scenes | M. Irani, P. Anandan | IEEE TPAMI 20(6):577–589, 1998 | 分层：单个 2D 参数变换 → 多层 → 平面 + 视差（plane+parallax）。可以给本系统「只建模纯平移」的局限定位 | B |
| Detection of Moving Objects with Non-stationary Cameras in 5.8ms | K. M. Yi, K. Yun, S. W. Kim, H. J. Chang, J. Y. Choi | CVPR Workshops 2013 | 双模单高斯背景模型 + 混合邻域模型来补偿相机运动；PC 上 5.8ms，手机实时。**实时移动相机运动检测的强参照** | B |
| Moving Objects Detection with a Moving Camera: A Comprehensive Review | M.-N. Chapel, T. Bouwmans | Computer Science Review 38:100310, 2020（arXiv:2001.05238） | 八类方法：全景背景减除、双相机、运动补偿、子空间分割、运动分割、平面 + 视差、多平面、分块 | A（摘要）/ 期刊卷号 B |
| BoT-SORT 的 CMC | 见 B.2 | 2022 | 稀疏光流 + RANSAC 仿射；检测框区域可排除在外 | A |
| Detection and Tracking of Large Number of Targets in Wide Area Surveillance | V. Reilly, H. Idrees, M. Shah | ECCV 2010 | 广域航拍：帧配准后做中值背景建模，用背景梯度剔除视差和配准误差造成的假检，**低帧率**输入 | B |

与本系统第 2 点（轨迹位移取中位数）的关系 [C]：中位数相当于 0 维（纯平移）的鲁棒估计，崩溃点 50%，作用与 RANSAC 一致；单应 / 仿射 + RANSAC 是「同一思路换更高维的模型」。论文里应当把它写成「检测框级别的轻量 CMC，模型阶数更低」，而不是新方法；实测「比相位相关更稳」可以作为一条工程观察报告。建议加一组对照：用 BoT-SORT 的 `gmc.py`（MIT）替换中位数估计。

### B.4 主动跟踪 / 自动取景（虚拟云台）

| 标题 | 作者 | 出处 · 年 | 要点 | 等级 |
|---|---|---|---|---|
| End-to-end Active Object Tracking via Reinforcement Learning | W. Luo, P. Sun, F. Zhong, W. Liu, T. Zhang, Y. Wang | ICML 2018 | 端到端 RL，直接从画面输出相机动作 | A（元数据） |
| Towards Distraction-Robust Active Visual Tracking | F. Zhong, P. Sun, W. Luo, T. Yan, Y. Wang | ICML 2021 | **主动跟踪 + 干扰物**：目标与干扰物组成合作-竞争多智能体博弈来训练跟踪器；干扰物的定义是外观相近 / 遮挡 | A（摘要） |
| Auto-Directed Video Stabilization with Robust L1 Optimal Camera Paths | M. Grundmann, V. Kwatra, I. Essa | CVPR 2011 | **在原视频里移动裁剪窗口**（虚拟相机路径），用线性规划同时最小化路径的一、二、三阶导，路径由常数 / 线性 / 抛物线段组成，并带约束（比如让显著区域留在窗口内）。离线、非因果。与第 6 点（限加速度、minimum-jerk 风格）在目标上很接近 | B |
| AutoFlip: An Open Source Framework for Intelligent Video Reframing | N. Frey, Z. Sun（Google） | Google AI Blog，2020-02 | 基于 MediaPipe 的显著性检测 + 自动裁剪跟拍；MediaPipe 仓库现为 google-ai-edge/mediapipe（Apache-2.0）[A] | B（博客）/ A（许可证） |
| The Coordination of Arm Movements: An Experimentally Confirmed Mathematical Model | T. Flash, N. Hogan | J. Neuroscience 5(7):1688–1703, 1985 | minimum-jerk 轨迹的出处 | B |

### B.5 帧率鲁棒性 / 时间归一化

| 标题 | 作者 | 出处 · 年 | 要点 | 等级 |
|---|---|---|---|---|
| Need for Speed: A Benchmark for Higher Frame Rate Object Tracking (NfS) | H. Kiani Galoogahi, A. Fagg, C. Huang, D. Ramanan, S. Lucey | ICCV 2017（会议名来自我的记忆） | 240fps 基准；结论是「高帧率下简单的相关滤波跟踪器反而胜过深度方法」 | A（摘要） |
| Towards Frame Rate Agnostic Multi-Object Tracking (FAPS) | W. Feng, L. Bai, Y. Yao, F. Yu, W. Ouyang | IJCV（DOI 10.1007/s11263-023-01943-2，arXiv 2209.11404） | 「所有近期 SOTA 跟踪器在输入帧率变化时精度都会大幅下降」；提出帧率无关关联模块 FAAM，**把帧率作为编码输入**，并配合周期训练 | A（摘要）/ 期刊 B |
| Collaborative Tracking Learning for Frame-Rate-Insensitive Multi-Object Tracking (ColTrack) | Y. Liu, J. Wu, Y. Fu | ICCV 2023 | 基于 query 的端到端 MOT，面向低帧率 | A |
| Real-Time Multiple Object Tracking – A Study on the Importance of Speed | S. Murray | 硕士论文，arXiv:1709.03572 | C++SORT；在 Okutama-Action 上按不同帧率采样，真正实时运行时多数指标下降约 50%，个别达 90% | A（摘要） |
| ByteTrack 源码 | 见 B.2 | — | `buffer_size` 按 `frame_rate/30` 缩放——**一个参数的时间归一化** | A |

---

## C. 新颖性评估

### C.1 第 4 点：排除名单挂在「背景坐标系的位置」上，而不是挂在轨迹 ID 上

**最接近的已有工作（按接近程度排序）：**

1. **雷达杂波图（clutter map）CFAR** [B]：Nitzberg, *Clutter map CFAR analysis*, IEEE Trans. AES, AES-22, 419–421, 1986；概念在 Skolnik《Radar Handbook》(1978) 里就已成熟。做法是：把空间划成**地理固定的分辨单元**，每个单元累积长期回波强度；固定散射体会「长进」杂波图而被抑制，运动目标只在短暂占据某个单元、且超过该单元存储水平时才被检出。
   - 共同点：**按空间位置**记忆「这里总有东西但它不动」；目标经过时只受局部抑制，离开后不受影响。和第 4 点的核心思想**高度同构**。
   - 差异 [C]：杂波图作用在**信号 / 检测层**（逐单元幅值门限），而且雷达通常知道自身位姿；第 4 点作用在**目标选择层**（决定跟哪条轨迹），坐标系来自视觉估计的背景平移（会漂移）；判据是「重心散布小」而不是「平均回波强度」；名单是永久的，而不是指数遗忘。
2. **MegaDetector 的重复检测剔除（Repeat Detection Elimination, RDE）** [A，文档页]：在同一台相机（同一文件夹）的图像序列里，按框的 IoU 把「同一位置反复出现的检测」聚成组，次数超过 `occurrenceThreshold` 就标为疑似假阳性（树枝、石头等），再人工复核。
   - 共同点：**按位置而不是按身份**标记静态干扰物，场景也是野生动物。
   - 差异：RDE 用于**固定相机**、**离线**、在**图像坐标**里、而且需要人工复核；第 4 点用于移动相机、在线、在背景坐标里，并且能在目标离开后自动恢复资格。
3. **SOT 里的干扰物感知**——DaSiamRPN（Zhu et al., ECCV 2018, arXiv:1808.06048）、KeepTrack（Mayer et al., ICCV 2021, arXiv:2103.16556，「keep track of what not to track」）[A，元数据]：显式记住干扰物，但干扰物按**外观 / 候选身份**来关联和传播，而不是按背景坐标里的位置。伪装场景下外观正是失效的那个线索 [C]。
4. **全景 / 运动补偿背景减除**（Chapel & Bouwmans 综述里的一类；Yi et al. 2013；Murray & Basu 1994）[B]：背景模型锚定在补偿后的坐标系里——但它是**像素级外观背景模型**，不是「检测级干扰物名单」。
5. **主动跟踪中的干扰物**（Zhong et al., ICML 2021）[A]：处理的是会动、外观相似的干扰物，用学习方法解决；没有位置锚定的记忆。

**结论 [C]**：「按位置抑制持续静止的东西」这个思想**不新**（雷达杂波图已有几十年，RDE 也是），论文里**必须**引用并正面承认。在我检索过的视觉 MOT / VCOD / 主动跟踪文献（B 节所列，以及上述搜索）里，**没找到**把它做成以下组合的工作：「移动相机下的检测级跟踪 + 背景坐标系锚定 + 对轨迹 ID 切换免疫 + 目标停留时暂时压制、离开后自动恢复」。合理的定位是：**把雷达杂波图的思想迁移到「移动相机、伪装、检测框级」的目标选择问题上**，贡献在于这一迁移以及「ID 无关」带来的鲁棒性（可以用实验证明：在 ID 频繁切换的条件下，ID 锚定的名单会失效，而位置锚定的不会）。这比声称「首创」安全得多。

需要在论文里主动讨论的风险 [C]：
- 只建模平移，背景坐标会随时间漂移，名单位置会跟着错位。需要报告漂移量，或者给名单位置加容差。
- 「永久」标记：如果目标本身在开场长时间静止，它会被写进名单。目前靠「离开该点即恢复」兜底，应当专门构造这种情形做测试。
- 视差：近处和远处的静物在纯平移模型下的背景坐标不一致。

### C.2 第 5 点：所有参数按物理时间定义，以抗帧率变化与帧间隔抖动

**已有的相关工作：**

1. **把 dt 写进 Kalman 转移矩阵**是教科书做法（Bar-Shalom 等跟踪教材的标准内容）[C，未逐本核实]。所以「速度、加速度按秒定义」本身不新。
2. **ByteTrack 按帧率缩放丢失缓冲**（`frame_rate / 30.0 * track_buffer`）[A]：一个参数按**标称帧率**（不是逐帧的真实时间戳）做了归一化，其他参数（比如 IoU 门限）仍然按帧定义。
3. **FAPS（IJCV）与 ColTrack（ICCV 2023）**：明确研究「帧率变化导致 MOT 性能骤降」[A]，但走的是**学习路线**——FAPS 把帧率编码成网络输入，ColTrack 用端到端 query 模型。
4. **NfS（2017）、Murray（2017）**：实证了帧率对跟踪性能的影响 [A]。
5. 广域航拍跟踪（Reilly et al. 2010）把低帧率当作挑战之一 [B]。

**我没找到的（在 arXiv 搜索、B.5 所列论文、以及上面的搜索结果里）**：
- 针对**非均匀帧间隔（抖动）**本身的系统性评测——也就是「平均帧率相同、但间隔不均匀」与「均匀间隔」的对照。本系统「非均匀 20Hz 比均匀 15Hz 还差」正是这类观察。
- 在**无学习**的跟踪-选择-控制流水线里，**所有**参数（包括时间窗、置信闸、运动门限、控制器加速度上限）统一用秒、px/s、px/s² 定义并按真实时间戳切窗口，同时报告各参数在不同帧率下的不变性。

**结论 [C]**：「时间归一化」作为原则是常识级别的，单独作为贡献会被审稿人认为不新；但下面两点有价值：
- (a) 一个**可复现的帧率 / 帧间隔抖动评测协议**：均匀降采样 30→15→8 Hz，加上「同平均帧率下的间隔抖动」，在 MoCA 上做，并且先处理 MoCA 自带的重复帧（见 A.2）；
- (b) 定量证明「按帧定义」与「按时间定义」在该协议下的差距（消融：同一系统把参数换回按帧定义）。

也就是说，**贡献应当写成「评测协议 + 实证发现」**，而不是「我们提出用秒做单位」。

另外，你们实测的 30→15→8Hz：93%→85%→33%，说明即使按时间定义，低帧率下仍会崩溃 [C]。这是一个诚实的「失效边界」结果，应当报告，并分析原因（例如每个时间窗里的观测数少于某个门限、关联门限被单帧位移撑破等）。

### C.3 建议的贡献点表述（1~3 条）

1. **（系统 + 问题设定）** 一个实时、检测框级别、无需光流的「运动显著目标」虚拟云台跟拍流水线，专门面向「移动相机 + 伪装目标 + 静态相似干扰物」。在 MoCA 上，以「目标落在取景窗内的时间占比」为指标，并与基于光流的离线运动分割方法（FlowSAM / Motion Grouping）和 MOT 基线比较精度-速度权衡。
   ——定位：实时性 + 闭环取景，这是 B.1 那条线普遍没覆盖的。
2. **（方法）** 背景坐标系锚定、与轨迹 ID 无关的干扰点抑制：把雷达杂波图思想迁移到检测级目标选择上。通过消融（ID 锚定名单 vs 位置锚定名单 vs 不用名单），并人为注入 ID 切换来展示鲁棒性。
   ——措辞上必须引用 clutter map 和 MegaDetector RDE，强调「迁移 + 在移动相机 / ID 不稳条件下的有效性」。
3. **（评测 / 实证）** 一个帧率与帧间隔抖动的评测协议，以及「按时间定义参数」与「按帧定义参数」的消融。核心发现是：帧间隔的不均匀性本身就有独立代价（非均匀 20Hz 差于均匀 15Hz），并指出 MoCA 的 24fps 重采样带来了重复帧这一数据伪影。

第 6 点（限加速度的平滑趋近）和第 7 点（时间占比指标）建议作为设计细节和评测选择来写，不单列为贡献：前者有 Grundmann 2011 和 minimum-jerk 文献在先，后者与 SOT 的 success / precision 指标接近 [C]。

---

## D. 基线建议（MoCA 为主，一个月内可跑通）

以下判断都是 [C]，许可证为 [A]。

| # | 基线 | 为什么选它 | 可复现性 | 怎么接入「跟拍时间占比」 |
|---|---|---|---|---|
| 1 | **OC-SORT + 简单选择规则**（同一个 YOLO 检测器；选择规则为「最大置信度」或「图像坐标位移最大」） | 证明「运动显著性选择 + 背景坐标 + 排除名单」带来的增益，而不是跟踪器本身的增益 | MIT，纯 CPU 可跑 | 每帧取被选轨迹的框中心，套用同一个取景控制器 |
| 2 | **BoT-SORT（开 CMC）+ 你们的选择规则**，或者只把你们的中位数估计换成 BoT-SORT 的 `gmc.py` | 直接对照第 2 点（中位数平移 vs 稀疏光流 + RANSAC 仿射） | MIT | 同上 |
| 3 | **FlowSAM 或 Motion Grouping**（基于光流的运动分割，在 MoCA filtered 上有公开 SR） | 代表「伪装 + 运动」这条线的 SOTA / 经典；给出精度上限和速度对照（需要 RAFT 光流，非实时） | FlowSAM：Apache-2.0；Motion Grouping：BSD-2-Clause | 掩码 → 外接框中心 → 同一控制器；同时报告 SR，方便和原文数字对齐 |
| 4（可选） | **一个通用 SOT 跟踪器**（例如 COTD 的 HiPTrack-MLS，Apache-2.0；或 pytracking 里的 KeepTrack） | 「给第一帧框」的上限式对照，说明不需要初始化框的价值 | HiPTrack-MLS 需要 COTD 训练权重；KeepTrack 在 pytracking 里（许可证未核实） | 直接输出框 |

建议的主表 [C]：MoCA（filtered 88 段 + 全量 141 段各报一次），指标为「时间占比 @ 半径 r」（r 取若干值）+ SR@0.5 + FPS；按 locomotion / deformation / still 分层。帧率实验只在你们的方法和基线 1、2 上做（3 是离线方法，没有意义）。

---

## E. 投稿去向（截止日期一律需查）

- **CVPR CV4Animals Workshop**（Computer Vision for Animal Behavior Tracking and Modeling）[A，CVPR 页面显示 2023、2024、2025 有举办，2026 有第 6 届页面]：有 4 页短文轨道，题材最契合。截稿日：需查。
- **WACV**（偏应用与系统，BIRDSAI 发在这里）[B]：截稿日需查。
- **BMVC / ACCV / ICPR**：中等规模的视觉会议；MoCA 本身就发在 ACCV [A]。截稿日需查。
- 机器人方向：**IROS / ICRA 的 workshop**（主动感知、野外机器人）[C]：需查。
- arXiv 分类建议：主类 cs.CV，交叉 cs.RO [C]。

---

## 附：本次未能核实、需要后续补的事项

1. CAD 的可用下载镜像（原页面 404，UMass 地址连接失败）。
2. MoCA-Mask 的数据许可证（项目页和 README 都没写）；MoCA 标注文件的具体格式。
3. COTD 的数据许可证、相机是否运动、能否从非百度渠道下载。
4. MSVCOD 是否已公开发布。
5. 标注为「会议名来自我的记忆」的几条（ByteTrack ECCV 2022、NfS ICCV 2017、FlowSAM ACCV 2024、Making and Breaking ICCV 2023、DeepSORT ICIP 2017）——投稿前应到各会议论文集核对。
6. 雷达杂波图文献只核实到书目信息（[B]），引用前应读 Nitzberg 1986 原文或 Skolnik 相关章节。
