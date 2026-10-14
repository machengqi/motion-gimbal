"""**MoCA**（Moving Camouflaged Animals）加载器。

出处：Lamdouar, Yang, Xie, Zisserman. *Betrayed by Motion: Camouflaged Object
Discovery via Motion Segmentation*. ACCV 2020。许可证 CC BY 4.0。
下载：https://thor.robots.ox.ac.uk/datasets/MoCA/MoCA.zip → 解压到 `data/moca/`。

# 标注

`Annotations/annotations.csv`（VGG Image Annotator 导出）：每 5 帧一个框
`[2, x, y, w, h]`（原图像素），外加运动标签 `0 locomotion / 1 subtle_motion / 2 still`。
其余帧按两端标注线性插值 —— 与原数据集及后续工作的做法一致。

# ⚠️ 重复帧

原作者把所有视频统一重采样到 24fps，源帧率更低的视频因此夹着复制帧。它们不是
真实的采样时刻 ⇒ 做帧率相关的实验前必须识别出来：[`duplicates`] 给出每帧是否为复制帧。

判据见 [`DUP_FRAC`]。⚠️ 不能用「平均差异很小」：镜头静止、目标不动时相邻帧的
平均差异也很低（实测低至 0.08 灰度级），那是真实采样，不能删。
"""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

FPS = 24.0
"""MoCA 统一的标称帧率。"""

DUP_FRAC = 0.005
"""复制帧判据：与上一帧相比，变化超过 2 个灰度级的像素占比低于此值。

2026-10-14 在 10 段视频上看这个占比的分布：完全相同的 199 帧（占比 0）；正常帧
绝大多数 > 1%；两者之间有一条稀疏的模糊带（1e-5 ~ 5e-3，约 30 帧），其中有按
6~7 帧周期出现的复制帧（复制后重新编码，只有极少数像素不同），也有 2 帧分不清
是复制帧还是真实的静止画面。模糊带不到全部帧的 0.1%，判错对结论的影响可以忽略。
⚠️ 「完全相同」作判据会漏掉重新编码过的复制帧。
"""

MOTION = {0: "locomotion", 1: "subtle_motion", 2: "still"}


def default_root() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "moca" / "MoCA"


@dataclass
class Sequence:
    name: str
    frames: list[Path]
    keys: dict[int, tuple[np.ndarray, int]] = field(default_factory=dict)
    """标注帧号 → (框 `[x, y, w, h]`, 运动标签)。"""

    def box(self, k: int) -> np.ndarray | None:
        """第 k 帧的框（插值）；不在首末标注之间返回 `None`。"""
        ks = sorted(self.keys)
        if not ks or k < ks[0] or k > ks[-1]:
            return None
        if k in self.keys:
            return self.keys[k][0]
        i = int(np.searchsorted(ks, k))
        k0, k1 = ks[i - 1], ks[i]
        a = (k - k0) / (k1 - k0)
        return (1 - a) * self.keys[k0][0] + a * self.keys[k1][0]

    def motion(self, k: int) -> int | None:
        """第 k 帧的运动标签：沿用不晚于 k 的最近一个**明确**标签。

        ⚠️ 标签是稀疏的：7617 个标注框里只有 1855 个带标签，其余为 `{}`（记作 -1）。
        「标签变化时才标、之后沿用」是根据数据形态做的**推断**，数据集文档没有写明；
        按运动类型分组的结果都依赖这个假设，报告时要注明。
        """
        ks = [x for x in sorted(self.keys) if x <= k and self.keys[x][1] >= 0]
        return self.keys[ks[-1]][1] if ks else None


def load(root: Path | None = None) -> dict[str, Sequence]:
    root = root or default_root()
    seqs: dict[str, Sequence] = {}
    for d in sorted((root / "JPEGImages").iterdir()):
        if d.is_dir():
            seqs[d.name] = Sequence(d.name, sorted(d.glob("*.jpg")))
    with open(root / "Annotations" / "annotations.csv", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            path, spatial, meta = row[1], row[4], row[5]
            name, fname = path.strip("/").split("/")
            shape = ast.literal_eval(spatial)
            if shape[0] != 2:
                continue
            motion = int(json.loads(meta).get("1", -1)) if meta else -1
            seqs[name].keys[int(Path(fname).stem)] = (np.array(shape[1:5], dtype=float), motion)
    return seqs


def duplicates(seq: Sequence, cache_dir: Path | None = None) -> np.ndarray:
    """每帧是否为复制帧（判据见 [`DUP_FRAC`]）。结果缓存到 `cache_dir`。"""
    cache = (cache_dir or default_root().parent / "cache") / f"dup{DUP_FRAC:g}_{seq.name}.npy"
    if cache.exists():
        return np.load(cache)
    out = np.zeros(len(seq.frames), dtype=bool)
    prev = None
    for k, p in enumerate(seq.frames):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.int16)
        if prev is not None and (np.abs(g - prev) > 2).mean() < DUP_FRAC:
            out[k] = True
        prev = g
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out
