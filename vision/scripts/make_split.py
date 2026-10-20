"""把 MoCA 141 段**按视频**划成开发集 / 测试集，写入 `vision/splits/moca_split.json`。

    uv run python scripts/make_split.py [--seed 20261020]

# 规则（确定性：同一种子、同一份标注 ⇒ 同一份划分）

```text
① 物种 = 视频名去掉末尾的 `_数字`（black_cat_1 → black_cat），共 67 种
② 同物种 ≥ 2 段：种内随机打乱，前 ⌊n/2⌋ 段进 dev、接着 ⌊n/2⌋ 段进 test，奇数余下的 1 段进「余池」
③ 余池（单段物种 + 奇数余段）：按该段的**主运动标签**（有真值帧上 locomotion / subtle_motion / still
   的多数；无标签记 unlabeled）再分层，层内随机打乱后逐段放进当前段数较少的一侧（相等时随机）
```

⇒ 物种层面尽量两边都有，单段物种按运动类型均衡；两边段数差 ≤ 1。

⚠️ 划分只依赖视频名与标注里的运动标签，不依赖任何方法的结果。此后参数与修复只在 dev 上选，
test 只在最后各跑一次。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from gimbal_vision.datasets import moca

OUT = Path(__file__).resolve().parents[1] / "splits" / "moca_split.json"


def species(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def main_motion(seq: moca.Sequence) -> str:
    ks = sorted(seq.keys)
    if not ks:
        return "unlabeled"
    c = Counter(seq.motion(k) for k in range(ks[0], ks[-1] + 1))
    c.pop(None, None)
    return moca.MOTION[c.most_common(1)[0][0]] if c else "unlabeled"


def make(seqs: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    by_sp = defaultdict(list)
    for n in sorted(seqs):
        by_sp[species(n)].append(n)
    dev, test, pool = [], [], []
    for sp in sorted(by_sp):
        v = list(by_sp[sp])
        rng.shuffle(v)
        h = len(v) // 2
        dev += v[:h]
        test += v[h : 2 * h]
        pool += v[2 * h :]
    by_m = defaultdict(list)
    for n in pool:
        by_m[main_motion(seqs[n])].append(n)
    for m in sorted(by_m):
        v = list(by_m[m])
        rng.shuffle(v)
        for n in v:
            if len(dev) != len(test):
                (dev if len(dev) < len(test) else test).append(n)
            else:
                (dev if rng.random() < 0.5 else test).append(n)
    motion = {n: main_motion(seqs[n]) for n in seqs}
    return {
        "seed": seed,
        "rule": "按物种分层（种内对半，奇数余段与单段物种进余池）；余池按主运动标签分层，逐段放进较少一侧",
        "script": "vision/scripts/make_split.py",
        "dev": sorted(dev),
        "test": sorted(test),
        "counts": {
            "dev": len(dev),
            "test": len(test),
            "dev_motion": dict(Counter(motion[n] for n in dev)),
            "test_motion": dict(Counter(motion[n] for n in test)),
            "species_in_both": sum(
                1 for sp in by_sp if any(n in dev for n in by_sp[sp]) and any(n in test for n in by_sp[sp])
            ),
            "species": len(by_sp),
        },
    }


def load_split(which: str) -> list[str]:
    """读已提交的划分：`which` ∈ {"dev", "test", "all"}。"""
    d = json.loads(OUT.read_text(encoding="utf-8"))
    return sorted(d["dev"] + d["test"]) if which == "all" else d[which]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261020)
    a = ap.parse_args()
    split = make(moca.load(), a.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(split["counts"], ensure_ascii=False))
    print(f"写入 {OUT}")


if __name__ == "__main__":
    main()
