"""帧率与帧间隔抖动实验（论文 6.5 节）：把 MoCA 降采样 / 注入抖动后，复用 E2 / E2b 协议比较两种门控。

    uv run python scripts/eval_fps.py --split dev [--conditions r24,r12,r8,r6,j1,j2,j3] [--workers 4]

输出 `runs/fps-<split>[-tag]-<时间>.json`（各条件 × 协议 × 取景窗 的汇总、逐段 time-in-frame、跟踪诊断、
时钟统计）并打印汇总表。⚠️ 划分同 E2（`vision/splits/moca_split.json`）：设定只在 dev 上检查，test 只跑一次。

# 帧的选取（全部在**去复制帧之前**的原始 24 Hz 帧号 `k` 上定义，时间戳 = `k / 24` 秒）

按一串间隔 `g_1, g_2, ...`（单位：原始帧）生成目标帧号 `k_0 = 0, k_j = k_{j-1} + g_j`。目标帧是复制帧时，
在 `±⌊g_j / 2⌋` 帧内取最近的非复制帧（等距时取较早的；必须晚于上一个保留帧），找不到则这个采样点丢失。
保留帧沿用真实时间戳，真值按保留帧评测（真值本身是每 5 帧标注、线性插值，见 `datasets.moca`）。

| 条件 | 间隔 `g` | 名义帧率 |
|---|---|---|
| `r24` | 恒为 1（⌊1/2⌋ = 0 ⇒ 复制帧直接丢掉 = 原 E2 / E2b 的帧序列） | 24 Hz |
| `r12` / `r8` / `r6` | 恒为 2 / 3 / 4 | 12 / 8 / 6 Hz |
| `j1`（轻） | {1, 2, 3}，概率 {0.25, 0.5, 0.25} | 平均 12 Hz |
| `j2`（中） | 几何分布 `P(g) = 0.5^g`（= 每个原始帧独立以 0.5 的概率保留，随机丢帧） | 平均 12 Hz |
| `j3`（重） | {1, 5}，概率 {0.75, 0.25}（多数帧按 24 Hz 到达，四分之一的间隔是 5 帧的卡顿） | 平均 12 Hz |

抖动档的平均间隔都是 2 帧 ⇒ 与 `r12` 同一平均帧率，只有间隔的分布不同。随机数种子
`[SEED, crc32(视频名), 条件序号]`，固定、与方法无关。抖动度量（逐段算，报中位数）：

- `jitter` = `Clock.jitter()` = 帧间隔的 90% 分位 ÷ 中位（1.0 = 均匀）；
- `cv` = 帧间隔标准差 ÷ 均值；`hz` = `Clock.hz()`（中位间隔的倒数）；`mean_hz` = (n−1) ÷ 时长。

⚠️ `j2` / `j3` 的中位间隔是 1 帧 ⇒ `Clock.hz()` 报 24 Hz，而平均帧率是 12 Hz：所有用 `Clock.frames_in`
（按中位间隔把秒换成帧数）的派生量会按 24 Hz 换算。这是被测系统本身的行为，不在实验里修正。

# 方法与协议

与 `eval_e2.py` 相同（`prepare` + `evaluate`，同一套候选、选择器、控制器、取景窗与指标）。默认方法：
`center`、`naive`、`closed_naive`、`core:fixed` / `core:speed`（开环，修复后系统 + 不确定性 / 速度门控）、
`closed:fixed` / `closed:speed`（闭环）。默认两种协议（`e2`、`e2b`）× 两种取景窗（0.5、0.33）。
E2b 的画布偏移在完整 24 Hz 序列上算（`runs/diag_cache/stab-*.pkl`），再取保留帧 —— 评测的「尺子」不随帧率变。

# 跟踪诊断（开环全幅候选，与协议、取景窗无关）

对两种门控各跑一遍 `mot.Tracker`（与 `core:fixed` / `core:speed` 同一配置），在「**机会**」上记录：第 k 帧
有真值、至少一个候选中心落在真值框内，且某条存活轨迹的上一次观测落在（它那一帧的）真值框内。对每个机会：

- 门宽：uncertainty 取 `min(√(GATE_CHI2 · tr(S)/2), GATE_PPS·dt)`；speed 取 `GATE_PPS·dt`（px，640 宽）；
- NIS：框内候选相对该轨迹预测的最小马氏距离平方（只对 uncertainty 有意义，模型自洽时 ~ χ²(2)）；
- 结果：`cont` 这条轨迹本帧配上了框内候选；`steal` 配上了**扩大一倍的真值框之外**的候选（目标轨迹被别的
  运动接走 = 误关联）；`ring` 配上了框外但在扩大框内的候选；`miss` 本帧没有配上任何候选。
  `rejected` = 最近的框内候选没通过门（距离上限或马氏距离）。
- 误关联另在「**目标轨迹帧**」上统计（第 k 帧有真值、轨迹上一次观测在真值框内，不要求本帧有框内候选）：
  `steal_any` = 其中配上扩大框之外候选的比例，`steal_per_min` = 次数 / 分钟。

- 干扰候选：每个有真值的帧上，真值中心到扩大框之外最近候选的距离；报有干扰候选的帧占比、距离四分位、
  落在该帧速度门宽 `GATE_PPS·dt` 之内的帧占比。

另记：轨迹出生率（新 ID / 秒）、目标身份切换（有新观测落在真值框内的轨迹集合里不再包含上一个目标轨迹的次数
/ 分钟）、选择器输出的 tid 切换次数 / 分钟与有新证据（track / jump）的帧占比。
"""

from __future__ import annotations

import os

# 多进程 × 多线程会过度订阅 CPU：每个进程单线程（须在 numpy / cv2 导入前设定，子进程继承环境变量）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import json
import pickle
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_e2 import RUNS, VARIANTS, evaluate, prepare, summarize  # noqa: E402
from make_split import load_split  # noqa: E402

from gimbal_vision.core import mot, timing  # noqa: E402
from gimbal_vision.datasets import moca  # noqa: E402

SEED = 20261026
CONDITIONS = {
    "r24": {"kind": "uniform", "step": 1},
    "r12": {"kind": "uniform", "step": 2},
    "r8": {"kind": "uniform", "step": 3},
    "r6": {"kind": "uniform", "step": 4},
    "j1": {"kind": "choice", "gaps": [1, 2, 3], "p": [0.25, 0.5, 0.25]},
    "j2": {"kind": "geometric", "p": 0.5},
    "j3": {"kind": "choice", "gaps": [1, 5], "p": [0.75, 0.25]},
}
DEFAULT_METHODS = "center,naive,closed_naive,core:fixed,core:speed,closed:fixed,closed:speed"
PAIRS = [("core:fixed", "core:speed"), ("closed:fixed", "closed:speed")]
"""逐段胜负：(不确定性门控, 速度门控)。"""


def low_priority():
    """当前进程单线程 + 低优先级（Windows：BELOW_NORMAL_PRIORITY_CLASS；其他系统 nice 10）。不影响结果数值。"""
    cv2.setNumThreads(1)
    try:
        if sys.platform == "win32":
            import ctypes  # noqa: PLC0415

            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)
        else:
            os.nice(10)
    except Exception:  # noqa: BLE001 — 设不上优先级不影响实验本身
        pass


# ── 帧的选取 ──────────────────────────────────────────────────────
def gap_source(cond: dict, rng: np.random.Generator):
    kind = cond["kind"]
    if kind == "uniform":
        return lambda: cond["step"]
    if kind == "choice":
        gaps, p = np.asarray(cond["gaps"]), np.asarray(cond["p"], float)
        return lambda: int(rng.choice(gaps, p=p))
    if kind == "geometric":
        return lambda: int(rng.geometric(cond["p"]))
    raise ValueError(kind)


def pick_frames(dup: np.ndarray, gap) -> list[int]:
    """按间隔源 `gap()` 在原始帧号上取帧（规则见模块文档）。返回保留的原始帧号（升序、全是非复制帧）。"""
    n, out, last, t, r = len(dup), [], -1, 0, 0
    while t < n:
        for d in sorted(range(-r, r + 1), key=lambda d: (abs(d), d)):
            c = t + d
            if 0 <= c < n and c > last and not dup[c]:
                out.append(c)
                last = c
                break
        g = gap()
        t, r = t + g, g // 2
    return out


def select(name: str, dup: np.ndarray, cond_name: str) -> list[int]:
    ci = list(CONDITIONS).index(cond_name)
    rng = np.random.default_rng([SEED, zlib.crc32(name.encode()), ci])
    return pick_frames(dup, gap_source(CONDITIONS[cond_name], rng))


def clock_stats(clk: timing.Clock) -> dict:
    d = np.diff(clk.t)
    return {
        "n": int(clk.n),
        "hz": float(clk.hz()),
        "mean_hz": float((clk.n - 1) / (clk.t[-1] - clk.t[0])) if clk.n > 1 else None,
        "jitter": float(clk.jitter()),
        "cv": float(d.std() / d.mean()) if len(d) else None,
        "ok": bool(clk.ok()[0]),
    }


# ── 跟踪诊断 ──────────────────────────────────────────────────────
def in_box(p, g, grow=0.0):
    if g is None:
        return False
    x, y, w, h = g
    return x - grow * w <= p[0] <= x + w + grow * w and y - grow * h <= p[1] <= y + h + grow * h


def track_diag(ctx: dict, gate: str) -> dict:
    """见模块文档「跟踪诊断」。同 `core.run` 的跟踪器配置（`meas_var=None` = 标定值）。"""
    clk, gts, obs, shifts = ctx["clk"], ctx["gts"], ctx["obs"], ctx["shifts"]
    W, H = ctx["W"], ctx["H"]
    tk = mot.Tracker((0, 0, W, H), clk, gate=gate, meas_var=VARIANTS["fixed" if gate == "uncertainty" else "speed"]["meas_var"])
    opp = []  # (r_eff, r_phys, nis, rejected, outcome, 有框内候选)
    cur_target, switches = None, 0
    for k, dets in enumerate(obs):
        g = gts[k]
        ing = [d for d in dets if in_box(d, g)]
        watch = {}
        if g is not None:
            dt = clk.dt(k)
            r_phys = clk.speed_raw(timing.GATE_PPS, k)
            for t in tk.tracks:
                if t.last_k is None or not in_box(t.last_obs, gts[t.last_k]):
                    continue
                p = copy.deepcopy(t)
                p.predict(dt)
                if gate == "uncertainty":
                    S = p.H @ p.P @ p.H.T + p.R
                    r_eff = min(float(np.sqrt(timing.GATE_CHI2 * np.trace(S) / 2)), r_phys)
                else:
                    r_eff = r_phys
                nis_min, rej = float("nan"), None
                if ing:
                    tx, ty = p.pos
                    dist = np.array([np.hypot(tx - d[0], ty - d[1]) for d in ing])
                    nis = np.array([p.mahal2(d[0], d[1]) for d in ing])
                    ok = (dist < r_phys) & (nis <= timing.GATE_CHI2) if gate == "uncertainty" else dist < r_phys
                    nis_min, rej = float(nis.min()), bool(not ok.any())
                watch[t.id] = (r_eff, r_phys, nis_min, rej)
        tk.step(k, dets, shifts[k])
        alive = {t.id: t for t in tk.tracks}
        for tid, w in watch.items():
            t = alive.get(tid)
            if t is None or t.last_k != k:
                out = "miss"
            elif in_box(t.last_obs, g):
                out = "cont"
            elif in_box(t.last_obs, g, 0.5):
                out = "ring"
            else:
                out = "steal"
            opp.append((*w, out, bool(ing)))
        # 目标身份切换
        on = [t for t in tk.tracks if t.last_k == k and in_box(t.last_obs, g)]
        if on:
            ids = {t.id for t in on}
            if cur_target is not None and cur_target not in ids:
                switches += 1
            if cur_target not in ids:
                cur_target = max(on, key=lambda t: t.hits).id
    dur = float(clk.t[-1] - clk.t[0]) if clk.n > 1 else 0.0
    return {"opp": opp, "births": int(tk.next_id), "target_switches": switches, "duration": dur}


def distractors(ctx: dict) -> list:
    """每个有真值的帧：真值中心到「扩大一倍的真值框之外」最近候选的距离（px，没有则 inf），以及该帧速度门宽
    `GATE_PPS·dt`。衡量目标附近有没有能被宽门抢配的其他运动。"""
    clk, gts, obs = ctx["clk"], ctx["gts"], ctx["obs"]
    out = []
    for k, g in enumerate(gts):
        if g is None:
            continue
        c = (g[0] + g[2] / 2, g[1] + g[3] / 2)
        ds = [float(np.hypot(d[0] - c[0], d[1] - c[1])) for d in obs[k] if not in_box(d, g, 0.5)]
        out.append((min(ds) if ds else float("inf"), clk.speed_raw(timing.GATE_PPS, k)))
    return out


def agg_distractors(items):
    a = np.array([x for it in items for x in it]) if any(items) else np.zeros((0, 2))
    fin = a[np.isfinite(a[:, 0]), 0] if len(a) else a
    return {
        "gt_frames": int(len(a)),
        "with_distractor": float(np.isfinite(a[:, 0]).mean()) if len(a) else None,
        "distance_quartiles": np.percentile(fin, [25, 50, 75]).round(1).tolist() if len(fin) else None,
        "within_speed_gate": float((a[:, 0] < a[:, 1]).mean()) if len(a) else None,
        "within_40px": float((a[:, 0] < 40).mean()) if len(a) else None,
    }


def sel_stats(out, dur):
    tids = [r[2] for r in out]
    sw = sum(1 for a, b in zip(tids, tids[1:], strict=False) if b is not None and a is not None and a != b)
    fresh = float(np.mean([r[0] in ("track", "jump") for r in out])) if out else 0.0
    return {"tid_switches": sw, "fresh_frac": fresh, "duration": dur}


# ── 一个 (视频, 条件) 的全部评测 ───────────────────────────────────
def work(name, cond, width, methods, protocols, fracs):
    seq = moca.load()[name]
    dup = moca.duplicates(seq)
    k_keep = select(name, dup, cond)
    nondup = np.flatnonzero(~dup)
    pos = {int(k): i for i, k in enumerate(nondup)}
    keep = [pos[k] for k in k_keep]
    ctx = prepare(name, width, keep=None if len(keep) == len(nondup) else keep, stab="e2b" in protocols)
    res, sel = {}, {}
    for prot in protocols:
        for frac in fracs:
            aux = {}
            r = evaluate(ctx, frac, prot, methods, aux)
            res[f"{prot}@{frac:g}"] = r
            if not sel:
                dur = float(ctx["ts"][-1] - ctx["ts"][0])
                sel = {m: sel_stats(o, dur) for m, o in aux.items()}
    diag = {g: track_diag(ctx, g) for g in mot.GATES}
    return {"name": name, "cond": cond, "res": res, "sel": sel, "diag": diag, "clock": clock_stats(ctx["clk"]),
            "distractors": distractors(ctx)}


# ── 汇总 ─────────────────────────────────────────────────────────
def agg_diag(items):
    opp = [o for it in items for o in it["opp"]]
    dur = sum(it["duration"] for it in items)
    out = {"opportunities": 0, "births_per_sec": sum(it["births"] for it in items) / dur,
           "target_switches_per_min": 60 * sum(it["target_switches"] for it in items) / dur}
    if opp:
        op = [o for o in opp if o[5]]
        a = np.array([(o[0], o[1], o[2]) for o in op]) if op else np.full((1, 3), np.nan)
        outc = [o[4] for o in op]
        out.update({
            "opportunities": len(op),
            "target_track_frames": len(opp),
            "gate_r_eff_median": float(np.median(a[:, 0])),
            "gate_r_eff_quartiles": np.percentile(a[:, 0], [25, 50, 75]).round(1).tolist(),
            "gate_r_phys_median": float(np.median(a[:, 1])),
            "nis_median": float(np.median(a[:, 2])),
            "nis_over_chi2": float(np.mean(a[:, 2] > timing.GATE_CHI2)),
            "rejected": float(np.mean([o[3] for o in op])) if op else None,
            **{x: float(np.mean([o == x for o in outc])) if op else None for x in ("cont", "steal", "ring", "miss")},
            # 不论本帧有没有框内候选：目标轨迹配上扩大框之外候选的比例与频率（误关联）
            "steal_any": float(np.mean([o[4] == "steal" for o in opp])),
            "steal_per_min": 60 * sum(o[4] == "steal" for o in opp) / dur,
            "gate_r_eff_median_all": float(np.median([o[0] for o in opp])),
        })
    return out


def pair_wins(vids, key, a, b, thr=0.01):
    """逐段 time-in-frame：a 比 b 高 / 低 超过 thr 的段数。"""
    va = [np.mean([x[a][0] for x in v["res"][key]["rows"]]) for v in vids if v["res"][key]["rows"]]
    vb = [np.mean([x[b][0] for x in v["res"][key]["rows"]]) for v in vids if v["res"][key]["rows"]]
    d = np.array(va) - np.array(vb)
    return [int((d > thr).sum()), int((d < -thr).sum())]


def condense(vids, methods):
    """一个条件下全部视频的结果 → 汇总 + 逐段精简记录（丢掉逐帧记录）。"""
    summary = {}
    for key in vids[0]["res"]:
        s = summarize([v["res"][key] for v in vids], methods)
        s["pairs_better_worse"] = {f"{a}_vs_{b}": pair_wins(vids, key, a, b) for a, b in PAIRS if a in methods and b in methods}
        summary[key] = s
    diag = {g: agg_diag([v["diag"][g] for v in vids]) for g in mot.GATES}
    diag["distractors"] = agg_distractors([v["distractors"] for v in vids])
    for m in vids[0]["sel"]:
        st = [v["sel"][m] for v in vids]
        dur = sum(x["duration"] for x in st)
        diag[f"selector:{m}"] = {
            "tid_switches_per_min": 60 * sum(x["tid_switches"] for x in st) / dur,
            "fresh_frac_mean": float(np.mean([x["fresh_frac"] for x in st])),
        }
    cl = [v["clock"] for v in vids]
    clock = {
        **{f"{k}_median": float(np.median([x[k] for x in cl])) for k in ("hz", "mean_hz", "jitter", "cv")},
        **{f"{k}_quartiles": np.percentile([x[k] for x in cl], [25, 50, 75]).round(3).tolist() for k in ("jitter", "cv")},
        "frames": int(sum(x["n"] for x in cl)),
        "clock_ok_false_videos": int(sum(not x["ok"] for x in cl)),
    }
    videos = [
        {
            "name": v["name"],
            "clock": v["clock"],
            "in_frame": {
                key: {m: float(np.mean([x[m][0] for x in r["rows"]])) if r["rows"] else None for m in methods}
                for key, r in v["res"].items()
            },
            "gt_frames": len(next(iter(v["res"].values()))["rows"]),
            "diag": {g: agg_diag([v["diag"][g]]) for g in mot.GATES},
            "sel": v["sel"],
        }
        for v in vids
    ]
    return {"summary": summary, "diag": diag, "clock": clock, "videos": videos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test", "all"))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--protocols", default="e2,e2b")
    ap.add_argument("--fracs", default="0.5,0.33")
    ap.add_argument("--methods", default=DEFAULT_METHODS)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    conds = args.conditions.split(",")
    for c in conds:
        if c not in CONDITIONS:
            raise SystemExit(f"未知条件 {c}")
    protocols, fracs, methods = args.protocols.split(","), [float(x) for x in args.fracs.split(",")], args.methods.split(",")
    names = load_split(args.split)
    if args.limit:
        names = names[: args.limit]
    tag = f"-{args.tag}" if args.tag else ""
    parts = RUNS / "fps_parts"
    parts.mkdir(parents=True, exist_ok=True)
    low_priority()
    t0 = time.time()
    summary, diag, clock, videos = {}, {}, {}, {}
    # 逐条件跑、逐条件汇总并落盘（`runs/fps_parts/`，同 split + tag + 条件 + 设定已存在就直接读）——
    # 逐帧记录只在内存里留一个条件，中断后可续跑
    key_cfg = f"{args.split}{tag}-n{len(names)}-{args.protocols}-{args.fracs}-{zlib.crc32(args.methods.encode())}"
    with ProcessPoolExecutor(args.workers, initializer=low_priority) as ex:
        for c in conds:
            pf = parts / f"{key_cfg}-{c}.pkl"
            if pf.exists():
                part = pickle.loads(pf.read_bytes())
            else:
                futs = [ex.submit(work, n, c, args.width, methods, protocols, fracs) for n in names]
                vids = [f.result() for f in futs]
                part = condense(vids, methods)
                del vids
                pf.write_bytes(pickle.dumps(part))
            summary[c], diag[c], clock[c], videos[c] = part["summary"], part["diag"], part["clock"], part["videos"]
            print(f"  [{c}] 完成  {time.time() - t0:.0f}s", flush=True)

    meta = {
        "split": args.split, "videos": len(names), "conditions": {c: CONDITIONS[c] for c in conds}, "seed": SEED,
        "protocols": protocols, "fracs": fracs, "methods": methods, "width": args.width,
        "variants": {k: VARIANTS[k] for k in ("fixed", "speed")}, "meas_var_calibrated": list(mot.MEAS_VAR),
        "wall_sec": round(time.time() - t0, 1),
    }
    out = RUNS / f"fps-{args.split}{tag}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(
        json.dumps({"meta": meta, "clock": clock, "diag": diag, "summary": summary, "videos": videos},
                   ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )
    # ── 打印 ──
    print(f"{args.split}  {len(names)} 段  用时 {meta['wall_sec']}s")
    for c in conds:
        cl = clock[c]
        print(f"[{c}] 帧 {cl['frames']}  hz(中位) {cl['hz_median']:.1f}  平均 {cl['mean_hz_median']:.1f}  "
              f"jitter {cl['jitter_median']:.2f}  cv {cl['cv_median']:.2f}  Clock.ok=False {cl['clock_ok_false_videos']} 段")
    for key in summary[conds[0]]:
        print(f"\n== {key}  time-in-frame %（全部帧）==")
        print(f"{'方法':14s}" + "".join(f"{c:>7s}" for c in conds))
        for m in methods:
            print(f"{m:14s}" + "".join(f"{100 * summary[c][key]['all_frames'][m]['in_frame']:7.1f}" for c in conds))
        for a, b in PAIRS:
            if a in methods and b in methods:
                print(f"{a} 对 {b} 优/劣 " + " ".join(str(summary[c][key]["pairs_better_worse"][f"{a}_vs_{b}"]) for c in conds))
    print("\n== 跟踪诊断 ==")
    for c in conds:
        for g in mot.GATES:
            d = diag[c][g]
            print(f"[{c}] {g:11s} 机会 {d['opportunities']:6d}  门宽 {d.get('gate_r_eff_median', float('nan')):6.1f}px  "
                  f"拒 {100 * d.get('rejected', 0):5.1f}%  续 {100 * d.get('cont', 0):5.1f}%  被接走 {100 * d.get('steal', 0):4.1f}%  "
                  f"丢 {100 * d.get('miss', 0):5.1f}%  误关联/min {d.get('steal_per_min', 0):4.1f}  NIS中位 {d.get('nis_median', float('nan')):6.1f}  "
                  f"出生/s {d['births_per_sec']:.2f}  目标切换/min {d['target_switches_per_min']:.1f}")
        dd = diag[c]["distractors"]
        print(f"      干扰候选：有 {100 * dd['with_distractor']:.1f}% 帧，距离四分位 {dd['distance_quartiles']}，"
              f"落在速度门内 {100 * dd['within_speed_gate']:.1f}%，40px 内 {100 * dd['within_40px']:.1f}%")
        for m in ("core:fixed", "core:speed"):
            if f"selector:{m}" in diag[c]:
                s = diag[c][f"selector:{m}"]
                print(f"      {m} 选择器 tid 切换/min {s['tid_switches_per_min']:.1f}  有新证据帧 {100 * s['fresh_frac_mean']:.1f}%")
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
