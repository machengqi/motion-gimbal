"""诊断 E2 的失败案例：瞄准点是怎么被带走的。

    uv run python scripts/diag_e2_failures.py cuttlefish_3 [--gate uncertainty] [--frames 0:80]

与 `core.run` 同一条流水线（Tracker → fill_gaps → Selector → Aim，离线口径），逐帧打印：
候选、轨迹（背景系 / 画面系位置、净位移、散布、是否被排除名单命中）、置信闸是否关闭、
选择器阶段 / pos / 印证状态、背景位移与累计值、瞄准点、真值中心。

⚠️ 不改核心代码：选择器内部的中间量（排除名单命中、净位移、置信闸）是**在 `step()` 之后**
用同一组公开函数（`saliency.net_disp` / `spread` / `ExclusionMap.hit`）和同一组常数重算的。
`step()` 里排除名单的标记发生在 ② 之前，所以 step 之后的名单就是 ② 用的那份。
重算结果与 `step()` 的阶段对不上会打印 `!!MISMATCH`。

另外逐帧写 `runs/diag_cache/<name>-trace.json`，供后续统计。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import CACHE, gt_center, load  # noqa: E402

from gimbal_vision.core import controller, mot, saliency, timing  # noqa: E402

# 2026-10-20：核心默认值已改（R 标定、chal 平移、screenfix、stale）；本脚本固定用修复前的口径，
# 以保持 2026-10-14 诊断结论可复现。
LEGACY_SEL = {"shift_chal": False, "screenfix": False, "stale": False}
LEGACY_R = 9.0


def in_box(p, g, pad=0.0):
    return g is not None and g[0] - pad <= p[0] <= g[0] + g[2] + pad and g[1] - pad <= p[1] <= g[1] + g[3] + pad


def run(name, gate="uncertainty", frames=None, quiet=False, write=True, selector_cls=None):
    d = load(name)
    clk, obs, shifts, gts = d["clk"], d["obs"], d["shifts"], d["gts"]
    n, W, H = len(obs), d["W"], d["H"]
    roi = (0, 0, W, H)
    hist = mot.track(obs, roi, clock=clk, gate=gate, shifts=shifts, meas_var=LEGACY_R)
    seq = mot.fill_gaps(mot.paths(hist), clk.frames_in(timing.FILL_SEC))
    sel, aim = (selector_cls or saliency.Selector)(clk, roi, **LEGACY_SEL), controller.Aim(clk)
    lo_f, hi_f = frames or (0, n)

    trace, prev_cum = [], np.zeros(2)
    mismatch = 0
    for k in range(n):
        cum = hist[k][2]
        bg = cum - prev_cum
        prev_cum = cum
        n_ex_before = len(sel.excluded.pts)
        pos_before = None if sel.pos is None else np.array(sel.pos)
        phase, target, tid, fresh = sel.step(k, hist, seq, bg)
        apos = aim.step(k, bg, cum, target, fresh)

        # ── 重算选择器的中间量 ──
        lo_s, n_s = clk.window(k, timing.STATIC_SEC)
        lo_b, n_b = clk.window(k, timing.WINDOW_SEC)
        g = gts[k]
        gc = gt_center(g)
        rows, cand_raw = [], {}
        for t, v in hist[k][1].items():
            fx, fy = v[0] + cum[0], v[1] + cum[1]  # 画面系
            ex = sel.excluded.hit(v[0], v[1])
            nd = saliency.net_disp(seq, k, t, lo_b, n_b)
            sp = saliency.spread(seq, k, t, lo_s, n_s)
            obs_now = v[5] == 0
            if obs_now and not ex and nd is not None:
                cand_raw[t] = nd
            rows.append(dict(tid=t, fx=fx, fy=fy, bx=v[0], by=v[1], missed=v[5], hits=v[4], ex=ex,
                             nd=nd, sp=sp, on_gt=bool(in_box((fx, fy), g)), obs=obs_now))
        gate_closed = False
        med = None
        if len(cand_raw) >= 3:
            med = float(np.median(list(cand_raw.values())))
            gate_closed = max(cand_raw.values()) < saliency.LEAD * max(med, 1e-6)
        cand = {} if gate_closed else cand_raw
        if phase != "wait" and not cand:
            mismatch += 1
        # 真值位置有没有被排除名单覆盖（背景系）
        gt_ex = None if gc is None else sel.excluded.hit(gc[0] - cum[0], gc[1] - cum[1])
        dets_on_gt = sum(in_box((dd[0], dd[1]), g) for dd in obs[k])
        rec = dict(
            k=k, fk=d["ks"][k], gt=None if gc is None else gc.tolist(), gtbox=None if g is None else list(map(float, g)), gt_ex=gt_ex,
            ndet=len(obs[k]), dets_on_gt=int(dets_on_gt), bg=bg.tolist(), cum=cum.tolist(),
            phase=phase, sel_pos=None if target is None else list(target), cur=tid, fresh=fresh,
            chal=None if sel.chal is None else sel.chal.tolist(), streak=sel.streak,
            aim=None if apos is None else list(apos), n_excl=len(sel.excluded.pts),
            new_excl=len(sel.excluded.pts) - n_ex_before, gate_closed=gate_closed, med=med,
            cand=cand_raw, tracks=rows,
            jump_lim=clk.speed(timing.JUMP_PPS, k),
            pos_before=None if pos_before is None else pos_before.tolist(),
        )
        trace.append(rec)
        if not quiet and lo_f <= k < hi_f:
            pr(rec)
    if not quiet:
        print(f"[{name}] gate={gate} 帧数 {n}  重算与阶段不一致 {mismatch} 帧  排除点总数 {len(sel.excluded.pts)}")
    if write:
        out = CACHE / f"{name}-{gate}-trace.json"
        out.write_text(json.dumps(trace, default=float), encoding="utf-8")
    return trace, sel


def f2(p):
    return "None" if p is None else f"({p[0]:6.1f},{p[1]:6.1f})"


def pr(r):
    print(
        f"k={r['k']:3d}(#{r['fk']:3d}) gt={f2(r['gt'])} gt_ex={r['gt_ex']!s:5} det={r['ndet']:2d}/onGT{r['dets_on_gt']} "
        f"bg=({r['bg'][0]:+5.1f},{r['bg'][1]:+5.1f}) cum={f2(r['cum'])} | {r['phase']:5} sel={f2(r['sel_pos'])} cur={r['cur']} "
        f"chal={f2(r['chal'])}x{r['streak']} aim={f2(r['aim'])} excl={r['n_excl']}(+{r['new_excl']}) "
        f"gate={'CLOSED' if r['gate_closed'] else 'open'} med={r['med'] and round(r['med'], 1)} lim={r['jump_lim']:.1f}"
    )
    for t in r["tracks"]:
        if not t["obs"] and not t["on_gt"]:
            continue
        print(
            f"     tid={t['tid']:3d} f=({t['fx']:6.1f},{t['fy']:6.1f}) miss={t['missed']} hits={t['hits']:3d} "
            f"{'EXCL' if t['ex'] else '    '} nd={t['nd'] and round(t['nd'], 1)} sp={t['sp'] and round(t['sp'], 1)}"
            f"{' <GT' if t['on_gt'] else ''}{' <cand' if t['tid'] in r['cand'] else ''}"
        )


def stats(name, trace, sel):
    """把逐帧记录归类：瞄准点为什么没有回到目标上。"""
    n_conf = sel.n_confirm
    first = next((r for r in trace if r["phase"] != "wait"), None)
    print(f"\n== {name}  帧数 {len(trace)}  n_confirm={n_conf}  CONFIRM_R≈{trace[-1]['jump_lim'] / 2:.1f}px  JUMP≈{trace[-1]['jump_lim']:.1f}px")
    if first:
        t0 = next(t for t in first["tracks"] if t["tid"] == first["cur"])
        print(f"首次锁定 k={first['k']}：sel={f2(first['sel_pos'])} 真值={f2(first['gt'])} 在真值框内={t0['on_gt']} "
              f"候选数={len(first['cand'])} 该轨迹 hits={t0['hits']} nd={t0['nd']:.1f}")
    cats: dict[str, int] = {}
    streak_on_gt_max, resets = 0, {"换到非目标候选": 0, "目标身上换了一块（>R）": 0}
    prev = None
    lost = 0
    for r in trace:
        if r["gt"] is None or r["sel_pos"] is None:
            continue
        g = r["gt"]
        on = next((t["on_gt"] for t in r["tracks"] if t["tid"] == r["cur"]), None)
        # 选择器 pos 是否在真值框内（用真值中心 + 框不可得时，复用轨迹判定；这里直接按中心距离 / 框）
        inside = in_box(r["sel_pos"], r["gtbox"])
        gtt = [t for t in r["tracks"] if t["on_gt"] and t["obs"]]
        gtc = [t for t in gtt if str(t["tid"]) in r["cand"] or t["tid"] in r["cand"]]
        if inside:
            cat = "pos 在目标上"
        elif r["phase"] != "wait":
            cat = "track/jump 但不在目标上"
        elif not gtt:
            cat = "wait：目标上没有观测到的轨迹（无候选）"
        elif all(t["ex"] for t in gtt):
            cat = "wait：目标上的轨迹全被排除名单命中"
        elif not gtc:
            cat = "wait：目标上有轨迹，但历史不够算净位移"
        elif r["gate_closed"]:
            cat = "wait：置信闸关闭"
        else:
            best = max(r["cand"], key=r["cand"].get)
            bt = next(t for t in r["tracks"] if str(t["tid"]) == str(best))
            cat = "wait：目标是赢家但远跳待印证" if bt["on_gt"] else "wait：赢家不是目标"
        cats[cat] = cats.get(cat, 0) + 1
        if not inside:
            lost += 1
        if r["chal"] is not None and prev is not None and prev["chal"] is not None and r["streak"] == 1 and prev["streak"] >= 1 and r["chal"] != prev["chal"]:
            ct = [t for t in r["tracks"] if abs(t["fx"] - r["chal"][0]) < 1e-6]
            if ct and ct[0]["on_gt"]:
                resets["目标身上换了一块（>R）"] += 1
            else:
                resets["换到非目标候选"] += 1
        if r["chal"] is not None and r["phase"] == "wait":
            ct = [t for t in r["tracks"] if abs(t["fx"] - r["chal"][0]) < 1e-6]
            if ct and ct[0]["on_gt"]:
                streak_on_gt_max = max(streak_on_gt_max, r["streak"])
        prev = r
    tot = sum(cats.values())
    for c, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c:32s} {v:4d}  {100 * v / tot:5.1f}%")
    print(f"  印证链被打断（streak 回到 1）: {resets}；挑战者在目标上时的最长 streak = {streak_on_gt_max}（需要 {n_conf}）")
    ex_hits = sum(1 for r in trace for t in r["tracks"] if t["on_gt"] and t["obs"] and t["ex"])
    print(f"  排除点总数 {trace[-1]['n_excl']}；目标上被排除的观测 {ex_hits} 次；真值中心落在排除点上 {sum(1 for r in trace if r['gt_ex'])} 帧")
    jumps = [r["k"] for r in trace if r["phase"] == "jump"]
    print(f"  jump 帧: {jumps}")
    return {"cats": cats, "resets": resets, "streak_on_gt_max": streak_on_gt_max, "n_confirm": n_conf, "excl": trace[-1]["n_excl"], "gt_ex_frames": sum(1 for r in trace if r["gt_ex"]), "ex_hits": ex_hits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="+")
    ap.add_argument("--gate", default="uncertainty")
    ap.add_argument("--frames", default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    fr = tuple(int(x) for x in a.frames.split(":")) if a.frames else None
    if a.name == ["ALL"]:
        aggregate(gate=a.gate)
        return
    for name in a.name:
        trace, sel = run(name, a.gate, fr, a.quiet or a.stats)
        if a.stats:
            stats(name, trace, sel)




def _one(name, gate="uncertainty"):
    import contextlib
    import io

    trace, sel = run(name, gate, None, True, write=False)
    with contextlib.redirect_stdout(io.StringIO()):
        return name, stats(name, trace, sel)


def aggregate(workers=6, gate="uncertainty"):
    """全集：各类「没回到目标上」的帧数合计。`uv run python scripts/diag_e2_failures.py ALL`"""
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    from gimbal_vision.datasets import moca  # noqa: PLC0415

    names = sorted(moca.load())
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_one, names, [gate] * len(names)))
    cats, resets = {}, {}
    for _, s in res:
        for c, v in s["cats"].items():
            cats[c] = cats.get(c, 0) + v
        for c, v in s["resets"].items():
            resets[c] = resets.get(c, 0) + v
    tot = sum(cats.values())
    print(f"全集 {len(res)} 段，有真值且选择器已有 pos 的帧 {tot}")
    for c, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c:32s} {v:6d}  {100 * v / tot:5.1f}%")
    print(f"  印证链被打断: {resets}")
    print(f"  排除点>0 的视频 {sum(s['excl'] > 0 for _, s in res)} 段；真值中心落在排除点上的帧合计 "
          f"{sum(s['gt_ex_frames'] for _, s in res)}；目标上的观测被排除合计 {sum(s['ex_hits'] for _, s in res)}")
    sm = [s["streak_on_gt_max"] for _, s in res]
    print(f"  挑战者在目标上时的最长 streak 分布: {dict(zip(*np.unique(sm, return_counts=True)))}（n_confirm 取值 {sorted({s['n_confirm'] for _, s in res})}）")


if __name__ == "__main__":
    main()
