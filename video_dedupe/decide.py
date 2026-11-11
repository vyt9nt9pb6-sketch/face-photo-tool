#!/usr/bin/env python3
"""decide.py —— 合并重复簇，按画质选保留项，生成移动计划

三个合并依据（任一成立即并入同簇）：
  1. 字节完全相同（md5check.py 的结果）
  2. 帧比对确认同一段画面（med <= --frame-med 且 corr >= --corr-min）
  3. 同名族（去掉 (0001) 之类后缀后同名）+ 同分辨率 + 时长差 <= --same-tol

硬约束：**时长差 > --max-dur-diff 一律不合并**。
同一段视频被裁过首尾时，内容范围不同，合并会丢素材。

保留哪一份：像素数 → 码率 → 体积 → 目录层级浅。
不能用时长——低码率版本往往时长略长，会把糊的那份留下。

用法:
  python decide.py --index vindex.pkl --exact exact.json --frames frames.json \
      --dup-dir ./重复视频 --out plan.json
"""
import argparse
import json
import os
import pickle
from collections import defaultdict

SUFFIXES = ("(0001)", "(0002)", "(0003)", "(0004)", "(1)", "(2)", "_dup")


def base_name(p):
    stem = os.path.splitext(os.path.basename(p))[0]
    changed = True
    while changed:
        changed = False
        for suf in SUFFIXES:
            if stem.lower().endswith(suf):
                stem = stem[: -len(suf)]
                changed = True
    return stem.lower()


def depth_of(p):
    return len(os.path.normpath(p).split(os.sep))


def main():
    ap = argparse.ArgumentParser(description="生成视频去重移动计划")
    ap.add_argument("--index", required=True, help="vindex.py 输出的 pickle")
    ap.add_argument("--exact", help="md5check.py 输出（字节相同组）")
    ap.add_argument("--frames", help="frames.py 输出（画面比对）")
    ap.add_argument("--dup-dir", default="重复视频", help="重复项去向（仅记录，不在此实际移动）")
    ap.add_argument("--out", default="plan.json", help="计划输出")
    ap.add_argument("--clusters-out", default="clusters.json", help="簇摘要输出")
    ap.add_argument("--frame-med", type=float, default=1.0, help="判定同画面的逐帧差中位数阈值")
    ap.add_argument("--corr-min", type=float, default=0.99, help="判定同画面的相关系数下限")
    ap.add_argument("--max-dur-diff", type=float, default=0.5, help="时长差超过此值不合并（秒）")
    ap.add_argument("--same-tol", type=float, default=0.2, help="同名族合并的时长容差（秒）")
    ap.add_argument("--no-basename", action="store_true", help="禁用同名族合并")
    a = ap.parse_args()

    rows = pickle.load(open(a.index, "rb"))
    probe = {r["p"]: r for r in rows}
    exact = json.load(open(a.exact, encoding="utf-8")) if a.exact and os.path.exists(a.exact) else []
    frames = json.load(open(a.frames, encoding="utf-8")) if a.frames and os.path.exists(a.frames) else []

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    def dur(p):
        return probe.get(p, {}).get("dur", 0.0) or 0.0

    # 1) 字节完全相同
    for g in exact:
        files = g["files"]
        for p in files[1:]:
            union(files[0], p)

    # 2) 帧比对确认
    confirmed = 0
    for r in frames:
        pa, pb = r.get("a"), r.get("b")
        if not pa or not pb or pa not in probe or pb not in probe:
            continue
        if r.get("med", 99) > a.frame_med or r.get("corr", 0) < a.corr_min:
            continue
        if abs(dur(pa) - dur(pb)) > a.max_dur_diff:
            continue
        union(pa, pb)
        confirmed += 1

    # 3) 同名族
    basename_edges = 0
    if not a.no_basename:
        by_base = defaultdict(list)
        for p in probe:
            by_base[base_name(p)].append(p)
        for base, items in by_base.items():
            if len(items) < 2:
                continue
            items.sort(key=lambda p: -dur(p))
            anchor = items[0]
            for p in items[1:]:
                if probe[p].get("w") != probe[anchor].get("w"):
                    continue
                if probe[p].get("h") != probe[anchor].get("h"):
                    continue
                if abs(dur(p) - dur(anchor)) > min(a.same_tol, a.max_dur_diff):
                    continue
                union(anchor, p)
                basename_edges += 1

    # 归簇
    clusters = defaultdict(list)
    for p in list(parent):
        if os.path.exists(p):
            clusters[find(p)].append(p)

    def qkey(p):
        """画质排序键：像素数 → 码率(kbps) → 体积 → 层级浅"""
        r = probe.get(p, {})
        px = (r.get("w", 0) or 0) * (r.get("h", 0) or 0)
        d = r.get("dur", 0) or 0
        br = round(r.get("sz", 0) * 8 / d / 1000, 1) if d else 0
        return (px, br, r.get("sz", 0), -depth_of(p))

    plan, summary = [], []
    for root, files in clusters.items():
        if len(files) < 2:
            continue
        files_sorted = sorted(files, key=qkey, reverse=True)
        keep = files_sorted[0]
        kq = qkey(keep)
        moved = []
        for p in files_sorted[1:]:
            pq = qkey(p)
            r = probe.get(p, {})
            kr = probe.get(keep, {})
            same_bytes = (r.get("sz") == kr.get("sz") and pq[0] == kq[0] and pq[1] == kq[1])
            plan.append({
                "src": p,
                "keep": keep,
                "size": r.get("sz", 0),
                "kind": "exact" if same_bytes else "reencode",
                "note": ("字节完全相同" if same_bytes else
                         f"同一段视频的较低画质版本（{r.get('w')}x{r.get('h')} "
                         f"{r.get('dur')}s {r.get('sz', 0) / 1048576:.1f}MB）"),
            })
            moved.append(p)
        summary.append({"keep": keep, "moved": moved, "n": len(files)})

    total = sum(r["size"] for r in plan)
    print(f"字节相同组 {len(exact)}；帧比对确认 {confirmed} 对；同名族并入 {basename_edges} 对")
    print(f"重复簇 {len(summary)}  待移动 {len(plan)} 个  {total / 1024 ** 3:.2f} GB")

    json.dump(plan, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(summary, open(a.clusters_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n计划已写入 {a.out}（未移动任何文件）")
    for s in sorted(summary, key=lambda x: -x["n"])[:20]:
        r = probe.get(s["keep"], {})
        print(f"  留 {r.get('w')}x{r.get('h')} {r.get('dur')}s "
              f"{r.get('sz', 0) / 1048576:8.2f}MB  {s['keep']}  (移 {len(s['moved'])})")


if __name__ == "__main__":
    main()
