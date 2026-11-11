#!/usr/bin/env python3
"""frames.py —— 第三层判重：时间轴抽帧 + 旋转不变的画面比对

对候选对抽取 n 帧灰度缩略图，逐帧取 4 个旋转方向的最小平均差，
用逐帧差的**中位数**判定是否同一段画面。

关键约束（实测踩坑）：
  * 必须按 CAP_PROP_POS_MSEC 定位，不能按帧序号——不同版本帧率不同，
    按帧号取「第 N 帧」会落在不同时刻。
  * 两段只能用共同的时间跨度 min(dur_a, dur_b) 比较。

用法:
  python frames.py --index vindex.pkl --out frames.json            # 自动挑候选对
  python frames.py --pairs pairs.json --out frames.json            # 指定对
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

# 候选对筛选：分辨率相同、时长差小、体积比接近，且不是同一路径
MAX_DUR_DIFF = 0.35
SIZE_RATIO = (0.55, 1.8)


def probe(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        dur = total / fps if fps > 0 and total > 0 else 0.0
        return {"w": w, "h": h, "dur": round(dur, 2), "fps": round(fps, 2), "nf": total}
    finally:
        cap.release()


def frames_sig(path, n=10, side=64, span=None):
    """按统一时间轴抽帧；span 给定时只在 0..span 内取点，便于不同时长对齐"""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        dur = total / fps if fps > 0 and total > 0 else 0.0
        use = span if span else dur
        outs = []
        if use > 0:
            for i in range(n):
                t = (i + 0.5) / n * use
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, fr = cap.read()
                if not ok:
                    outs.append(np.zeros((side, side), np.uint8))
                    continue
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                outs.append(cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA))
        elif total > 0:
            for fi in [int(total * (i + 0.5) / n) for i in range(n)]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, fr = cap.read()
                if not ok:
                    outs.append(np.zeros((side, side), np.uint8))
                    continue
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                outs.append(cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA))
        if not outs:
            return None
        return np.stack(outs).astype(np.uint8)
    finally:
        cap.release()


def compare(path_a, path_b, n=10):
    """返回逐帧差统计。med 是主要判据。"""
    ma, mb = probe(path_a), probe(path_b)
    if not ma or not mb:
        return None
    span = min(ma["dur"], mb["dur"]) or None
    sa = frames_sig(path_a, n=n, span=span)
    sb = frames_sig(path_b, n=n, span=span)
    if sa is None or sb is None:
        return None
    k = min(len(sa), len(sb))
    per = []
    for i in range(k):
        fa = sa[i].astype(np.float32)
        fb = sb[i].astype(np.float32)
        per.append(min(float(np.abs(fa - np.rot90(fb, r)).mean()) for r in range(4)))
    arr = np.array(per)
    x = sa[:k].astype(np.float32).ravel()
    y = sb[:k].astype(np.float32).ravel()
    mx, my = x - x.mean(), y - y.mean()
    corr = float(np.dot(mx, my) / ((np.linalg.norm(mx) * np.linalg.norm(my)) + 1e-6))
    return {"a": path_a, "b": path_b, "meta_a": ma, "meta_b": mb,
            "mean": round(float(arr.mean()), 2), "med": round(float(np.median(arr)), 2),
            "max": round(float(arr.max()), 2), "corr": round(corr, 4),
            "per": [round(float(v), 2) for v in per]}


def build_pairs(rows, exact):
    """自动挑候选对：同分辨率、时长接近、体积比合理"""
    covered = set()
    for g in exact:
        for p in g["files"]:
            covered.add(p)
    cand = [r for r in rows if r.get("dur") and r["p"] not in covered]
    pairs = {}
    # 按分辨率分桶：只有同分辨率的才可能是同一段的不同编码
    buckets = {}
    for r in cand:
        buckets.setdefault((r.get("w"), r.get("h")), []).append(r)
    for key, items in buckets.items():
        items.sort(key=lambda r: r["dur"])
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[j]["dur"] - items[i]["dur"] > MAX_DUR_DIFF:
                    break
                ra, rb = items[i]["sz"], items[j]["sz"]
                ratio = ra / rb if rb else 0
                if SIZE_RATIO[0] <= ratio <= SIZE_RATIO[1]:
                    pairs[(items[i]["p"], items[j]["p"])] = True
    return list(pairs.keys())


def main():
    ap = argparse.ArgumentParser(description="时间轴抽帧的画面比对")
    ap.add_argument("--index", help="vindex.py 输出的 pickle（自动挑候选对时需要）")
    ap.add_argument("--exact", help="md5check.py 输出的 exact.json")
    ap.add_argument("--pairs", help="指定候选对 JSON：[[a, b], ...]")
    ap.add_argument("--out", default="frames.json", help="比对结果输出")
    ap.add_argument("--nframes", type=int, default=10, help="每对抽几帧")
    a = ap.parse_args()

    if a.pairs:
        pairs = json.load(open(a.pairs, encoding="utf-8"))
    else:
        if not a.index:
            raise SystemExit("需要 --index 或 --pairs")
        rows = pickle.load(open(a.index, "rb"))
        exact = json.load(open(a.exact, encoding="utf-8")) if a.exact else []
        pairs = build_pairs(rows, exact)
    print(f"候选对: {len(pairs)}", flush=True)

    out = []
    t0 = time.time()
    for i, item in enumerate(pairs, 1):
        pa, pb = (item["a"], item["b"]) if isinstance(item, dict) else item
        try:
            r = compare(pa, pb, a.nframes)
        except Exception as e:
            print("  ERR", os.path.basename(pa), e, flush=True)
            r = None
        if r:
            out.append(r)
            flag = "同" if r["med"] <= 1 else ("疑似" if r["med"] <= 5 else "不同")
            print(f"  [{flag}] med={r['med']:6.2f} corr={r['corr']:.3f}  "
                  f"{os.path.basename(pa)} | {os.path.basename(pb)}", flush=True)
        if i % 5 == 0:
            print(f"  -- {i}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)

    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n完成 {len(out)} 对，用时 {time.time() - t0:.0f}s → {a.out}")


if __name__ == "__main__":
    main()
