#!/usr/bin/env python3
"""vindex.py —— 视频索引：体积 + 头/中/尾三段采样哈希（可选帧指纹）

第一层判重。只有采样哈希相同的文件才可能是同一份内容，
这一步把 N 个文件的全量比对压缩到少数几组。

用法:
  python vindex.py --root /path/to/videos --out vindex.pkl
  python vindex.py --root /path/to/videos --out vindex.pkl --frames
"""
import argparse
import hashlib
import os
import pickle
import time

VID = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".3gp", ".mpg", ".mpeg", ".wmv",
       ".flv", ".webm", ".ts", ".mts", ".m2ts", ".rmvb", ".rm", ".vob", ".asf",
       ".f4v", ".divx"}

EXC = {"system volume information", "$recycle.bin", "recycler"}


def list_files(root, skip_top=()):
    """遍历视频文件；skip_top 中的顶层目录整棵跳过"""
    skip = {s.lower() for s in skip_top}
    out = []
    for dp, dn, fn in os.walk(root):
        rel = os.path.relpath(dp, root).split(os.sep)
        if rel and rel[0].lower() in skip:
            dn[:] = []
            continue
        dn[:] = [d for d in dn if d.lower() not in EXC and not d.startswith("$")]
        for f in fn:
            if os.path.splitext(f)[1].lower() in VID:
                out.append(os.path.normpath(os.path.join(dp, f)))
    return sorted(out)


def sample_hash(path, size, block=1 << 18):
    """头 + 中 + 尾三段采样哈希。体积也参与，避免不同长度的文件撞哈希。"""
    h = hashlib.blake2b(digest_size=16)
    h.update(str(size).encode())
    with open(path, "rb") as f:
        starts = [0, max(0, size // 2 - block // 2), max(0, size - block)]
        for s in starts:
            f.seek(s)
            h.update(f.read(block))
    return h.hexdigest()


def frame_sig(path, n=12, side=64):
    """均匀抽 n 帧灰度缩略图，返回 (字节, 元信息, 时长)"""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None, 0.0
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        dur = total / fps if fps > 0 and total > 0 else 0.0
        idxs = [int(total * (i + 0.5) / n) for i in range(n)] if total > 0 else []
        thumbs = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                thumbs.append(np.zeros((side, side), dtype=np.uint8))
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            thumbs.append(cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA))
        if not thumbs:
            return None, (w, h, dur, fps, total), 0.0
        stack = np.stack(thumbs).astype(np.uint8)
        return stack.tobytes(), (w, h, dur, fps, total), float(dur)
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser(description="视频索引（体积 + 采样哈希）")
    ap.add_argument("--root", required=True, help="要扫描的根目录")
    ap.add_argument("--out", default="vindex.pkl", help="索引输出文件")
    ap.add_argument("--skip", default="", help="跳过的顶层目录，逗号分隔")
    ap.add_argument("--frames", action="store_true", help="额外抽取帧指纹（慢）")
    ap.add_argument("--nframes", type=int, default=12, help="每个视频抽几帧")
    a = ap.parse_args()

    files = list_files(a.root, [s for s in a.skip.split(",") if s])
    print(f"待索引视频: {len(files)}", flush=True)

    rows = []
    t0 = time.time()
    for i, p in enumerate(files, 1):
        try:
            sz = os.path.getsize(p)
        except OSError as e:
            print("  size err", p, e, flush=True)
            continue
        r = {"p": p, "sz": sz}
        try:
            r["sh"] = sample_hash(p, sz)
        except OSError as e:
            r["sh"] = None
            r["err"] = f"read:{e}"
        if a.frames:
            try:
                sig, meta, dur = frame_sig(p, a.nframes)
                r["sig"] = sig
                if meta:
                    r["w"], r["h"], r["dur"], r["fps"], r["nf"] = meta
            except Exception as e:
                r["err"] = f"frame:{e}"
        rows.append(r)
        if i % 25 == 0 or i == len(files):
            el = time.time() - t0
            rest = el / i * (len(files) - i) if i else 0
            print(f"  {i}/{len(files)}  {el:.0f}s  剩余约 {rest:.0f}s", flush=True)

    with open(a.out, "wb") as f:
        pickle.dump(rows, f)
    print("已保存", a.out, len(rows), "条")


if __name__ == "__main__":
    main()
