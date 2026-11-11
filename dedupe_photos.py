# -*- coding: utf-8 -*-
"""dedupe_photos.py —— 全盘/目录级照片查重，重复的移到指定文件夹

判定两道关：
  1) 感知指纹（16x16 dHash，256bit）汉明距离 <= --ham，且长宽比差异 <= --ar-tol
  2) 64x64 灰度像素级复核：平均绝对差 <= --mad 且相关系数 >= --corr
这样连拍的不同瞬间、相似版式的截图/文档不会被误判成重复。

用法:
  python dedupe_photos.py --root D:\\ --dup-dir D:\\重复照片            # 预演
  python dedupe_photos.py --root D:\\ --dup-dir D:\\重复照片 --apply    # 执行
"""
import os
import sys
import json
import time
import argparse
import multiprocessing as mp

import numpy as np

import fp_core as C


def _task(p):
    return C.analyze(p, crop_dir=None, detect=False)


def scan(paths, work, workers, done):
    os.makedirs(work, exist_ok=True)
    out_path = os.path.join(work, "scan_noface.jsonl")
    todo = [p for p in paths if p not in done]
    print(f"待扫描 {len(todo)} / 共 {len(paths)}")
    t0 = time.time()
    recs, failed = [], []
    with open(out_path, "a", encoding="utf-8") as out, mp.Pool(workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_task, todo, chunksize=8), 1):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec.get("ok"):
                recs.append(rec)
            else:
                failed.append(rec["p"])
            if i % 500 == 0:
                out.flush()
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  已用 {el/60:.1f} min  剩余约 {el/i*(len(todo)-i)/60:.1f} min")
        out.flush()

    # 并发读 U 盘时偶发 Errno 22 / 卷被更改，串行重试，避免整批文件被静默跳过
    if failed:
        print(f"读取失败 {len(failed)} 个，串行重试…")
        still = []
        with open(out_path, "a", encoding="utf-8") as out:
            for i, p in enumerate(failed, 1):
                ok = None
                for _ in range(3):
                    r = _task(p)
                    if r.get("ok"):
                        ok = r
                        break
                    time.sleep(0.6)
                out.write(json.dumps(ok or r, ensure_ascii=False) + "\n")
                if ok:
                    recs.append(ok)
                else:
                    still.append((p, r.get("err", "")))
                if i % 20 == 0:
                    out.flush()
        print(f"  重试成功 {len(failed)-len(still)}，仍失败 {len(still)}")
        for p, e in still[:5]:
            print(f"  ! {p}  {str(e)[:70]}")
    return recs


def main():
    ap = argparse.ArgumentParser(description="照片查重并归集重复文件")
    ap.add_argument("--root", required=True)
    ap.add_argument("--dup-dir", required=True, help="重复照片移到这里")
    ap.add_argument("--work", default="./_dupwork")
    ap.add_argument("--ham", type=int, default=8, help="指纹汉明阈值，默认 8")
    ap.add_argument("--ar-tol", type=float, default=0.03, help="长宽比容差，默认 0.03")
    ap.add_argument("--mad", type=float, default=9.0, help="像素平均绝对差上限，默认 9")
    ap.add_argument("--corr", type=float, default=0.96, help="相关系数下限，默认 0.96")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    abs_dup = os.path.abspath(args.dup_dir)
    scan_jsonl = os.path.join(args.work, "scan_noface.jsonl")
    done, cache = set(), []
    if os.path.exists(scan_jsonl):
        with open(scan_jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("ok") and r["p"] not in done:
                    done.add(r["p"])
                    cache.append(r)

    paths = [p for p in C.iter_images(args.root) if not os.path.abspath(p).startswith(abs_dup)]
    recs = cache + scan(paths, args.work, args.workers, done)
    print(f"参与比对 {len(recs)} 张")

    groups = C.cluster(recs, ham=args.ham, ar_tol=args.ar_tol)
    print(f"疑似重复组 {len(groups)}")

    cache_px, keep, dups = {}, [], []

    def px(p):
        if p not in cache_px:
            try:
                cache_px[p] = C.pixel_vec(p)
            except Exception:
                cache_px[p] = None
        return cache_px[p]

    for g in groups:
        g.sort(key=C.quality_key, reverse=True)
        anchor = px(g[0]["p"])
        same = [g[0]]
        for m in g[1:]:
            v = px(m["p"])
            if anchor is None or v is None:
                same.append(m)
                continue
            mad, corr = C.verify(anchor, v)
            if mad <= args.mad and corr >= args.corr:
                same.append(m)
        if len(same) > 1:
            keep.append(same[0])
            dups.extend(same[1:])

    print(f"复核后：保留 {len(keep)} 张，重复 {len(dups)} 张，涉及 {len(keep)+len(dups)} 个文件")
    if dups:
        sizes = sorted([len(d) for d in [dups]], reverse=True)
        print(f"重复文件总大小 {sum(d['sz'] for d in dups)/1024/1024/1024:.2f} GB")

    if not args.apply:
        print("\n预演完成，未移动任何文件。加 --apply 执行移动。")
        return

    moved, errs = C.move_files([(d["p"], d) for d in dups], args.dup_dir)
    C.write_manifest(os.path.join(args.work, "move_manifest.csv"),
                     [["dup", s, d, m.get("w", ""), m.get("h", ""), m.get("sz", ""), ""]
                      for s, d, m in moved])
    print(f"已移动 {len(moved)} 张到 {args.dup_dir}（失败 {len(errs)}）")
    for e in errs[:10]:
        print("  ERR", e)


if __name__ == "__main__":
    mp.freeze_support()
    main()
