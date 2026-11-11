#!/usr/bin/env python3
"""restore.py —— 按清单把移走的视频搬回原位

用法:
  python restore.py --manifest video_dedupe_manifest.csv           # 预演
  python restore.py --manifest video_dedupe_manifest.csv --apply   # 执行
"""
import argparse
import csv
import os
import shutil


def main():
    ap = argparse.ArgumentParser(description="按清单还原视频位置")
    ap.add_argument("--manifest", required=True, help="apply.py 生成的清单 CSV")
    ap.add_argument("--apply", action="store_true", help="真正执行；不加则只预演")
    a = ap.parse_args()

    with open(a.manifest, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    ok, skip, miss = [], [], []
    for r in rows:
        src, dst = r["原路径"], r["新路径"]
        if not os.path.exists(dst):
            miss.append((dst, "清单里的新路径已不存在"))
            continue
        if os.path.exists(src):
            skip.append((src, "原路径已被占用，跳过以免覆盖"))
            continue
        ok.append((dst, src))

    print(f"清单 {len(rows)} 条：可还原 {len(ok)}，跳过 {len(skip)}，缺失 {len(miss)}")
    for p, why in (skip + miss)[:10]:
        print("  ", why, p)
    if not a.apply:
        for dst, src in ok[:10]:
            print(f"   {dst}  ->  {src}")
        print("\n确认无误后加 --apply 执行")
        return

    done = failed = 0
    for dst, src in ok:
        try:
            os.makedirs(os.path.dirname(src), exist_ok=True)
            shutil.move(dst, src)
            done += 1
        except Exception as e:
            failed += 1
            print("  失败", src, e)
    print(f"\n已还原 {done}，失败 {failed}")


if __name__ == "__main__":
    main()
