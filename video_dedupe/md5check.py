#!/usr/bin/env python3
"""md5check.py —— 第二层判重：对采样哈希相同的组做完整 MD5

用法:
  python md5check.py --index vindex.pkl --out exact.json
  python md5check.py --index vindex.pkl --out exact.json --groups my_groups.json
"""
import argparse
import hashlib
import json
import os
import pickle
import time


def md5(p, chunk=4 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def groups_from_index(rows):
    """按采样哈希分组，只保留 >=2 个文件的组"""
    by = {}
    for r in rows:
        if r.get("sh"):
            by.setdefault(r["sh"], []).append(r["p"])
    return [v for v in by.values() if len(v) > 1]


def main():
    ap = argparse.ArgumentParser(description="完整 MD5 确认字节一致的视频组")
    ap.add_argument("--index", required=True, help="vindex.py 输出的 pickle")
    ap.add_argument("--out", default="exact.json", help="字节相同组输出")
    ap.add_argument("--groups", help="自定义候选组 JSON（[[路径,...], ...]），省略则用索引自动分组")
    a = ap.parse_args()

    if a.groups:
        groups = json.load(open(a.groups, encoding="utf-8"))
    else:
        rows = pickle.load(open(a.index, "rb"))
        groups = groups_from_index(rows)
        print(f"采样哈希相同的组: {len(groups)}")

    files = [p for g in groups for p in g]
    if not files:
        print("没有候选组，无需 MD5")
        json.dump([], open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return

    tot = sum(os.path.getsize(p) for p in files)
    print(f"校验 {len(files)} 个文件，共 {tot / 1024 ** 3:.2f} GB", flush=True)

    hashes = {}
    done = 0
    t0 = time.time()
    for i, p in enumerate(files, 1):
        try:
            hashes[p] = md5(p)
        except OSError as e:
            hashes[p] = f"ERR:{e}"
        done += os.path.getsize(p)
        if i % 10 == 0:
            el = time.time() - t0
            sp = done / el / 1024 / 1024 if el else 0
            rest = (tot - done) / max(sp, 1) / 60
            print(f"  {i}/{len(files)}  {el:.0f}s  {sp:.1f} MB/s  剩余约 {rest:.1f} min", flush=True)

    out = []
    for g in groups:
        by = {}
        for p in g:
            by.setdefault(hashes.get(p), []).append(p)
        for h, v in by.items():
            if len(v) > 1:
                out.append({"md5": h, "files": v, "sz": os.path.getsize(v[0])})

    movable = sum(len(x["files"]) - 1 for x in out)
    print(f"\n字节完全相同的组: {len(out)}，涉及 {sum(len(x['files']) for x in out)} 个，可移走 {movable}")
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("已写入", a.out)


if __name__ == "__main__":
    main()
