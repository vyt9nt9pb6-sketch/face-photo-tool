#!/usr/bin/env python3
"""apply.py —— 执行移动计划，生成可还原清单

只移动，不删除。同盘 move 是即时改名，不占额外空间、可秒级还原。

用法:
  python apply.py --plan plan.json --dup-dir ./重复视频
  python apply.py --plan plan.json --dup-dir ./重复视频 --index vindex.pkl
"""
import argparse
import csv
import json
import os
import pickle
import shutil
import time

FIELDS = ["原路径", "新路径", "保留路径", "类型", "字节", "分辨率", "时长秒", "帧率", "说明"]


def free_name(dst_dir, base):
    """目标重名时追加 _dup1/_dup2，绝不覆盖"""
    dst = os.path.join(dst_dir, base)
    if not os.path.exists(dst):
        return dst
    stem, ext = os.path.splitext(base)
    n = 1
    while os.path.exists(os.path.join(dst_dir, f"{stem}_dup{n}{ext}")):
        n += 1
    return os.path.join(dst_dir, f"{stem}_dup{n}{ext}")


def main():
    ap = argparse.ArgumentParser(description="执行视频去重移动")
    ap.add_argument("--plan", required=True, help="decide.py 输出的计划")
    ap.add_argument("--dup-dir", required=True, help="重复视频的去向目录")
    ap.add_argument("--index", help="vindex.py 输出的 pickle（用于补充分辨率等元信息）")
    ap.add_argument("--manifest", default="video_dedupe_manifest.csv", help="清单输出")
    ap.add_argument("--apply", action="store_true", help="真正执行；不加则只预演")
    a = ap.parse_args()

    plan = json.load(open(a.plan, encoding="utf-8"))
    probe = {}
    if a.index and os.path.exists(a.index):
        probe = {r["p"]: r for r in pickle.load(open(a.index, "rb"))}

    if not a.apply:
        total = sum(r.get("size", 0) for r in plan)
        print(f"[预演] 将移动 {len(plan)} 个文件，共 {total / 1024 ** 3:.2f} GB")
        print(f"[预演] 去向: {os.path.abspath(a.dup_dir)}")
        for r in plan[:15]:
            print(f"   {r['src']}  ->  {r['keep']}")
        if len(plan) > 15:
            print(f"   ... 另外 {len(plan) - 15} 条")
        print("\n确认无误后加 --apply 执行")
        return

    os.makedirs(a.dup_dir, exist_ok=True)
    rows, failed = [], []
    t0 = time.time()
    for i, r in enumerate(plan, 1):
        src, keep = r["src"], r["keep"]
        if not os.path.exists(src):
            failed.append((src, "源文件不存在"))
            continue
        dst = free_name(a.dup_dir, os.path.basename(src))
        try:
            size = os.path.getsize(src)
            shutil.move(src, dst)
        except Exception as e:
            failed.append((src, str(e)))
            continue
        pr = probe.get(src, {}) or {}
        rows.append({
            "原路径": src, "新路径": dst, "保留路径": keep,
            "类型": r.get("kind", ""), "字节": size,
            "分辨率": f"{pr.get('w', '')}x{pr.get('h', '')}" if pr else "",
            "时长秒": pr.get("dur", ""), "帧率": pr.get("fps", ""),
            "说明": r.get("note", ""),
        })
        if i % 20 == 0:
            print(f"  {i}/{len(plan)}  {time.time() - t0:.0f}s", flush=True)

    with open(a.manifest, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    moved_bytes = sum(r["字节"] for r in rows)
    print(f"\n移动成功 {len(rows)}，失败 {len(failed)}，共 {moved_bytes / 1024 ** 3:.2f} GB，"
          f"用时 {time.time() - t0:.0f}s")
    for s, e in failed[:10]:
        print("  失败", s, e)
    print("清单:", os.path.abspath(a.manifest))


if __name__ == "__main__":
    main()
