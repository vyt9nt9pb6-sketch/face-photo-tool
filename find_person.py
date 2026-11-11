# -*- coding: utf-8 -*-
"""find_person.py —— 从整个磁盘/目录里找出某个人的全部照片，并按画质自动去重

用法示例:
  python find_person.py --root D:\\ --ref 参考照.jpg --out D:\\人物照片 --apply
  python find_person.py --root D:\\ --ref ref.jpg --out D:\\人物照片 --threshold 0.45

流程:
  1) 遍历所有图片，YuNet 检测人脸 -> 五点对齐裁成 112x112
  2) SFace 提 128 维特征，与参考照算余弦相似度
  3) 相似度 >= 阈值判定为同一人（建议 0.42，可用 --review 导出中间档人工确认）
  4) 命中的照片内部按感知指纹聚类去重，保留画质最好的一张
  5) --apply 时移动文件（保留者进 --out，重复者进 --dup-dir）
"""
import os
import sys
import json
import time
import argparse
import multiprocessing as mp

import numpy as np
import cv2

import fp_core as C


def scan_all(paths, work, yunet, workers, done):
    """多进程扫描，结果写入 work/scan.jsonl（可断点续跑）"""
    crops = os.path.join(work, "crops")
    os.makedirs(crops, exist_ok=True)
    out_path = os.path.join(work, "scan.jsonl")
    todo = [p for p in paths if p not in done]
    print(f"待扫描 {len(todo)} / 共 {len(paths)}")
    t0 = time.time()
    recs = []
    with open(out_path, "a", encoding="utf-8") as out, mp.Pool(
            workers, initializer=C.worker_init, initargs=(yunet,)) as pool:
        for i, rec in enumerate(pool.imap_unordered(_task, [(p, crops) for p in todo], chunksize=8), 1):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            recs.append(rec)
            if i % 200 == 0:
                out.flush()
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  已用 {el/60:.1f} min  剩余约 {el/i*(len(todo)-i)/60:.1f} min")
        out.flush()
    return recs


def _task(args):
    p, crops = args
    return C.analyze(p, crop_dir=crops)


def ref_embedding(ref_path, yunet, sface, work=None):
    """从参考照片里取最大人脸的特征"""
    rgb = np.asarray(C.load_rgb(ref_path))
    det = C.get_detector(yunet)
    h, w = rgb.shape[:2]
    scale = min(1.0, 1100.0 / max(w, h))
    img = rgb if scale == 1.0 else cv2.resize(rgb, (int(w * scale), int(h * scale)))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    det.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, faces = det.detect(bgr)
    if faces is None or not len(faces):
        raise SystemExit("参考照片里没有检测到人脸，换一张正脸清晰的")
    f = max(faces, key=lambda x: x[2] * x[3])
    lms = np.array(f[4:14], np.float32).reshape(5, 2) / scale
    crop = C.align_crop(rgb, lms)
    dst_dir = work or os.path.dirname(os.path.abspath(ref_path))
    C.imwrite_unicode(os.path.join(dst_dir, "ref_aligned.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    rec = C.get_recognizer(sface)
    return C.l2norm(rec.feature(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))), crop


def embed_crops(recs, work, sface, batch=128):
    rec = C.get_recognizer(sface)
    crops_dir = os.path.join(work, "crops")
    names, feats = [], []
    todo = [(r["p"], f["crop"]) for r in recs for f in r.get("faces", [])]
    print(f"人脸数 {len(todo)}")
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        arr = []
        for p, c in chunk:
            img = C.imread_unicode(os.path.join(crops_dir, c))
            if img is None:
                continue
            try:
                arr.append((p, C.l2norm(rec.feature(img))[0]))
            except cv2.error:
                continue
        if not arr:
            continue
        F = np.stack([a[1] for a in arr])
        names.extend([a[0] for a in arr])
        feats.append(F)
        if (i // batch) % 4 == 0:
            print(f"  特征 {i}/{len(todo)}", flush=True)
    return names, (np.vstack(feats) if feats else np.zeros((0, 128), np.float32))


def main():
    ap = argparse.ArgumentParser(description="按参考照片找人像并去重整理")
    ap.add_argument("--root", required=True, help="要搜索的根目录，如 D:\\")
    ap.add_argument("--ref", required=True, help="参考照片（这个人的一张清晰正脸照）")
    ap.add_argument("--out", required=True, help="命中的照片移动到这里")
    ap.add_argument("--dup-dir", default=None, help="重复照片移到这里，默认 <out>/重复照片")
    ap.add_argument("--work", default="./_fpwork", help="缓存目录（可断点续跑）")
    ap.add_argument("--model-dir", default="./models")
    ap.add_argument("--threshold", type=float, default=0.42, help="同人判定阈值，默认 0.42")
    ap.add_argument("--ham", type=int, default=10, help="去重指纹汉明阈值，默认 10")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--apply", action="store_true", help="真正移动文件（不加只做预演）")
    ap.add_argument("--review", nargs=2, metavar=("LO", "HI"),
                    help="把该相似度区间的照片导出成清单，供人工核对")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    dup_dir = args.dup_dir or os.path.join(args.out, "重复照片")

    models = C.ensure_models(args.model_dir)
    yunet, sface = models["yunet.onnx"], models["sface.onnx"]

    scan_jsonl = os.path.join(args.work, "scan.jsonl")
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
        print(f"复用缓存 {len(cache)} 条")

    paths = [p for p in C.iter_images(args.root) if not p.startswith(os.path.abspath(args.out))]
    new = scan_all(paths, args.work, yunet, args.workers, done)
    recs = cache + [r for r in new if r.get("ok")]
    print(f"可用记录 {len(recs)}")

    ref_f, _ = ref_embedding(args.ref, yunet, sface, work=args.work)
    names, F = embed_crops(recs, args.work, sface)
    if not len(F):
        raise SystemExit("没有可用人脸特征")
    sims = F @ ref_f[0]
    best = {}
    for p, s in zip(names, sims.tolist()):
        if s > best.get(p, -1):
            best[p] = s

    scored = []
    for r in recs:
        s = best.get(r["p"])
        if s is None:
            continue
        scored.append({**{k: r.get(k) for k in ("p", "w", "h", "sz", "sharp", "dh", "mt")},
                       "sim": round(float(s), 4), "nfaces": len(r.get("faces", []))})
    scored.sort(key=lambda x: -x["sim"])
    json.dump(scored, open(os.path.join(args.work, "match_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    arr = np.array([s["sim"] for s in scored])
    print("\n相似度分布（命中候选）:")
    for lo in [0.80, 0.70, 0.60, 0.50, 0.42, 0.38, 0.34]:
        print(f"  >= {lo:.2f}: {int((arr >= lo).sum())}")

    if args.review:
        lo, hi = float(args.review[0]), float(args.review[1])
        band = [s for s in scored if lo <= s["sim"] < hi]
        p = os.path.join(args.work, f"review_{lo}_{hi}.json")
        json.dump(band, open(p, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"待人工核对 {len(band)} 张 -> {p}")

    matched = [s for s in scored if s["sim"] >= args.threshold]
    print(f"命中 {len(matched)} 张（阈值 {args.threshold}）")

    groups = C.cluster(matched, ham=args.ham)
    keep, dups = [], []
    for g in groups:
        g.sort(key=C.quality_key, reverse=True)
        keep.append(g[0])
        dups.extend(g[1:])
    kept = {k["p"] for k in keep}
    keep += [m for m in matched if m["p"] not in kept]
    print(f"去重后保留 {len(keep)} 张，重复 {len(dups)} 张")

    if not args.apply:
        print("\n预演完成，未移动任何文件。加 --apply 执行移动。")
        return

    moved_k, err_k = C.move_files([(k["p"], k) for k in keep], args.out)
    moved_d, err_d = C.move_files([(d["p"], d) for d in dups], dup_dir)
    rows = [["keep", s, d, m.get("w", ""), m.get("h", ""), m.get("sz", ""), m.get("sim", "")]
            for s, d, m in moved_k + moved_d]
    C.write_manifest(os.path.join(os.path.dirname(os.path.normpath(args.out)), "整理记录_原路径对照.csv"), rows)
    print(f"已移动 {len(rows)} 张（失败 {len(err_k) + len(err_d)}）")


if __name__ == "__main__":
    mp.freeze_support()
    main()
