# -*- coding: utf-8 -*-
"""fp_core.py —— 公共能力：模型管理 / 图片遍历 / 感知指纹 / 聚类 / 安全移动

依赖: opencv-contrib-python-headless, numpy, pillow, pillow-heif
"""
import os
import json
import math
import hashlib
import shutil
import urllib.request

import numpy as np
import cv2
import pillow_heif
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()
Image.MAX_IMAGE_PIXELS = None

# OpenCV Zoo 官方模型：YuNet 人脸检测 + SFace 人脸识别
MODELS = {
    "yunet.onnx": [
        "https://hf-mirror.com/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ],
    "sface.onnx": [
        "https://hf-mirror.com/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ],
}
MODEL_SIZES = {"yunet.onnx": 232589, "sface.onnx": 38696353}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
SKIP_DIRS = {"system volume information", "$recycle.bin", "recycler"}

# ArcFace / SFace 112x112 五点参考坐标
REF_PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# ---------------------------------------------------------------- 模型
def ensure_models(model_dir, quiet=False):
    """缺失则下载模型（先国内镜像，失败再走 GitHub）"""
    os.makedirs(model_dir, exist_ok=True)
    paths = {}
    for name, urls in MODELS.items():
        dst = os.path.join(model_dir, name)
        want = MODEL_SIZES.get(name, 0)
        if os.path.exists(dst) and os.path.getsize(dst) >= want * 0.99:
            paths[name] = dst
            continue
        ok = False
        for url in urls:
            try:
                if not quiet:
                    print(f"下载 {name} <- {url.split('/')[2]}")
                tmp = dst + ".part"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(262144)
                        if not chunk:
                            break
                        f.write(chunk)
                os.replace(tmp, dst)
                ok = True
                break
            except Exception as e:
                if not quiet:
                    print(f"  失败: {type(e).__name__} {str(e)[:80]}")
        if not ok:
            raise RuntimeError(f"模型下载失败: {name}，请手动下载后放入 {model_dir}")
        paths[name] = dst
    return paths


_det = None
_rec = None


def get_detector(model_path):
    global _det
    if _det is None:
        _det = cv2.FaceDetectorYN.create(model_path, "", (640, 640), 0.6, 0.3, 5000)
    return _det


def get_recognizer(model_path):
    global _rec
    if _rec is None:
        _rec = cv2.FaceRecognizerSF.create(model_path, "")
    return _rec


def l2norm(v):
    v = np.asarray(v, dtype=np.float32).reshape(1, -1)
    return v / (np.linalg.norm(v) + 1e-9)


# ---------------------------------------------------------------- 遍历 / 读取
def iter_images(root, skip_dirs=SKIP_DIRS):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d.lower() not in skip_dirs and not d.startswith("$")]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                yield os.path.join(dirpath, fn)


def load_rgb(path):
    """读图并校正 EXIF 方向，统一转 RGB"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return im


def imread_unicode(path):
    """cv2.imread 不支持中文路径，用 numpy 中转"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, bgr, quality=95):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(path)
    return ok


# ---------------------------------------------------------------- 指纹
def dhash(gray, size=16):
    """16x16 差值哈希 -> 256bit 十六进制串"""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return np.packbits(diff.flatten()).tobytes().hex()


def sharpness(gray, cap=800):
    g = gray
    if max(g.shape) > cap:
        f = cap / max(g.shape)
        g = cv2.resize(g, (int(g.shape[1] * f), int(g.shape[0] * f)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def hamming_hex(a, b):
    if not a or not b or len(a) != len(b):
        return 999
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def align_crop(rgb, landmarks):
    """按五点做相似变换，输出 112x112 对齐人脸"""
    M, _ = cv2.estimateAffinePartial2D(np.asarray(landmarks, np.float32), REF_PTS, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(rgb, M, (112, 112), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def quality_key(rec):
    """画质排序键：像素数 > 文件体积 > 清晰度"""
    return (rec.get("w", 0) * rec.get("h", 0), rec.get("sz", 0), rec.get("sharp", 0))


# ---------------------------------------------------------------- 单图分析（子进程用）
_DET_PATH = None
_REC_PATH = None


def worker_init(yunet_path, sface_path=None):
    global _DET_PATH, _REC_PATH
    _DET_PATH, _REC_PATH = yunet_path, sface_path
    get_detector(yunet_path)
    if sface_path:
        get_recognizer(sface_path)


def analyze(path, crop_dir=None, max_faces=4, det_side=1100, detect=True):
    """检测人脸 + 计算指纹，可选写出对齐人脸裁剪；detect=False 时只算指纹"""
    rec = {"p": path}
    try:
        st = os.stat(path)
        rec["sz"] = st.st_size
        rec["mt"] = int(st.st_mtime)
        pil = load_rgb(path)
        W, H = pil.size
        rec["w"], rec["h"] = W, H
        rgb = np.asarray(pil)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rec["dh"] = dhash(gray)
        rec["sharp"] = round(sharpness(gray), 2)

        if not detect:
            rec["faces"] = []
            rec["ok"] = 1
            return rec

        det = get_detector(_DET_PATH)
        scale = 1.0
        if max(W, H) > det_side:
            scale = det_side / max(W, H)
        img = rgb if scale == 1.0 else cv2.resize(
            rgb, (int(round(W * scale)), int(round(H * scale))), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        det.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = det.detect(bgr)

        out = []
        if faces is not None and len(faces):
            faces = sorted(faces, key=lambda f: -f[4] * f[2] * f[3])[:max_faces]
            for i, f in enumerate(faces):
                lms = np.array(f[4:14], np.float32).reshape(5, 2) / scale
                crop = align_crop(rgb, lms)
                if crop is None or crop_dir is None:
                    continue
                cid = hashlib.md5(path.encode("utf-8")).hexdigest()[:16] + f"_{i}.jpg"
                imwrite_unicode(os.path.join(crop_dir, cid), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                cg = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                out.append({
                    "crop": cid,
                    "score": round(float(f[14]), 4),
                    "face_sharp": round(float(cv2.Laplacian(cg, cv2.CV_64F).var()), 2),
                    "face_px": int(round(f[2] / scale)),
                })
        rec["faces"] = out
        rec["ok"] = 1
    except Exception as e:
        rec["ok"] = 0
        rec["err"] = f"{type(e).__name__}: {str(e)[:120]}"
    return rec


# ---------------------------------------------------------------- 聚类
def dh_ints(h):
    b = bytes.fromhex(h)
    return [int.from_bytes(b[i:i + 8], "big") for i in range(0, 32, 8)]


def cluster(records, ham=10, ar_tol=0.03, max_bucket=400):
    """感知指纹聚类

    召回策略（两道都做，取并集）：
      1) LSH：256bit 切成 8 段 32bit，任一段相同即成为候选。
         汉明距离 <=7 时必有一段完全相同（鸽巢原理），因此不会漏。
      2) 同文件字节数分桶：完全相同的文件大小必然一致，作为兜底。
    再用精确汉明距离 + 长宽比容差过滤。
    """
    n = len(records)
    if n == 0:
        return []
    K = np.array([[int.from_bytes(bytes.fromhex(r["dh"])[i:i + 4], "big")
                   for i in range(0, 32, 4)] for r in records], dtype=np.uint64)

    pairs = set()

    def add_pairs(idxs):
        if 1 < len(idxs) <= max_bucket:
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    pairs.add((idxs[a], idxs[b]))

    # 1) 单段 LSH
    for c in range(8):
        buckets = {}
        for i in range(n):
            buckets.setdefault(int(K[i, c]), []).append(i)
        for idxs in buckets.values():
            add_pairs(idxs)

    # 2) 同字节数兜底
    by_size = {}
    for i, r in enumerate(records):
        by_size.setdefault(r.get("sz"), []).append(i)
    for idxs in by_size.values():
        add_pairs(idxs)

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ars = [r["w"] / max(1, r["h"]) for r in records]
    for i, j in pairs:
        if abs(math.log(ars[i] / ars[j])) > ar_tol:
            continue
        x = K[i] ^ K[j]
        if int(sum(bin(int(v)).count("1") for v in x)) <= ham:
            a, b = find(i), find(j)
            if a != b:
                parent[b] = a

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(records[i])
    return [g for g in groups.values() if len(g) > 1]


def pixel_vec(path, n=64):
    """64x64 灰度向量，用于像素级复核"""
    im = load_rgb(path).convert("L").resize((n, n), Image.BOX)
    return np.asarray(im, dtype=np.float32).ravel()


def verify(a, b):
    """返回 (平均绝对差, 相关系数)"""
    ma, mb = a - a.mean(), b - b.mean()
    corr = float(np.dot(ma, mb) / ((np.linalg.norm(ma) * np.linalg.norm(mb)) + 1e-6))
    return float(np.mean(np.abs(a - b))), corr


# ---------------------------------------------------------------- 移动
def unique_dest(dst_dir, name, used):
    stem, ext = os.path.splitext(name)
    cand, i = name, 1
    while cand.lower() in used or os.path.exists(os.path.join(dst_dir, cand)):
        cand = f"{stem}_{i}{ext}"
        i += 1
    used.add(cand.lower())
    return os.path.join(dst_dir, cand)


def move_files(pairs, dst_dir, tag="dup"):
    """pairs: [(src, meta)]；返回 (成功列表[(src,dst)], 错误列表)"""
    os.makedirs(dst_dir, exist_ok=True)
    used, moved, errs = set(), [], []
    for src, meta in pairs:
        dst = unique_dest(dst_dir, os.path.basename(src), used)
        try:
            shutil.move(src, dst)
            moved.append((src, dst, meta))
        except Exception as e:
            errs.append((src, str(e)[:100]))
    return moved, errs


def write_manifest(path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["类型", "原路径", "新路径", "宽", "高", "字节", "相似度"])
        w.writerows(rows)
