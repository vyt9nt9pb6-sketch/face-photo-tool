# -*- coding: utf-8 -*-
"""下载 OpenCV Zoo 的人脸检测/识别模型（YuNet + SFace）到 ./models"""
import argparse
import fp_core as C

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="./models")
    args = ap.parse_args()
    paths = C.ensure_models(args.model_dir)
    for k, v in paths.items():
        print(f"{k}: {v}  ({__import__('os').path.getsize(v)/1024/1024:.1f} MB)")
