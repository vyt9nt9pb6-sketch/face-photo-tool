# face-photo-tool

纯本地运行的**人像检索 + 照片查重**小工具，基于 OpenCV（YuNet 人脸检测 + SFace 人脸识别），无需联网、不上传任何照片。

解决两个具体问题：

1. **给一张参考照，把整个目录里这个人的照片全部找出来**，并按画质自动去重（重复的副本单独归到「重复照片」）。
2. **全盘照片查重**，重复的挑出画质最好的一张留下，其余移到指定文件夹。

---

## 安装

```bash
python -m pip install -r requirements.txt
python download_models.py          # 下载 YuNet(0.2MB) + SFace(38MB) 到 ./models
```

> 模型来自 [OpenCV Zoo](https://github.com/opencv/opencv_zoo)，脚本会优先走 hf-mirror 镜像，失败再回退 GitHub。

## 用法一：找出某人的照片

```bash
# 预演（只看结果，不动文件）
python find_person.py --root /path/to/photos --ref reference.jpg --out ./output

# 确认没问题后执行移动
python find_person.py --root /path/to/photos --ref reference.jpg --out ./output --apply
```

结果：

```
./output/              # 去重后画质最好的照片
./output/重复照片/      # 同一张照片的其它副本
整理记录_原路径对照.csv # 每个文件从哪来、到哪去，随时可还原
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--threshold` | 0.42 | 同人判定阈值（余弦相似度），调高更严、调低更松 |
| `--review 0.34 0.42` | — | 把中间档照片导出成清单，人工确认后再决定 |
| `--dup-dir` | `<out>/重复照片` | 重复照片的去处 |
| `--ham` | 10 | 去重时指纹汉明阈值，越大越激进 |
| `--workers` | CPU/2 | 并行进程数 |
| `--work` | `./_fpwork` | 扫描缓存，中断可续跑 |

## 用法二：全盘查重

```bash
python dedupe_photos.py --root /path/to/photos --dup-dir ./duplicates          # 预演
python dedupe_photos.py --root /path/to/photos --dup-dir ./duplicates --apply  # 执行
```

---

## 原理

### 人脸检索

| 步骤 | 做法 |
|---|---|
| 读图 | PIL + pillow-heif（支持 iPhone 的 HEIC），按 EXIF 校正方向 |
| 检测 | YuNet，长边缩到 1100px 再推理，兼顾小脸与速度 |
| 对齐 | 用 5 个关键点做相似变换，裁成 112×112（与 SFace 训练口径一致） |
| 特征 | SFace 输出 128 维向量，L2 归一化后取余弦相似度 |
| 判定 | 相似度 ≥ 0.42 判为同一人；一张图有多张脸时取最高分 |

同一人相似度通常在 0.42 以上；长得较像的人一般低于此值。0.34~0.42 区间建议用 `--review` 人工确认。

### 查重

1. **第一关：感知指纹**。16×16 差分哈希（256 bit），要求汉明距离 ≤ 8 且长宽比差异 ≤ 3%。
2. **第二关：像素复核**。把两张图统一缩到 64×64 灰度，要求平均绝对差 ≤ 9 且相关系数 ≥ 0.96。

只靠感知指纹可能会把版式相似的截图或连拍误判成重复，加上像素复核可降低误判。

**注意事项：**

- 读取失败的文件会被跳过。并发读取时可能出现错误，脚本会重试并列出仍然失败的文件，需要单独处理。
- UI 截图之间不能只看相关系数。对 `.png` 截图建议把阈值收紧到 `--mad 0.5`，或先预演查看分组结果。

### 保留哪一张

排序键：**像素数 → 文件体积 → 拉普拉斯清晰度**。优先留下分辨率最高、压缩损失最小、最锐利的那张。

---

## 说明与限制

- **只移动，不删除**：所有操作都是同盘 rename，速度极快且可逆；整理记录能让你原样搬回去。
- 人脸太小的远景照、被遮挡/大侧脸的照片可能召回不到。
- 证件扫描件、海报、二维码、聊天截图里如果嵌着目标人的证件照，会被算作含此人，是否收进去按需自己筛。
- 加密文件无法读取，会自动跳过。

## 目录

```
fp_core.py          公共能力：模型管理/遍历/指纹/聚类/安全移动
find_person.py      人像检索 + 画质去重
dedupe_photos.py    全盘查重
download_models.py  模型下载
video_dedupe/       视频查重工具（同一套「只移动不删除 + 清单可还原」策略）
```

视频查重单独放在 [`video_dedupe/`](video_dedupe/README.md)：体积+采样哈希 → 全量 MD5 → 时间轴抽帧画面比对，
三层判重，按画质留一份，其余移位并生成可还原清单。

## 许可

MIT。模型权重版权归 OpenCV Zoo 各自作者所有。
