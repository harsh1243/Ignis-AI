"""Shared helpers: constants, config loading, crop extraction, timing."""
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import yaml

CLASS_NAMES = ["fire", "smoke"]          # final detector class ids: 0 = fire, 1 = smoke
EVENT_LABELS = ("fire", "smoke")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv", ".webm", ".3gp", ".ts"}

# Names found in other datasets that we map onto our two classes. Anything else is dropped.
LABEL_ALIASES = {
    "fire": "fire", "flame": "fire", "flames": "fire", "fire_flame": "fire", "burning": "fire",
    "smoke": "smoke", "smokes": "smoke", "fumes": "smoke",
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def fmt_secs(s):
    s = int(max(0, s))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def resolve_path(p, base_dir):
    """Paths in config.yaml are relative to the config file's folder."""
    if p is None:
        return None
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else (Path(base_dir) / p).resolve()


def list_files(root, exts):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts
                  and "__MACOSX" not in p.parts and not p.name.startswith("._"))


def square_crop(img, box, pad=0.3, min_side=16):
    """Square crop centred on box (x1,y1,x2,y2 in pixels) with `pad` context on each side.
    Out-of-image areas are filled by edge replication. Identical at training and inference."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, 1.0) * (1 + 2 * pad)
    side = max(side, min_side)
    half = side / 2
    a, b = int(math.floor(cx - half)), int(math.floor(cy - half))
    c, d = int(math.ceil(cx + half)), int(math.ceil(cy + half))
    pl, pt, pr, pb = max(0, -a), max(0, -b), max(0, c - w), max(0, d - h)
    crop = img[max(0, b):min(h, d), max(0, a):min(w, c)]
    if crop.size == 0:
        return None
    if pl or pt or pr or pb:
        import cv2
        crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    return crop


def box_iou(a, b):
    """IoU between arrays of boxes a (N,4) and b (M,4) in xyxy -> (N,M)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def read_yolo_label(path, w=1.0, h=1.0):
    """Read a YOLO label file -> (classes list, boxes xyxy scaled by w,h)."""
    cls, boxes = [], []
    p = Path(path)
    if not p.exists():
        return cls, np.zeros((0, 4), dtype=np.float32)
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
        cls.append(c)
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return cls, np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0
