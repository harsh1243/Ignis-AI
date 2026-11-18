"""Detector evaluation at image level, hard-negative mining on normal videos, and the crop dataset
for the verifier (true fire/smoke crops vs. the detector's own false positives)."""
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .common import CLASS_NAMES, IMG_EXTS, VID_EXTS, box_iou, fmt_secs, list_files, read_yolo_label, square_crop


def _imread(p):
    import cv2
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def predict_images(model, paths, conf=0.1, imgsz=640, batch=64, device=0, log_every=2000, what="images"):
    """Yields (path, image, boxes_xyxy_px, cls_ids, confs) for each readable image."""
    t0 = time.time()
    n = len(paths)
    with ThreadPoolExecutor(8) as ex:
        for s in range(0, n, batch):
            chunk = paths[s:s + batch]
            imgs = list(ex.map(_imread, chunk))
            ok = [(p, im) for p, im in zip(chunk, imgs) if im is not None]
            if not ok:
                continue
            kw = dict(conf=conf, imgsz=imgsz, device=device, verbose=False, batch=len(ok))
            if str(device) not in ("cpu",):
                kw["quantize"] = 16
            res = model.predict([im for _, im in ok], **kw)
            for (p, im), r in zip(ok, res):
                if r.boxes is None or len(r.boxes) == 0:
                    yield p, im, np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)
                else:
                    yield (p, im, r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                           r.boxes.conf.cpu().numpy())
            done = min(n, s + batch)
            if done // log_every != (done - len(chunk)) // log_every or done == n:
                el = time.time() - t0
                print(f"   {what}: {done:,}/{n:,} ({100 * done / n:.0f}%) elapsed {fmt_secs(el)} ETA {fmt_secs(el / done * (n - done))}")


def _label_path(img_path):
    p = Path(img_path)
    parts = list(p.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    return Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")


def image_level_eval(weights, data_dir, split="test", device=0, imgsz=640, max_images=None,
                     thresholds=(0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)):
    """How often is fire/smoke found in images that contain it, and how often does the detector fire on
    negative images? (closer to the alarm use case than mAP)"""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    paths = list_files(Path(data_dir) / "images" / split, IMG_EXTS)
    if max_images and len(paths) > max_images:
        paths = random.Random(0).sample(paths, max_images)
    recs = []
    for p, im, bx, cl, cf in predict_images(model, paths, conf=min(thresholds), imgsz=imgsz, device=device, what=f"eval {split}"):
        h, w = im.shape[:2]
        gc, gb = read_yolo_label(_label_path(p), w, h)
        recs.append((np.array(gc, int), gb, bx, cl, cf))
    rows = []
    for t in thresholds:
        c = {"fire_img": 0, "fire_hit": 0, "smoke_img": 0, "smoke_hit": 0, "neg_img": 0, "neg_fp": 0}
        for gc, gb, bx, cl, cf in recs:
            keep = cf >= t
            bx2, cl2 = bx[keep], cl[keep]
            if len(gc) == 0:
                c["neg_img"] += 1
                c["neg_fp"] += int(len(bx2) > 0)
                continue
            for k, name in enumerate(CLASS_NAMES):
                if (gc == k).any():
                    c[f"{name}_img"] += 1
                    iou = box_iou(gb[gc == k], bx2[cl2 == k]) if (cl2 == k).any() else np.zeros((1, 0))
                    c[f"{name}_hit"] += int(iou.size > 0 and iou.max() >= 0.1)
        rows.append({"conf": t,
                     "fire_image_recall": c["fire_hit"] / max(1, c["fire_img"]),
                     "smoke_image_recall": c["smoke_hit"] / max(1, c["smoke_img"]),
                     "negative_image_fp_rate": c["neg_fp"] / max(1, c["neg_img"]),
                     "n_fire_imgs": c["fire_img"], "n_smoke_imgs": c["smoke_img"], "n_neg_imgs": c["neg_img"]})
    print(f"\nImage-level results on '{split}' ({len(recs):,} images)")
    print(f"{'conf':>6} {'fire recall':>12} {'smoke recall':>13} {'FP on negatives':>16}")
    for r in rows:
        print(f"{r['conf']:6.2f} {r['fire_image_recall']:12.3f} {r['smoke_image_recall']:13.3f} {r['negative_image_fp_rate']:16.3f}")
    return rows


def _save_crop(img, box, path, pad=0.3, max_side=256):
    import cv2
    c = square_crop(img, box, pad)
    if c is None or c.shape[0] < 8:
        return False
    if c.shape[0] > max_side:
        c = cv2.resize(c, (max_side, max_side), interpolation=cv2.INTER_AREA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(path), c, [cv2.IMWRITE_JPEG_QUALITY, 92])


def mine_video_false_positives(weights, video_dirs, out_dir, every_s=1.0, conf=0.25, max_per_video=300,
                               device=0, imgsz=640, max_side=960, repeat=2, crop_dir=None, crop_pad=0.3):
    """Run the detector on NORMAL videos (no fire). Every frame where it fires is a hard negative:
    saved as a background image (empty label, `repeat` copies) + its boxes as not_fire crops."""
    import cv2
    from ultralytics import YOLO
    from .data_prep import _resize
    model = YOLO(str(weights))
    out_img, out_lbl = Path(out_dir) / "images" / "mined", Path(out_dir) / "labels" / "mined"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    vids = [v for d in video_dirs for v in list_files(d, VID_EXTS)]
    total_frames = total_fp = 0
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 1 <= fps <= 240 else 25.0
        step = max(1, int(round(fps * every_s)))
        i = n_fp = n_seen = 0
        while n_fp < max_per_video:
            if not cap.grab():
                break
            if i % step == 0:
                ok, fr = cap.retrieve()
                if ok and fr is not None:
                    fr = _resize(fr, max_side)
                    n_seen += 1
                    kw = dict(conf=conf, imgsz=imgsz, device=device, verbose=False)
                    r = model.predict(fr, **kw)[0]
                    if r.boxes is not None and len(r.boxes):
                        n_fp += 1
                        for k in range(repeat):
                            name = f"mined_{v.stem}_f{i:07d}_r{k}"
                            cv2.imwrite(str(out_img / f"{name}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                            (out_lbl / f"{name}.txt").write_text("")
                        if crop_dir:
                            for j, b in enumerate(r.boxes.xyxy.cpu().numpy()):
                                _save_crop(fr, b, Path(crop_dir) / "train" / "not_fire" / f"vid_{v.stem}_{i}_{j}.jpg", crop_pad)
            i += 1
        cap.release()
        total_frames += n_seen
        total_fp += n_fp
        print(f"   {v.name}: {n_seen} frames checked, {n_fp} with false detections")
    print(f"Mined {total_fp} hard-negative frames from {len(vids)} normal videos ({total_frames} frames checked)")
    return total_fp


def build_verifier_dataset(weights, data_dir, out_dir, conf=0.1, device=0, imgsz=640, pad=0.3,
                           max_images={"train": 30000, "val": 4000}, caps={"train": 15000, "val": 2000}, seed=0):
    """Crops for the verifier, per split:
       fire / smoke  : ground-truth boxes + detector boxes that overlap them (IoU >= 0.4)
       not_fire      : detector boxes with IoU < 0.1 to every ground-truth box (its own false positives)."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    rng = random.Random(seed)
    stats = {}
    for split in ("train", "val"):
        paths = list_files(Path(data_dir) / "images" / split, IMG_EXTS)
        if split == "train":
            paths += list_files(Path(data_dir) / "images" / "mined", IMG_EXTS)
        rng.shuffle(paths)
        paths = paths[:max_images[split]]
        cnt = {"fire": 0, "smoke": 0, "not_fire": 0}
        cap = caps[split]
        for p, im, bx, cl, cf in predict_images(model, paths, conf=conf, imgsz=imgsz, device=device, what=f"crops {split}"):
            h, w = im.shape[:2]
            lp = _label_path(p)
            gc, gb = read_yolo_label(lp, w, h)
            stem = Path(p).stem
            for k, (c, b) in enumerate(zip(gc, gb)):          # ground truth crops
                name = CLASS_NAMES[c]
                if cnt[name] < cap and rng.random() < 0.7:
                    cnt[name] += _save_crop(im, b, Path(out_dir) / split / name / f"{stem}_gt{k}.jpg", pad)
            if len(bx):
                iou = box_iou(bx, gb) if len(gb) else np.zeros((len(bx), 0))
                for k, b in enumerate(bx):
                    best = iou[k].max() if iou.shape[1] else 0.0
                    if best >= 0.4:
                        name = CLASS_NAMES[int(gc[int(iou[k].argmax())])]
                        if cnt[name] < cap:
                            cnt[name] += _save_crop(im, b, Path(out_dir) / split / name / f"{stem}_tp{k}.jpg", pad)
                    elif best < 0.1 and cnt["not_fire"] < cap:
                        cnt["not_fire"] += _save_crop(im, b, Path(out_dir) / split / "not_fire" / f"{stem}_fp{k}.jpg", pad)
            if len(gc) == 0 and cnt["not_fire"] < cap and rng.random() < 0.15:   # some random background crops
                s = rng.uniform(0.06, 0.35) * max(w, h)
                x, y = rng.uniform(0, max(1, w - s)), rng.uniform(0, max(1, h - s))
                cnt["not_fire"] += _save_crop(im, [x, y, x + s, y + s], Path(out_dir) / split / "not_fire" / f"{stem}_rnd.jpg", pad)
            if all(v >= cap for v in cnt.values()):
                break
        # existing crops (e.g. mined from videos) count too
        stats[split] = {c: len(list_files(Path(out_dir) / split / c, IMG_EXTS)) for c in ("fire", "smoke", "not_fire")}
        print(f"Verifier crops [{split}]: {stats[split]}")
    (Path(out_dir) / "crop_stats.json").write_text(json.dumps(stats, indent=2))
    return stats



def mine_image_false_positives(weights, image_dir, out_dir, conf=0.25, repeat=2, max_images=None, device=0,
                               imgsz=640, max_side=960, crop_dir=None, crop_pad=0.3):
    """Like the video version, for a folder of FIRE-FREE images: every image on which the detector fires is a
    hard negative (saved as background image, `repeat` copies) and its boxes become not_fire crops."""
    import cv2
    from ultralytics import YOLO
    from .data_prep import _resize
    model = YOLO(str(weights))
    paths = list_files(image_dir, IMG_EXTS)
    if max_images:
        paths = paths[:max_images]
    out_img, out_lbl = Path(out_dir) / "images" / "mined", Path(out_dir) / "labels" / "mined"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    n_fp = 0
    for p, im, bx, cl, cf in predict_images(model, paths, conf=conf, imgsz=imgsz, device=device, what="mining street frames"):
        if len(bx) == 0:
            continue
        n_fp += 1
        fr = _resize(im, max_side)
        sc = fr.shape[1] / im.shape[1]
        for k in range(repeat):
            name = f"mined_img_{Path(p).stem}_r{k}"
            cv2.imwrite(str(out_img / f"{name}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out_lbl / f"{name}.txt").write_text("")
        if crop_dir:
            for j, b in enumerate(bx):
                _save_crop(fr, np.asarray(b) * sc, Path(crop_dir) / "train" / "not_fire" / f"img_{Path(p).stem}_{j}.jpg", crop_pad)
    print(f"Mined {n_fp} hard-negative images out of {len(paths)} fire-free frames")
    return n_fp, len(paths)


def street_fp_rate(weights, image_dir, thresholds=(0.1, 0.2, 0.25, 0.3, 0.4, 0.5), device=0, imgsz=640, max_images=None):
    """Share of fire-free images on which the detector reports anything, per confidence threshold."""
    from ultralytics import YOLO
    paths = list_files(image_dir, IMG_EXTS)
    if max_images:
        paths = paths[:max_images]
    if not paths:
        return None
    model = YOLO(str(weights))
    mc = []
    for p, im, bx, cl, cf in predict_images(model, paths, conf=min(thresholds), imgsz=imgsz, device=device, what="false-alarm test"):
        mc.append(float(cf.max()) if len(cf) else 0.0)
    mc = np.asarray(mc)
    return {"n_images": int(len(mc)), "fp_rate": {str(t): float((mc >= t).mean()) for t in thresholds}}
