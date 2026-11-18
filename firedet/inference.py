"""Inference components: detector, crop verifier and the sequential video runner that writes
predictions.jsonl, alarms.jsonl and run_summary.json."""
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from .common import CLASS_NAMES, square_crop
from .temporal import AlarmManager

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_device(device):
    """'cuda:0' | 'cuda' | '0' | 'cpu' | 'mps' -> (torch_device_str, warning or None)."""
    import torch
    d = str(device or "cpu").strip().lower()
    if d.isdigit():
        d = f"cuda:{d}"
    if d == "cuda":
        d = "cuda:0"
    if d.startswith("cuda"):
        if not torch.cuda.is_available():
            return "cpu", f"requested {device} but CUDA is not available - falling back to CPU"
        idx = int(d.split(":")[1]) if ":" in d else 0
        if idx >= torch.cuda.device_count():
            return "cuda:0", f"requested {device} but only {torch.cuda.device_count()} GPU(s) - using cuda:0"
        return d, None
    if d == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", None
        return "cpu", "requested mps but it is not available - falling back to CPU"
    if d != "cpu":
        return "cpu", f"unknown device '{device}' - using CPU"
    return "cpu", None


class Detector:
    def __init__(self, weights, device="cpu", imgsz=640, conf=0.2, iou=0.5, max_det=50, half=True):
        from ultralytics import YOLO
        self.model = YOLO(str(weights), task="detect")
        self.device = device
        self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, verbose=False)
        if half and device.startswith("cuda"):
            self.kw["quantize"] = 16
        names = self.model.names
        self.id2label = {int(k): str(v).lower() for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}

    def __call__(self, frame_bgr):
        r = self.model.predict(frame_bgr, **self.kw)[0]
        out = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for b, c, k in zip(xyxy, conf, cls):
            label = self.id2label.get(int(k), str(k))
            if label not in CLASS_NAMES:
                continue
            out.append({"label": label, "det_confidence": float(c), "bbox": [float(v) for v in b]})
        return out


class Verifier:
    """EfficientNet-B0 crop classifier -> probability that a detected box is real fire/smoke."""

    def __init__(self, weights, device="cpu"):
        import torch
        import torchvision
        ck = torch.load(str(weights), map_location="cpu", weights_only=False)
        self.classes = list(ck["classes"])
        self.imgsz = int(ck.get("imgsz", 224))
        self.pad = float(ck.get("pad", 0.3))
        m = torchvision.models.efficientnet_b0(weights=None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, len(self.classes))
        m.load_state_dict(ck["state_dict"])
        self.model = m.to(device).eval()
        self.device = device
        self.event_idx = [i for i, c in enumerate(self.classes) if c in CLASS_NAMES]
        self.torch = torch

    def preprocess(self, crops):
        import cv2
        arr = []
        for c in crops:
            c = cv2.resize(c, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA if c.shape[0] > self.imgsz else cv2.INTER_LINEAR)
            c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            arr.append((c - IMAGENET_MEAN) / IMAGENET_STD)
        x = np.stack(arr).transpose(0, 3, 1, 2)
        return self.torch.from_numpy(np.ascontiguousarray(x)).to(self.device)

    def __call__(self, frame_bgr, boxes):
        """Returns array (N, n_classes) of class probabilities for each box."""
        if not boxes:
            return np.zeros((0, len(self.classes)), dtype=np.float32)
        crops = [square_crop(frame_bgr, b, self.pad) for b in boxes]
        valid = [i for i, c in enumerate(crops) if c is not None]
        probs = np.zeros((len(boxes), len(self.classes)), dtype=np.float32)
        if not valid:
            return probs
        with self.torch.inference_mode():
            x = self.preprocess([crops[i] for i in valid])
            p = self.torch.softmax(self.model(x).float(), 1).cpu().numpy()
        probs[valid] = p
        return probs

    def event_prob(self, probs):
        return probs[:, self.event_idx].sum(1) if len(probs) else np.zeros(0)


class Pipeline:
    """Detector + verifier for a single frame (no temporal state)."""

    def __init__(self, cfg, base_dir, device):
        from .common import resolve_path
        m, d = cfg["models"], cfg["detector"]
        self.cfg = cfg
        self.detector = Detector(resolve_path(m["detector"], base_dir), device, imgsz=cfg["input"]["imgsz"],
                                 conf=d["conf"], iou=d.get("iou", 0.5), max_det=d.get("max_det", 50),
                                 half=d.get("half", True))
        self.verifier = None
        if m.get("use_verifier", True) and m.get("verifier"):
            vp = resolve_path(m["verifier"], base_dir)
            if vp.exists():
                self.verifier = Verifier(vp, device)
            else:
                print(f"WARNING: verifier weights {vp} not found - running detector only")
        v = cfg.get("verifier", {})
        self.min_prob = float(v.get("min_prob", 0.5))
        self.min_box = float(cfg["detector"].get("min_box_px", 4))

    def raw(self, frame):
        """Detections with verifier probability attached (nothing filtered by the verifier yet)."""
        dets = [d for d in self.detector(frame)
                if (d["bbox"][2] - d["bbox"][0]) >= self.min_box and (d["bbox"][3] - d["bbox"][1]) >= self.min_box]
        if self.verifier is not None and dets:
            pe = self.verifier.event_prob(self.verifier(frame, [d["bbox"] for d in dets]))
            for d, p in zip(dets, pe):
                d["verifier_prob"] = float(p)
        else:
            for d in dets:
                d["verifier_prob"] = 1.0
        return dets

    @staticmethod
    def finalize(dets, min_prob):
        """Keep verified detections; confidence = geometric mean of detector and verifier scores."""
        out = []
        for d in dets:
            if d["verifier_prob"] < min_prob:
                continue
            e = dict(d)
            e["confidence"] = float(np.sqrt(d["det_confidence"] * d["verifier_prob"]))
            out.append(e)
        return out


def _draw(frame, dets, alarms_active):
    import cv2
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        col = (0, 0, 255) if d["label"] == "fire" else (200, 200, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"{d['label']} {d['confidence']:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    if alarms_active:
        cv2.putText(frame, "ALARM: " + ",".join(sorted(alarms_active)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return frame


def run_video(source, pipeline, cfg, f_pred, f_alarm, save_video_dir=None, log=print):
    """Process one video sequentially. Writes records to the open files. Returns a summary dict."""
    import cv2
    t0 = time.time()
    source = Path(source)
    sid = source.stem
    s = dict(source=str(source), source_id=sid, status="failed", frames_read=0, frames_processed=0,
             frames_skipped=0, frames_failed=0, alarms=0, fps=None, errors=[])
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        s["errors"].append(f"cannot open video: {source}")
        s["runtime_s"] = round(time.time() - t0, 3)
        return s
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_ok = 1.0 <= fps <= 240.0
    if not fps_ok:
        s["errors"].append(f"invalid FPS metadata ({fps}); using timestamps from the decoder / 25 fps")
        fps = 25.0
    s["fps"] = round(float(fps), 3)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = float(cfg["input"].get("target_fps", 5))
    stride = max(1, int(round(fps / target))) if target > 0 else 1
    s["frame_stride"] = stride
    temporal = AlarmManager(cfg["temporal"], sid, sample_fps=fps / stride)
    min_prob = float(cfg.get("verifier", {}).get("min_prob", 0.5))
    log_unverified = bool(cfg.get("output", {}).get("log_unverified", False))
    max_frames = cfg["input"].get("max_frames")
    writer = None
    active = {}
    frame_index, consecutive_fail = -1, 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # premature failure? try to skip over a corrupt frame a few times
            if n_total and frame_index + 1 < n_total - 1 and consecutive_fail < 25:
                consecutive_fail += 1
                frame_index += 1
                s["frames_failed"] += 1
                continue
            break
        consecutive_fail = 0
        frame_index += 1
        s["frames_read"] += 1
        if max_frames and frame_index >= max_frames:
            break
        if frame_index % stride != 0:
            s["frames_skipped"] += 1
            continue
        ts = frame_index / fps
        if not fps_ok:
            pos = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = pos / 1000.0 if pos and pos > 0 else frame_index / fps
        try:
            raw = pipeline.raw(frame)
            dets = Pipeline.finalize(raw, min_prob)
            alarms = temporal.update(frame_index, ts, dets)
            preds = [{"label": d["label"], "confidence": round(d["confidence"], 4),
                      "bbox_xyxy": [round(v, 1) for v in d["bbox"]],
                      "det_confidence": round(d["det_confidence"], 4),
                      "verifier_prob": round(d["verifier_prob"], 4),
                      "track_id": d.get("track_id")} for d in dets]
            rec = {"source_id": sid, "frame_index": frame_index, "timestamp_s": round(ts, 3), "predictions": preds}
            if log_unverified:
                rec["rejected"] = [{"label": d["label"], "det_confidence": round(d["det_confidence"], 4),
                                    "verifier_prob": round(d["verifier_prob"], 4)}
                                   for d in raw if d["verifier_prob"] < min_prob]
            f_pred.write(json.dumps(rec) + "\n")
            for a in alarms:
                f_alarm.write(json.dumps(a) + "\n")
                f_alarm.flush()
                s["alarms"] += 1
                active[a["label"]] = ts
                log(f"  ALARM {a['label'].upper()} at t={a['decision_timestamp_s']:.2f}s "
                    f"(frame {a['decision_frame_index']}, conf {a['confidence']:.2f}, onset {a['onset_timestamp_s']:.2f}s)")
            s["frames_processed"] += 1
            if save_video_dir is not None:
                if writer is None:
                    Path(save_video_dir).mkdir(parents=True, exist_ok=True)
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(Path(save_video_dir) / f"{sid}_annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h))
                on = {k for k, t in active.items() if ts - t < 5}
                writer.write(_draw(frame.copy(), dets, on))
        except Exception as e:  # noqa: BLE001
            s["errors"].append(f"frame {frame_index}: {type(e).__name__}: {e}")
            f_pred.write(json.dumps({"source_id": sid, "frame_index": frame_index, "timestamp_s": round(ts, 3),
                                     "predictions": [], "error": str(e)}) + "\n")
            if len(s["errors"]) == 1:
                log(traceback.format_exc())
            if len(s["errors"]) > 50:
                s["errors"].append("too many errors - aborting this video")
                break
        if s["frames_processed"] % 200 == 0 and s["frames_processed"]:
            f_pred.flush()
    cap.release()
    if writer is not None:
        writer.release()
    s["frames_skipped"] += s["frames_failed"]
    s["duration_s"] = round((frame_index + 1) / fps, 3) if frame_index >= 0 else 0.0
    s["runtime_s"] = round(time.time() - t0, 3)
    s["realtime_factor"] = round(s["duration_s"] / s["runtime_s"], 2) if s["runtime_s"] > 0 else None
    errs_frames = sum(1 for e in s["errors"] if e.startswith("frame "))
    s["status"] = "completed" if s["frames_read"] > 0 and errs_frames == 0 else (
        "completed_with_errors" if s["frames_processed"] > 0 else "failed")
    if s["frames_read"] == 0:
        s["errors"].append("no frames could be decoded")
    return s
