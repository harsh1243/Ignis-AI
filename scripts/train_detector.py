#!/usr/bin/env python
"""Step 2: train the YOLO26 detector.
    python scripts/train_detector.py --train-config train_config.yaml --stage v1   # from COCO weights
    python scripts/train_detector.py --train-config train_config.yaml --stage v2   # fine-tune with mined hard negatives"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from firedet.common import load_yaml
from firedet.train_det import train_detector

ap = argparse.ArgumentParser()
ap.add_argument("--train-config", default="train_config.yaml")
ap.add_argument("--stage", choices=["v1", "v2"], default="v1")
a = ap.parse_args()
cfg = load_yaml(a.train_config)
p, d = cfg["paths"], cfg["detector"]
common = dict(project=p["runs_dir"], imgsz=d["imgsz"], batch=d["batch"], workers=d["workers"], device=d["device"])
if a.stage == "v1":
    best, _ = train_detector(Path(p["dataset_dir"]) / "data.yaml", name="det_v1", model=d["model"], hours=d["hours_v1"],
                             **common, **d.get("augment", {}))
else:
    v1 = Path(p["runs_dir"]) / "det_v1" / "weights" / "best.pt"
    aug = dict(d.get("augment", {}), close_mosaic=2)
    best, _ = train_detector(Path(p["dataset_dir"]) / "data_v2.yaml", name="det_v2", model=str(v1), hours=d["hours_v2"],
                             optimizer="SGD", lr0=d.get("lr0_v2", 0.003), lrf=0.1, warmup_epochs=0.5, **common, **aug)
print("best weights:", best)
