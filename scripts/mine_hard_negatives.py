#!/usr/bin/env python
"""Step 3: run detector v1 on normal (no-fire) videos, save every frame it fires on as a hard negative,
and write data_v2.yaml (train = train + mined).
    python scripts/mine_hard_negatives.py --train-config train_config.yaml"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from firedet.common import load_yaml
from firedet.data_prep import write_data_yaml
from firedet.mining import mine_video_false_positives

ap = argparse.ArgumentParser()
ap.add_argument("--train-config", default="train_config.yaml")
a = ap.parse_args()
cfg = load_yaml(a.train_config)
p, m = cfg["paths"], cfg["mining"]
v1 = Path(p["runs_dir"]) / "det_v1" / "weights" / "best.pt"
dirs = [Path(p["raw_dir"]) / x for x in m["video_dirs"]]
n = mine_video_false_positives(v1, dirs, p["dataset_dir"], every_s=m["every_s"], conf=m["conf"],
                               max_per_video=m["max_per_video"], repeat=m["repeat"], device=cfg["detector"]["device"],
                               crop_dir=p["crops_dir"])
st = cfg["data"].get("street", {})
if st.get("enabled"):
    from firedet.data_prep import get_street_frames
    from firedet.mining import mine_image_false_positives
    fr = get_street_frames(p["raw_dir"], st["n_train"], st["n_mine"], st["n_test"])
    if "mine" in fr:
        n += mine_image_false_positives(v1, fr["mine"], p["dataset_dir"], conf=m["conf"], repeat=m["repeat"],
                                        device=cfg["detector"]["device"], crop_dir=p["crops_dir"])[0]
write_data_yaml(p["dataset_dir"], extra_train_dirs=["images/mined"])
print("mined frames:", n)
