#!/usr/bin/env python
"""Step 1: download D-Fire / FASDD (+ your own negatives) and build the merged YOLO dataset.
    python scripts/prepare_data.py --train-config train_config.yaml"""
import argparse

import _bootstrap  # noqa: F401
from firedet.common import load_yaml
from firedet.data_prep import prepare_all

ap = argparse.ArgumentParser()
ap.add_argument("--train-config", default="train_config.yaml")
a = ap.parse_args()
cfg = load_yaml(a.train_config)
p = cfg["paths"]
prepare_all(cfg, p["raw_dir"], p["work_dir"], p["dataset_dir"])
