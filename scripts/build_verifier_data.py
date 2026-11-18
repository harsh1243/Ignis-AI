#!/usr/bin/env python
"""Step 4: cut fire / smoke / not_fire crops for the verifier using the final detector.
    python scripts/build_verifier_data.py --train-config train_config.yaml --detector weights/detector.pt"""
import argparse

import _bootstrap  # noqa: F401
from firedet.common import load_yaml
from firedet.mining import build_verifier_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--train-config", default="train_config.yaml")
ap.add_argument("--detector", required=True)
a = ap.parse_args()
cfg = load_yaml(a.train_config)
p, v = cfg["paths"], cfg["verifier"]
build_verifier_dataset(a.detector, p["dataset_dir"], p["crops_dir"], conf=v["mine_conf"], device=cfg["detector"]["device"],
                       pad=v["pad"], max_images=v["max_images"], caps=v["caps"])
