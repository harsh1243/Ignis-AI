#!/usr/bin/env python
"""Step 5: train the EfficientNet-B0 verifier on the crops.
    python scripts/train_verifier.py --train-config train_config.yaml --out weights/verifier.pt"""
import argparse

import _bootstrap  # noqa: F401
from firedet.common import load_yaml
from firedet.train_verifier import train_verifier

ap = argparse.ArgumentParser()
ap.add_argument("--train-config", default="train_config.yaml")
ap.add_argument("--out", default="weights/verifier.pt")
a = ap.parse_args()
cfg = load_yaml(a.train_config)
p, v = cfg["paths"], cfg["verifier"]
train_verifier(p["crops_dir"], a.out, epochs=v["epochs"], batch=v["batch"], lr=v["lr"], imgsz=v["imgsz"],
               workers=cfg["detector"]["workers"], device=v["device"], pad=v["pad"], show_plots=False)
