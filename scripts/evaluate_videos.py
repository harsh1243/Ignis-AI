#!/usr/bin/env python
"""Step 6: evaluate the full pipeline on labelled videos and (optionally) tune the alarm parameters.
    python scripts/evaluate_videos.py --config config.yaml --eval-dir EVAL_DIR [--tune --write-config]
EVAL_DIR contains fire/, smoke/ (event videos) and normal/ (no fire) sub-folders."""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from firedet.common import load_yaml, save_yaml
from firedet.evaluate import evaluate, score, tune
from firedet.inference import Pipeline, resolve_device

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--eval-dir", required=True)
ap.add_argument("--cache-dir", default="eval_cache")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--tune", action="store_true")
ap.add_argument("--write-config", action="store_true")
a = ap.parse_args()
cfg_path = Path(a.config).resolve()
cfg = load_yaml(cfg_path)
dev, _ = resolve_device(a.device)
pipe = Pipeline(cfg, cfg_path.parent, dev)
res, per_video, data = evaluate(pipe, cfg, a.eval_dir, a.cache_dir)
print(json.dumps(res, indent=2))
if a.tune and data:
    new_cfg, best, _ = tune(*data, cfg)
    print("best:", json.dumps(best, indent=2))
    print("with tuned params:", json.dumps(score(*data, new_cfg["temporal"], new_cfg["verifier"]["min_prob"]), indent=2))
    if a.write_config:
        save_yaml(new_cfg, cfg_path)
        print("config updated:", cfg_path)
