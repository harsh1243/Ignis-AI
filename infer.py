#!/usr/bin/env python
"""Fire / smoke alarm inference.

    python infer.py --source INPUT.mp4 --config config.yaml --output-dir results --device cuda:0 --no-display

--source   a video file, a folder of videos, or a glob ("videos/*.mp4"); temporal state is reset per video
--device   cuda:0 | cuda:1 | cuda | 0 | cpu | mps   (falls back to CPU, with a logged warning, if unavailable)
Outputs in --output-dir: predictions.jsonl, alarms.jsonl, run_summary.json
Runs fully offline: no downloads, no GUI, no interactive input.
"""
import os

# offline + quiet before any ultralytics import
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from firedet.common import VID_EXTS, load_yaml  # noqa: E402


def collect_sources(src):
    p = Path(src)
    if p.is_dir():
        return sorted(q for q in p.iterdir() if q.suffix.lower() in VID_EXTS)
    if any(ch in str(src) for ch in "*?["):
        return sorted(Path(q) for q in glob.glob(str(src)))
    return [p]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-display", action="store_true", help="accepted for compatibility; nothing is ever displayed")
    ap.add_argument("--save-video", action="store_true", help="also write an annotated mp4 (debugging)")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"status": "failed", "frames_read": 0, "frames_processed": 0, "frames_skipped": 0,
               "total_runtime_s": 0.0, "errors": [], "warnings": [], "alarms_raised": 0,
               "command": " ".join(sys.argv), "config": str(Path(args.config).resolve()),
               "device_requested": args.device, "videos": []}
    f_pred = open(out / "predictions.jsonl", "w")
    f_alarm = open(out / "alarms.jsonl", "w")
    try:
        cfg_path = Path(args.config).resolve()
        cfg = load_yaml(cfg_path)
        import torch

        from firedet.inference import Pipeline, resolve_device, run_video
        device, warn = resolve_device(args.device)
        if warn:
            print("WARNING:", warn)
            summary["warnings"].append(warn)
        summary["device_used"] = device
        summary["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                                  "cuda_available": torch.cuda.is_available(), "platform": platform.platform()}
        pipe = Pipeline(cfg, cfg_path.parent, device)
        sources = collect_sources(args.source)
        if not sources:
            raise FileNotFoundError(f"no video found for --source {args.source}")
        for src in sources:
            print(f"Processing {src} ...")
            if not Path(src).exists():
                s = {"source": str(src), "source_id": Path(src).stem, "status": "failed",
                     "errors": [f"file not found: {src}"], "frames_read": 0, "frames_processed": 0, "frames_skipped": 0}
            else:
                s = run_video(src, pipe, cfg, f_pred, f_alarm, save_video_dir=out / "annotated" if args.save_video else None)
            f_pred.flush()
            summary["videos"].append(s)
            print(f"  done: {s['status']} | read {s['frames_read']} | processed {s['frames_processed']} | "
                  f"skipped {s['frames_skipped']} | alarms {s.get('alarms', 0)} | {s.get('runtime_s', 0):.1f}s")
        for k in ("frames_read", "frames_processed", "frames_skipped"):
            summary[k] = sum(v.get(k, 0) for v in summary["videos"])
        summary["alarms_raised"] = sum(v.get("alarms", 0) for v in summary["videos"])
        for v in summary["videos"]:
            summary["errors"] += [f"{v['source_id']}: {e}" for e in v.get("errors", [])]
        st = {v["status"] for v in summary["videos"]}
        summary["status"] = "completed" if st == {"completed"} else (
            "failed" if st == {"failed"} else "completed_with_errors")
        if len(summary["videos"]) == 1:   # flat fields for the single-video case
            v = summary["videos"][0]
            summary.update(source_id=v["source_id"], source=v["source"], fps=v.get("fps"),
                           frame_stride=v.get("frame_stride"), duration_s=v.get("duration_s"))
    except Exception as e:  # noqa: BLE001
        summary["errors"].append(f"{type(e).__name__}: {e}")
        summary["traceback"] = traceback.format_exc()
        print(summary["traceback"])
    finally:
        f_pred.close()
        f_alarm.close()
        summary["total_runtime_s"] = round(time.time() - t0, 3)
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Status: {summary['status']} | outputs in {out.resolve()}")
    sys.exit(0 if summary["status"] != "failed" else 1)


if __name__ == "__main__":
    main()
