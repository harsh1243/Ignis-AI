"""Video-level evaluation and alarm-parameter tuning.

Put labelled clips in   <eval_dir>/fire/*.mp4  <eval_dir>/smoke/*.mp4  (event videos)
                        <eval_dir>/normal/*.mp4                       (no fire/smoke at all)
Detector + verifier run once per video (cached); the cheap temporal logic is then re-simulated for
many parameter combinations. Metrics: event recall (event videos with >= 1 alarm), false alarms on
normal videos (and per hour), time to first alarm.
"""
import copy
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .common import IMG_EXTS, VID_EXTS, fmt_secs, list_files
from .inference import Pipeline
from .temporal import AlarmManager


def list_eval_videos(eval_dir):
    vids = []
    for kind in ("fire", "smoke", "normal"):
        for v in list_files(Path(eval_dir) / kind, VID_EXTS):
            vids.append((v, "normal" if kind == "normal" else "event"))
    return vids


def cache_video(pipeline, video, target_fps, cache_dir, refresh=False, max_s=None):
    """Run detector + verifier on one video (sampled at target_fps) and cache raw detections."""
    import cv2
    cache = Path(cache_dir) / f"{Path(video).stem}_{hashlib.md5(str(video).encode()).hexdigest()[:8]}_{int(max_s or 0)}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 240 else 25.0
    stride = max(1, int(round(fps / target_fps)))
    frames, i = [], -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        i += 1
        if max_s and i / fps > max_s:
            break
        if i % stride:
            continue
        ok, fr = cap.retrieve()
        if not ok or fr is None:
            continue
        frames.append([i, i / fps, pipeline.raw(fr)])
    cap.release()
    d = {"video": str(video), "fps": fps, "stride": stride, "duration_s": (i + 1) / fps, "frames": frames}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(d))
    return d


def simulate(cached, temporal_cfg, min_prob):
    am = AlarmManager(temporal_cfg, Path(cached["video"]).stem, cached["fps"] / cached["stride"])
    alarms = []
    for fi, ts, raw in cached["frames"]:
        alarms += am.update(fi, ts, Pipeline.finalize([dict(d) for d in raw], min_prob))
    return alarms


def score(cached_list, kinds, temporal_cfg, min_prob):
    first, fa, hours, missed, fa_videos = [], 0, 0.0, [], []
    for c, k in zip(cached_list, kinds):
        al = simulate(c, temporal_cfg, min_prob)
        if k == "event":
            first.append(al[0]["decision_timestamp_s"] if al else None)
            if not al:
                missed.append(Path(c["video"]).name)
        else:
            fa += len(al)
            hours += c["duration_s"] / 3600
            if al:
                fa_videos.append(Path(c["video"]).name)
    n_ev = len(first)
    hit = [t for t in first if t is not None]
    return {"event_recall": len(hit) / n_ev if n_ev else float("nan"), "events": n_ev, "events_detected": len(hit),
            "false_alarms": fa, "false_alarms_per_hour": fa / hours if hours else float("nan"),
            "normal_hours": round(hours, 3), "mean_time_to_alarm_s": float(np.mean(hit)) if hit else float("nan"),
            "missed_event_videos": missed, "videos_with_false_alarms": fa_videos}


def evaluate(pipeline, cfg, eval_dir, cache_dir, refresh=False, max_s=None):
    vids = list_eval_videos(eval_dir)
    if not vids:
        print(f"No evaluation videos in {eval_dir}/{{fire,smoke,normal}}/ - skipping.")
        return None, None, None
    t0 = time.time()
    cached, kinds = [], []
    for n, (v, k) in enumerate(vids, 1):
        cached.append(cache_video(pipeline, v, cfg["input"]["target_fps"], cache_dir, refresh, max_s))
        kinds.append(k)
        el = time.time() - t0
        print(f"   [{n}/{len(vids)}] {k:6s} {v.name}  ({fmt_secs(el)} elapsed, ETA {fmt_secs(el / n * (len(vids) - n))})")
    res = score(cached, kinds, cfg["temporal"], cfg["verifier"]["min_prob"])
    per_video = []
    for c, k in zip(cached, kinds):
        al = simulate(c, cfg["temporal"], cfg["verifier"]["min_prob"])
        per_video.append({"video": Path(c["video"]).name, "kind": k, "duration_s": round(c["duration_s"], 1),
                          "alarms": [(a["label"], a["decision_timestamp_s"]) for a in al]})
    return res, per_video, (cached, kinds)


def tune(cached, kinds, cfg, fire_grid=None, smoke_grid=None, prob_grid=(0.3, 0.5, 0.7), max_fa_per_hour=None,
         slack_s=1.5):
    """Grid search. Labels are independent, so fire and smoke parameters are searched separately and
    then combined. Rule: highest event recall, then fewest false alarms, then fastest alarm."""
    fire_grid = fire_grid or {"window_s": [1.0, 1.6, 2.4], "min_hit_ratio": [0.4, 0.6, 0.8], "min_mean_conf": [0.25, 0.35, 0.45]}
    smoke_grid = smoke_grid or {"window_s": [2.0, 3.2, 4.8], "min_hit_ratio": [0.4, 0.6, 0.8], "min_mean_conf": [0.2, 0.3, 0.4]}
    base = cfg["temporal"]
    ev_idx = [i for i, k in enumerate(kinds) if k == "event"]
    nm_idx = [i for i, k in enumerate(kinds) if k == "normal"]
    hours = sum(cached[i]["duration_s"] for i in nm_idx) / 3600 or float("nan")

    def run_label(label, params, prob):
        t = {k: v for k, v in copy.deepcopy(base).items() if k not in ("fire", "smoke")}
        lc = copy.deepcopy(base[label])
        lc.update(params)
        lc["moving_min_mean_conf"] = round(max(lc["min_mean_conf"] + 0.2, base[label].get("moving_min_mean_conf", 0.6)), 3)
        t[label] = lc
        first = np.full(len(cached), np.inf)
        fa = 0
        for i, c in enumerate(cached):
            al = simulate(c, t, prob)
            if i in nm_idx:
                fa += len(al)
            elif al:
                first[i] = al[0]["decision_timestamp_s"]
        return first, fa, lc

    t0 = time.time()
    results = []
    for prob in prob_grid:
        fires = [run_label("fire", dict(zip(fire_grid, v)), prob) for v in itertools.product(*fire_grid.values())]
        smokes = [run_label("smoke", dict(zip(smoke_grid, v)), prob) for v in itertools.product(*smoke_grid.values())]
        for (ff, fa1, fc), (sf, fa2, sc) in itertools.product(fires, smokes):
            first = np.minimum(ff, sf)[ev_idx]
            hit = np.isfinite(first)
            results.append({"min_prob": prob, "fire": fc, "smoke": sc,
                            "event_recall": float(hit.mean()) if len(hit) else float("nan"),
                            "false_alarms": fa1 + fa2, "fa_per_hour": (fa1 + fa2) / hours,
                            "mean_time_to_alarm_s": float(first[hit].mean()) if hit.any() else float("inf")})
        print(f"   min_prob={prob}: searched {len(fires)}x{len(smokes)} combos ({fmt_secs(time.time() - t0)})")
    ok = results
    if max_fa_per_hour is not None:
        ok = [r for r in results if r["fa_per_hour"] <= max_fa_per_hour] or results
    top = sorted(ok, key=lambda r: (-r["event_recall"], r["false_alarms"], r["mean_time_to_alarm_s"]))[0]
    # among equally good settings, take the STRICTEST one that alarms at most `slack_s` later than the fastest:
    # the eval set is small, so a stricter setting generalises better to unseen videos
    pool = [r for r in ok if r["event_recall"] == top["event_recall"] and r["false_alarms"] == top["false_alarms"]
            and r["mean_time_to_alarm_s"] <= top["mean_time_to_alarm_s"] + slack_s]
    best = max(pool, key=lambda r: (r["min_prob"], r["fire"]["min_mean_conf"] + r["smoke"]["min_mean_conf"],
                                    r["fire"]["min_hit_ratio"] + r["smoke"]["min_hit_ratio"]))
    new_cfg = copy.deepcopy(cfg)
    new_cfg["verifier"]["min_prob"] = best["min_prob"]
    new_cfg["temporal"]["fire"] = best["fire"]
    new_cfg["temporal"]["smoke"] = best["smoke"]
    return new_cfg, best, results



def street_image_fp(pipeline, image_dir, min_prob):
    """Fire-free images: share with any detection, and share still alarming after the verifier."""
    import cv2
    paths = list_files(image_dir, IMG_EXTS)
    if not paths:
        return None
    n_det = n_ver = 0
    for p in paths:
        im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        raw = pipeline.raw(im)
        n_det += int(len(raw) > 0)
        n_ver += int(len(Pipeline.finalize(raw, min_prob)) > 0)
    n = len(paths)
    return {"n_images": n, "detector_rate": n_det / n, "after_verifier_rate": n_ver / n}
