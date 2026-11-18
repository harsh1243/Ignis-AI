"""Temporal alarm logic: track verified detections over time and raise an alarm only when a
fire/smoke region persists. Uses only the current and past frames (sequential, causal)."""
import math
from collections import deque

import numpy as np

from .common import box_iou


class Track:
    __slots__ = ("tid", "label", "box", "hits", "first_ts", "first_frame", "last_hit_ts", "alarmed")

    def __init__(self, tid, label, box, ts, frame_index):
        self.tid, self.label, self.box = tid, label, list(box)
        self.hits = deque()                 # (ts, score, cx, cy, size)
        self.first_ts, self.first_frame = ts, frame_index
        self.last_hit_ts = ts
        self.alarmed = False

    def add_hit(self, box, ts, score):
        self.box = list(box)
        x1, y1, x2, y2 = box
        self.hits.append((ts, score, (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1, 1.0)))
        self.last_hit_ts = ts

    def prune(self, now, keep_s):
        while self.hits and self.hits[0][0] < now - keep_s:
            self.hits.popleft()


class AlarmManager:
    """One instance per video. Call `update(frame_index, ts, detections)` for every processed frame
    (in order). `detections` = list of dicts with keys label, confidence, bbox (xyxy pixels) - only the
    detections that passed the verifier. Returns the list of alarms raised at this frame."""

    def __init__(self, cfg, source_id, sample_fps):
        self.cfg = cfg
        self.source_id = source_id
        self.sample_fps = max(0.5, float(sample_fps))
        self.tracks = []
        self.next_tid = 1
        self.n_events = 0
        self.event_active = {}       # label -> bool (an alarm was raised and the incident is ongoing)
        self.last_hit_ts = {}        # label -> ts of last verified detection
        self.match_iou = cfg.get("match_iou", 0.1)
        self.match_center = cfg.get("match_center_dist", 1.0)
        self.max_missed_s = cfg.get("max_missed_s", 2.0)
        self.event_timeout_s = cfg.get("event_timeout_s", 10.0)

    def _lc(self, label):
        return self.cfg[label]

    def min_hits(self, label):
        c = self._lc(label)
        return max(1, int(math.ceil(c["min_hit_ratio"] * c["window_s"] * self.sample_fps)))

    def _match(self, label, dets):
        tr = [t for t in self.tracks if t.label == label]
        if not tr or not dets:
            return [], list(range(len(dets))), tr
        tb = np.array([t.box for t in tr], dtype=np.float32)
        db = np.array([d["bbox"] for d in dets], dtype=np.float32)
        iou = box_iou(tb, db)
        tc = np.stack([(tb[:, 0] + tb[:, 2]) / 2, (tb[:, 1] + tb[:, 3]) / 2], 1)
        dc = np.stack([(db[:, 0] + db[:, 2]) / 2, (db[:, 1] + db[:, 3]) / 2], 1)
        tsz = np.maximum(tb[:, 2] - tb[:, 0], tb[:, 3] - tb[:, 1]).clip(1)
        dist = np.linalg.norm(tc[:, None] - dc[None], axis=2) / tsz[:, None]
        score = np.where(iou > 0, 1 + iou, 1 - np.minimum(dist, 1))
        cand = [(score[i, j], i, j) for i in range(len(tr)) for j in range(len(dets))
                if iou[i, j] >= self.match_iou or dist[i, j] <= self.match_center]
        cand.sort(reverse=True)
        used_t, used_d, pairs = set(), set(), []
        for _, i, j in cand:
            if i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            pairs.append((tr[i], j))
        return pairs, [j for j in range(len(dets)) if j not in used_d], tr

    def update(self, frame_index, ts, detections):
        alarms = []
        for label in ("fire", "smoke"):
            if label not in self.cfg:
                continue
            c = self._lc(label)
            dets = [d for d in detections if d["label"] == label]
            if dets:
                self.last_hit_ts[label] = ts
            # incident over? -> allow a new alarm for this label later
            if self.event_active.get(label) and ts - self.last_hit_ts.get(label, -1e9) > self.event_timeout_s:
                self.event_active[label] = False
            pairs, unmatched, _ = self._match(label, dets)
            for t, j in pairs:
                t.add_hit(dets[j]["bbox"], ts, dets[j]["confidence"])
                dets[j]["track_id"] = t.tid
            for j in unmatched:
                t = Track(self.next_tid, label, dets[j]["bbox"], ts, frame_index)
                self.next_tid += 1
                t.add_hit(dets[j]["bbox"], ts, dets[j]["confidence"])
                dets[j]["track_id"] = t.tid
                self.tracks.append(t)
            # decide (only tracks that were hit in THIS frame -> decision ts = this frame)
            need = self.min_hits(label)
            for t in self.tracks:
                if t.label != label or t.last_hit_ts != ts:
                    continue
                t.prune(ts, c["window_s"])
                if len(t.hits) < need:
                    continue
                scores = [h[1] for h in t.hits]
                mean_conf = float(np.mean(scores))
                first, last = t.hits[0], t.hits[-1]
                size = float(np.mean([h[4] for h in t.hits]))
                moved = math.hypot(last[2] - first[2], last[3] - first[3]) / max(size, 1.0)
                thr = c["moving_min_mean_conf"] if moved > c.get("move_thresh", 1.0) else c["min_mean_conf"]
                if mean_conf < thr or self.event_active.get(label):
                    continue
                self.n_events += 1
                self.event_active[label] = True
                t.alarmed = True
                alarms.append({
                    "source_id": self.source_id,
                    "event_id": f"{self.source_id}-{self.n_events:04d}",
                    "label": label,
                    "decision_timestamp_s": round(float(ts), 3),
                    "decision_frame_index": int(frame_index),
                    "confidence": round(mean_conf, 4),
                    "onset_timestamp_s": round(float(t.first_ts), 3),
                    "onset_frame_index": int(t.first_frame),
                    "track_id": t.tid,
                    "hits_in_window": len(t.hits),
                    "bbox_xyxy": [round(float(v), 1) for v in t.box],
                })
        # drop stale tracks
        self.tracks = [t for t in self.tracks if ts - t.last_hit_ts <= self.max_missed_s]
        return alarms
