# Video-Based Fire and Smoke Detection

This course project turns video into timestamped **fire** and **smoke** alarms. **YOLO26-m locates possible events, EfficientNet-B0 checks each proposed region, and temporal rules decide when the evidence is strong enough to raise an alarm.** The program processes video in order and runs offline with the included YOLO26-m detector and updated EfficientNet-B0 verifier.



The [technical report](Report/report.pdf) contains the full methodology, experiments, and individual contributions.

**Contents** · [Architecture](#1-system-architecture) · [Installation](#2-environment-and-installation) · [Run a video](#3-running-video-inference) · [Verify the package](#5-bundled-verification-examples) · [Training](#7-datasets-and-training) · [Results](#8-evaluation-results)

## 1. System architecture

**Video → Sampled frames → YOLO26-m detector → EfficientNet-B0 verifier → Temporal rules → Alarm**

1. **Sample the video.** The program reads frames in order and passes a subset to the detector. It chooses an integer frame stride to target 5 FPS and keeps each sampled frame’s original index and timestamp.
2. **Locate candidates with YOLO26-m.** YOLO processes the full sampled frame at an input size of 800 pixels. It returns a bounding box, a fire or smoke label, and a detector confidence for each candidate. The detector cutoff is 0.20, and boxes smaller than 4 pixels in either dimension are discarded.
3. **Check each candidate with EfficientNet-B0.** The program makes a padded square crop around each proposed box and resizes it to 224 × 224 pixels. The verifier classifies the crop as fire, smoke, or background. It keeps the box when the sum of the fire and smoke probabilities is at least 0.70. **The YOLO label stays unchanged.** The tracker receives `sqrt(detector_confidence × verifier_event_probability)` as the box confidence.
4. **Confirm the alarm over time.** The tracker connects verified boxes of the same class using box overlap or center distance. An alarm requires enough hits in the class’s time window, a high enough mean score, and a hit in the current sampled frame. Tracking starts fresh for each video and uses no future frames.

YOLO answers **where the event might be and what class it is**. The verifier asks **whether the crop contains credible fire or smoke evidence at all**. This second check can reject a look-alike such as a reflection before it contributes to an alarm. It also has limits. It cannot recover a box YOLO missed or correct its class, and it can reject a real event. The temporal rules then filter brief detections without a third model.

### Fixed inference settings

The released settings are in [config.yaml](config.yaml). Leave them unchanged when reproducing the results.

| Setting | Value |
|---|---|
| Detector and verifier weights | `weights/detector.pt` and `weights/verifier.pt` |
| Detector input size / target sampling rate | 800 px / 5 FPS |
| Detector confidence / IoU setting | 0.20 / 0.50 |
| Minimum box side / maximum detections | 4 px / 50 |
| Verifier crop / event probability cutoff | 224 × 224 px / 0.70 |
| Same-class track matching | IoU ≥ 0.10 or normalized center distance ≤ 1.0 |
| Track expiration / class-alarm reset | 2.0 s without a track hit / more than 10.0 s without a verified detection of that class |

| Class | Time window | Minimum hit ratio | Minimum hits at 5 FPS | Minimum mean score | Minimum moving score |
|---|---:|---:|---:|---:|---:|
| Fire | 1.0 s | 0.80 | 4 | 0.50 | 0.80 |
| Smoke | 2.0 s | 0.60 | 6 | 0.40 | 0.70 |

The minimum hit count is `max(1, ceil(hit_ratio × window_seconds × effective_sample_fps))`. At about 5 FPS, four consecutive qualifying hits on a fire track can first raise an alarm about 0.60 s into a clip if the track starts at the first frame. This is the **decision timestamp**, not the estimated onset.

## 2. Environment and installation

| Component | Reference setup |
|---|---|
| OS / Python | Linux x86-64 / Python 3.13.5 in the offline submission check |
| Verified demo inference | CPU, PyTorch 2.11.0+cpu, and Ultralytics 8.4.194 |
| Later detector fine-tuning | NVIDIA T4 and PyTorch 2.11.0+cu130 |
| CPU inference | No CUDA required |
| NVIDIA GPU inference | Requires a CUDA-enabled PyTorch wheel and compatible NVIDIA driver |

The full package pins are in [requirements.txt](requirements.txt). These commands describe a clean Linux setup. The recorded demo verifies inference in the reference CPU environment, not every dependency pin on other platforms.

**Step 1.** Extract the ZIP and open a terminal in the folder containing `infer.py` and `config.yaml`. Create and activate a Python 3.13 virtual environment.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

**Step 2.** Install the PyTorch and Torchvision wheel pair. Choose **one** command.

CPU:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
```

NVIDIA GPU with the CUDA 13.0 wheel:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
```

**Step 3.** Install the remaining dependencies and check the environment.

```bash
python -m pip install -r requirements.txt
python -m pip check
```

If importing OpenCV on minimal Debian/Ubuntu reports a missing `libGL.so.1` or GLib library, install `libgl1` or the distribution’s GLib runtime as needed. Package installation can require internet access. Inference does not.

## 3. Running video inference

**Required command format**

```bash
python infer.py --source INPUT.mp4 --config config.yaml --output-dir results --device cuda:0 --no-display
```

**CPU example**

```bash
python infer.py --source INPUT.mp4 --config config.yaml --output-dir results --device cpu --no-display
```

| Argument | Meaning |
|---|---|
| `--source` | Local video file, folder of videos, or a quoted glob such as `"videos/*.mp4"` |
| `--config` | Settings file. Weight paths are relative to this file. |
| `--output-dir` | Destination for prediction and alarm logs and the run summary |
| `--device` | `cpu`, `cuda`, a CUDA index such as `cuda:0` or `0`, or `mps` on supported Apple systems |
| `--no-display` | Run without a display or interaction |
| `--save-video` | Optional annotated video for debugging |

If CUDA is unavailable, the runner falls back to CPU and records a warning. Check `device_used` in the run summary. Keep both included weights in `weights/` and use local video inputs for offline inference. No separate tracking model, account, or external API is required.

## 4. Reading the output files

### predictions.jsonl — boxes by sampled frame

One JSON record per sampled frame contains `source_id`, `frame_index`, `timestamp_s`, and the verified `predictions` list. Each detection includes its label, fused confidence, bounding box, detector confidence, verifier probability, and track ID. A sampled frame without detections still gets a record.

```json
{"source_id":"normal_example","frame_index":0,"timestamp_s":0.0,"predictions":[]}
```

### alarms.jsonl — one record per newly raised alarm

Each record contains the source and event IDs, the alarm label, decision time and frame index, and confidence. The alarm confidence is the track’s mean verified score within the decision window. The estimated onset is recorded separately as `onset_timestamp_s`. A clip with no alarms contributes no records.

### run_summary.json — completion and diagnostics

Check the run status, actual device used, frames read/processed/skipped, runtime, alarm count, warnings, and errors. A successful demo run has status `completed` and no errors. Use a separate output directory for each comparison run.

## 5. Bundled verification examples

```bash
python infer.py --source examples/clips/event_example.mp4 --config config.yaml --output-dir results/event_example --device cpu --no-display
python infer.py --source examples/clips/normal_example.mp4 --config config.yaml --output-dir results/normal_example --device cpu --no-display
```

Run both commands offline after installation. To use the first NVIDIA GPU instead, replace `--device cpu` with `--device cuda:0`.

| Clip | Frames read / processed / skipped | Expected alarms |
|---|---|---|
| `event_example.mp4` | 510 / 85 / 425 | One fire alarm at 0.601 s (frame 18) |
| `normal_example.mp4` | 480 / 96 / 384 | None |

Reference outputs generated with the shipped detector and updated verifier are in [examples/expected_outputs/](examples/expected_outputs/). Both bundled clips were rerun offline on CPU from the extracted submission ZIP. The predictions and alarms matched the bundled outputs. Runtime, absolute paths and floating-point confidence can differ across hardware. The event and normal clips are a reproducibility check, not the whole evaluation suite.

## 6. Project layout and checkpoints

| Path | Contents |
|---|---|
| [infer.py](infer.py) and [config.yaml](config.yaml) | Command-line runner and released configuration |
| [firedet/](firedet/) | Data preparation, inference, crop verification, tracking, and alarms |
| [weights/detector.pt](weights/detector.pt) | YOLO26-m, detector epoch 11 of fine-tuning run 3. Original export `detector_last_epoch.pt`. |
| [weights/verifier.pt](weights/verifier.pt) | Updated EfficientNet-B0 verifier loaded by the submitted configuration. |
| [scripts/](scripts/), [notebooks/](notebooks/), and [train_config.yaml](train_config.yaml) | Training, evaluation, and threshold experiments |
| [examples/](examples/) | Bundled example videos and reference outputs |
| [reports/](reports/) | Dataset audits, epoch logs, evaluation CSVs, and figures |
| [Report/report.pdf](Report/report.pdf) | Final report with current results and individual contributions |
| [annotaed_videos/](annotaed_videos/) | Supplementary side-by-side detections and alarms. These videos are not required for inference. |

## 7. Datasets and training

The detector uses YOLO-format boxes, with class ID 0 for fire and 1 for smoke. Empty label files identify backgrounds. An image can contain both event classes.

| Stage | Training data and change | Validation / test |
|---|---|---|
| Initial training | 41,954 training images prepared from D-Fire (GAIASD), FASDD-CV and SentinelEdge street backgrounds. A second stage used mined hard negatives. | Original val 3,829. Original held-out test 1,958. |
| Fine-tuning run 1 | 12,258 training images. Original replay 10,000 plus `fireandsmoke_part1` (981) and `fireandsmoke_part2` (1,277). The mix has 9,304 positives and 2,954 backgrounds. | Original val 3,829 and hard val 869. Held-out original test 1,958 and hard test 491 were kept in the archive. |
| Fine-tuning run 2 | Started from the first run’s detector with the same training mix, a smaller batch and a lower LR. | Same split sizes as the first run. |
| Fine-tuning run 3 | Replaced the positives in the staged pool with 15,000 train-only Ultralytics RPi5 Fire Smoke positives and kept 15,186 existing backgrounds. Total 30,186. The verifier was also fine-tuned. | Filtered val 3,764 after removing 65 exact train/validation matches. Hard val remained 869. |

The original detector was initialized from COCO-pretrained YOLO26-m and trained in two stages on an A100 (26 epochs, then 10). The original verifier started from ImageNet-pretrained EfficientNet-B0 weights. The first fine-tuning run completed 16 epochs on an A100 and selected epoch 1. The second fine-tuning run completed 20 epochs on Kaggle T4 and kept its best detector. The third fine-tuning run stopped after 11 of 12 planned detector epochs to reserve verifier time. The release uses that last detector epoch, not the best detector checkpoint from that run, together with the updated EfficientNet-B0 verifier.

Training is separate from the grader's offline inference command. Prepared training datasets and historical checkpoints are not bundled. Original preparation (downloads and merging), hard-negative mining, detector/verifier training, and video scoring are in `scripts/` and `firedet/`. Install `requirements-train.txt` in the training environment when needed.

- **Initial training.** See [the initial training notebook](notebooks/0_fire_smoke_training.ipynb) and the scripts below. Adjust the data/work paths in `train_config.yaml` for training only.
- **Fine-tuning runs 1 and 2.** Use [the first](notebooks/1st_finetune.ipynb) and [second](notebooks/2nd_finetune.ipynb) fine-tuning notebooks with their documented source weights and `finetune_v4.tar`. Fine-tuning outputs are distinct from the submitted inference weights.
- **Fine-tuning run 3.** Use [the third fine-tuning notebook](notebooks/3rd_finetune.ipynb) or the executable `scripts/train_ft3.py` with a directory containing `fire_smoke_rpi5_10000fire_5000smoke_train_only.zip`, `finetune_v4.tar`, `detector_v4_kaggle.pt`, and the **original** `verifier.pt`. The script audits data and removes exact train/validation overlap from validation, trains YOLO, builds verifier crops, and fine-tunes the verifier. The fixed number of completed detector epochs in the recorded run resulted from its time budget. Outputs of a rerun can differ.
- [The threshold-tuning notebook](notebooks/4_threshold_tuning.ipynb) and [the reference evaluation notebook](notebooks/05_final_evaluation_results.ipynb) contain the tuning and evaluation source and recorded results. They use the training notebook state or the documented weights, datasets and videos. Historical CSVs and figures are kept under [reports/](reports/). The current results are in [the final report](Report/report.pdf).

Original stage commands:

```bash
python scripts/prepare_data.py --train-config train_config.yaml
python scripts/train_detector.py --train-config train_config.yaml --stage v1
python scripts/mine_hard_negatives.py --train-config train_config.yaml
python scripts/train_detector.py --train-config train_config.yaml --stage v2
python scripts/build_verifier_data.py --train-config train_config.yaml --detector TRAINED_V2_BEST.pt
python scripts/train_verifier.py --train-config train_config.yaml --out ORIGINAL_VERIFIER.pt
```

Third fine-tuning command (not needed for inference):

```bash
python scripts/train_ft3.py --inputs TRAINING_INPUTS --output-dir training_runs --device 0
```

Do not write training outputs into the submitted `weights/` directory. The historical CSVs and figures describe the checkpoints and test settings used at the time of those evaluations.

## 8. Evaluation results

### Submitted pipeline on 3,764 validation images

These results are for the included YOLO26-m detector and updated EfficientNet-B0 verifier. The evaluation used the fixed detector confidence cutoff of 0.20 and verifier cutoff of 0.70. Precision, recall and F1 use one-to-one box matching at IoU 0.50. The overall row pools counts for precision, recall and F1, and averages AP across the two classes for mAP.

| Class or aggregate | Precision | Recall | F1 | AP50 / mAP50 | AP50–95 / mAP50–95 |
|---|---:|---:|---:|---:|---:|
| Fire | 69.17% | 56.56% | 62.23% | 50.07% | 29.26% |
| Smoke | 84.40% | 61.37% | 71.07% | 58.62% | 39.92% |
| **Overall** | **74.89%** | **58.50%** | **65.69%** | **54.34%** | **34.59%** |

### Retest on three smoke-containing clips

| Clip | Clip contents | Alarm with the submitted verifier |
|---|---|---|
| `s1.mp4` | Smoke | Smoke at 1.042 s |
| `s2.mp4` | Fire and smoke | Fire at 0.626 s. No smoke alarm. |
| `s3.mp4` | Smoke | Smoke at 4.000 s |

The submitted pipeline raised an alarm on all three clips, with two smoke alarms and one fire alarm on the mixed clip. Times are measured from clip start, not from an independently annotated event onset. A verified box in one frame does not necessarily trigger an alarm.

### Historical results and limitations

The CSVs in [reports/evaluation/](reports/evaluation/), including `final_model_evaluation.csv` and the earlier 12-clip logs, describe earlier model configurations. The [final report](Report/report.pdf) separates those benchmarks from the updated verifier's results. The [12 evaluation videos on Google Drive](https://drive.google.com/drive/folders/1iZoIqJKZsJeuSlNnQlF7IiQowa4yV3gp?usp=drive_link) are supplementary and are not needed for offline inference.

The stricter mAP50–95 result shows that precise box localization remains a limitation. The verifier cannot create a box YOLO missed or change its class. The small video retest also does not establish a general false-alarm rate. Longer background videos and more clips with annotated event onsets would allow a stronger assessment.
