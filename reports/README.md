# Fire and Smoke Detection — Experiment Results

The original CSVs, JSON audits, and images are unchanged. The `visuals/` folder adds PNG charts for the CSVs and an annotated view of the saved confusion counts. Start with the final model metrics and the model-pair comparison. The epoch plots document training, not the performance of the complete alarm pipeline.

## Evaluation

| Open the figure | What it shows |
|---|---|
| [01 · Final model metrics](visuals/01_final_model_metrics.png) | Precision, recall, F1, and AP for fire and smoke. The overall AP bars are mAP across both classes. |
| [02 · Model-pair comparison](visuals/02_model_pair_comparison.png) | Two model pairs on the same 3,764 validation images, including how many of the 1,615 clean images were flagged. |
| [03 · Threshold comparison](visuals/03_threshold_comparison.png) | Baseline versus tuned cutoffs on both validation splits. |
| [04 · Detection confusion matrices](visuals/04_detection_confusion_matrices.png) | Four matrices from saved box counts at IoU ≥ 0.50. Rows are actual and columns are predicted. Background-to-background is undefined. |
| [05 · Verifier comparison](visuals/05_verifier_comparison.png) | How often each crop verifier kept fire and smoke crops and rejected background crops. |
| [06 · Verifier gate confusion counts](visuals/06_verifier_gate_confusion_counts.png) | Accepted versus rejected crops by actual class. Each count is uniquely determined by its saved percentage and denominator. |
| [07 · Positive video alarms](visuals/07_positive_video_alarms.png) | Expected alarm coverage and timing in the six fire and smoke clips. Times are measured from clip start. |
| [08 · Traffic video flags](visuals/08_traffic_video_false_positives.png) | Detector flags, verifier flags, and alarm outcomes on six traffic clips. A flagged frame is not an alarm. |

## Training history

| Open the figure | Source data |
|---|---|
| [09 · Detector fine-tuning run 1](visuals/09_fine_tuning_run_1_epochs.png) | [16 detector epochs](training/ft1_epochs.csv) |
| [10 · Detector fine-tuning run 2](visuals/10_fine_tuning_run_2_epochs.png) | [20 detector epochs](training/ft2_epochs.csv) |
| [11 · Detector fine-tuning run 3](visuals/11_fine_tuning_run_3_epochs.png) | [11 detector epochs](training/ft3_epochs.csv) |
| [12 · Verifier fine-tuning](visuals/12_verifier_epochs.png) | [6 verifier epochs](training/verifier_epochs.csv) |
| [13 · Initial detector stage 1](visuals/13_initial_detector_stage_1.png) | [26 detector epochs](original_model/detector_v1_results.csv) |
| [14 · Initial detector stage 2](visuals/14_initial_detector_stage_2.png) | [10 detector epochs](original_model/detector_v2_results.csv) |

The preserved combined plots and annotation images remain in [figures/](figures/). The original model logs and plots remain in [original_model/](original_model/).

## Regenerate just the images

Run `python visuals/build_charts.py` from this folder with NumPy, pandas, and Matplotlib installed. The script reads the saved CSVs and [recorded confusion counts](visuals/detection_confusion_counts.json). It does not rerun inference or training. Summary precision and recall alone do not determine a class-by-class confusion matrix, so the detection matrices use the actual saved counts.
