# Fire and Smoke Detection Notebooks

The notebooks document **one initial training run followed by three fine-tuning runs**. Read the training notebooks in this order.

| Stage | Notebook | What we did |
| --- | --- | --- |
| Initial training | [0_fire_smoke_training.ipynb](0_fire_smoke_training.ipynb) | Trained the initial YOLO26-m fire and smoke detector and the EfficientNet-B0 crop verifier. |
| Fine-tuning 1 | [1st_finetune.ipynb](1st_finetune.ipynb) | Continued training the initial detector using replay data and hard negatives. |
| Fine-tuning 2 | [2nd_finetune.ipynb](2nd_finetune.ipynb) | Continued from the selected detector checkpoint from run 1 using the same prepared dataset. |
| Fine-tuning 3 | [3rd_finetune.ipynb](3rd_finetune.ipynb) | Updated the detector with a new source of positive training images, then fine-tuned the crop verifier. |

Two notebooks follow the training runs. [4_threshold_tuning.ipynb](4_threshold_tuning.ipynb) tests confidence thresholds without changing model weights. [05_final_evaluation_results.ipynb](05_final_evaluation_results.ipynb) compares the second and third model pairs on validation images and evaluation videos.
