# Ultra-Short rPPG Heart Rate Estimation

PyTorch research code for **Towards Accurate Heart Rate Measurement from Ultra-Short Video Clips via Periodicity-Guided rPPG Estimation and Signal Reconstruction**.

This code estimates heart rate from ultra-short face clips, e.g. 60 frames at 30 FPS. The pipeline trains a PhysNet-based rPPG estimator, then trains a generator that reconstructs longer rPPG representations from short rPPG signals to reduce spectral leakage during HR estimation.

## Method

- Periodicity-guided rPPG estimation for 2-second video clips.
- Weighted spectral cross-entropy and maximized periodic similarity losses.
- Synthetic PPG supervision generated from ground-truth HR.
- Short-to-long rPPG feature reconstruction with a conditional generator.

## Setup

```bash
pip install -r requirements.txt
```

Use a PyTorch build that matches your CUDA environment.

## Layout

```text
scripts/   Training and evaluation entry points
models/    PhysNet and reconstruction generator
datasets/  Dataset loaders
loss/      Training losses and metrics
util/      Config, logging, and signal utilities
config/    Placeholder plus local dataset split files
```

## Data

Datasets, local dataset split files, checkpoints, logs, caches, and paper PDFs are not included.

Set the dataset root with `RPPG_DATA_ROOT` or `--data_root`. Put local split files under `config/`. Dataset aliases:

| Alias | Dataset |
| --- | --- |
| `C` | COHFACE |
| `P` | PURE |
| `U` | UBFC-rPPG |
| `V` | VIPL-HR |

Expected layout:

```text
data/
|-- COHFACE/crop_MTCNN_30fps/<subject>/*.png
|-- COHFACE/crop_MTCNN_30fps/<subject>/ground_truth.txt
|-- pure/crop_MTCNN/<subject>/*.png
|-- pure/crop_MTCNN/<subject>/ground_truth.txt
|-- UBFC/crop_MTCNN/<subject>/*.png
|-- UBFC/crop_MTCNN/<subject>/ground_truth.txt
`-- VIPL/
    |-- RGB_crop/preprocess_data_fold*/<subject>/*.png
    `-- GT/<subject>/ground_truth.txt
```

VIPL paths can also be passed with `--vipl_root`, `--vipl_gt_root`, and `--vipl_bg_root`.

## Run

Default reproduction setting: train on PURE + UBFC and test on VIPL-HR.

```bash
RPPG_DATA_ROOT=/path/to/data bash train.sh
```

Equivalent script order:

```text
scripts/train_label.py -> scripts/test_label.py -> scripts/train_cGAN.py -> scripts/train_adapt_G.py -> scripts/test_cGAN_G.py
```

Useful path options:

```text
--data_root   Dataset root
--output_dir  Checkpoint/result root, default ./results
--log_dir     Log root, default ./logs
--cache_dir   Preprocessed cache root, default ./cache/preprocessed
--test_seq    Test clip length in frames, default 60
```

## Outputs

```text
results/<dataset>/<run_name>/weight/
results/<dataset>/<run_name>/weight/trainG/
logs/<phase>/<dataset>/
cache/preprocessed/
```

## Notes

- `config/PUT_DATASET_SPLIT_FILES_HERE` is tracked only to keep the folder in Git. Other files under `config/` are local-only and ignored.
- Input videos must be cropped into frame folders before training.
- `torchprofile` is optional; FLOPs logging is skipped if it is unavailable.
