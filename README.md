# cyclops

**Comparative analysis of monocular depth estimation methods, with robustness evaluation under degraded visual conditions.**

Monocular depth estimation predicts a per-pixel distance map from a single RGB image — no lidar, no stereo. This project compares five approaches (a supervised ResNet-50 baseline, frozen Stable Diffusion / I-JEPA features, their fusion, and DepthAnything as a SOTA reference) on NYU Depth V2 and KITTI, and measures how each holds up under synthetically degraded inputs (fog, blur, under-exposure).

Authors: Jovan Cvijanović (SV83/2024), Vukan Radojević (SV67/2023)

The phased implementation plan is in [ROADMAP.md](ROADMAP.md). Config-driven experiments are described in [configs/README.md](configs/README.md).

## Setup

```bash
conda create -n cyclops python=3.11 -y
conda activate cyclops
pip install -e .            # installs the cyclops package and its dependencies
```

## Getting the dataset (NYU Depth V2)

The data is not in the repo (it is large and git-ignored). After cloning, download it once:

```bash
# 1. extra tools needed only for downloading + extracting the data
pip install gdown h5py scipy numpy opencv-python Pillow

# 2. download NYU into data/nyu_depth_v2/  (train ~4 GB, plus a 2.8 GB .mat for the test set)
bash scripts/download_nyu.sh
```

What the script does (it is safe to re-run — finished steps are skipped):
1. downloads the BTS-preprocessed **training** set (`sync.zip`, ~50k RGB-depth pairs) and unzips it,
2. downloads the official labeled `.mat` (the source of the test split),
3. extracts the **654 Eigen test** images with [scripts/extract_nyu_test.py](scripts/extract_nyu_test.py), then deletes the `.mat`.

Resulting layout:
```
data/nyu_depth_v2/
  sync/                 ~50k training pairs (rgb_xxxxx.jpg + sync_depth_xxxxx.png, 16-bit mm depth)
  official_splits/test/ 654 test pairs
```

Download somewhere else with `DATA_DIR=/path/to/nyu bash scripts/download_nyu.sh` (then point the config there). On Windows, keep the data **outside** OneDrive to avoid syncing gigabytes.

### Verify it loaded

```bash
python scripts/check_data.py --config configs/experiments/01_resnet50_nyu.yaml --split train
```
Prints the sample count and the valid depth range (should be ~0–10 m) and saves a preview to `outputs/data_check.png`.
