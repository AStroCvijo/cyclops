# Roadmap

Implementation plan for [cyclops](README.md). Guiding principle: **get one model working end-to-end on NYU first, then widen.** Don't build all five encoders in parallel before the training loop runs.

## Phase 0 — Foundations
- [x] `utils/config.py` — YAML loader with `_base_` inheritance (list-merge in order, later keys win)
- [x] `utils/logging.py` — W&B init + console logger + step logging
- [x] `utils/checkpoint.py` — save/load, `save_best` on `monitor=abs_rel` (`mode=min`)
- [x] `metrics/depth.py` — `abs_rel`, `rmse`, `delta1` (masked)
- [x] smoke-test the config loader against an experiment file (`tests/test_config.py`, 11 passing)

## Phase 1 — Data (NYU first)
- [x] `data/datasets.py` — NYU dataset: RGB+depth, resize, clamp, `valid_mask`. **KITTI dropped (out of scope — time).**
- [x] `data/transforms.py` — normalization + train-only augmentation (kept separate from degradations)
- [x] DataLoader sanity check (visualize a batch) — `scripts/check_data.py`; train 36k / test 654, depth ~0–10 m, RGB↔depth aligned

## Phase 2 — First model end-to-end (Approach 1, baseline)
- [x] `models/encoders/base.py` — `Encoder` interface (returns multi-scale feature maps)
- [x] `models/encoders/resnet50.py` — torchvision ResNet-50, features at strides 4/8/16/32
- [x] `models/decoders/lightweight.py` — feature pyramid → dense depth
- [x] `models/build.py` — factory assembling encoder + decoder (+ fusion later) from config
- [x] `models/losses.py` — SiLog loss
- [x] `engine/trainer.py` — AMP, grad clip, cosine + warmup, `trainable` filter, W&B, best checkpoint
- [x] `engine/evaluator.py` — metrics, eval crop (eigen), optional median align
- [x] `scripts/train.py`, `scripts/evaluate.py` — thin CLI wrappers
- [x] **run `01_resnet50_nyu` — baseline trained (RunPod 4090)**. Converged ~epoch 11: **abs_rel 0.138, rmse 0.495, delta1 0.822** (NYU clean test). Plateaued (epochs 11→15 flat).

## Phase 3 — Frozen encoders (Approaches 2 & 3)
- [x] `models/encoders/sd_unet.py` — frozen SD UNet, single forward at `timestep=1`, hook `feature_blocks`, auto-probe channels. Uses ungated SD-1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`; SD-2.1 repos are gated). Running `02_sd_unet_nyu` (RunPod).
- [x] `models/encoders/ijepa.py` — frozen I-JEPA ViT-H/14, taps layers `[8,16,24,32]`, tokens → grids
- [x] `models/decoders/dpt.py` — DPT-style reassembly of ViT tokens → pyramid → top-down fusion. Wired in `build.py`, config `03_ijepa_nyu`. **Trained: abs_rel 0.131 (ep12), beats baseline.**

## Phase 4 — Fusion (Approach 4)
- [x] `models/fusion.py` — `ConcatFusion` (parameter-free, default) and `CrossAttentionFusion` (SD queries attend to I-JEPA). `build.py` handles a `model.encoders` list → `FusionModel` with trainable `fusion` + `decoder`. Config `04_fusion_nyu`. **Not yet trained.**

## Phase 5 — SOTA reference (Approach 5, eval-only)
- [x] `models/depth_anything.py` — DepthAnything V2 zero-shot (`DepthAnythingModel`, HF `AutoModelForDepthEstimation`), inverts disparity → depth, `align: median`, no checkpoint. Config `05_depth_anything_nyu`. V3: swap `model_id` once its HF checkpoint is up.

## Phase 6 — Robustness
- [x] `scripts/generate_degraded.py` — writes fog/blur/exposure × light/moderate/severe test sets to disk (deterministic, CPU-only), depth copied unchanged
- [x] `scripts/evaluate_robustness.py` — evaluates any/all models on clean + 9 degraded sets, no retraining; `evaluate.py --degradation/--severity` for a single condition. Writes `outputs/robustness.json`.

## Phase 7 — Analysis & write-up
- [ ] `scripts/aggregate_results.py` — collect metrics, trainable params, train time, inference time per image
- [ ] qualitative depth-map visualizations (esp. degraded before/after)
- [ ] *(optional)* fine-tune selected models on a mix of original + degraded data
- [ ] report

## Critical notes
- **Scope: NYU Depth V2 only.** KITTI is dropped — time-bound, no budget to retrain every model on a second 0–80 m dataset. All five approaches are compared on NYU.
- **Masks are mandatory.** NYU has invalid pixels (depth = 0). Compute every metric and the loss only over `valid_mask`.
- The milestone is finishing the baseline's full cycle on NYU; everything after is adding encoders, not new infrastructure.
