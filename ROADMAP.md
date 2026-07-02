# Roadmap

Implementation plan for [cyclops](README.md). Guiding principle: **get one model working end-to-end on NYU first, then widen.** Don't build all five encoders in parallel before the training loop runs.

## Phase 0 — Foundations
- [x] `utils/config.py` — YAML loader with `_base_` inheritance (list-merge in order, later keys win)
- [x] `utils/logging.py` — W&B init + console logger + step logging
- [x] `utils/checkpoint.py` — save/load, `save_best` on `monitor=abs_rel` (`mode=min`)
- [x] `metrics/depth.py` — `abs_rel`, `rmse`, `delta1` (masked)
- [x] smoke-test the config loader against an experiment file (`tests/test_config.py`, 11 passing)

## Phase 1 — Data (NYU first)
- [x] `data/datasets.py` — NYU dataset: RGB+depth, resize, clamp, `valid_mask`. **KITTI deferred.**
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
- [x] **run `01_resnet50_nyu` end-to-end — pipeline works** (1-epoch smoke: loss 6.8→1.2, abs_rel 0.155, delta1 0.773, checkpoint + W&B ok). Rerun full 20 epochs for the reported baseline.

## Phase 3 — Frozen encoders (Approaches 2 & 3)
- [ ] `models/encoders/sd_unet.py` — frozen SD-2.1 UNet, single forward at `timestep=1`, tap `feature_blocks` → run `02_sd_unet_nyu`
- [ ] `models/encoders/ijepa.py` — frozen I-JEPA ViT, layers `[5,11,17,23]`
- [ ] `models/decoders/dpt.py` — DPT-style reassembly of ViT tokens → run `03_ijepa_nyu`

## Phase 4 — Fusion (Approach 4)
- [ ] `models/fusion.py` — `concat` (first) and `cross_attention`; `build.py` supports an `encoders` list → run `04_fusion_nyu`

## Phase 5 — SOTA reference (Approach 5, eval-only)
- [ ] `models/encoders/depth_anything.py` — DepthAnything V2 then V3 zero-shot, `align: median`, eval-only

## Phase 6 — Robustness
- [ ] `scripts/generate_degraded.py` — write degraded test sets to disk once (deterministic)
- [ ] evaluate all 5 models on original + 3 degradations × 3 severities, no retraining

## Phase 7 — KITTI replication
- [ ] add KITTI to the dataset (sparse mask, Garg crop, 0–80 m), then re-run Phases 2–6 with `*_kitti` configs

## Phase 8 — Analysis & write-up
- [ ] `scripts/aggregate_results.py` — collect metrics, trainable params, train time, inference time per image
- [ ] qualitative depth-map visualizations (esp. degraded before/after)
- [ ] *(optional)* fine-tune selected models on a mix of original + degraded data
- [ ] report

## Critical notes
- **Masks are mandatory everywhere.** KITTI gt is sparse, and NYU also has invalid pixels (depth = 0). Compute every metric and the loss only over `valid_mask`.
- **Don't touch KITTI until the full NYU pipeline runs** — mixing 0–10 m and 0–80 m ranges is the most common source of silent bugs.
- The milestone is finishing the baseline's full cycle on NYU; everything after is adding encoders, not new infrastructure.
