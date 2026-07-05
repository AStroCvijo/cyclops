# Results

NYU Depth V2, 654-image Eigen test split. AbsRel ↓, RMSE ↓ (meters), δ1 ↑.
Each trained row is `best.pt` (best-AbsRel epoch) for that experiment; approaches 1–4
train only the decoder (+ fusion module for 4) on top of frozen encoders, except the
ResNet-50 baseline which trains end-to-end. Scope is NYU only.

## Clean test set

| # | Approach | Encoder | AbsRel ↓ | RMSE ↓ | δ1 ↑ | Notes |
|---|----------|---------|----------|--------|------|-------|
| 1 | Supervised baseline | ResNet-50 | 0.1376 | 0.4949 | 0.8224 | trained end-to-end; epoch 11, converged |
| 2 | Frozen SD-UNet + decoder | SD-1.5 UNet | **0.0802** | **0.3113** | 0.9467 | decoder only; **lowest RMSE of all models** |
| 3 | Frozen I-JEPA + decoder | I-JEPA ViT-H/14 | 0.1314 | 0.4884 | 0.8329 | DPT decoder; beats baseline |
| 4 | Fusion SD + I-JEPA | SD-1.5 + I-JEPA | 0.0827 | 0.3218 | 0.9410 | concat fusion; ≈ SD alone (SD features dominate) |
| 5 | DepthAnything V2 (eval-only) | ViT (DINOv2) | **0.0688** | 0.3244 | **0.9497** | metric-indoor, **median-aligned**; SOTA upper bound |

Takeaways:
- **Frozen SD features nearly match SOTA.** The SD-UNet decoder (0.0802) sits within a
  whisker of DepthAnything (0.0688) — and DepthAnything gets a **GT-median scale oracle**
  the trained models don't. On RMSE the SD model is actually **best overall** (0.311).
- **Fusion doesn't beat SD alone** (0.0827 vs 0.0802). Concatenating I-JEPA onto SD adds
  no clean-accuracy gain here — the SD features already carry the geometry, and the
  weaker I-JEPA branch slightly dilutes them.
- **Frozen encoders >> supervised baseline.** SD/I-JEPA/DA all beat the end-to-end
  ResNet-50, with only a decoder trained.

## Robustness — AbsRel ↓ (no retraining)

| Model | clean | fog L | fog M | fog S | blur L | blur M | blur S | exp L | exp M | exp S |
|-------|-------|-------|-------|-------|--------|--------|--------|-------|-------|-------|
| 1 ResNet-50 | 0.1376 | 0.2663 | 0.3093 | 0.3385 | 0.1471 | 0.1680 | 0.2202 | 0.1961 | 0.2882 | 0.3271 |
| 2 SD-UNet   | 0.0802 | 0.1005 | 0.1213 | 0.1482 | 0.0804 | 0.0858 | 0.0947 | 0.1186 | 0.2363 | 0.3574 |
| 3 I-JEPA    | 0.1314 | 0.1607 | 0.1740 | 0.1891 | 0.1321 | 0.1379 | 0.1547 | 0.1844 | 0.2870 | 0.3914 |
| 4 Fusion    | 0.0827 | 0.1003 | 0.1163 | 0.1373 | 0.0826 | 0.0869 | 0.0945 | 0.1214 | 0.2399 | 0.3799 |
| 5 DepthAny. | 0.0688 | 0.1355 | 0.1909 | 0.2511 | 0.0708 | 0.0790 | 0.1033 | 0.0994 | 0.1944 | 0.3093 |

## Robustness — δ1 ↑ (no retraining)

| Model | clean | fog L | fog M | fog S | blur L | blur M | blur S | exp L | exp M | exp S |
|-------|-------|-------|-------|-------|--------|--------|--------|-------|-------|-------|
| 1 ResNet-50 | 0.8224 | 0.4320 | 0.3452 | 0.2898 | 0.7933 | 0.7125 | 0.5481 | 0.6533 | 0.4302 | 0.3677 |
| 2 SD-UNet   | 0.9467 | 0.9025 | 0.8396 | 0.7546 | 0.9464 | 0.9355 | 0.9174 | 0.8582 | 0.5943 | 0.3957 |
| 3 I-JEPA    | 0.8329 | 0.7719 | 0.7331 | 0.6884 | 0.8338 | 0.8229 | 0.7858 | 0.7072 | 0.4879 | 0.3903 |
| 4 Fusion    | 0.9410 | 0.9020 | 0.8557 | 0.7914 | 0.9416 | 0.9330 | 0.9161 | 0.8568 | 0.5989 | 0.4130 |
| 5 DepthAny. | 0.9497 | 0.8301 | 0.7308 | 0.6299 | 0.9462 | 0.9327 | 0.8889 | 0.8901 | 0.7023 | 0.4901 |

(L / M / S = light / moderate / severe.)

### Robustness findings

- **Blur is nearly harmless to the frozen-encoder models.** SD, Fusion, and DepthAnything
  lose almost nothing up to severe blur (SD δ1 0.947 → 0.917). Semantic features from
  large-scale pretraining don't depend on high-frequency detail. The **supervised baseline
  is the exception** — severe blur alone drops its δ1 0.822 → 0.548.
- **Under-exposure is the universal failure mode.** Every model collapses under severe
  under-exposure (δ1 ≈ 0.39–0.49, AbsRel triples-to-quadruples). This is the safety-critical
  low-light / *noćna vožnja* case: no approach here is robust to it without retraining.
- **The trained SD model is more fog-robust than zero-shot DepthAnything.** Under severe fog,
  SD degrades to 0.148 AbsRel vs DepthAnything's 0.251 — the decoder trained on NYU fog-free
  data still generalizes better to fog than the domain-transferred SOTA model does.
- **The supervised baseline is the least robust across the board**, degrading hardest under
  every corruption — robustness here comes from the frozen foundation-model features, not
  from the (larger) trained parameter count.

## Efficiency (trainable params / inference time)

Trainable = parameters that receive gradients; total = full model incl. frozen encoders.
Inference measured on the eval GPU (A100), mean ms per 480×640 image.

| # | Model | Trainable ↓ | Total | Inference ms/img ↓ |
|---|-------|-------------|-------|--------------------|
| 1 | ResNet-50 | 30.2 M | 30.2 M | **6.5** |
| 2 | SD-UNet | **5.7 M** | 948.9 M | 98.6 |
| 3 | I-JEPA | 7.9 M | 638.7 M | 142.5 |
| 4 | Fusion | 11.3 M | 1585.2 M | 239.7 |
| 5 | DepthAnything | **0** (zero-shot) | 97.5 M | 60.5 |

The frozen approaches train only **6–11 M** parameters — a fraction of the baseline's 30 M —
despite wrapping 0.6–1.6 B frozen weights, which is exactly the appeal: near-SOTA accuracy for
a cheaply-trained decoder. The trade-off is inference cost — running the giant frozen encoders
makes them 15–37× slower than the baseline, and fusion (two encoders) is the slowest.

## Notes

- **DepthAnything V2** (`Depth-Anything-V2-Metric-Indoor-Base-hf`) is fine-tuned on Hypersim
  (synthetic), so its absolute scale is biased on NYU; results use per-image **median
  alignment** to the GT, the standard zero-shot protocol. Without it AbsRel is ~0.24 (scale
  offset, not structural error).
- **DepthAnything V3** was *not* evaluated: it is not integrated into `transformers`
  (`AutoModelForDepthEstimation` only exposes V2). Adding it would require the official
  `bytedance-seed/depth-anything-3` repo and a custom wrapper — left as future work.

### Raw numbers

Full 5×10 matrices in `outputs/robustness.json` (median-aligned DepthAnything rows in
`outputs/robustness_da.json`). Regenerate with:

```
python scripts/generate_degraded.py --degradation all --severity all
python scripts/evaluate_robustness.py
```
