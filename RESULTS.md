# Results

Metrics on the original (clean) test splits. AbsRel ↓, RMSE ↓ (meters), δ1 ↑.
Each row is `best.pt` (best AbsRel epoch) for that experiment.

## NYU Depth V2

| # | Approach | Encoder | AbsRel ↓ | RMSE ↓ | δ1 ↑ | Notes |
|---|----------|---------|----------|--------|------|-------|
| 1 | Supervised baseline | ResNet-50 | **0.1376** | **0.4950** | **0.8224** | epoch 11, converged; RunPod 4090 |
| 2 | Frozen SD-UNet + decoder | SD 2.1 UNet | — | — | — | todo |
| 3 | Frozen I-JEPA + decoder | I-JEPA ViT-H | — | — | — | todo |
| 4 | Fusion SD + I-JEPA | SD + I-JEPA | — | — | — | todo |
| 5 | DepthAnything (eval-only) | ViT | — | — | — | todo (SOTA upper bound) |

## KITTI

| # | Approach | Encoder | AbsRel ↓ | RMSE ↓ | δ1 ↑ | Notes |
|---|----------|---------|----------|--------|------|-------|
| 1 | Supervised baseline | ResNet-50 | — | — | — | todo |
| 2 | Frozen SD-UNet + decoder | SD 2.1 UNet | — | — | — | todo |
| 3 | Frozen I-JEPA + decoder | I-JEPA ViT-H | — | — | — | todo |
| 4 | Fusion SD + I-JEPA | SD + I-JEPA | — | — | — | todo |
| 5 | DepthAnything (eval-only) | ViT | — | — | — | todo |

## Robustness (degraded test sets)

Same trained models, no retraining, on fog / blur / exposure at light|moderate|severe.
To be filled after Phase 6.

---

### Raw numbers (for reference)

- **01_resnet50_nyu** (best.pt, epoch 11):
  `{'abs_rel': 0.13760100123358937, 'rmse': 0.49495159780106895, 'delta1': 0.8223643273842044}`
