# Results

Metrics on the original (clean) test splits. AbsRel ↓, RMSE ↓ (meters), δ1 ↑.
Each row is `best.pt` (best AbsRel epoch) for that experiment.

## NYU Depth V2

| # | Approach | Encoder | AbsRel ↓ | RMSE ↓ | δ1 ↑ | Notes |
|---|----------|---------|----------|--------|------|-------|
| 1 | Supervised baseline | ResNet-50 | 0.1376 | 0.4950 | 0.8224 | epoch 11, converged; RunPod 4090 |
| 2 | Frozen SD-UNet + decoder | SD 2.1 UNet | — | — | — | todo |
| 3 | Frozen I-JEPA + decoder | I-JEPA ViT-H | **0.1314** | **0.4884** | **0.8329** | epoch 12 (best of 20, plateaued); **beats baseline**; DPT decoder, RunPod |
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
- **03_ijepa_nyu** (best.pt, epoch 12; full 20-epoch run, plateaued after ep12):
  `{'abs_rel': 0.13141962177142863, 'rmse': 0.4883969404348513, 'delta1': 0.83289836456136}`
  Frozen I-JEPA ViT-H/14 encoder, trainable DPT decoder, batch 16, lr 2e-4. Trajectory: ep0 0.199 → ep4 0.138 → **ep12 0.131 (best)** → ep13–19 flat (0.132–0.135). Beats the ResNet-50 baseline on all three metrics with only the decoder trained.
