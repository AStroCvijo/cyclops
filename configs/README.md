# Configs

One file in `experiments/` is one experiment (one row in the results table).

## Layout
- `_base/`: shared defaults. `default.yaml` for training/eval/logging, plus one dataset file per benchmark (`nyu.yaml`, `kitti.yaml`).
- `degradations/`: fog, blur, and exposure definitions, each with `light`, `moderate`, and `severe` levels.
- `experiments/`: the 5 approaches. Each includes `_base` files via `_base_` and sets only what differs (mostly the `model` block).

## Inheritance
`_base_` lists files merged in order; later keys override earlier ones. For `02_sd_unet_nyu.yaml` the order is `default.yaml`, then `nyu.yaml`, then the file itself.

## Running
```
python scripts/train.py    --config configs/experiments/02_sd_unet_nyu.yaml
python scripts/evaluate.py --config configs/experiments/02_sd_unet_nyu.yaml --degradation fog --severity severe
```

## Adding an approach
Add a file in `experiments/`, set its `model` block, and include a `_base` dataset. No engine changes are needed as long as the encoder implements the `Encoder` interface.
