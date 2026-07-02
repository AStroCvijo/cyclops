"""Train one experiment.

    python scripts/train.py --config configs/experiments/01_resnet50_nyu.yaml
"""

import argparse

from cyclops.engine.trainer import train
from cyclops.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
