from pathlib import Path

from cyclops.utils.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_inheritance_and_override():
    cfg = load_config(CONFIGS / "experiments" / "02_sd_unet_nyu.yaml")

    assert cfg["training"]["epochs"] == 20
    assert cfg["dataset"]["name"] == "nyu_depth_v2"
    assert cfg["model"]["encoder"]["name"] == "sd_unet"

    assert cfg["training"]["optimizer"]["lr"] == 2.0e-4
    assert cfg["training"]["optimizer"]["weight_decay"] == 1.0e-2


def test_all_experiment_configs_load():
    for path in (CONFIGS / "experiments").glob("*.yaml"):
        cfg = load_config(path)
        assert cfg["experiment"]["name"]
        assert "model" in cfg
