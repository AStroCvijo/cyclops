from pathlib import Path

import yaml


# Function for merging `extra` into `base` (nested dicts merge one level deeper, otherwise overwrite)
def _merge(base, extra):
    for key, value in extra.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


# Function for loading a YAML config and applying its `_base_` includes
def load_config(path):
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    cfg = {}
    for base in raw.pop("_base_", []):
        _merge(cfg, load_config(path.parent / base))
    _merge(cfg, raw)
    return cfg
