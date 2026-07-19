"""Placeholder so CI has a passing suite before Phase 1+ tests exist. Remove once real tests land."""

import yaml


def test_config_loads():
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["seed"] == 42
    assert config["dataset"]["split"] == "HI-Small"
