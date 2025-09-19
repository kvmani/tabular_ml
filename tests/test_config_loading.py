from __future__ import annotations

import config as config_module


def test_config_layering(tmp_path, monkeypatch):
    config_module.load_settings.cache_clear()
    local_cfg = tmp_path / "config.local.yaml"
    local_cfg.write_text('app:\n  log_level: "DEBUG"\n')
    monkeypatch.setattr(config_module, "LOCAL_CONFIG_PATH", local_cfg)
    monkeypatch.setenv("TABULAR_ML__APP__NAME", "Layered Test")

    settings = config_module.load_settings()

    assert settings.app.log_level == "DEBUG"
    assert settings.app.name == "Layered Test"

    config_module.load_settings.cache_clear()
