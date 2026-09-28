"""Tests for config loading (src/idxlab/config.py)."""

from idxlab.config import load_config

YAML_TEXT = """
data:
  tickers: [AAA.JK, BBB.JK]
  start_date: "2024-01-01"
  end_date: "2024-06-30"
logging:
  level: DEBUG
"""


def test_load_config_parses_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("IDXLAB_LOG_LEVEL", raising=False)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(YAML_TEXT, encoding="utf-8")

    cfg = load_config(cfg_file)

    assert cfg.tickers == ["AAA.JK", "BBB.JK"]
    assert cfg.start_date == "2024-01-01"
    assert cfg.log_level == "DEBUG"


def test_env_var_overrides_log_level(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(YAML_TEXT, encoding="utf-8")
    monkeypatch.setenv("IDXLAB_LOG_LEVEL", "warning")

    cfg = load_config(cfg_file)

    assert cfg.log_level == "WARNING"