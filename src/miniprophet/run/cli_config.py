"""CLI-only configuration (separate from agent YAML configs)."""

from __future__ import annotations

from pathlib import Path

import yaml
from platformdirs import user_config_dir

CLI_CONFIG_DIR = Path(user_config_dir("miniprophet"))
CLI_CONFIG_FILE = CLI_CONFIG_DIR / "cli.yaml"

DEFAULTS: dict = {
    "default_market_service": "kalshi",
    "kalshi_api_base": "https://api.elections.kalshi.com/trade-api/v2",
}


def load_cli_config() -> dict:
    """Load CLI config from disk, falling back to defaults."""
    config = dict(DEFAULTS)
    if CLI_CONFIG_FILE.exists():
        try:
            on_disk = yaml.safe_load(CLI_CONFIG_FILE.read_text()) or {}
            config.update(on_disk)
        except Exception:
            pass
    return config
