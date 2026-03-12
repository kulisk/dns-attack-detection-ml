"""
Configuration loader – reads YAML config files and provides
dot-access to nested settings.  Supports environment variable
overrides via a ``.env`` file.
"""
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """Load and provide access to project YAML configuration.

    Example::

        cfg = ConfigLoader()
        batch_size = cfg.get("supervised.mlp.batch_size", default=256)
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        load_dotenv()
        self._config_path = Path(config_path)
        self._config: dict[str, Any] = self._load()
        logger.info("Configuration loaded", extra={"path": str(self._config_path)})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a config value using dot-separated key.

        Args:
            key: Dot-separated path, e.g. ``"supervised.random_forest.n_estimators"``.
            default: Value returned when key is not found.

        Returns:
            Config value or *default*.
        """
        keys = key.split(".")
        value: Any = self._config
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                return default
            value = value[k]
        return value

    def get_section(self, section: str) -> dict[str, Any]:
        """Return an entire top-level section as a dict."""
        return self._config.get(section, {})

    def all(self) -> dict[str, Any]:
        """Return the full configuration dictionary."""
        return self._config

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if not self._config_path.exists():
            logger.warning(
                "Config file not found – using defaults",
                extra={"path": str(self._config_path)},
            )
            return {}
        with open(self._config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        # Override with environment variables (prefix DNS_)
        data = self._apply_env_overrides(data)
        return data

    @staticmethod
    def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
        """Allow ``DNS_SECTION__KEY=value`` env vars to override config."""
        for env_key, env_val in os.environ.items():
            if not env_key.startswith("DNS_"):
                continue
            parts = env_key[4:].lower().split("__")
            node = config
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = env_val
        return config
