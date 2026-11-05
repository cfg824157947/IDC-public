"""Canonical serialization and hashing for resolved configuration objects."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (ValueError, TypeError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"configuration contains unsupported value {type(value).__name__}")


def canonical_config_json(value: Any) -> str:
    """Serialize JSON configuration with sorted keys and stable compact form."""
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def canonical_config_hash(value: Any) -> str:
    """SHA-256 for the canonical representation used by fits and run plans."""
    return hashlib.sha256(canonical_config_json(value).encode("utf-8")).hexdigest()


def resolved_run_hashes(scientific_config: Any, execution_config: Any) -> dict[str, str]:
    """Hash scientific state, execution settings, and their resolved pairing."""
    science = _json_value(scientific_config)
    execution = _json_value(execution_config)
    return {
        "scientific_config_sha256": canonical_config_hash(science),
        "execution_config_sha256": canonical_config_hash(execution),
        "run_config_sha256": canonical_config_hash(
            {"scientific_config": science, "execution_config": execution}
        ),
    }
