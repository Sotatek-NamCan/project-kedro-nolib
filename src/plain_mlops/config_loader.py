"""Utilities for loading pipeline configuration files."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


_YAML_SUFFIXES = {".yml", ".yaml"}


class ConfigFormatError(ValueError):
    """Raised when the configuration file cannot be parsed."""


def _infer_project_root(config_path: Path) -> Path:
    """Guess the project root based on the config path."""
    config_dir = config_path.parent
    if config_dir.name.lower() == "config":
        return config_dir.parent
    return config_dir


def _load_from_file(config_path: Path) -> Dict[str, Any]:
    """Load a JSON or YAML file into a dictionary."""
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix in _YAML_SUFFIXES:
        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ConfigFormatError(
            f"Unsupported config format '{suffix}'. Use .yml, .yaml, or .json."
        )

    if not isinstance(data, dict):
        raise ConfigFormatError("Top-level configuration must be a mapping.")
    return data


@dataclass(frozen=True)
class ConfigBundle:
    data: Dict[str, Any]
    config_path: Path
    project_root: Path

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a relative path against the project root."""
        path = Path(value)
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        return path


def load_config(
    config_path: str | Path, *, project_root: str | Path | None = None
) -> ConfigBundle:
    """Load the config file and return the parsed content with helpful context."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at '{path}'.")

    data = _load_from_file(path)
    root = (
        Path(project_root).expanduser().resolve()
        if project_root
        else _infer_project_root(path)
    )
    return ConfigBundle(data=data, config_path=path, project_root=root)


def extract_sections(
    bundle: ConfigBundle, *section_names: str
) -> Tuple[Dict[str, Any], ...]:
    """Return requested sections ensuring they exist."""
    sections = []
    for name in section_names:
        try:
            section = bundle.data[name]
        except KeyError as exc:
            raise KeyError(f"Missing '{name}' section in configuration.") from exc
        if not isinstance(section, dict):
            raise ConfigFormatError(f"Section '{name}' must be a mapping.")
        sections.append(section)
    return tuple(sections)

