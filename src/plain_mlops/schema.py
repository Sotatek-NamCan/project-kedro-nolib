"""Schema loading helpers built on top of pandera."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml
from pandera import Check, Column, DataFrameSchema


_COLUMN_OPTION_KEYS = {
    "nullable",
    "allow_duplicates",
    "required",
    "unique",
    "regex",
    "coerce",
}

_SCHEMA_OPTION_KEYS = {
    "coerce",
    "ordered",
    "strict",
    "report_duplicates",
    "unique_column_names",
}

_DTYPE_ALIASES = {
    "int": int,
    "int64": int,
    "integer": int,
    "float": float,
    "float64": float,
    "double": float,
    "str": str,
    "string": str,
    "object": object,
    "any": object,
    "bool": bool,
    "boolean": bool,
}


def _resolve_dtype(dtype: Any) -> Any:
    if dtype is None:
        return object
    if isinstance(dtype, str):
        normalized = dtype.lower()
        return _DTYPE_ALIASES.get(normalized, dtype)
    return dtype


def _build_checks(checks_config: Any) -> Iterable[Check]:
    if not checks_config:
        return ()

    if isinstance(checks_config, dict):
        items = checks_config.items()
    else:
        items = []
        for entry in checks_config:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(
                    "Each check configuration must be a mapping with a single key."
                )
            items.append(next(iter(entry.items())))

    checks = []
    for check_name, check_args in items:
        if not hasattr(Check, check_name):
            raise ValueError(f"Unsupported check '{check_name}'.")
        check_factory = getattr(Check, check_name)
        if isinstance(check_args, dict):
            checks.append(check_factory(**check_args))
        elif isinstance(check_args, list):
            try:
                checks.append(check_factory(*check_args))
            except TypeError:
                checks.append(check_factory(check_args))
        elif check_args is None:
            checks.append(check_factory())
        else:
            checks.append(check_factory(check_args))
    return tuple(checks)


def _build_schema(schema_config: Dict[str, Any]) -> DataFrameSchema:
    columns_cfg = schema_config.get("columns")
    if not columns_cfg:
        raise ValueError("Schema configuration must define 'columns'.")

    schema_kwargs = {
        key: schema_config[key]
        for key in _SCHEMA_OPTION_KEYS
        if key in schema_config
    }

    columns = {}
    for column_name, column_config in columns_cfg.items():
        dtype = _resolve_dtype(column_config.get("dtype"))
        checks = _build_checks(column_config.get("checks"))
        column_kwargs = {
            key: column_config[key]
            for key in _COLUMN_OPTION_KEYS
            if key in column_config
        }
        columns[column_name] = Column(dtype, checks=checks or None, **column_kwargs)

    return DataFrameSchema(columns, **schema_kwargs)


def load_schema_definition(config: Dict[str, Any], *, project_root: Path) -> Dict[str, Any]:
    file_path = config.get("file_path")
    if not file_path:
        raise ValueError("Missing 'file_path' in schema configuration parameters.")

    path = Path(file_path)
    if not path.is_absolute():
        path = (project_root / path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Schema file not found at '{path}'.")

    with path.open("r", encoding="utf-8") as stream:
        schema_def = yaml.safe_load(stream) or {}

    if not isinstance(schema_def, dict):
        raise ValueError("Schema definition file must contain a mapping at the top level.")

    return schema_def


def resolve_schema(
    schema_definitions: Dict[str, Any], *, schema_key: str | None
) -> DataFrameSchema:
    if "columns" in schema_definitions:
        return _build_schema(schema_definitions)
    if schema_key:
        try:
            return _build_schema(schema_definitions[schema_key])
        except KeyError as exc:
            available = ", ".join(sorted(schema_definitions))
            raise KeyError(
                f"Schema '{schema_key}' not found. Available schemas: {available}"
            ) from exc
    if len(schema_definitions) == 1:
        return _build_schema(next(iter(schema_definitions.values())))
    raise ValueError(
        "Multiple schemas provided but no 'schema_key' specified in parameters."
    )

