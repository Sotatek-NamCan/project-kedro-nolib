"""Plain Python pipeline orchestrator inspired by demo-2 Kedro project."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib

from .config_loader import ConfigBundle, extract_sections
from .ingestion import ingest_data
from .processing import DatasetSplits, run_training_workflow
from .schema import load_schema_definition, resolve_schema
from .storage import (
    ObjectStorageConfigurationError,
    ObjectStorageOperationError,
    build_storage_client,
)


def _ensure_tuple(
    section: Dict[str, Any],
    primary_key: str,
    fallback_key: str | None = None,
    *,
    required: bool = True,
) -> tuple:
    raw = section.get(primary_key)
    if raw is None and fallback_key:
        raw = section.get(fallback_key)
    if raw is None:
        if required:
            raise KeyError(
                f"Missing required key '{primary_key}' in configuration section."
            )
        return tuple()
    if isinstance(raw, str):
        return (raw,)
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"Expected a list/tuple for '{primary_key}', received {type(raw).__name__}."
        )
    return tuple(raw)


@dataclass(frozen=True)
class PipelineResult:
    model: Any
    metrics: Dict[str, float]
    splits: DatasetSplits


class DataPipeline:
    def __init__(self, bundle: ConfigBundle) -> None:
        self.bundle = bundle

    def _load_schema(self, schema_cfg: Dict[str, Any]):
        schema_def = load_schema_definition(
            schema_cfg, project_root=self.bundle.project_root
        )
        schema_key = schema_cfg.get("schema_key")
        return resolve_schema(schema_def, schema_key=schema_key)

    def run(self) -> PipelineResult:
        ingest_cfg, schema_cfg, prepare_cfg = extract_sections(
            self.bundle, "ingest", "schema", "prepare"
        )
        model_cfg = self.bundle.data.get("model") or {}
        output_cfg = self.bundle.data.get("output") or {}

        dataset = ingest_data(ingest_cfg, project_root=self.bundle.project_root)
        schema = self._load_schema(schema_cfg)

        drop_columns = _ensure_tuple(prepare_cfg, "drop_columns", "drop_cols", required=False)
        feature_cols = _ensure_tuple(prepare_cfg, "feature_cols")
        numeric_cols = _ensure_tuple(prepare_cfg, "numeric_cols")
        categorical_cols = _ensure_tuple(prepare_cfg, "categorical_cols")
        label_col = prepare_cfg.get("label_col")
        if not label_col:
            raise KeyError("prepare.label_col must be provided.")

        split_cfg = {
            "test_size": prepare_cfg.get("test_size", 0.2),
            "random_state": prepare_cfg.get("random_state", 42),
        }

        model, metrics, splits = run_training_workflow(
            dataset,
            schema=schema,
            drop_columns=drop_columns,
            feature_cols=feature_cols,
            label_col=label_col,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            splitter_cfg=split_cfg,
            model_cfg=model_cfg,
        )

        self._persist_artifacts(metrics, model, output_cfg)
        return PipelineResult(model=model, metrics=metrics, splits=splits)

    def _persist_artifacts(
        self, metrics: Dict[str, float], model: Any, output_cfg: Dict[str, Any]
    ) -> None:
        metrics_path_cfg = output_cfg.get("metrics_path")
        model_path_cfg = output_cfg.get("model_path")
        metrics_file: Path | None = None
        model_file: Path | None = None

        if metrics_path_cfg:
            metrics_file = self.bundle.resolve_path(metrics_path_cfg)
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if model_path_cfg:
            model_file = self.bundle.resolve_path(model_path_cfg)
            model_file.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_file)

        self._mirror_to_object_storage(
            metrics_file=metrics_file,
            model_file=model_file,
            output_cfg=output_cfg,
        )

    def _mirror_to_object_storage(
        self,
        *,
        metrics_file: Path | None,
        model_file: Path | None,
        output_cfg: Dict[str, Any],
    ) -> None:
        storage_cfg = output_cfg.get("object_storage") or {}
        bucket_override = (
            storage_cfg.get("bucket")
            or os.getenv("OBJECT_STORAGE_OUTPUT_BUCKET")
            or os.getenv("OBJECT_STORAGE_BUCKET")
        )
        prefix = storage_cfg.get("prefix") or os.getenv("OBJECT_STORAGE_OUTPUT_PREFIX")
        metrics_key_override = storage_cfg.get("metrics_key")
        model_key_override = storage_cfg.get("model_key")
        enabled = storage_cfg.get("enabled", False)

        should_upload = enabled or any(
            [
                prefix,
                metrics_key_override,
                model_key_override,
                bucket_override,
            ]
        )
        if not should_upload:
            return

        try:
            client = build_storage_client(bucket=bucket_override)
        except ObjectStorageConfigurationError:
            if enabled:
                raise RuntimeError(
                    "Object storage mirroring is enabled but OBJECT_STORAGE_* variables are missing."
                )
            return

        def _build_key(default_name: str, override: str | None) -> str:
            if override:
                return override.lstrip("/")
            if not prefix:
                return default_name
            cleaned = prefix.strip().strip("/")
            return f"{cleaned}/{default_name}" if cleaned else default_name

        def _upload(file_path: Path | None, key_override: str | None) -> None:
            if not file_path or not file_path.is_file():
                return
            key = _build_key(file_path.name, key_override)
            try:
                client.upload(source=file_path, object_key=key, bucket=bucket_override)
            except ObjectStorageOperationError as exc:
                raise RuntimeError(
                    f"Failed to upload '{file_path}' to object storage."
                ) from exc

        _upload(metrics_file, metrics_key_override)
        _upload(model_file, model_key_override)
