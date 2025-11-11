"""Plain Python pipeline orchestrator inspired by demo-2 Kedro project."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

import pandas as pd

import joblib

from .config_loader import ConfigBundle, extract_sections
from .ingestion import ingest_data
from .processing import DatasetSplits, evaluate, run_training_workflow, train_model
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

        precomputed_cfg = prepare_cfg.get("precomputed_splits") or {}
        use_precomputed = self._should_use_precomputed_splits(precomputed_cfg)

        if use_precomputed:
            splits = self._load_precomputed_splits(precomputed_cfg)
            model = train_model(
                splits.X_train,
                splits.y_train,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                model_cfg=model_cfg,
            )
            metrics = evaluate(model, splits.X_test, splits.y_test)
        else:
            dataset = ingest_data(ingest_cfg, project_root=self.bundle.project_root)
            schema = self._load_schema(schema_cfg)
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

        self._persist_artifacts(metrics, model, output_cfg, splits)
        return PipelineResult(model=model, metrics=metrics, splits=splits)

    def _persist_artifacts(
        self,
        metrics: Dict[str, float],
        model: Any,
        output_cfg: Dict[str, Any],
        splits: DatasetSplits | None,
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
            splits=splits,
        )

    def _mirror_to_object_storage(
        self,
        *,
        metrics_file: Path | None,
        model_file: Path | None,
        output_cfg: Dict[str, Any],
        splits: DatasetSplits | None,
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
        splits_cfg = storage_cfg.get("splits") or {}
        splits_bucket_override = (
            splits_cfg.get("bucket")
            or os.getenv("OBJECT_STORAGE_SPLITS_BUCKET")
            or bucket_override
        )
        splits_prefix = (
            splits_cfg.get("prefix")
            or os.getenv("OBJECT_STORAGE_SPLITS_PREFIX")
            or prefix
        )
        splits_format = (
            splits_cfg.get("format")
            or os.getenv("OBJECT_STORAGE_SPLITS_FORMAT")
            or "csv"
        ).lower()
        splits_requested = bool(splits) and (
            splits_cfg.get("enabled", False)
            or any(
                splits_cfg.get(name)
                for name in (
                    "x_train_key",
                    "y_train_key",
                    "x_test_key",
                    "y_test_key",
                    "bucket",
                    "prefix",
                )
            )
        )

        should_upload = enabled or any(
            [
                prefix,
                metrics_key_override,
                model_key_override,
                bucket_override,
            ]
        ) or splits_requested
        if not should_upload:
            return

        try:
            client = build_storage_client(
                bucket=bucket_override or splits_bucket_override
            )
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

        if not splits or not splits_requested:
            return

        if splits_format not in {"csv", "parquet"}:
            raise ValueError(
                f"Unsupported splits serialization format '{splits_format}'. "
                "Supported formats: csv, parquet."
            )

        split_overrides = {
            "X_train": splits_cfg.get("x_train_key"),
            "y_train": splits_cfg.get("y_train_key"),
            "X_test": splits_cfg.get("x_test_key"),
            "y_test": splits_cfg.get("y_test_key"),
        }

        def _build_split_key(default_name: str, override: str | None) -> str:
            if override:
                return override.lstrip("/")
            target_prefix = splits_prefix or ""
            cleaned = target_prefix.strip().strip("/")
            split_dir = f"{cleaned}/splits" if cleaned else "splits"
            return f"{split_dir}/{default_name}"

        with TemporaryDirectory() as tmp_dir:
            tmp_base = Path(tmp_dir)

            def _serialize_and_upload(
                name: str, data: pd.DataFrame | pd.Series, override: str | None
            ) -> None:
                filename = f"{name}.{splits_format}"
                path = tmp_base / filename
                # Ensure Series are converted to DataFrames for consistent serialization.
                serializable = (
                    data.to_frame(name=data.name or "value")
                    if isinstance(data, pd.Series)
                    else data
                )
                if splits_format == "csv":
                    serializable.to_csv(path, index=False)
                else:
                    serializable.to_parquet(path, index=False)
                key = _build_split_key(filename, override)
                try:
                    client.upload(
                        source=path,
                        object_key=key,
                        bucket=splits_bucket_override,
                    )
                except ObjectStorageOperationError as exc:
                    raise RuntimeError(
                        f"Failed to upload dataset split '{name}' to object storage."
                    ) from exc

            _serialize_and_upload("X_train", splits.X_train, split_overrides["X_train"])
            _serialize_and_upload("y_train", splits.y_train, split_overrides["y_train"])
            _serialize_and_upload("X_test", splits.X_test, split_overrides["X_test"])
            _serialize_and_upload("y_test", splits.y_test, split_overrides["y_test"])

    @staticmethod
    def _should_use_precomputed_splits(cfg: Dict[str, Any]) -> bool:
        if not cfg:
            return False
        return cfg.get("enabled", False) or any(
            cfg.get(name)
            for name in (
                "x_train_key",
                "y_train_key",
                "x_test_key",
                "y_test_key",
                "bucket",
                "prefix",
            )
        )

    def _load_precomputed_splits(self, cfg: Dict[str, Any]) -> DatasetSplits:
        splits_bucket = (
            cfg.get("bucket")
            or os.getenv("OBJECT_STORAGE_SPLITS_INPUT_BUCKET")
            or os.getenv("OBJECT_STORAGE_SPLITS_BUCKET")
            or os.getenv("OBJECT_STORAGE_BUCKET")
        )
        splits_prefix = (
            cfg.get("prefix")
            or os.getenv("OBJECT_STORAGE_SPLITS_INPUT_PREFIX")
            or os.getenv("OBJECT_STORAGE_SPLITS_PREFIX")
        )
        splits_format = (
            cfg.get("format")
            or os.getenv("OBJECT_STORAGE_SPLITS_INPUT_FORMAT")
            or os.getenv("OBJECT_STORAGE_SPLITS_FORMAT")
            or "csv"
        ).lower()
        if splits_format not in {"csv", "parquet"}:
            raise ValueError(
                f"Unsupported precomputed splits format '{splits_format}'. "
                "Supported formats: csv, parquet."
            )

        overrides = {
            "X_train": cfg.get("x_train_key"),
            "y_train": cfg.get("y_train_key"),
            "X_test": cfg.get("x_test_key"),
            "y_test": cfg.get("y_test_key"),
        }

        def _default_key(name: str) -> str:
            default_filename = f"{name}.{splits_format}"
            if overrides[name]:
                return overrides[name].lstrip("/")
            target_prefix = splits_prefix or ""
            cleaned = target_prefix.strip().strip("/")
            split_dir = f"{cleaned}/splits" if cleaned else "splits"
            return f"{split_dir}/{default_filename}"

        try:
            client = build_storage_client(bucket=splits_bucket)
        except ObjectStorageConfigurationError as exc:
            raise RuntimeError(
                "Failed to configure object storage client for precomputed splits."
            ) from exc

        loaded: Dict[str, pd.DataFrame | pd.Series] = {}
        with TemporaryDirectory() as tmp_dir:
            tmp_base = Path(tmp_dir)

            def _download_and_load(name: str, expects_series: bool) -> None:
                filename = f"{name}.{splits_format}"
                destination = tmp_base / filename
                object_key = _default_key(name)
                try:
                    client.download(
                        object_key=object_key,
                        destination=destination,
                        bucket=splits_bucket,
                    )
                except ObjectStorageOperationError as exc:
                    raise RuntimeError(
                        f"Unable to download precomputed split '{name}' "
                        f"from object storage (key='{object_key}')."
                    ) from exc

                if splits_format == "csv":
                    frame = pd.read_csv(destination)
                else:
                    frame = pd.read_parquet(destination)
                if expects_series:
                    if frame.shape[1] == 0:
                        raise ValueError(f"Split '{name}' has no columns.")
                    series = frame.iloc[:, 0].copy()
                    loaded[name] = series.reset_index(drop=True)
                else:
                    loaded[name] = frame.reset_index(drop=True)

            _download_and_load("X_train", expects_series=False)
            _download_and_load("y_train", expects_series=True)
            _download_and_load("X_test", expects_series=False)
            _download_and_load("y_test", expects_series=True)

        return DatasetSplits(
            X_train=loaded["X_train"],
            X_test=loaded["X_test"],
            y_train=loaded["y_train"],
            y_test=loaded["y_test"],
        )
