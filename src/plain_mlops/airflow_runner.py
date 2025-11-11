"""Stage-oriented CLI to support Airflow DockerOperator tasks."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .config_loader import load_config
from .ingestion import DataIngestorFactory
from .processing import (
    evaluate,
    split_features_label,
    split_train_test,
    train_model,
    validate_and_clean,
)
from .schema import load_schema_definition, resolve_schema
from .storage import build_storage_client


def _ensure_tuple(values) -> Tuple[str, ...]:
    if not values:
        return tuple()
    if isinstance(values, str):
        return (values,)
    if isinstance(values, (list, tuple)):
        return tuple(values)
    raise ValueError(f"Expected list/tuple/str, received {type(values).__name__}.")


def _read_metadata(path: Path) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found at '{path}'.")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_metadata(path: Path, payload: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_bundle(config: str, project_root: str | None):
    return load_config(config, project_root=project_root)


def stage_download(args: argparse.Namespace) -> str:
    bundle = _load_bundle(args.config, args.project_root)
    ingest_cfg = bundle.data.get("ingest") or {}
    storage_cfg = ingest_cfg.get("object_storage") or {}
    dataset_key = storage_cfg.get("object_key") or os.getenv("OBJECT_STORAGE_DATASET_KEY")
    bucket = (
        storage_cfg.get("bucket")
        or os.getenv("OBJECT_STORAGE_DATASET_BUCKET")
        or os.getenv("OBJECT_STORAGE_BUCKET")
    )
    file_path = ingest_cfg.get("file_path")

    download_dir = Path(args.download_dir).resolve()
    download_dir.mkdir(parents=True, exist_ok=True)

    if dataset_key:
        destination = download_dir / Path(dataset_key).name
        client = build_storage_client(bucket=bucket)
        client.download(object_key=dataset_key, destination=destination, bucket=bucket)
    else:
        if not file_path:
            raise ValueError("Neither 'file_path' nor 'object_key' configured for ingestion.")
        source = bundle.resolve_path(file_path)
        destination = download_dir / source.name
        shutil.copy2(source, destination)

    metadata_path = Path(args.metadata_path).resolve()
    _write_metadata(metadata_path, {"dataset_path": str(destination)})
    print(str(destination))
    return str(destination)


def stage_ingest(args: argparse.Namespace) -> str:
    bundle = _load_bundle(args.config, args.project_root)
    ingest_cfg = bundle.data.get("ingest") or {}
    metadata = _read_metadata(Path(args.download_metadata).resolve())
    dataset_path = Path(metadata["dataset_path"])

    extension = ingest_cfg.get("file_extension") or dataset_path.suffix
    if not extension:
        raise ValueError(
            "Unable to determine dataset file extension. Please configure 'file_extension'."
        )
    ingestor = DataIngestorFactory.get_data_ingestor(extension)
    df = ingestor.ingest(dataset_path)

    output_path = Path(args.ingest_output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    metadata_path = Path(args.metadata_path).resolve()
    _write_metadata(metadata_path, {"ingested_path": str(output_path)})
    print(str(output_path))
    return str(output_path)


def stage_split(args: argparse.Namespace) -> Dict[str, str]:
    bundle = _load_bundle(args.config, args.project_root)
    prepare_cfg = bundle.data.get("prepare") or {}
    schema_cfg = bundle.data.get("schema") or {}

    ingest_metadata = _read_metadata(Path(args.ingest_metadata).resolve())
    ingested_path = Path(ingest_metadata["ingested_path"])
    df = pd.read_parquet(ingested_path)

    schema_def = load_schema_definition(schema_cfg, project_root=bundle.project_root)
    schema = resolve_schema(schema_def, schema_key=schema_cfg.get("schema_key"))

    drop_columns = _ensure_tuple(
        prepare_cfg.get("drop_columns") or prepare_cfg.get("drop_cols")
    )
    feature_cols = _ensure_tuple(prepare_cfg.get("feature_cols"))
    numeric_cols = _ensure_tuple(prepare_cfg.get("numeric_cols"))
    categorical_cols = _ensure_tuple(prepare_cfg.get("categorical_cols"))
    label_col = prepare_cfg.get("label_col")
    if not label_col:
        raise ValueError("prepare.label_col must be provided in the configuration.")

    cleaned = validate_and_clean(df, schema=schema, drop_columns=drop_columns or tuple())
    X, y = split_features_label(cleaned, feature_cols=feature_cols, label_col=label_col)

    split_cfg = {
        "test_size": prepare_cfg.get("test_size", 0.2),
        "random_state": prepare_cfg.get("random_state", 42),
    }
    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
    )

    splits_dir = Path(args.splits_dir).resolve()
    splits_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "X_train": splits_dir / "X_train.parquet",
        "X_test": splits_dir / "X_test.parquet",
        "y_train": splits_dir / "y_train.parquet",
        "y_test": splits_dir / "y_test.parquet",
    }

    X_train.to_parquet(outputs["X_train"], index=False)
    X_test.to_parquet(outputs["X_test"], index=False)
    y_train.to_frame(name=label_col).to_parquet(outputs["y_train"], index=False)
    y_test.to_frame(name=label_col).to_parquet(outputs["y_test"], index=False)

    payload = {name: str(path) for name, path in outputs.items()}
    payload["label_col"] = label_col
    payload["numeric_cols"] = list(numeric_cols)
    payload["categorical_cols"] = list(categorical_cols)
    metadata_path = Path(args.metadata_path).resolve()
    _write_metadata(metadata_path, payload)

    print(json.dumps(payload))
    return payload


def stage_train(args: argparse.Namespace) -> Dict[str, float]:
    bundle = _load_bundle(args.config, args.project_root)
    model_cfg = bundle.data.get("model") or {}
    output_cfg = bundle.data.get("output") or {}

    splits_metadata = _read_metadata(Path(args.splits_metadata).resolve())
    label_col = splits_metadata.get("label_col")
    if not label_col:
        raise ValueError("Split metadata is missing 'label_col'.")

    X_train = pd.read_parquet(splits_metadata["X_train"])
    X_test = pd.read_parquet(splits_metadata["X_test"])
    y_train = pd.read_parquet(splits_metadata["y_train"])[label_col]
    y_test = pd.read_parquet(splits_metadata["y_test"])[label_col]

    numeric_cols = tuple(splits_metadata.get("numeric_cols") or ())
    categorical_cols = tuple(splits_metadata.get("categorical_cols") or ())

    model = train_model(
        X_train,
        y_train,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        model_cfg=model_cfg,
    )
    metrics = evaluate(model, X_test, y_test)

    metrics_path = output_cfg.get("metrics_path")
    if metrics_path:
        metrics_file = bundle.resolve_path(metrics_path)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    model_path = output_cfg.get("model_path")
    if model_path:
        model_file = bundle.resolve_path(model_path)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        import joblib  # Local import to avoid unnecessary dependency when unused.

        joblib.dump(model, model_file)

    print(json.dumps(metrics))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Utility runner for orchestrating plain-mlops stages from Airflow."
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.yml",
        help="Path to the pipeline configuration file inside the container.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root used to resolve relative paths.",
    )

    subparsers = parser.add_subparsers(dest="task", required=True)

    download = subparsers.add_parser("download", help="Download raw dataset.")
    download.add_argument("--download-dir", required=True, help="Directory to store the dataset.")
    download.add_argument(
        "--metadata-path",
        required=True,
        help="Path to write metadata referencing the downloaded dataset.",
    )

    ingest = subparsers.add_parser("ingest", help="Ingest raw dataset into a clean table.")
    ingest.add_argument(
        "--download-metadata",
        required=True,
        help="Metadata file produced by the download stage.",
    )
    ingest.add_argument(
        "--ingest-output",
        required=True,
        help="Path to write the ingested dataset (Parquet).",
    )
    ingest.add_argument(
        "--metadata-path",
        required=True,
        help="Path to write metadata referencing the ingested dataset.",
    )

    split = subparsers.add_parser("split", help="Split ingested dataset into train/test.")
    split.add_argument(
        "--ingest-metadata",
        required=True,
        help="Metadata file produced by the ingest stage.",
    )
    split.add_argument(
        "--splits-dir",
        required=True,
        help="Directory to save the split Parquet files.",
    )
    split.add_argument(
        "--metadata-path",
        required=True,
        help="Path to write metadata describing the split file locations.",
    )

    train = subparsers.add_parser("train", help="Train model from prepared splits.")
    train.add_argument(
        "--splits-metadata",
        required=True,
        help="Metadata file produced by the split stage.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.task == "download":
        stage_download(args)
    elif args.task == "ingest":
        stage_ingest(args)
    elif args.task == "split":
        stage_split(args)
    elif args.task == "train":
        stage_train(args)
    else:
        raise ValueError(f"Unsupported task '{args.task}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
