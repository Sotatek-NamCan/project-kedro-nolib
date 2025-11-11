"""Configurable data ingestion utilities."""
from __future__ import annotations

import os
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import pandas as pd

from .storage import (
    ObjectStorageConfigurationError,
    ObjectStorageOperationError,
    build_storage_client,
)


def _normalize_extension(extension: str) -> str:
    ext = extension.lower().strip()
    if not ext:
        raise ValueError("File extension cannot be empty.")
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext


def _validate_file_exists(file_path: str | Path) -> Path:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"No file found at path: {path}")
    return path


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str | Path) -> pd.DataFrame:
        raise NotImplementedError


class PandasReaderIngestor(DataIngestor):
    """Generic ingestor wrapper around a pandas reader callable."""

    def __init__(
        self,
        reader: Callable[..., pd.DataFrame],
        supported_extensions: Iterable[str],
        reader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._reader = reader
        self._extensions = tuple(
            _normalize_extension(ext) for ext in supported_extensions
        )
        self._reader_kwargs = reader_kwargs or {}

    def ingest(self, file_path: str | Path) -> pd.DataFrame:
        path = _validate_file_exists(file_path)
        extension = path.suffix
        if not extension:
            raise ValueError(
                f"Unable to determine the file extension for '{path.name}'. "
                f"Supported extensions: {self._extensions}"
            )

        normalized_extension = _normalize_extension(extension)
        if normalized_extension not in self._extensions:
            raise ValueError(
                f"Unsupported file extension '{normalized_extension}' for {self.__class__.__name__}. "
                f"Expected one of {self._extensions}."
            )
        return self._reader(path, **self._reader_kwargs)


class CSVDataIngestor(PandasReaderIngestor):
    def __init__(self) -> None:
        super().__init__(reader=pd.read_csv, supported_extensions=(".csv",))


class TSVDataIngestor(PandasReaderIngestor):
    def __init__(self) -> None:
        super().__init__(
            reader=pd.read_csv, supported_extensions=(".tsv",), reader_kwargs={"sep": "\t"}
        )


class JSONDataIngestor(PandasReaderIngestor):
    def __init__(self) -> None:
        super().__init__(reader=pd.read_json, supported_extensions=(".json",))


class ExcelDataIngestor(PandasReaderIngestor):
    def __init__(self) -> None:
        super().__init__(reader=pd.read_excel, supported_extensions=(".xlsx", ".xls"))


class ParquetDataIngestor(PandasReaderIngestor):
    def __init__(self) -> None:
        super().__init__(reader=pd.read_parquet, supported_extensions=(".parquet",))


class ZipDataIngestor(DataIngestor):
    """Extract a ZIP archive and delegate ingestion to an inner ingestor."""

    def __init__(self, extract_dir: Optional[str] = None) -> None:
        self._extract_dir = Path(extract_dir or "extracted_data")

    def ingest(self, file_path: str | Path) -> pd.DataFrame:
        path = _validate_file_exists(file_path)
        normalized_extension = _normalize_extension(path.suffix)
        if normalized_extension != ".zip":
            raise ValueError("The provided file is not a .zip file.")

        self._extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self._extract_dir)

        supported_extensions = set(
            DataIngestorFactory.supported_extensions(exclude_archive=True)
        )
        extracted_files = []
        for root, _, files in os.walk(self._extract_dir):
            for file_name in files:
                extension = Path(file_name).suffix
                if not extension:
                    continue
                normalized = _normalize_extension(extension)
                if normalized in supported_extensions:
                    extracted_files.append(Path(root) / file_name)

        if not extracted_files:
            raise FileNotFoundError("No supported data file found in the extracted data.")
        if len(extracted_files) > 1:
            raise ValueError(
                "Multiple supported files found in the ZIP archive. "
                "Please specify which file to use."
            )

        inner_file_path = extracted_files[0]
        inner_ingestor = DataIngestorFactory.get_data_ingestor(inner_file_path.suffix)
        return inner_ingestor.ingest(inner_file_path)


class DataIngestorFactory:
    """Factory responsible for instantiating the appropriate DataIngestor."""

    _INGESTOR_REGISTRY: Dict[str, Callable[[], DataIngestor]] = {
        ".csv": CSVDataIngestor,
        ".tsv": TSVDataIngestor,
        ".json": JSONDataIngestor,
        ".xlsx": ExcelDataIngestor,
        ".xls": ExcelDataIngestor,
        ".parquet": ParquetDataIngestor,
        ".zip": ZipDataIngestor,
    }

    @classmethod
    def get_data_ingestor(
        cls, file_extension: str | Path, *, zip_extract_dir: Optional[str] = None
    ) -> DataIngestor:
        normalized_extension = _normalize_extension(str(file_extension))
        try:
            ingestor_cls = cls._INGESTOR_REGISTRY[normalized_extension]
        except KeyError as exc:
            supported = ", ".join(sorted(cls._INGESTOR_REGISTRY))
            raise ValueError(
                f"No ingestor available for file extension: {normalized_extension}. "
                f"Supported extensions: {supported}"
            ) from exc

        if normalized_extension == ".zip":
            return ingestor_cls(zip_extract_dir)
        return ingestor_cls()

    @classmethod
    def supported_extensions(cls, *, exclude_archive: bool = False) -> Tuple[str, ...]:
        if exclude_archive:
            return tuple(ext for ext in cls._INGESTOR_REGISTRY if ext != ".zip")
        return tuple(cls._INGESTOR_REGISTRY)


def _resolve_cache_dir(value: Optional[str], *, project_root: Path) -> Path:
    if not value:
        return project_root / "data" / "99_object_storage"
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path


def _strip_leading_slash(value: str) -> str:
    return value.lstrip("/\\")


def ingest_data(config: Dict[str, Any], *, project_root: Path) -> pd.DataFrame:
    """Load a dataset either from the filesystem or from object storage."""
    storage_cfg = config.get("object_storage") or {}
    dataset_key = storage_cfg.get("object_key") or os.getenv("OBJECT_STORAGE_DATASET_KEY")
    bucket_override = (
        storage_cfg.get("bucket")
        or os.getenv("OBJECT_STORAGE_DATASET_BUCKET")
        or os.getenv("OBJECT_STORAGE_BUCKET")
    )
    use_object_storage = storage_cfg.get("enabled", False) or bool(dataset_key)

    extract_dir = config.get("zip_extract_dir")
    if extract_dir and not Path(extract_dir).is_absolute():
        extract_dir = str(project_root / extract_dir)

    if use_object_storage:
        if not dataset_key:
            raise ValueError(
                "Object storage ingestion is enabled but no dataset object key was provided."
            )
        cache_dir = _resolve_cache_dir(storage_cfg.get("cache_dir"), project_root=project_root)
        relative_key = Path(_strip_leading_slash(dataset_key))
        resolved_path = cache_dir / relative_key
        try:
            client = build_storage_client(bucket=bucket_override)
            client.download(object_key=dataset_key, destination=resolved_path, bucket=bucket_override)
        except ObjectStorageConfigurationError as exc:
            raise RuntimeError(
                "Failed to configure object storage client for ingestion. "
                "Ensure OBJECT_STORAGE_* variables are set."
            ) from exc
        except ObjectStorageOperationError as exc:
            raise RuntimeError(
                f"Unable to download dataset '{dataset_key}' from object storage."
            ) from exc
        extension = config.get("file_extension") or resolved_path.suffix or Path(dataset_key).suffix
    else:
        file_path = config.get("file_path")
        if not file_path:
            raise ValueError(
                "Missing 'file_path' in ingestion configuration and no object storage dataset configured."
            )
        resolved_path = (
            project_root / file_path if not Path(file_path).is_absolute() else Path(file_path)
        )
        extension = config.get("file_extension") or Path(resolved_path).suffix

    ingestor = DataIngestorFactory.get_data_ingestor(
        extension, zip_extract_dir=extract_dir
    )
    return ingestor.ingest(resolved_path)
