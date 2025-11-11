"""Helpers to interact with an S3-compatible object storage."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError


class ObjectStorageConfigurationError(RuntimeError):
    """Raised when the object storage client is not fully configured."""


class ObjectStorageOperationError(RuntimeError):
    """Raised when a storage operation fails."""


@dataclass(frozen=True)
class ObjectStorageConfig:
    endpoint_url: str
    access_key: str
    secret_key: str
    default_bucket: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ObjectStorageConfig":
        endpoint = os.getenv("OBJECT_STORAGE_ENDPOINT_URL") or os.getenv(
            "OBJECT_STORAGE_URL"
        )
        access_key = os.getenv("OBJECT_STORAGE_ACCESS_KEY")
        secret_key = os.getenv("OBJECT_STORAGE_SECRET_KEY")
        default_bucket = os.getenv("OBJECT_STORAGE_BUCKET")

        missing = [
            name
            for name, value in [
                ("OBJECT_STORAGE_ENDPOINT_URL", endpoint),
                ("OBJECT_STORAGE_ACCESS_KEY", access_key),
                ("OBJECT_STORAGE_SECRET_KEY", secret_key),
            ]
            if not value
        ]
        if missing:
            joined = ", ".join(missing)
            raise ObjectStorageConfigurationError(
                f"Missing required environment variables for object storage: {joined}."
            )
        return cls(
            endpoint_url=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            default_bucket=default_bucket,
        )


class ObjectStorageClient:
    """Small wrapper around boto3 for S3-compatible endpoints."""

    def __init__(
        self, config: ObjectStorageConfig, *, default_bucket: str | None = None
    ) -> None:
        self._client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
        )
        self._default_bucket = default_bucket or config.default_bucket

    def _resolve_bucket(self, bucket: str | None) -> str:
        bucket_name = bucket or self._default_bucket
        if not bucket_name:
            raise ObjectStorageConfigurationError(
                "Bucket name must be provided either via OBJECT_STORAGE_BUCKET or "
                "per-operation overrides."
            )
        return bucket_name

    def download(
        self, *, object_key: str, destination: Path, bucket: str | None = None
    ) -> Path:
        """Download an object into ``destination`` and return the path."""
        try:
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            bucket_name = self._resolve_bucket(bucket)
            self._client.download_file(bucket_name, object_key, str(destination))
            return destination
        except (BotoCoreError, ClientError) as exc:
            raise ObjectStorageOperationError(
                f"Failed to download '{object_key}' from bucket '{bucket_name}': {exc}"
            ) from exc

    def upload(
        self, *, source: Path, object_key: str, bucket: str | None = None
    ) -> None:
        """Upload a local ``source`` file to ``object_key``."""
        try:
            bucket_name = self._resolve_bucket(bucket)
            self._client.upload_file(str(source), bucket_name, object_key)
        except (BotoCoreError, ClientError) as exc:
            raise ObjectStorageOperationError(
                f"Failed to upload '{source}' to '{bucket_name}/{object_key}': {exc}"
            ) from exc


def build_storage_client(*, bucket: str | None = None) -> ObjectStorageClient:
    """Instantiate a storage client using environment variables."""
    config = ObjectStorageConfig.from_env()
    return ObjectStorageClient(config, default_bucket=bucket or config.default_bucket)

