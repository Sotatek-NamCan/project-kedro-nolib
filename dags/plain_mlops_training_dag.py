"""Airflow DAG that orchestrates the plain-mlops pipeline inside Docker containers."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator


HERE = Path(__file__).resolve()
HOST_PROJECT_ROOT = Path(
    os.environ.get("PLAIN_MLOPS_HOST_PROJECT_ROOT", HERE.parents[1])
).resolve()
CONTAINER_PROJECT_ROOT = os.environ.get(
    "PLAIN_MLOPS_CONTAINER_PROJECT_ROOT", "/opt/plain-mlops"
)
CONFIG_PATH = os.environ.get(
    "PLAIN_MLOPS_CONFIG_PATH",
    f"{CONTAINER_PROJECT_ROOT}/config/pipeline.yml",
)
IMAGE = os.environ.get("PLAIN_MLOPS_IMAGE", "plain-mlops:latest")

STATE_DIR = os.environ.get(
    "PLAIN_MLOPS_STATE_DIR", f"{CONTAINER_PROJECT_ROOT}/data/.airflow_state"
)
DOWNLOAD_DIR = os.environ.get(
    "PLAIN_MLOPS_DOWNLOAD_DIR", f"{CONTAINER_PROJECT_ROOT}/data/00_download"
)
INGEST_OUTPUT = os.environ.get(
    "PLAIN_MLOPS_INGEST_OUTPUT",
    f"{CONTAINER_PROJECT_ROOT}/data/02_intermediate/ingested_dataset.parquet",
)
SPLITS_DIR = os.environ.get(
    "PLAIN_MLOPS_SPLITS_DIR", f"{CONTAINER_PROJECT_ROOT}/data/03_primary"
)

DOWNLOAD_METADATA = os.environ.get(
    "PLAIN_MLOPS_DOWNLOAD_METADATA", f"{STATE_DIR}/download.json"
)
INGEST_METADATA = os.environ.get(
    "PLAIN_MLOPS_INGEST_METADATA", f"{STATE_DIR}/ingest.json"
)
SPLITS_METADATA = os.environ.get(
    "PLAIN_MLOPS_SPLITS_METADATA", f"{STATE_DIR}/splits.json"
)

DOCKER_URL = os.environ.get("PLAIN_MLOPS_DOCKER_URL", "unix://var/run/docker.sock")
DOCKER_NETWORK = os.environ.get("PLAIN_MLOPS_DOCKER_NETWORK")

_volumes_raw = os.environ.get(
    "PLAIN_MLOPS_DOCKER_VOLUMES",
    f"{HOST_PROJECT_ROOT}:{CONTAINER_PROJECT_ROOT}",
)
VOLUMES = [entry for entry in _volumes_raw.split(";") if entry]

ENV_PASSTHROUGH_KEYS = [
    "OBJECT_STORAGE_ENDPOINT_URL",
    "OBJECT_STORAGE_ACCESS_KEY",
    "OBJECT_STORAGE_SECRET_KEY",
    "OBJECT_STORAGE_BUCKET",
    "OBJECT_STORAGE_DATASET_BUCKET",
    "OBJECT_STORAGE_DATASET_KEY",
    "OBJECT_STORAGE_OUTPUT_BUCKET",
    "OBJECT_STORAGE_OUTPUT_PREFIX",
    "OBJECT_STORAGE_SPLITS_BUCKET",
    "OBJECT_STORAGE_SPLITS_PREFIX",
    "OBJECT_STORAGE_SPLITS_FORMAT",
    "OBJECT_STORAGE_SPLITS_INPUT_BUCKET",
    "OBJECT_STORAGE_SPLITS_INPUT_PREFIX",
    "OBJECT_STORAGE_SPLITS_INPUT_FORMAT",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
]
ENV_VARS = {key: os.environ[key] for key in ENV_PASSTHROUGH_KEYS if key in os.environ}


def _runner_cmd(stage: str, **options) -> list[str]:
    cmd = [
        "python",
        "-m",
        "plain_mlops.airflow_runner",
        stage,
        f"--config={CONFIG_PATH}",
        f"--project-root={CONTAINER_PROJECT_ROOT}",
    ]
    for key, value in options.items():
        if value is None:
            continue
        flag = key.replace("_", "-")
        cmd.append(f"--{flag}={value}")
    return cmd


default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="plain_mlops_training",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 6 * * *",
    catchup=False,
    tags=["mlops", "plain-python"],
) as dag:
    download = DockerOperator(
        task_id="download_data",
        image=IMAGE,
        command=_runner_cmd(
            "download",
            download_dir=DOWNLOAD_DIR,
            metadata_path=DOWNLOAD_METADATA,
        ),
        api_version="auto",
        docker_url=DOCKER_URL,
        network_mode=DOCKER_NETWORK,
        mount_tmp_dir=False,
        volumes=VOLUMES,
        environment=ENV_VARS,
        auto_remove=True,
        tty=False,
    )

    ingest = DockerOperator(
        task_id="ingest_data",
        image=IMAGE,
        command=_runner_cmd(
            "ingest",
            download_metadata=DOWNLOAD_METADATA,
            ingest_output=INGEST_OUTPUT,
            metadata_path=INGEST_METADATA,
        ),
        api_version="auto",
        docker_url=DOCKER_URL,
        network_mode=DOCKER_NETWORK,
        mount_tmp_dir=False,
        volumes=VOLUMES,
        environment=ENV_VARS,
        auto_remove=True,
        tty=False,
    )

    split = DockerOperator(
        task_id="split_data",
        image=IMAGE,
        command=_runner_cmd(
            "split",
            ingest_metadata=INGEST_METADATA,
            splits_dir=SPLITS_DIR,
            metadata_path=SPLITS_METADATA,
        ),
        api_version="auto",
        docker_url=DOCKER_URL,
        network_mode=DOCKER_NETWORK,
        mount_tmp_dir=False,
        volumes=VOLUMES,
        environment=ENV_VARS,
        auto_remove=True,
        tty=False,
    )

    train = DockerOperator(
        task_id="train_model",
        image=IMAGE,
        command=_runner_cmd(
            "train",
            splits_metadata=SPLITS_METADATA,
        ),
        api_version="auto",
        docker_url=DOCKER_URL,
        network_mode=DOCKER_NETWORK,
        mount_tmp_dir=False,
        volumes=VOLUMES,
        environment=ENV_VARS,
        auto_remove=True,
        tty=False,
    )

    download >> ingest >> split >> train
