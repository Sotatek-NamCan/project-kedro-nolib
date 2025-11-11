# Plain Python MLOps Demo

This project reproduces the `demo-2` Kedro pipeline using only standard Python
modules, so every behaviour is controlled via YAML/JSON configs instead of Kedro
hooks. You can describe where the data lives, how to process it, and which model
to train just by editing `config/pipeline.yml` (or pointing the CLI to a JSON
file with the same structure). The dataset schema is defined separately under
`config/schema/` so you keep data contracts close to the code. Because
everything lives behind a single CLI entry point you can drop this repo into a
container image and reuse it across multiple Airflow DAGs—each DAG only needs to
mount its own config/env file before triggering the run.

## Layout

- `config/pipeline.yml` – end-to-end pipeline definition (ingestion, features,
  split, model, output paths) for the churn dataset.
- `config/pipeline_marketing.yml` – alternate configuration that targets a
  marketing-leads dataset with a different schema/model setup.
- `config/schema/churn_customers.yml` – Pandera schema for the churn dataset.
- `config/schema/marketing_leads.yml` – Pandera schema for the marketing dataset.
- `data/01_raw/marketing_leads.csv` – lightweight marketing-leads sample to
  showcase how the same codebase adapts to another domain.
- `src/plain_mlops/` – plain Python implementation of ingest/validation/modeling.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -e .
```

All dependencies are defined in `pyproject.toml`.

## Running the pipeline

```bash
python -m plain_mlops --config config/pipeline.yml
```

By default the CLI resolves paths relative to the project root. Use
`--project-root` if your config lives elsewhere, and `--metrics-json` to mirror
metrics to an extra location.

The marketing sample can be executed with:

```bash
python -m plain_mlops --config config/pipeline_marketing.yml
```

The default config writes to `outputs/metrics.json` and `outputs/model.joblib`,
while the marketing profile stores artifacts under `outputs/marketing/`. The
CLI also prints the metrics to stdout so you can quickly see how a config change
affects performance.

## Airflow / container-friendly workflow

Use the project as a reusable “training runtime” container for multiple DAGs:

1. **Bake the code once** – build an image (or publish a wheel) that contains
   this repository and its dependencies. The only runtime inputs are the config
   file plus environment variables.
2. **Create DAG-specific configs** – for each DAG, add a YAML/JSON config (and
   optional `.env`) describing ingestion, schema, model, outputs, and
   object-storage credentials. Store them in your scheduler’s dataset bucket or
   the DAG repository.
3. **Mount and run inside the task** – in an Airflow `DockerOperator` /
   `KubernetesPodOperator` / `PythonVirtualenvOperator`, mount or download the
   config/env, then call:
   ```bash
   python -m plain_mlops --config /path/to/dag-config.yml \
          --project-root /path/within/container
   ```
   The CLI returns a 0/1 exit code that Airflow can use for retries.
4. **Parameterise dynamically** – DAG params/variables can point to different
   configs, override `OBJECT_STORAGE_*` env vars, or provide alternative
   `--metrics-json` mirrors so the same container image powers multiple training
   jobs without rebuilding.

### Packaging as a reusable code base

1. **Base image** – create a Dockerfile that installs system deps, copies this
   repo, and runs `pip install -e .`. Example:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /opt/plain-mlops
   COPY . .
   RUN pip install --no-cache-dir -e .
   ENTRYPOINT ["python", "-m", "plain_mlops"]
   ```
2. **Config injection** – at runtime, mount a volume (or download from object
   storage) containing `pipeline.yml` plus optional `.env`.
3. **Runtime contract** – your orchestrator only needs to provide three things:
   the config file path, optional `--project-root`, and any env vars (object
   storage credentials, experiment metadata, etc.).

Build once (`docker build -t plain-mlops:latest .`) and push to the registry
used by your orchestrator. Every DAG/job reuses the same artifact.

### Example Airflow `DockerOperator`

```python
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime

with DAG("train_churn_model", start_date=datetime(2024, 1, 1), schedule="@daily") as dag:
    train = DockerOperator(
        task_id="run_training",
        image="registry.example.com/plain-mlops:latest",
        command="python -m plain_mlops --config /configs/churn.yml",
        environment={
            "OBJECT_STORAGE_ENDPOINT_URL": "https://minio.internal",
            "OBJECT_STORAGE_ACCESS_KEY": "{{ var.value.ml_access_key }}",
            "OBJECT_STORAGE_SECRET_KEY": "{{ var.value.ml_secret_key }}",
        },
        mounts=["/airflow/configs/churn.yml:/configs/churn.yml:ro"],
        auto_remove=True,
    )
```

Swap the mount path, config, or env vars per DAG/task to launch different
pipelines without changing the base image.

## Object storage integration

The pipeline can pull raw datasets and push trained artefacts to any
S3-compatible object storage (AWS S3, MinIO, etc.).

1. Duplicate `.env` if needed and fill in `OBJECT_STORAGE_ENDPOINT_URL`,
   `OBJECT_STORAGE_ACCESS_KEY`, `OBJECT_STORAGE_SECRET_KEY`, and the bucket /
   object key fields. The package automatically loads `.env` via
   [`python-dotenv`](https://pypi.org/project/python-dotenv/).
2. Enable remote ingestion by setting `ingest.object_storage.enabled: true`.
   Optionally provide `object_key`, `bucket`, or `cache_dir` overrides in the
   config; otherwise the values in `.env` are used. The dataset is downloaded to
   the cache directory before the existing validation/feature steps run.
3. Mirror outputs by toggling `output.object_storage.enabled: true` (or by
   defining `OBJECT_STORAGE_OUTPUT_PREFIX`). You can override the bucket, prefix,
   or explicit object keys for metrics/model via `output.object_storage`.

When the object storage blocks are disabled the project falls back to the local
paths defined under `ingest.file_path` and `output.*`, so you can continue to
develop locally without a remote bucket.

## Customising via config

- `ingest.file_path` / `file_extension` – point to any CSV/TSV/JSON/Parquet/ZIP
  file and the project will auto-select the reader.
- `schema.file_path` / `schema_key` – choose which Pandera schema to apply.
- `prepare` – control dropped columns, label, feature lists, and the split
  parameters.
- `model` – choose `type` (`logreg` or `rf`) and hyper-parameters. You can define
  multiple options and switch between them by changing `model.selected`.
- `output` – where to store the trained model and metrics artefacts.

Because everything is plain Python you can import `plain_mlops.pipeline` inside
notebooks, swap out components, or embed the pipeline in a larger orchestration
tool without taking a Kedro dependency.

