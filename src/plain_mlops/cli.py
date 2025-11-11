"""Command line interface for running the plain Kedro-free pipeline."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from .config_loader import load_config
from .pipeline import DataPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the config-driven churn pipeline without Kedro."
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.yml",
        help="Path to the YAML/JSON pipeline configuration file.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Override the project root used to resolve relative paths.",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional path to dump metrics separate from the config-defined outputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    bundle = load_config(args.config, project_root=args.project_root)
    pipeline = DataPipeline(bundle)
    result = pipeline.run()

    pretty = json.dumps(result.metrics, indent=2)
    print("Pipeline finished successfully. Metrics:")
    print(pretty)

    if args.metrics_json:
        path = bundle.resolve_path(args.metrics_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pretty, encoding="utf-8")

    return 0

