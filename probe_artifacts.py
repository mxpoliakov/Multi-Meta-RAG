"""Shared `artifacts/` paths and the `--model` flag for the probe pipeline."""

import argparse
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"
ARTIFACT_DIR = Path("artifacts")


def model_slug(model_name: str) -> str:
    # Qwen/Qwen2.5-1.5B -> Qwen2.5-1.5B
    return model_name.rsplit("/", 1)[-1]


def artifact_path(stem: str, suffix: str, model_name: str) -> Path:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    return ARTIFACT_DIR / f"{stem}_{model_slug(model_name)}.{suffix}"


def hidden_states_path(pooling: str, model_name: str) -> Path:
    return artifact_path(f"hidden_states_{pooling}", "npy", model_name)


def figure_path(model_name: str) -> Path:
    return artifact_path("accuracy_vs_depth", "png", model_name)


def results_path(model_name: str) -> Path:
    return artifact_path("probe_results", "json", model_name)


def probe_filters_path(model_name: str) -> Path:
    return artifact_path("query_probe_filters", "json", model_name)


def parse_model() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id, e.g. Qwen/Qwen2.5-0.5B",
    )
    return parser.parse_args().model
