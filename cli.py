"""Offline CLI for managing datasets and training models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from config import load_dataset_registry

from backend.app.core import config as runtime_config
from backend.app.services import evaluation as eval_utils
from backend.app.services.data_manager import data_manager
from backend.app.services.model_training import model_trainer
from backend.app.services.run_tracker import RunSummary, run_tracker

ALGORITHM_ALIASES = {
    "rf": "random_forest",
    "random_forest": "random_forest",
    "logreg": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "linear": "linear_regression",
    "linear_regression": "linear_regression",
}

if model_trainer.torch_available:
    ALGORITHM_ALIASES.update(
        {
            "nn": "neural_network",
            "neural_network": "neural_network",
        }
    )


def _registry() -> Dict[str, Any]:
    return {key: entry.model_dump() for key, entry in load_dataset_registry().items()}


def _resolve_dataset_path(filename: str) -> Path:
    base = runtime_config.DATA_DIR
    return (base / filename).resolve()


def cmd_datasets_list(_: argparse.Namespace) -> None:
    registry = load_dataset_registry()
    for key, entry in registry.items():
        name = entry.name or key
        print(f"{key:30s} | {name:30s} | task={entry.task} | target={entry.target}")


def _load_dataframe(name: str) -> pd.DataFrame:
    registry = load_dataset_registry()
    if name not in registry:
        raise SystemExit(
            f"Unknown dataset '{name}'. Use 'datasets list' to view options."
        )
    entry = registry[name]
    path = _resolve_dataset_path(entry.file)
    if not path.exists():
        raise SystemExit(f"Dataset file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def cmd_datasets_preview(args: argparse.Namespace) -> None:
    df = _load_dataframe(args.name)
    rows = min(args.rows, len(df))
    preview = df.head(rows)
    print(preview.to_string(index=False))


def cmd_train(args: argparse.Namespace) -> None:
    dataset_key = args.name
    algo = ALGORITHM_ALIASES.get(args.algo.lower())
    if algo is None:
        raise SystemExit(f"Unknown algorithm alias '{args.algo}'.")
    registry = load_dataset_registry()
    if dataset_key not in registry:
        raise SystemExit(f"Dataset '{dataset_key}' not found in registry.")
    dataset_meta = data_manager.load_sample_dataset(dataset_key)
    df = data_manager.get_dataset(dataset_meta.dataset_id)
    target_column = registry[dataset_key].target
    task_type = args.task
    result = model_trainer.train(
        dataset_id=dataset_meta.dataset_id,
        df=df,
        target_column=target_column,
        task_type=task_type,
        algorithm=algo,
    )
    data_manager.store_split(result.split)
    data_manager.store_model(result.artifact)
    run_tracker.record(
        RunSummary(
            run_id=result.artifact.model_id,
            dataset_id=dataset_meta.dataset_id,
            dataset_name=dataset_meta.name,
            algorithm=algo,
            task_type=task_type,
            metrics=result.artifact.metrics,
            created_at=result.artifact.created_at,
        )
    )
    print(
        json.dumps(
            {"model_id": result.artifact.model_id, "metrics": result.artifact.metrics},
            indent=2,
        )
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    try:
        model_artifact = data_manager.get_model(args.run_id)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    split = data_manager.get_split(model_artifact.split_id)
    X_test = split.X_test.reset_index(drop=True)
    y_test = split.y_test.reset_index(drop=True)
    preprocessor = model_artifact.model_object.get("preprocessor")
    if preprocessor is not None:
        X_test_proc = preprocessor.transform(X_test)
    else:
        X_test_proc = X_test.to_numpy()
    preds, proba = model_trainer._predict(model_artifact.model_object, X_test_proc)
    if model_artifact.task_type == "classification":
        metrics = eval_utils.classification_metrics(y_test, preds, proba)
    else:
        metrics = eval_utils.regression_metrics(y_test, preds)
    print(json.dumps({"metrics": metrics}, indent=2))


def cmd_info(_: argparse.Namespace) -> None:
    info = {
        "settings": runtime_config.as_dict(),
        "datasets": _registry(),
        "last_run": run_tracker.get_last().to_dict()
        if run_tracker.get_last()
        else None,
    }
    print(json.dumps(info, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline tabular ML control plane")
    subparsers = parser.add_subparsers(dest="command")

    datasets_parser = subparsers.add_parser("datasets", help="Dataset utilities")
    datasets_sub = datasets_parser.add_subparsers(dest="datasets_cmd")
    list_parser = datasets_sub.add_parser("list", help="List available datasets")
    list_parser.set_defaults(func=cmd_datasets_list)
    preview_parser = datasets_sub.add_parser(
        "preview", help="Preview a dataset from the registry"
    )
    preview_parser.add_argument(
        "--name", required=True, help="Dataset key from registry"
    )
    preview_parser.add_argument("--rows", type=int, default=20, help="Rows to preview")
    preview_parser.set_defaults(func=cmd_datasets_preview)

    train_parser = subparsers.add_parser(
        "train", help="Train a model using registry dataset"
    )
    train_parser.add_argument("--name", required=True, help="Dataset key")
    algo_help = "Algorithm alias (logreg, rf, linear"
    if model_trainer.torch_available:
        algo_help += ", nn"
    algo_help += ")"
    train_parser.add_argument("--algo", required=True, help=algo_help)
    train_parser.add_argument(
        "--task",
        required=True,
        choices=["classification", "regression"],
        help="Task type",
    )
    train_parser.set_defaults(func=cmd_train)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a stored model run")
    eval_parser.add_argument("--run-id", required=True, help="Model/run identifier")
    eval_parser.set_defaults(func=cmd_evaluate)

    info_parser = subparsers.add_parser("info", help="Show effective configuration")
    info_parser.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
