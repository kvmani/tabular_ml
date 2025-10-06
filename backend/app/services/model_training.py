"""Model training orchestration."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from packaging import version
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import settings
from backend.app.models.storage import DataSplit, ModelArtifact
from backend.app.services import evaluation as eval_utils
from backend.app.services.preprocess import split_dataset


def _torch_available() -> bool:
    """Return True when PyTorch can be imported."""

    return importlib.util.find_spec("torch") is not None


TORCH_AVAILABLE = _torch_available()

# Algorithms that benefit from feature scaling
SCALE_NUMERIC_ALGORITHMS = {"logistic_regression", "neural_network"}


@dataclass
class TrainingResult:
    """Container for the outcome of a model training run."""

    artifact: ModelArtifact
    split: DataSplit


class ModelTrainer:
    """Coordinates preprocessing, training, and metric calculation."""

    def __init__(self) -> None:
        self.torch_available = TORCH_AVAILABLE
        self.algorithm_catalog = {
            "logistic_regression": {
                "label": "Logistic Regression",
                "task_types": ["classification"],
            },
            "random_forest": {
                "label": "Random Forest",
                "task_types": ["classification", "regression"],
            },
            "linear_regression": {
                "label": "Linear Regression",
                "task_types": ["regression"],
            },
        }
        self.algorithm_catalog["neural_network"] = {
            "label": (
                "Feedforward Neural Network"
                if self.torch_available
                else "Feedforward Neural Network (sklearn)"
            ),
            "task_types": ["classification", "regression"],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def available_algorithms(self) -> List[Dict[str, object]]:
        return [
            {"key": key, **details}
            for key, details in sorted(self.algorithm_catalog.items())
        ]

    def train(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        algorithm: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        split: Optional[DataSplit] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> TrainingResult:
        if algorithm not in self.algorithm_catalog:
            raise ValueError(f"Unsupported algorithm '{algorithm}'")
        if task_type not in self.algorithm_catalog[algorithm]["task_types"]:
            raise ValueError(
                f"Algorithm '{algorithm}' does not support task type '{task_type}'"
            )
        hyperparameters = hyperparameters or {}

        if split is None:
            split = split_dataset(
                df,
                target_column=target_column,
                task_type=task_type,
                test_size=hyperparameters.pop("test_size", 0.2),
                val_size=hyperparameters.pop("val_size", 0.2),
                random_state=hyperparameters.get("random_state", 42),
                stratify=hyperparameters.pop("stratify", task_type == "classification"),
            )
        split.dataset_id = dataset_id

        X_train = split.X_train.reset_index(drop=True)
        X_val = split.X_val.reset_index(drop=True)
        X_test = split.X_test.reset_index(drop=True)
        y_train = split.y_train.reset_index(drop=True)
        y_val = split.y_val.reset_index(drop=True)
        y_test = split.y_test.reset_index(drop=True)

        scale_numeric = algorithm in SCALE_NUMERIC_ALGORITHMS
        preprocessor = self._build_preprocessor(X_train, scale_numeric=scale_numeric)
        self._emit_progress(progress_callback, {"type": "status", "stage": "preprocessing"})
        preprocessor.fit(X_train)
        X_train_proc = preprocessor.transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
        X_test_proc = preprocessor.transform(X_test)

        history: List[Dict[str, Any]] = []
        metrics: Dict[str, Dict[str, float]]
        model_object: Dict[str, Any]

        if algorithm == "logistic_regression":
            artifact_metrics, history = self._train_logistic(
                hyperparameters,
                X_train_proc,
                y_train,
                X_val_proc,
                y_val,
                progress_callback,
            )
            model_object = {
                "type": "sklearn",
                "model": artifact_metrics.pop("model"),
                "preprocessor": preprocessor,
            }
            metrics = artifact_metrics
        elif algorithm == "random_forest":
            artifact_metrics, history = self._train_random_forest(
                task_type,
                hyperparameters,
                X_train_proc,
                y_train,
                X_val_proc,
                y_val,
                progress_callback,
            )
            model_object = {
                "type": "sklearn",
                "model": artifact_metrics.pop("model"),
                "preprocessor": preprocessor,
            }
            metrics = artifact_metrics
        elif algorithm == "xgboost":
            artifact_metrics, history = self._train_xgboost(
                task_type,
                hyperparameters,
                X_train_proc,
                y_train,
                X_val_proc,
                y_val,
                progress_callback,
            )
            model_object = {
                "type": "sklearn",
                "model": artifact_metrics.pop("model"),
                "preprocessor": preprocessor,
                "label_encoder": artifact_metrics.pop("label_encoder", None),
            }
            metrics = artifact_metrics
        elif algorithm == "neural_network":
            artifact_metrics, history, model_type = self._train_neural_network(
                task_type,
                hyperparameters,
                X_train_proc,
                y_train,
                X_val_proc,
                y_val,
                progress_callback,
            )
            model_object = {
                "type": model_type,
                "model": artifact_metrics.pop("model"),
                "preprocessor": preprocessor,
                "label_encoder": artifact_metrics.pop("label_encoder", None),
            }
            if model_type == "pytorch":
                model_object["input_dim"] = X_train_proc.shape[1]
            metrics = artifact_metrics
        elif algorithm == "linear_regression":
            artifact_metrics, history = self._train_linear_regression(
                hyperparameters,
                X_train_proc,
                y_train,
                X_val_proc,
                y_val,
                progress_callback,
            )
            model_object = {
                "type": "sklearn",
                "model": artifact_metrics.pop("model"),
                "preprocessor": preprocessor,
            }
            metrics = artifact_metrics
        else:
            raise ValueError(f"Algorithm '{algorithm}' is not implemented")

        # Evaluate on test data using stored objects
        y_test_pred, proba_test = self._predict(model_object, X_test_proc)
        if task_type == "classification":
            test_metrics = eval_utils.classification_metrics(
                y_test, y_test_pred, proba_test
            )
        else:
            test_metrics = eval_utils.regression_metrics(y_test, y_test_pred)
        metrics["test"] = test_metrics

        model_id = uuid4().hex
        artifact = ModelArtifact(
            model_id=model_id,
            algorithm=algorithm,
            task_type=task_type,
            target_column=target_column,
            feature_columns=split.feature_columns,
            model_object=model_object,
            metrics=metrics,
            history=history,
            split_id=split.split_id,
        )
        self._emit_progress(
            progress_callback,
            {
                "type": "completed",
                "model_id": model_id,
                "metrics": metrics,
                "history_length": len(history),
            },
        )
        return TrainingResult(artifact=artifact, split=split)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_preprocessor(
        self, X: pd.DataFrame, scale_numeric: bool = False
    ) -> ColumnTransformer:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in X.columns if col not in numeric_cols]

        numeric_steps: List[Tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median"))
        ]
        if scale_numeric and numeric_cols:
            numeric_steps.append(("scaler", StandardScaler()))
        categorical_steps: List[Tuple[str, Any]] = []
        if categorical_cols:
            categorical_steps = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", self._one_hot_encoder()),
            ]
        transformers: List[Tuple[str, Any, List[str]]] = []
        if numeric_cols:
            transformers.append(("numeric", Pipeline(numeric_steps), numeric_cols))
        if categorical_cols:
            transformers.append(
                ("categorical", Pipeline(categorical_steps), categorical_cols)
            )
        if not transformers:
            # Default passthrough when dataset has no features (edge case)
            transformers.append(("identity", "passthrough", X.columns.tolist()))
        return ColumnTransformer(transformers)

    @staticmethod
    def _one_hot_encoder() -> OneHotEncoder:
        kwargs = {"handle_unknown": "ignore"}
        if version.parse(sklearn_version) < version.parse("1.4"):
            kwargs["sparse"] = settings.ml.sklearn_onehot_sparse
        else:
            kwargs["sparse_output"] = settings.ml.sklearn_onehot_sparse
        return OneHotEncoder(**kwargs)

    def _emit_progress(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]],
        event: Dict[str, Any],
    ) -> None:
        if callback is None:
            return
        try:
            callback(event)
        except Exception:  # pragma: no cover - defensive against user callbacks
            pass

    def _train_logistic(
        self,
        hyperparameters: Dict[str, Any],
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
        params = {"max_iter": 500, "solver": "lbfgs", "multi_class": "auto"}
        params.update(hyperparameters)
        model = LogisticRegression(**params)
        start = perf_counter()
        model.fit(X_train, y_train)
        train_duration = perf_counter() - start
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)
        val_metrics = eval_utils.classification_metrics(y_val, y_val_pred, y_val_proba)
        history_entry = {
            "stage": "fit",
            "train_score": float(model.score(X_train, y_train)),
            "validation_accuracy": float(val_metrics["accuracy"]),
            "elapsed_seconds": train_duration,
            "metrics": val_metrics,
        }
        history = [history_entry]
        self._emit_progress(progress_callback, {"type": "history", "entry": history_entry})
        return {"validation": val_metrics, "model": model}, history

    def _train_random_forest(
        self,
        task_type: str,
        hyperparameters: Dict[str, Any],
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
        defaults = {
            "n_estimators": 200,
            "random_state": 42,
            "n_jobs": settings.ml.n_jobs,
        }
        defaults.update(hyperparameters)
        if task_type == "classification":
            model = RandomForestClassifier(**defaults)
        else:
            model = RandomForestRegressor(**defaults)
        start = perf_counter()
        model.fit(X_train, y_train)
        duration = perf_counter() - start
        y_val_pred = model.predict(X_val)
        if task_type == "classification":
            y_val_proba = model.predict_proba(X_val)
            val_metrics = eval_utils.classification_metrics(
                y_val, y_val_pred, y_val_proba
            )
        else:
            val_metrics = eval_utils.regression_metrics(y_val, y_val_pred)
        history_entry = {
            "stage": "fit",
            "validation_score": val_metrics["accuracy"]
            if task_type == "classification"
            else val_metrics["r2"],
            "elapsed_seconds": duration,
            "metrics": val_metrics,
        }
        history = [history_entry]
        self._emit_progress(progress_callback, {"type": "history", "entry": history_entry})
        return {"validation": val_metrics, "model": model}, history

    def _train_xgboost(
        self,
        task_type: str,
        hyperparameters: Dict[str, Any],
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
        from xgboost import XGBClassifier, XGBRegressor

        params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6}
        params.update(hyperparameters)
        label_encoder = None
        y_train_input = y_train
        y_val_input = y_val
        if task_type == "classification":
            from sklearn.preprocessing import LabelEncoder

            label_encoder = LabelEncoder()
            y_train_input = label_encoder.fit_transform(y_train)
            y_val_input = label_encoder.transform(y_val)
            model = XGBClassifier(
                use_label_encoder=False,
                eval_metric=params.pop("eval_metric", "logloss"),
                **params,
            )
        else:
            model = XGBRegressor(**params)
        eval_set = [(X_train, y_train_input), (X_val, y_val_input)]
        model.fit(X_train, y_train_input, eval_set=eval_set, verbose=False)
        if task_type == "classification":
            val_pred = label_encoder.inverse_transform(
                model.predict(X_val).round().astype(int)
            )
            val_proba = model.predict_proba(X_val)
            val_metrics = eval_utils.classification_metrics(y_val, val_pred, val_proba)
        else:
            val_pred = model.predict(X_val)
            val_metrics = eval_utils.regression_metrics(y_val, val_pred)
        history = []
        if hasattr(model, "evals_result"):
            eval_history = model.evals_result()
            if "validation_0" in eval_history and "validation_1" in eval_history:
                metric_names = list(eval_history["validation_0"].keys())
                if metric_names:
                    metric_name = metric_names[0]
                    train_series = eval_history["validation_0"].get(metric_name, [])
                    val_series = eval_history["validation_1"].get(metric_name, [])
                    for i, (train_metric, val_metric) in enumerate(
                        zip(train_series, val_series)
                    ):
                        history_entry = {
                            "iteration": i,
                            "metric": metric_name,
                            "train_metric": train_metric,
                            "validation_metric": val_metric,
                        }
                        history.append(history_entry)
                        self._emit_progress(
                            progress_callback,
                            {"type": "history", "entry": history_entry},
                        )
        summary_entry = {"iteration": len(history), "metrics": val_metrics}
        history.append(summary_entry)
        self._emit_progress(
            progress_callback,
            {"type": "history", "entry": summary_entry, "final": True},
        )
        result = {"validation": val_metrics, "model": model}
        if label_encoder is not None:
            result["label_encoder"] = label_encoder
        return result, history

    def _train_neural_network(
        self,
        task_type: str,
        hyperparameters: Dict[str, Any],
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]], str]:
        if not TORCH_AVAILABLE:
            return self._train_neural_network_sklearn(
                task_type,
                hyperparameters,
                X_train,
                y_train,
                X_val,
                y_val,
                progress_callback,
            )
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        class FeedForwardNet(nn.Module):
            """Simple feedforward neural network for tabular data."""

            def __init__(
                self,
                input_dim: int,
                hidden_layers: List[int],
                output_dim: int,
                dropout: float = 0.1,
            ) -> None:
                super().__init__()
                layers: List[nn.Module] = []
                prev_dim = input_dim
                for hidden_dim in hidden_layers:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, output_dim))
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.model(x)

        params = {
            "hidden_layers": [128, 64],
            "epochs": 25,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "dropout": 0.2,
        }
        params.update(hyperparameters)
        device = torch.device("cpu")
        input_dim = X_train.shape[1]
        history: List[Dict[str, Any]] = []
        label_encoder = None

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)

        if task_type == "classification":
            from sklearn.preprocessing import LabelEncoder

            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_val_encoded = label_encoder.transform(y_val)
            num_classes = len(label_encoder.classes_)
            if num_classes == 2:
                y_train_tensor = torch.tensor(
                    y_train_encoded.reshape(-1, 1), dtype=torch.float32, device=device
                )
                y_val_tensor = torch.tensor(
                    y_val_encoded.reshape(-1, 1), dtype=torch.float32, device=device
                )
                criterion = nn.BCEWithLogitsLoss()
                output_dim = 1
            else:
                y_train_tensor = torch.tensor(
                    y_train_encoded, dtype=torch.long, device=device
                )
                y_val_tensor = torch.tensor(
                    y_val_encoded, dtype=torch.long, device=device
                )
                criterion = nn.CrossEntropyLoss()
                output_dim = num_classes
        else:
            y_train_tensor = torch.tensor(
                y_train.values, dtype=torch.float32, device=device
            )
            y_val_tensor = torch.tensor(
                y_val.values, dtype=torch.float32, device=device
            )
            criterion = nn.MSELoss()
            output_dim = 1

        model = FeedForwardNet(
            input_dim=input_dim,
            hidden_layers=params["hidden_layers"],
            output_dim=output_dim,
            dropout=params.get("dropout", 0.1),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=params["batch_size"], shuffle=True
        )

        for epoch in range(1, params["epochs"] + 1):
            model.train()
            epoch_loss = 0.0
            for features, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(features)
                if task_type == "classification" and outputs.shape[1] == 1:
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * features.size(0)
            epoch_loss /= len(train_dataset)

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                if task_type == "classification" and val_outputs.shape[1] == 1:
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_probs = torch.sigmoid(val_outputs).cpu().numpy()
                    val_pred = (val_probs > 0.5).astype(int).ravel()
                    val_pred_labels = label_encoder.inverse_transform(val_pred)
                    val_metrics = eval_utils.classification_metrics(
                        y_val, val_pred_labels, val_probs
                    )
                elif task_type == "classification":
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_probs = torch.softmax(val_outputs, dim=1).cpu().numpy()
                    val_pred = val_probs.argmax(axis=1)
                    val_pred_labels = label_encoder.inverse_transform(val_pred)
                    val_metrics = eval_utils.classification_metrics(
                        y_val, val_pred_labels, val_probs
                    )
                else:
                    val_outputs = val_outputs.squeeze()
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_pred = val_outputs.cpu().numpy()
                    val_metrics = eval_utils.regression_metrics(y_val, val_pred)
            history_entry = {
                "epoch": epoch,
                "train_loss": float(epoch_loss),
                "val_loss": float(val_loss),
                "metrics": val_metrics,
            }
            history.append(history_entry)
            self._emit_progress(progress_callback, {"type": "history", "entry": history_entry})
        result = {"validation": history[-1]["metrics"], "model": model}
        if label_encoder is not None:
            result["label_encoder"] = label_encoder
        return result, history, "pytorch"

    def _train_neural_network_sklearn(
        self,
        task_type: str,
        hyperparameters: Dict[str, Any],
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]], str]:
        params: Dict[str, Any]
        history: List[Dict[str, Any]]
        start = perf_counter()
        if task_type == "classification":
            params = {
                "hidden_layer_sizes": (64, 32),
                "activation": "relu",
                "solver": "adam",
                "max_iter": 300,
                "random_state": 42,
                "early_stopping": True,
            }
            params.update(hyperparameters)
            model = MLPClassifier(**params)
            model.fit(X_train, y_train)
            duration = perf_counter() - start
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)
            val_metrics = eval_utils.classification_metrics(
                y_val, y_val_pred, y_val_proba
            )
            history_entry = {
                "stage": "fit",
                "elapsed_seconds": duration,
                "metrics": val_metrics,
            }
            history = [history_entry]
            self._emit_progress(progress_callback, {"type": "history", "entry": history_entry})
        else:
            params = {
                "hidden_layer_sizes": (64, 32),
                "activation": "relu",
                "solver": "adam",
                "max_iter": 300,
                "random_state": 42,
                "early_stopping": True,
            }
            params.update(hyperparameters)
            model = MLPRegressor(**params)
            model.fit(X_train, y_train)
            duration = perf_counter() - start
            y_val_pred = model.predict(X_val)
            val_metrics = eval_utils.regression_metrics(y_val, y_val_pred)
            history_entry = {
                "stage": "fit",
                "elapsed_seconds": duration,
                "metrics": val_metrics,
            }
            history = [history_entry]
            self._emit_progress(progress_callback, {"type": "history", "entry": history_entry})
        return {"validation": val_metrics, "model": model}, history, "sklearn"

    def _train_linear_regression(
        self,
        hyperparameters: Dict[str, Any],
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
        model = LinearRegression(**hyperparameters)
        start = perf_counter()
        model.fit(X_train, y_train)
        duration = perf_counter() - start
        val_pred = model.predict(X_val)
        val_metrics = eval_utils.regression_metrics(y_val, val_pred)
        history_entry = {
            "stage": "fit",
            "validation_r2": float(val_metrics["r2"]),
            "elapsed_seconds": duration,
            "metrics": val_metrics,
        }
        history = [history_entry]
        self._emit_progress(progress_callback, {"type": "history", "entry": history_entry})
        return {"validation": val_metrics, "model": model}, history

    def _predict(
        self,
        model_object: Dict[str, Any],
        X: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        model = model_object["model"]
        if model_object["type"] == "sklearn":
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
            else:
                proba = None
            preds = model.predict(X)
            label_encoder = model_object.get("label_encoder")
            if label_encoder is not None:
                preds = label_encoder.inverse_transform(preds.astype(int))
                if proba is not None:
                    # Ensure probabilities align with original class order
                    proba = proba
            return preds, proba
        elif model_object["type"] == "pytorch":
            if not TORCH_AVAILABLE:
                raise RuntimeError(
                    "PyTorch is required to perform predictions for this model."
                )
            import torch

            model.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(X, dtype=torch.float32)
                outputs = model(tensor_x)
                if model_object.get("label_encoder") is not None:
                    if outputs.shape[1] == 1:
                        probs = torch.sigmoid(outputs).cpu().numpy()
                        preds = (probs > 0.5).astype(int).ravel()
                    else:
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()
                        preds = probs.argmax(axis=1)
                    label_encoder = model_object["label_encoder"]
                    preds = label_encoder.inverse_transform(preds)
                    return preds, probs
                else:
                    preds = outputs.cpu().numpy().squeeze()
                    return preds, None
        else:
            raise ValueError("Unsupported model type for prediction")


model_trainer = ModelTrainer()
