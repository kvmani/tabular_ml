"""Capability-rich registry for CPU-friendly classification and regression."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..errors import ApiError

Task = Literal["classification", "regression"]
ParameterKind = Literal["integer", "number", "choice", "boolean"]


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    label: str
    kind: ParameterKind
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] = ()
    nullable: bool = False
    help: str = ""

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["choices"] = list(self.choices)
        return payload


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    id: str
    label: str
    family: str
    tasks: tuple[Task, ...]
    summary: str
    strengths: str
    cautions: str
    speed: Literal["fast", "moderate", "slow"]
    interpretability: Literal["high", "medium", "low"]
    scale_sensitive: bool
    requires_dense: bool
    supports_probabilities: bool
    parameters: tuple[ParameterSpec, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "family": self.family,
            "tasks": list(self.tasks),
            "summary": self.summary,
            "strengths": self.strengths,
            "cautions": self.cautions,
            "speed": self.speed,
            "interpretability": self.interpretability,
            "scale_sensitive": self.scale_sensitive,
            "requires_dense": self.requires_dense,
            "supports_probabilities": self.supports_probabilities,
            "parameters": [item.public_dict() for item in self.parameters],
        }


def _integer(
    name: str,
    label: str,
    default: int | None,
    minimum: int,
    maximum: int,
    help_text: str,
    *,
    nullable: bool = False,
) -> ParameterSpec:
    return ParameterSpec(
        name,
        label,
        "integer",
        default,
        minimum,
        maximum,
        nullable=nullable,
        help=help_text,
    )


def _number(
    name: str,
    label: str,
    default: float,
    minimum: float,
    maximum: float,
    help_text: str,
) -> ParameterSpec:
    return ParameterSpec(name, label, "number", default, minimum, maximum, help=help_text)


def _choice(
    name: str,
    label: str,
    default: Any,
    choices: tuple[Any, ...],
    help_text: str,
) -> ParameterSpec:
    return ParameterSpec(name, label, "choice", default, choices=choices, help=help_text)


ALGORITHMS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        "linear",
        "Linear / Logistic",
        "Linear",
        ("classification", "regression"),
        "A regularized linear baseline with coefficients that are easy to inspect.",
        "Fast, stable, and a strong first benchmark.",
        "Cannot learn nonlinear interactions unless features express them.",
        "fast",
        "high",
        True,
        False,
        True,
        (
            _number(
                "regularization",
                "Regularization",
                1.0,
                0.0001,
                100.0,
                "Smaller values regularize regression more; classification uses inverse strength.",
            ),
            _choice(
                "class_weight",
                "Class weighting",
                None,
                (None, "balanced"),
                "Balanced weighting helps uneven classification targets.",
            ),
        ),
    ),
    AlgorithmSpec(
        "knn",
        "k-Nearest Neighbours",
        "Neighbour",
        ("classification", "regression"),
        "Predict from the most similar prepared training rows.",
        "Flexible local boundaries and intuitive behavior.",
        "Prediction slows as data grows and irrelevant features can dominate distance.",
        "moderate",
        "medium",
        True,
        False,
        True,
        (
            _integer("n_neighbors", "Neighbours", 7, 1, 100, "Number of nearby rows used."),
            _choice(
                "weights",
                "Weighting",
                "distance",
                ("uniform", "distance"),
                "Distance weighting emphasizes closer rows.",
            ),
            _choice("p", "Distance", 2, (1, 2), "1 is Manhattan; 2 is Euclidean distance."),
        ),
    ),
    AlgorithmSpec(
        "decision_tree",
        "Decision Tree",
        "Tree",
        ("classification", "regression"),
        "Learn human-readable threshold rules.",
        "Captures nonlinear interactions without scaling.",
        "A deep tree can overfit small changes in the training data.",
        "fast",
        "high",
        False,
        False,
        True,
        (
            _integer("max_depth", "Maximum depth", 8, 1, 80, "Limits rule depth.", nullable=True),
            _integer(
                "min_samples_leaf",
                "Minimum leaf rows",
                3,
                1,
                100,
                "Larger leaves smooth the model.",
            ),
            _choice(
                "class_weight", "Class weighting", None, (None, "balanced"), "Classification only."
            ),
        ),
    ),
    AlgorithmSpec(
        "random_forest",
        "Random Forest",
        "Ensemble",
        ("classification", "regression"),
        "Average many randomized decision trees.",
        "Strong general-purpose nonlinear model with robust defaults.",
        "Larger models train and predict more slowly and are less transparent than one tree.",
        "moderate",
        "medium",
        False,
        False,
        True,
        (
            _integer(
                "n_estimators",
                "Trees",
                250,
                50,
                1_000,
                "More trees reduce variance at additional cost.",
            ),
            _integer(
                "max_depth",
                "Maximum depth",
                None,
                1,
                100,
                "None grows trees until other limits stop them.",
                nullable=True,
            ),
            _integer(
                "min_samples_leaf",
                "Minimum leaf rows",
                2,
                1,
                100,
                "Larger leaves reduce overfitting.",
            ),
            _choice(
                "max_features",
                "Features per split",
                "sqrt",
                ("sqrt", "log2", None),
                "Random feature sampling decorrelates trees.",
            ),
            _choice(
                "class_weight",
                "Class weighting",
                None,
                (None, "balanced", "balanced_subsample"),
                "Classification only.",
            ),
        ),
    ),
    AlgorithmSpec(
        "extra_trees",
        "Extra Trees",
        "Ensemble",
        ("classification", "regression"),
        "Average highly randomized decision trees.",
        "Often faster than random forests and strong on noisy nonlinear data.",
        "Additional randomization can increase bias.",
        "moderate",
        "medium",
        False,
        False,
        True,
        (
            _integer("n_estimators", "Trees", 250, 50, 1_000, "Number of randomized trees."),
            _integer(
                "max_depth", "Maximum depth", None, 1, 100, "Limits tree depth.", nullable=True
            ),
            _integer(
                "min_samples_leaf", "Minimum leaf rows", 2, 1, 100, "Larger leaves reduce variance."
            ),
            _choice(
                "max_features",
                "Features per split",
                "sqrt",
                ("sqrt", "log2", None),
                "Features considered at each split.",
            ),
            _choice(
                "class_weight", "Class weighting", None, (None, "balanced"), "Classification only."
            ),
        ),
    ),
    AlgorithmSpec(
        "gradient_boosting",
        "Gradient Boosting",
        "Boosting",
        ("classification", "regression"),
        "Build shallow trees sequentially to correct previous errors.",
        "Excellent accuracy on many medium-sized tabular datasets.",
        "Sensitive to learning-rate/depth choices and slower than bagged trees.",
        "moderate",
        "medium",
        False,
        False,
        True,
        (
            _integer(
                "n_estimators", "Boosting stages", 150, 25, 800, "Number of sequential trees."
            ),
            _number(
                "learning_rate",
                "Learning rate",
                0.05,
                0.005,
                1.0,
                "Smaller steps usually need more stages.",
            ),
            _integer("max_depth", "Tree depth", 3, 1, 12, "Depth of each weak learner."),
            _number(
                "subsample",
                "Row fraction",
                1.0,
                0.3,
                1.0,
                "Values below one add stochastic regularization.",
            ),
        ),
    ),
    AlgorithmSpec(
        "hist_gradient_boosting",
        "Histogram Gradient Boosting",
        "Boosting",
        ("classification", "regression"),
        "Fast histogram-based boosting for larger datasets.",
        "Scales well and models complex nonlinear effects.",
        "Requires dense prepared data and careful iteration/leaf control.",
        "moderate",
        "medium",
        False,
        True,
        True,
        (
            _integer("max_iter", "Iterations", 200, 25, 1_000, "Maximum boosting iterations."),
            _number(
                "learning_rate", "Learning rate", 0.08, 0.005, 1.0, "Contribution of each stage."
            ),
            _integer("max_leaf_nodes", "Maximum leaves", 31, 2, 255, "Complexity of each tree."),
            _number(
                "l2_regularization",
                "L2 regularization",
                0.0,
                0.0,
                100.0,
                "Penalizes extreme leaf values.",
            ),
        ),
    ),
    AlgorithmSpec(
        "ada_boost",
        "AdaBoost",
        "Boosting",
        ("classification", "regression"),
        "Focus successive weak learners on hard examples.",
        "Simple, effective boosting baseline.",
        "Can chase noisy labels and extreme outliers.",
        "moderate",
        "medium",
        False,
        False,
        True,
        (
            _integer("n_estimators", "Estimators", 150, 25, 800, "Number of weak learners."),
            _number(
                "learning_rate", "Learning rate", 0.5, 0.01, 2.0, "Contribution of each learner."
            ),
        ),
    ),
    AlgorithmSpec(
        "svm",
        "Support Vector Machine",
        "Kernel",
        ("classification", "regression"),
        "Fit a maximum-margin boundary, optionally using an RBF kernel.",
        "Powerful on medium-sized, scaled datasets.",
        "Can be slow on large row counts and sensitive to C/gamma.",
        "slow",
        "low",
        True,
        False,
        True,
        (
            _number("c", "C", 1.0, 0.001, 1_000.0, "Penalty for training errors."),
            _choice(
                "kernel",
                "Kernel",
                "rbf",
                ("rbf", "linear", "poly"),
                "Shape of the decision boundary.",
            ),
            _choice(
                "gamma", "Gamma", "scale", ("scale", "auto"), "RBF/poly kernel radius behavior."
            ),
            _choice(
                "class_weight", "Class weighting", None, (None, "balanced"), "Classification only."
            ),
        ),
    ),
    AlgorithmSpec(
        "sgd",
        "Stochastic Gradient Descent",
        "Linear",
        ("classification", "regression"),
        "Fit a regularized linear model with incremental gradient updates.",
        "Very fast on large, wide prepared tables.",
        "Needs scaling and may require alpha tuning for stable convergence.",
        "fast",
        "high",
        True,
        False,
        True,
        (
            _number("alpha", "Regularization", 0.0001, 0.000001, 1.0, "Penalty strength."),
            _choice(
                "penalty", "Penalty", "l2", ("l2", "l1", "elasticnet"), "Coefficient regularizer."
            ),
        ),
    ),
    AlgorithmSpec(
        "mlp",
        "Neural Network (MLP)",
        "Neural network",
        ("classification", "regression"),
        "Learn nonlinear interactions with a compact feed-forward network.",
        "Flexible function approximation for sufficiently large datasets.",
        "Needs scaling, can be slow, and is less interpretable.",
        "slow",
        "low",
        True,
        True,
        True,
        (
            _integer("hidden_units", "Hidden units", 64, 8, 512, "Width of each hidden layer."),
            _integer("hidden_layers", "Hidden layers", 2, 1, 4, "Number of hidden layers."),
            _number("alpha", "L2 regularization", 0.0001, 0.000001, 1.0, "Weight penalty."),
            _number(
                "learning_rate_init",
                "Initial learning rate",
                0.001,
                0.00001,
                0.1,
                "Optimizer starting step size.",
            ),
        ),
    ),
    AlgorithmSpec(
        "naive_bayes",
        "Gaussian Naive Bayes",
        "Probabilistic",
        ("classification",),
        "Combine per-feature Gaussian likelihoods under conditional independence.",
        "Extremely fast and useful as a probabilistic baseline.",
        "Independence and Gaussian assumptions can be unrealistic.",
        "fast",
        "medium",
        True,
        True,
        True,
        (
            _number(
                "var_smoothing",
                "Variance smoothing",
                1e-9,
                1e-12,
                0.1,
                "Numerical stability added to variances.",
            ),
        ),
    ),
    AlgorithmSpec(
        "lasso",
        "Lasso Regression",
        "Linear",
        ("regression",),
        "Linear regression with sparse L1-regularized coefficients.",
        "Performs embedded feature selection and remains interpretable.",
        "Unstable among strongly correlated features.",
        "fast",
        "high",
        True,
        False,
        False,
        (
            _number(
                "alpha",
                "L1 regularization",
                0.1,
                0.00001,
                100.0,
                "Strength of coefficient sparsity.",
            ),
        ),
    ),
    AlgorithmSpec(
        "elastic_net",
        "Elastic Net",
        "Linear",
        ("regression",),
        "Blend L1 and L2 regularization in a linear regression.",
        "Handles correlated features more smoothly than pure Lasso.",
        "Still limited to additive linear effects.",
        "fast",
        "high",
        True,
        False,
        False,
        (
            _number("alpha", "Regularization", 0.1, 0.00001, 100.0, "Overall penalty strength."),
            _number(
                "l1_ratio", "L1 ratio", 0.5, 0.0, 1.0, "Zero is Ridge-like; one is Lasso-like."
            ),
        ),
    ),
)

ALGORITHM_INDEX = {spec.id: spec for spec in ALGORITHMS}


def algorithm_catalog(task: Task | None = None) -> list[dict[str, Any]]:
    return [spec.public_dict() for spec in ALGORITHMS if task is None or task in spec.tasks]


def _validate_value(spec: ParameterSpec, value: Any) -> Any:
    if value is None and spec.nullable:
        return None
    if spec.kind == "boolean":
        if not isinstance(value, bool):
            raise ValueError("must be true or false")
        return value
    if spec.kind == "choice":
        if value not in spec.choices:
            raise ValueError(f"must be one of {list(spec.choices)}")
        return value
    if spec.kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("must be an integer")
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a number")
    if spec.minimum is not None and value < spec.minimum:
        raise ValueError(f"must be at least {spec.minimum}")
    if spec.maximum is not None and value > spec.maximum:
        raise ValueError(f"must be at most {spec.maximum}")
    return value


def validated_parameters(spec: AlgorithmSpec, supplied: dict[str, Any] | None) -> dict[str, Any]:
    raw = supplied or {}
    known = {item.name: item for item in spec.parameters}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ApiError(
            code="algorithm.unknown_parameters",
            message=f"Unknown parameters for {spec.label}.",
            details={"parameters": unknown},
        )
    output: dict[str, Any] = {}
    for name, parameter in known.items():
        value = raw.get(name, parameter.default)
        try:
            output[name] = _validate_value(parameter, value)
        except ValueError as exc:
            raise ApiError(
                code="algorithm.invalid_parameter",
                message=f"{parameter.label} {exc}.",
                details={"parameter": name, "value": value},
            ) from exc
    return output


def estimator_for(
    algorithm_id: str,
    task: Task,
    supplied: dict[str, Any] | None,
    *,
    random_seed: int,
) -> tuple[AlgorithmSpec, BaseEstimator, dict[str, Any]]:
    spec = ALGORITHM_INDEX.get(algorithm_id)
    if spec is None:
        raise ApiError(code="algorithm.not_found", message=f"Unknown algorithm: {algorithm_id}")
    if task not in spec.tasks:
        raise ApiError(
            code="algorithm.incompatible_task",
            message=f"{spec.label} does not support {task}.",
        )
    params = validated_parameters(spec, supplied)
    estimator: BaseEstimator
    if algorithm_id == "linear":
        regularization = params["regularization"]
        estimator = (
            LogisticRegression(
                C=regularization,
                class_weight=params["class_weight"],
                max_iter=1_000,
                random_state=random_seed,
            )
            if task == "classification"
            else Ridge(alpha=regularization, random_state=random_seed)
        )
    elif algorithm_id == "knn":
        common = {
            "n_neighbors": params["n_neighbors"],
            "weights": params["weights"],
            "p": params["p"],
        }
        estimator = (
            KNeighborsClassifier(**common)
            if task == "classification"
            else KNeighborsRegressor(**common)
        )
    elif algorithm_id == "decision_tree":
        common = {
            "max_depth": params["max_depth"],
            "min_samples_leaf": params["min_samples_leaf"],
            "random_state": random_seed,
        }
        estimator = (
            DecisionTreeClassifier(class_weight=params["class_weight"], **common)
            if task == "classification"
            else DecisionTreeRegressor(**common)
        )
    elif algorithm_id in {"random_forest", "extra_trees"}:
        common = {
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "min_samples_leaf": params["min_samples_leaf"],
            "max_features": params["max_features"],
            "random_state": random_seed,
            "n_jobs": 1,
        }
        if algorithm_id == "random_forest":
            estimator = (
                RandomForestClassifier(class_weight=params["class_weight"], **common)
                if task == "classification"
                else RandomForestRegressor(**common)
            )
        else:
            estimator = (
                ExtraTreesClassifier(class_weight=params["class_weight"], **common)
                if task == "classification"
                else ExtraTreesRegressor(**common)
            )
    elif algorithm_id == "gradient_boosting":
        common = {
            "n_estimators": params["n_estimators"],
            "learning_rate": params["learning_rate"],
            "max_depth": params["max_depth"],
            "subsample": params["subsample"],
            "random_state": random_seed,
        }
        estimator = (
            GradientBoostingClassifier(**common)
            if task == "classification"
            else GradientBoostingRegressor(**common)
        )
    elif algorithm_id == "hist_gradient_boosting":
        common = {
            "max_iter": params["max_iter"],
            "learning_rate": params["learning_rate"],
            "max_leaf_nodes": params["max_leaf_nodes"],
            "l2_regularization": params["l2_regularization"],
            "random_state": random_seed,
        }
        estimator = (
            HistGradientBoostingClassifier(**common)
            if task == "classification"
            else HistGradientBoostingRegressor(**common)
        )
    elif algorithm_id == "ada_boost":
        common = {
            "n_estimators": params["n_estimators"],
            "learning_rate": params["learning_rate"],
            "random_state": random_seed,
        }
        estimator = (
            AdaBoostClassifier(**common)
            if task == "classification"
            else AdaBoostRegressor(**common)
        )
    elif algorithm_id == "svm":
        common = {"C": params["c"], "kernel": params["kernel"], "gamma": params["gamma"]}
        estimator = (
            SVC(
                probability=True,
                class_weight=params["class_weight"],
                random_state=random_seed,
                **common,
            )
            if task == "classification"
            else SVR(**common)
        )
    elif algorithm_id == "sgd":
        common = {
            "alpha": params["alpha"],
            "penalty": params["penalty"],
            "max_iter": 2_000,
            "tol": 1e-4,
            "random_state": random_seed,
        }
        estimator = (
            SGDClassifier(loss="log_loss", **common)
            if task == "classification"
            else SGDRegressor(loss="squared_error", **common)
        )
    elif algorithm_id == "mlp":
        hidden = tuple([params["hidden_units"]] * params["hidden_layers"])
        common = {
            "hidden_layer_sizes": hidden,
            "alpha": params["alpha"],
            "learning_rate_init": params["learning_rate_init"],
            "max_iter": 600,
            "early_stopping": True,
            "random_state": random_seed,
        }
        estimator = MLPClassifier(**common) if task == "classification" else MLPRegressor(**common)
    elif algorithm_id == "naive_bayes":
        estimator = GaussianNB(var_smoothing=params["var_smoothing"])
    elif algorithm_id == "lasso":
        estimator = Lasso(alpha=params["alpha"], max_iter=5_000, random_state=random_seed)
    elif algorithm_id == "elastic_net":
        estimator = ElasticNet(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            max_iter=5_000,
            random_state=random_seed,
        )
    elif task == "regression":
        estimator = LinearRegression()
    else:  # pragma: no cover - index and branches are exhaustive
        raise ApiError(code="algorithm.not_found", message=f"Unknown algorithm: {algorithm_id}")
    return spec, estimator, params
