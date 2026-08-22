from __future__ import annotations

import json
import tomllib
from pathlib import Path

FORBIDDEN_ACCELERATOR_PACKAGES = {
    "catboost",
    "cudf",
    "cuml",
    "cupy",
    "jax",
    "jaxlib",
    "lightgbm",
    "onnxruntime-gpu",
    "rapids",
    "tensorflow",
    "tensorflow-gpu",
    "torch",
    "torchvision",
    "xgboost",
}


def _dependency_name(requirement: str) -> str:
    return requirement.split(";", 1)[0].split("@", 1)[0].split("[", 1)[0].strip().lower()


def test_python_dependency_contract_has_no_gpu_or_accelerator_runtime() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = [*project["dependencies"]]
    for group in project.get("optional-dependencies", {}).values():
        requirements.extend(group)

    names = {_dependency_name(requirement) for requirement in requirements}
    assert not names & FORBIDDEN_ACCELERATOR_PACKAGES


def test_frontend_dependency_contract_has_no_gpu_runtime() -> None:
    package = json.loads(Path("frontend/package.json").read_text(encoding="utf-8"))
    names = {*(package.get("dependencies", {})), *(package.get("devDependencies", {}))}
    assert not {name.lower() for name in names} & FORBIDDEN_ACCELERATOR_PACKAGES


def test_frontend_does_not_request_webgl_plot_traces() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in Path("frontend/src").rglob("*.tsx")
    )
    assert "scattergl" not in sources.lower()
    assert "webgl" not in sources.lower()
