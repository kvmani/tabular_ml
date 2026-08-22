# CPU-only execution policy

Tabular ML is deliberately a CPU-only application. It does not detect, reserve,
initialize, or require a GPU. This applies equally to standalone and
`ml_server`-mounted operation.

## Runtime contract

- Model training, cross-validation, preprocessing, outlier analysis,
  permutation importance, and inference use NumPy, SciPy, pandas, and
  scikit-learn CPU implementations.
- The estimator catalog contains only `sklearn.*` estimators. The MLP option is
  scikit-learn's CPU neural network; it is not a PyTorch or TensorFlow model.
- Parallel estimator, cross-validation, outlier, and permutation-importance
  calls are bounded with `n_jobs=1` where the library exposes that control.
- Browser figures use regular Plotly scatter/bar/box/heatmap traces and do not
  request WebGL trace types. Plot interaction has no GPU runtime prerequisite.
- CUDA, ROCm, PyTorch, TensorFlow, JAX, CuPy, RAPIDS, XGBoost, LightGBM,
  CatBoost, and GPU-specific ONNX runtimes are not dependencies.

The test suite rejects accelerator package names in both Python and frontend
dependency manifests, rejects WebGL trace requests in authored frontend code,
and verifies that every registered estimator is implemented by scikit-learn.

CPU math libraries may use optimized CPU instructions or multiple BLAS threads
provided by the operating system. That remains CPU execution and does not
change the no-GPU contract.
