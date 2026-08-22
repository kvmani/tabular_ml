# HTTP API

The standalone service and the `ml_server` blueprint expose the same routes
under `/tabular_ml/api/v1`. JSON responses use either
`{"ok": true, "data": ...}` or
`{"ok": false, "error": {"code": ..., "message": ..., "details": ...}}`.

## Main routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Service and CPU policy health check |
| `GET` | `/datasets` | Discover vendored datasets and metadata |
| `POST` | `/datasets/load` | Load a vendored dataset by slug |
| `POST` | `/datasets/upload` | Upload CSV/XLSX multipart data |
| `GET` | `/sessions/{id}` | Retrieve the current profile and state |
| `PATCH` | `/sessions/{id}/schema` | Revise semantic feature roles |
| `POST` | `/sessions/{id}/target` | Choose target and task |
| `POST` | `/sessions/{id}/prepare/preview` | Validate a preparation plan |
| `POST` | `/sessions/{id}/outliers/detect` | Score outliers |
| `POST` | `/sessions/{id}/outliers/apply` | Apply or reset treatment |
| `POST` | `/sessions/{id}/visualizations` | Compute bounded chart data |
| `GET` | `/algorithms` | List task-aware algorithms and schemas |
| `POST` | `/sessions/{id}/train` | Train and evaluate one model |
| `POST` | `/sessions/{id}/compare` | Compare several algorithms |
| `POST` | `/runs/{id}/predict` | Infer from one or more records |
| `GET` | `/runs/{id}/report` | Download the JSON model report |
| `GET` | `/runs/{id}/artifact` | Download the trusted joblib artifact |

## Minimal workflow

Load a demo dataset:

~~~http
POST /tabular_ml/api/v1/datasets/load
Content-Type: application/json

{"slug":"palmer_penguins"}
~~~

Select a target using the returned session ID:

~~~http
POST /tabular_ml/api/v1/sessions/{session_id}/target
Content-Type: application/json

{"target":"species","task":"classification"}
~~~

Train with defaults or explicit configuration:

~~~http
POST /tabular_ml/api/v1/sessions/{session_id}/train
Content-Type: application/json

{
  "algorithm":"random_forest",
  "parameters":{"n_estimators":200,"max_depth":8},
  "test_size":0.2,
  "cv_folds":5,
  "random_state":42
}
~~~

The response includes the run ID, split and fold summaries, metrics,
diagnostics, feature importance, resolved preparation plan, and warnings.

## Runtime limits

The service bounds upload size, retained sessions, chart points, correlation
columns, and stored model runs. Defaults can be changed through the documented
Flask configuration/environment settings in `tabular_ml_service.config`.
Sessions expire after the configured TTL and are not durable storage.

All training and inference is CPU-only. Parallel estimator, validation, and
importance paths are explicitly constrained to one worker where scikit-learn
exposes `n_jobs`.
