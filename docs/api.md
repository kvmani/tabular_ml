# API Reference

This document captures the subset of FastAPI endpoints touched or relied upon by the React UI during the default Titanic workflow.

## Base URL

The frontend honours `VITE_API_BASE_URL` when provided. Otherwise it derives the API base URL by examining the current location: during development it swaps the Vite port `5173` (or `-5173` host suffix on Codespaces) to `8000`, and finally falls back to the current origin. As a result manual configuration is only required for cross-domain deployments.

## Dataset catalogue

### `GET /data/datasets`
Returns the list of in-memory datasets and the default dataset ID.

```json
{
  "datasets": [
    {
      "dataset_id": "f5c4c18493c84c46a8c8a1b0f406f0a2",
      "name": "Titanic Survival",
      "source": "sample:titanic",
      "description": "Passenger data with survival labels",
      "created_at": "2024-05-20T19:05:16.540212+00:00",
      "num_rows": 1309,
      "num_columns": 12,
      "columns": [
        "PassengerId",
        "Survived",
        "Pclass",
        "Name"
      ],
      "extras": {
        "task": "classification",
        "target_column": "Survived"
      }
    }
  ],
  "default_dataset_id": "f5c4c18493c84c46a8c8a1b0f406f0a2"
}
```

### `GET /data/{dataset_id}/preview`
Returns an array of row objects. The UI limits previews to the top 50 rows by default.

### `GET /data/{dataset_id}/summary`
Returns a per-column statistics map mirroring `DataFrame.describe(include="all")`.

### `GET /data/{dataset_id}/columns`
Lists the ordered column names with dtype strings. The hook powering the UI combines this with the preview and summary to drive visualisations, preprocessing, and training configuration widgets.

## Sample datasets

### `GET /data/samples`
Returns the registry of built-in datasets exposed in the **Sample datasets** panel.

### `POST /data/samples/{key}`
Loads the dataset identified by `{key}` (e.g., `titanic`) into the in-memory store and returns its metadata. The frontend automatically invokes this endpoint during initialisation if the data store is empty so that users immediately see the Titanic dataset without manual intervention.

## System configuration

### `GET /system/config`
Returns the active runtime configuration merged from YAML + environment overrides. The `settings.app.allow_file_uploads` flag drives the "Uploads enabled" banner in the UI, and the dataset registry mirror helps users discover bundled datasets.

## Algorithms

### `GET /model/algorithms`
Returns the list of training algorithms available for the current deployment. Each entry contains the algorithm key, its display label, and supported task types.

## Evaluation

### `POST /model/evaluate`
Triggers evaluation for the supplied model ID. The UI uses this endpoint to populate metric tables and diagnostic charts after a training run completes.
