import type {
  Algorithm,
  DatasetMetadata,
  DatasetSummary,
  TrainingRun,
  Workspace,
} from "../types";

const API_ROOT = "/tabular_ml/api/v1";

interface Envelope<T> {
  success: boolean;
  data: T;
  error?: { code: string; message: string; details?: Record<string, unknown> };
}

export class ApiFailure extends Error {
  code: string;
  details?: Record<string, unknown>;

  constructor(
    code: string,
    message: string,
    details?: Record<string, unknown>,
  ) {
    super(message);
    this.name = "ApiFailure";
    this.code = code;
    this.details = details;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
  });
  const payload = (await response.json()) as Envelope<T>;
  if (!response.ok || !payload.success) {
    const error = payload.error ?? {
      code: "request.failed",
      message: "The local service could not complete this request.",
    };
    throw new ApiFailure(error.code, error.message, error.details);
  }
  return payload.data;
}

export const api = {
  datasets: () =>
    request<{ datasets: DatasetSummary[] }>("/datasets").then(
      (value) => value.datasets,
    ),
  datasetMetadata: (key: string) =>
    request<DatasetMetadata>(`/datasets/${encodeURIComponent(key)}`),
  loadDataset: (key: string) =>
    request<Workspace>(`/datasets/${encodeURIComponent(key)}/load`, {
      method: "POST",
    }),
  uploadDataset: (file: File) => {
    const form = new FormData();
    form.set("file", file);
    return request<Workspace>("/datasets/upload", {
      method: "POST",
      body: form,
    });
  },
  algorithms: (task?: "classification" | "regression") =>
    request<{ algorithms: Algorithm[] }>(
      `/algorithms${task ? `?task=${task}` : ""}`,
    ).then((value) => value.algorithms),
  preparationPlan: (
    sessionId: string,
    target: string,
    config: Record<string, unknown>,
  ) =>
    request<Record<string, unknown>>(
      `/sessions/${sessionId}/preparation/plan`,
      { method: "POST", body: JSON.stringify({ target, config }) },
    ),
  outliers: (
    sessionId: string,
    target: string,
    config: Record<string, unknown>,
  ) =>
    request<Record<string, unknown>>(`/sessions/${sessionId}/outliers/detect`, {
      method: "POST",
      body: JSON.stringify({ target, config }),
    }),
  applyOutliers: (
    sessionId: string,
    target: string,
    config: Record<string, unknown>,
    action: string,
  ) =>
    request<Workspace>(`/sessions/${sessionId}/outliers/apply`, {
      method: "POST",
      body: JSON.stringify({ target, config, action }),
    }),
  reset: (sessionId: string) =>
    request<Workspace>(`/sessions/${sessionId}/reset`, { method: "POST" }),
  visualize: (sessionId: string, config: Record<string, unknown>) =>
    request<{ type: string; data: Record<string, unknown> }>(
      `/sessions/${sessionId}/visualizations`,
      { method: "POST", body: JSON.stringify(config) },
    ),
  train: (sessionId: string, config: Record<string, unknown>) =>
    request<TrainingRun>(`/sessions/${sessionId}/runs`, {
      method: "POST",
      body: JSON.stringify(config),
    }),
  run: (sessionId: string, runId: string) =>
    request<TrainingRun>(`/sessions/${sessionId}/runs/${runId}`),
  compare: (sessionId: string, config: Record<string, unknown>) =>
    request<{
      leaderboard: Array<Record<string, unknown>>;
      failures: Array<Record<string, unknown>>;
    }>(`/sessions/${sessionId}/comparisons`, {
      method: "POST",
      body: JSON.stringify(config),
    }),
  infer: (sessionId: string, runId: string, rows: Record<string, unknown>[]) =>
    request<{
      rows: Array<Record<string, unknown>>;
      task: string;
      target: string;
    }>(`/sessions/${sessionId}/runs/${runId}/infer`, {
      method: "POST",
      body: JSON.stringify({ rows }),
    }),
  artifactUrl: (sessionId: string, runId: string) =>
    `${API_ROOT}/sessions/${sessionId}/runs/${runId}/artifact`,
};
