export type Task = "classification" | "regression";

export interface DatasetSummary {
  key: string;
  name: string;
  task: Task;
  target: string;
  rows: number;
  columns: number;
  summary: string;
  license: string;
  tags: string[];
}

export interface DatasetMetadata {
  key: string;
  name: string;
  description: string;
  task: Task;
  target: string;
  learning_goals: string[];
  recommended_exclusions: string[];
  ethical_notes: string;
  license: { name: string; url: string; attribution: string };
  source: { page: string; download: string; retrieved: string };
}

export interface ColumnProfile {
  name: string;
  storage_type: string;
  semantic_type: string;
  suggested_role: "feature" | "identifier" | "ignore";
  non_missing: number;
  missing: number;
  missing_ratio: number;
  unique: number;
  unique_ratio: number;
  sample_values: unknown[];
  statistics: Record<string, unknown>;
  warnings: string[];
}

export interface Workspace {
  session_id: string;
  dataset: {
    name: string;
    source: string;
    rows: number;
    columns: number;
    metadata: Partial<DatasetMetadata>;
  };
  profile: {
    overview: {
      rows: number;
      columns: number;
      missing_cells: number;
      missing_ratio: number;
      duplicate_rows: number;
      memory_bytes: number;
      health_score: number;
    };
    columns: ColumnProfile[];
    target_candidates: Array<{
      column: string;
      task: Task;
      confidence: number;
      reason: string;
    }>;
    warnings: Array<{ code: string; severity: string; message: string }>;
  };
  preview: { columns: string[]; rows: Record<string, unknown>[]; shown: number };
  history: Array<Record<string, unknown>>;
}

export interface ParameterSpec {
  name: string;
  label: string;
  kind: "integer" | "number" | "choice" | "boolean";
  default: unknown;
  minimum?: number;
  maximum?: number;
  choices: unknown[];
  nullable: boolean;
  help: string;
}

export interface Algorithm {
  id: string;
  label: string;
  family: string;
  tasks: Task[];
  summary: string;
  strengths: string;
  cautions: string;
  speed: "fast" | "moderate" | "slow";
  interpretability: "high" | "medium" | "low";
  scale_sensitive: boolean;
  requires_dense: boolean;
  supports_probabilities: boolean;
  parameters: ParameterSpec[];
}

export interface TrainingRun {
  run_id: string;
  task: Task;
  target: string;
  algorithm: Algorithm;
  parameters: Record<string, unknown>;
  primary_metric: string;
  metrics: Record<string, number | null>;
  cross_validation: {
    folds: number;
    metrics: Record<
      string,
      { mean: number; std: number; fold_scores: number[] }
    >;
  };
  diagnostics: Record<string, unknown>;
  importance: {
    native: Array<{ feature: string; importance: number; kind: string }>;
    permutation: Array<{
      feature: string;
      importance: number;
      std: number;
      kind: string;
    }>;
  };
}

export interface ChartSettings {
  title: string;
  xAxisTitle: string;
  yAxisTitle: string;
  palette: string;
  fontFamily: string;
  fontSize: number;
  height: number;
  background: string;
  showGrid: boolean;
  showLegend: boolean;
  showAnnotations: boolean;
  exportFormat: "svg" | "png";
  exportScale: number;
}
