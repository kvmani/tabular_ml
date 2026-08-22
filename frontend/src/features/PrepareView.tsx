import { AlertTriangle, Filter, Sparkles } from "lucide-react";
import { useMemo, useState } from "react";
import { HelpButton } from "../components/HelpButton";
import { api } from "../lib/api";
import type { Task, Workspace } from "../types";

export interface PreparationState {
  numeric_imputer: string;
  categorical_imputer: string;
  scaling: string;
  encoding: string;
  add_missing_indicators: boolean;
  date_strategy: string;
  rare_category_threshold: number;
  exclude_columns: string[];
}

interface PrepareViewProps {
  workspace: Workspace;
  target: string;
  task: Task;
  preparation: PreparationState;
  onTarget: (target: string) => void;
  onPreparation: (state: PreparationState) => void;
  onWorkspace: (workspace: Workspace) => void;
  perform: (message: string, operation: () => Promise<void>) => Promise<void>;
}

export function PrepareView({
  workspace,
  target,
  task,
  preparation,
  onTarget,
  onPreparation,
  onWorkspace,
  perform,
}: PrepareViewProps) {
  const [plan, setPlan] = useState<Record<string, unknown> | null>(null);
  const [outlierConfig, setOutlierConfig] = useState({
    method: "iqr",
    threshold: 1.5,
    contamination: 0.05,
    columns: [] as string[],
  });
  const [outlierReport, setOutlierReport] =
    useState<Record<string, unknown> | null>(null);
  const numericColumns = useMemo(
    () =>
      workspace.profile.columns.filter((column) =>
        ["integer", "continuous"].includes(column.semantic_type),
      ),
    [workspace],
  );

  function update<K extends keyof PreparationState>(
    key: K,
    value: PreparationState[K],
  ) {
    onPreparation({ ...preparation, [key]: value });
    setPlan(null);
  }

  function toggleExcluded(column: string) {
    const excluded = preparation.exclude_columns.includes(column)
      ? preparation.exclude_columns.filter((item) => item !== column)
      : [...preparation.exclude_columns, column];
    update("exclude_columns", excluded);
  }

  return (
    <div className="view-stack">
      <section className="panel target-panel">
        <div>
          <p className="eyebrow">OUTCOME DEFINITION</p>
          <h2>What should the model learn to predict?</h2>
          <p>
            The target determines whether this is classification or regression.
            Confirm it before any split or preprocessing is fitted.
          </p>
        </div>
        <label className="large-field">
          Target column
          <select value={target} onChange={(event) => onTarget(event.target.value)}>
            {workspace.profile.columns.map((column) => (
              <option key={column.name} value={column.name}>
                {column.name} · {column.semantic_type}
              </option>
            ))}
          </select>
        </label>
        <div className={`task-callout ${task}`}>
          <strong>{task === "classification" ? "Classification" : "Regression"}</strong>
          <p>
            {task === "classification"
              ? "The model will assign one of a finite set of outcome labels."
              : "The model will estimate a continuous numeric outcome."}
          </p>
        </div>
      </section>

      <section className="panel">
        <header className="panel-heading">
          <div>
            <p className="eyebrow">PREPARATION RECIPE</p>
            <h2>Make every transformation explicit</h2>
            <p>Each step is fitted only on training folds to prevent validation leakage.</p>
          </div>
          <HelpButton title="Leakage-safe preparation">
            <p>
              Imputation, scaling, category learning, and date expansion are part of
              one sklearn pipeline. During cross-validation, each fold learns those
              values from its training portion only.
            </p>
          </HelpButton>
        </header>
        <div className="control-grid">
          <label>
            Numeric missing values
            <select
              value={preparation.numeric_imputer}
              onChange={(event) => update("numeric_imputer", event.target.value)}
            >
              <option value="median">Median (robust default)</option>
              <option value="mean">Mean</option>
              <option value="most_frequent">Most frequent</option>
              <option value="knn">KNN imputation</option>
            </select>
            <small>How absent numeric values are estimated.</small>
          </label>
          <label>
            Categorical missing values
            <select
              value={preparation.categorical_imputer}
              onChange={(event) => update("categorical_imputer", event.target.value)}
            >
              <option value="most_frequent">Most frequent</option>
              <option value="constant">Explicit “missing” level</option>
            </select>
            <small>Preserve absence as information or use the mode.</small>
          </label>
          <label>
            Numeric scaling
            <select
              value={preparation.scaling}
              onChange={(event) => update("scaling", event.target.value)}
            >
              <option value="standard">Standard (z-score)</option>
              <option value="robust">Robust (median / IQR)</option>
              <option value="minmax">Min–max (0 to 1)</option>
              <option value="none">None</option>
            </select>
            <small>Important for neighbours, SVMs, linear models, and MLPs.</small>
          </label>
          <label>
            Category encoding
            <select
              value={preparation.encoding}
              onChange={(event) => update("encoding", event.target.value)}
            >
              <option value="onehot">One-hot</option>
              <option value="ordinal">Ordinal integer codes</option>
            </select>
            <small>One-hot is safer when levels have no natural order.</small>
          </label>
          <label>
            Date handling
            <select
              value={preparation.date_strategy}
              onChange={(event) => update("date_strategy", event.target.value)}
            >
              <option value="components">Year, month, day, weekday, day of year</option>
              <option value="drop">Exclude dates</option>
            </select>
            <small>Calendar components let models use temporal structure.</small>
          </label>
          <label>
            Rare category threshold
            <input
              type="range"
              min="0"
              max="0.2"
              step="0.005"
              value={preparation.rare_category_threshold}
              onChange={(event) =>
                update("rare_category_threshold", Number(event.target.value))
              }
            />
            <span className="range-value">
              {(preparation.rare_category_threshold * 100).toFixed(1)}%
            </span>
            <small>Combine levels appearing less often than this fraction.</small>
          </label>
        </div>
        <label className="switch-row prominent-switch">
          <input
            type="checkbox"
            checked={preparation.add_missing_indicators}
            onChange={(event) =>
              update("add_missing_indicators", event.target.checked)
            }
          />
          <span><strong>Add missingness indicators</strong><small>Let the model learn whether absence itself carries signal.</small></span>
        </label>

        <div className="feature-selector">
          <header>
            <div><h3>Feature roles</h3><p>Unchecked columns are excluded from training.</p></div>
            <span>{workspace.profile.columns.length - preparation.exclude_columns.length - 1} active</span>
          </header>
          <div className="feature-chip-grid">
            {workspace.profile.columns
              .filter((column) => column.name !== target)
              .map((column) => {
                const active = !preparation.exclude_columns.includes(column.name);
                return (
                  <label className={`feature-chip ${active ? "active" : ""}`} key={column.name}>
                    <input
                      type="checkbox"
                      checked={active}
                      onChange={() => toggleExcluded(column.name)}
                    />
                    <span><strong>{column.name}</strong><small>{column.semantic_type}</small></span>
                    {column.warnings.length ? <AlertTriangle size={14} aria-label={column.warnings.join(", ")} /> : null}
                  </label>
                );
              })}
          </div>
        </div>

        <div className="action-row">
          <button
            type="button"
            className="button primary"
            onClick={() =>
              void perform("Validating the leakage-safe feature plan…", async () => {
                setPlan(
                  await api.preparationPlan(
                    workspace.session_id,
                    target,
                    preparation as unknown as Record<string, unknown>,
                  ),
                );
              })
            }
          >
            <Sparkles size={17} /> Preview feature plan
          </button>
          {plan ? (
            <div className="inline-result">
              <strong>{String(plan.estimated_output_features)} estimated features</strong>
              <span>{String(plan.usable_rows)} usable rows · {String(plan.task)}</span>
            </div>
          ) : null}
        </div>
      </section>

      <section className="panel">
        <header className="panel-heading">
          <div>
            <p className="eyebrow">OUTLIER LAB</p>
            <h2>Detect first. Treat only with a reason.</h2>
            <p>Outliers can be errors, rare valid cases, or the most important observations.</p>
          </div>
          <HelpButton title="Outliers are contextual">
            <p>
              IQR and z-score methods flag unusual values per feature. Isolation
              Forest and Local Outlier Factor inspect combinations of features.
              Always inspect domain meaning before dropping rows.
            </p>
          </HelpButton>
        </header>
        <div className="outlier-layout">
          <div className="control-grid compact">
            <label>
              Detector
              <select
                value={outlierConfig.method}
                onChange={(event) =>
                  setOutlierConfig({ ...outlierConfig, method: event.target.value })
                }
              >
                <option value="iqr">IQR fences</option>
                <option value="zscore">Z-score</option>
                <option value="modified_zscore">Modified z-score (MAD)</option>
                <option value="isolation_forest">Isolation Forest</option>
                <option value="lof">Local Outlier Factor</option>
              </select>
            </label>
            {["iqr", "zscore", "modified_zscore"].includes(outlierConfig.method) ? (
              <label>
                Threshold
                <input
                  type="number"
                  min="0.1"
                  step="0.1"
                  value={outlierConfig.threshold}
                  onChange={(event) =>
                    setOutlierConfig({
                      ...outlierConfig,
                      threshold: Number(event.target.value),
                    })
                  }
                />
              </label>
            ) : (
              <label>
                Expected fraction
                <input
                  type="number"
                  min="0.01"
                  max="0.25"
                  step="0.01"
                  value={outlierConfig.contamination}
                  onChange={(event) =>
                    setOutlierConfig({
                      ...outlierConfig,
                      contamination: Number(event.target.value),
                    })
                  }
                />
              </label>
            )}
          </div>
          <div className="outlier-columns">
            <strong>Numeric columns</strong>
            <div>
              {numericColumns
                .filter((column) => column.name !== target)
                .map((column) => (
                  <label key={column.name}>
                    <input
                      type="checkbox"
                      checked={outlierConfig.columns.includes(column.name)}
                      onChange={() =>
                        setOutlierConfig({
                          ...outlierConfig,
                          columns: outlierConfig.columns.includes(column.name)
                            ? outlierConfig.columns.filter((item) => item !== column.name)
                            : [...outlierConfig.columns, column.name],
                        })
                      }
                    />
                    {column.name}
                  </label>
                ))}
            </div>
            <small>No selection means all numeric features except the target.</small>
          </div>
        </div>
        <div className="action-row">
          <button
            type="button"
            className="button secondary"
            onClick={() =>
              void perform("Scanning the working table for unusual rows…", async () => {
                setOutlierReport(
                  await api.outliers(workspace.session_id, target, outlierConfig),
                );
              })
            }
          >
            <Filter size={17} /> Detect outliers
          </button>
          {outlierReport ? (
            <>
              <div className="inline-result">
                <strong>{String(outlierReport.outlier_rows)} rows flagged</strong>
                <span>{(Number(outlierReport.outlier_ratio) * 100).toFixed(2)}% of the table</span>
              </div>
              <div className="button-cluster">
                {[
                  ["winsorize", "Cap extremes"],
                  ["mask", "Mark missing"],
                  ["drop", "Drop rows"],
                ].map(([action, label]) => (
                  <button
                    type="button"
                    className={action === "drop" ? "button danger" : "button ghost"}
                    key={action}
                    onClick={() =>
                      void perform(`Applying ${label.toLowerCase()}…`, async () => {
                        onWorkspace(
                          await api.applyOutliers(
                            workspace.session_id,
                            target,
                            outlierConfig,
                            action,
                          ),
                        );
                        setOutlierReport(null);
                      })
                    }
                  >
                    {label}
                  </button>
                ))}
              </div>
            </>
          ) : null}
        </div>
      </section>
    </div>
  );
}
