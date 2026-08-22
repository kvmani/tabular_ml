import { Activity, Award, ShieldCheck } from "lucide-react";
import type { ReactNode } from "react";
import type { Data } from "plotly.js";
import { ChartPanel } from "../components/ChartPanel";
import { HelpButton } from "../components/HelpButton";
import type { TrainingRun } from "../types";

export function EvaluationView({ run }: { run: TrainingRun }) {
  const metrics = Object.entries(run.metrics).filter((entry): entry is [string, number] => typeof entry[1] === "number");
  const primary = run.metrics[run.primary_metric];
  const predictions = Array.isArray(run.diagnostics.predictions) ? run.diagnostics.predictions as Array<Record<string, unknown>> : [];
  const native = run.importance.native.slice(0, 20);
  const permutation = run.importance.permutation.slice(0, 20);
  const cv = run.cross_validation.metrics[run.primary_metric];
  return <div className="view-stack">
    <section className="report-hero">
      <div><p className="eyebrow">TRAINING COMPLETE</p><h2>{run.algorithm.label}</h2><p>{run.algorithm.summary}</p></div>
      <div className="hero-score"><span>{run.primary_metric.replaceAll("_", " ")}</span><strong>{formatMetric(primary)}</strong><small>untouched holdout</small></div>
      <div className="report-trust"><ShieldCheck /><strong>Leakage-safe pipeline</strong><span>Run {run.run_id.slice(0, 8)}</span></div>
    </section>
    <section className="metric-grid" aria-label="Model metrics">
      {metrics.map(([name, value], index) => <article className={name === run.primary_metric ? "primary" : ""} key={name}><span>{index === 0 ? <Award /> : <Activity />}</span><small>{name.replaceAll("_", " ")}</small><strong>{formatMetric(value)}</strong></article>)}
    </section>
    <section className="panel interpretation-note"><HelpButton title="Reading validation metrics"><p>Compare the holdout score with the cross-validation distribution. A large gap can indicate an unstable split or overfitting. Metric meaning depends on the task and class balance.</p></HelpButton><div><strong>Cross-validation stability</strong><p>{cv ? `${run.cross_validation.folds} folds: mean ${formatMetric(cv.mean)} ± ${formatMetric(cv.std)}.` : "No fold scores were returned."}</p></div></section>

    {cv ? <ChartPanel title={`${run.primary_metric.replaceAll("_", " ")} across validation folds`} subtitle={`Mean ${formatMetric(cv.mean)} · standard deviation ${formatMetric(cv.std)}`} traces={[{ type: "bar", x: cv.fold_scores.map((_, index) => `Fold ${index + 1}`), y: cv.fold_scores, text: cv.fold_scores.map((value) => value.toFixed(3)), textposition: "outside", hovertemplate: "%{x}<br>%{y:.4f}<extra></extra>" }]} layout={{ yaxis: { title: { text: run.primary_metric.replaceAll("_", " ") } } }} settingsKey={`${run.run_id}:cv`} /> : null}

    {run.task === "classification" ? <ClassificationDiagnostics run={run} /> : <RegressionDiagnostics run={run} predictions={predictions} />}

    {permutation.length || native.length ? <div className="chart-grid two-column">
      {permutation.length ? <ChartPanel title="Permutation feature importance" subtitle="Drop in validation performance after shuffling each original feature." traces={[{ type: "bar", orientation: "h", y: permutation.map((item) => item.feature).reverse(), x: permutation.map((item) => item.importance).reverse(), error_x: { type: "data", array: permutation.map((item) => item.std).reverse(), visible: true }, hovertemplate: "%{y}<br>importance %{x:.4f}<extra></extra>" }]} layout={{ margin: { l: 150 }, xaxis: { title: { text: "Performance decrease" } } }} settingsKey={`${run.run_id}:permutation`} /> : null}
      {native.length ? <ChartPanel title="Model-native feature importance" subtitle={`${native[0]?.kind ?? "Native"} importance on transformed features; useful within this model only.`} traces={[{ type: "bar", orientation: "h", y: native.map((item) => item.feature).reverse(), x: native.map((item) => Math.abs(item.importance)).reverse(), hovertemplate: "%{y}<br>%{x:.4f}<extra></extra>" }]} layout={{ margin: { l: 170 }, xaxis: { title: { text: "Absolute importance" } } }} settingsKey={`${run.run_id}:native`} /> : null}
    </div> : null}

    <section className="panel">
      <header className="panel-heading"><div><p className="eyebrow">AUDIT TABLE</p><h2>Holdout predictions</h2><p>A bounded sample of the observations behind the summary metrics.</p></div></header>
      <div className="table-scroll audit-scroll"><table><thead><tr><th>Row</th><th>Actual</th><th>Predicted</th>{run.task === "regression" ? <th>Residual</th> : null}</tr></thead><tbody>{predictions.slice(0, 100).map((row, index) => <tr key={`${String(row.row)}-${index}`}><td>{String(row.row)}</td><td>{String(row.actual)}</td><td>{String(row.predicted)}</td>{run.task === "regression" ? <td>{formatMetric(row.residual)}</td> : null}</tr>)}</tbody></table></div>
    </section>
  </div>;
}

function ClassificationDiagnostics({ run }: { run: TrainingRun }) {
  const confusion = run.diagnostics.confusion_matrix as { labels: unknown[]; matrix: number[][] } | undefined;
  const roc = run.diagnostics.roc_curve as Record<string, unknown> | undefined;
  const pr = run.diagnostics.precision_recall_curve as Record<string, unknown> | undefined;
  const charts: ReactNode[] = [];
  if (confusion) charts.push(<ChartPanel key="confusion" title="Confusion matrix" subtitle="Rows are actual classes; columns are predicted classes." traces={[{ type: "heatmap", x: confusion.labels.map(String), y: confusion.labels.map(String), z: confusion.matrix, colorscale: [[0, "#f6f2e8"], [1, "#0f766e"]], hovertemplate: "Actual %{y}<br>Predicted %{x}<br>%{z} rows<extra></extra>" }]} layout={{ xaxis: { title: { text: "Predicted class" } }, yaxis: { title: { text: "Actual class" }, autorange: "reversed" } }} settingsKey={`${run.run_id}:confusion`} />);
  if (roc) charts.push(<ChartPanel key="roc" title="ROC curve" subtitle={`Positive class: ${String(roc.positive_class)}. The diagonal represents chance ranking.`} traces={[{ type: "scatter", mode: "lines", name: `Model · AUC ${formatMetric(run.metrics.roc_auc)}`, x: numberArray(roc.fpr), y: numberArray(roc.tpr), line: { width: 3 } }, { type: "scatter", mode: "lines", name: "Chance", x: [0, 1], y: [0, 1], line: { dash: "dot", color: "#71827d" } }]} layout={{ xaxis: { title: { text: "False positive rate" }, range: [0, 1] }, yaxis: { title: { text: "True positive rate" }, range: [0, 1] } }} settingsKey={`${run.run_id}:roc`} />);
  if (pr) charts.push(<ChartPanel key="pr" title="Precision–recall curve" subtitle={`Positive class: ${String(pr.positive_class)}. Especially informative when classes are imbalanced.`} traces={[{ type: "scatter", mode: "lines", name: "Model", x: numberArray(pr.recall), y: numberArray(pr.precision), line: { width: 3 } }]} layout={{ xaxis: { title: { text: "Recall" }, range: [0, 1] }, yaxis: { title: { text: "Precision" }, range: [0, 1] } }} settingsKey={`${run.run_id}:pr`} />);
  return charts.length ? <div className="chart-grid two-column">{charts}</div> : null;
}

function RegressionDiagnostics({ run, predictions }: { run: TrainingRun; predictions: Array<Record<string, unknown>> }) {
  const actual = predictions.map((row) => Number(row.actual)); const predicted = predictions.map((row) => Number(row.predicted)); const residual = predictions.map((row) => Number(row.residual));
  const low = Math.min(...actual, ...predicted); const high = Math.max(...actual, ...predicted);
  const traces: Data[] = [{ type: "scatter", mode: "markers", name: "Holdout rows", x: actual, y: predicted, marker: { size: 7, opacity: 0.7 }, hovertemplate: "actual %{x:.4g}<br>predicted %{y:.4g}<extra></extra>" }, { type: "scatter", mode: "lines", name: "Perfect prediction", x: [low, high], y: [low, high], line: { dash: "dash", color: "#71827d" } }];
  return <div className="chart-grid two-column"><ChartPanel title="Actual vs predicted" subtitle="Points near the diagonal indicate close predictions." traces={traces} layout={{ xaxis: { title: { text: "Actual" } }, yaxis: { title: { text: "Predicted" } } }} settingsKey={`${run.run_id}:actual-predicted`} /><ChartPanel title="Residuals vs predicted" subtitle="A healthy residual cloud has no obvious curve or widening funnel." traces={[{ type: "scatter", mode: "markers", x: predicted, y: residual, marker: { size: 7, opacity: 0.7 }, hovertemplate: "predicted %{x:.4g}<br>residual %{y:.4g}<extra></extra>" }, { type: "scatter", mode: "lines", x: [Math.min(...predicted), Math.max(...predicted)], y: [0, 0], line: { dash: "dash", color: "#b4473d" }, hoverinfo: "skip" }]} layout={{ xaxis: { title: { text: "Predicted" } }, yaxis: { title: { text: "Actual − predicted" } }, showlegend: false }} settingsKey={`${run.run_id}:residuals`} /></div>;
}

function numberArray(value: unknown): number[] { return Array.isArray(value) ? value.map(Number) : []; }
function formatMetric(value: unknown): string { const number = Number(value); return Number.isFinite(number) ? number.toFixed(4) : "—"; }
