import { Download, FileJson, Plus, Send, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { HelpButton } from "../components/HelpButton";
import { api } from "../lib/api";
import type { TrainingRun, Workspace } from "../types";

interface InferenceResult { rows: Array<Record<string, unknown>>; task: string; target: string }

export function PredictView({ workspace, run, perform }: {
  workspace: Workspace;
  run: TrainingRun;
  perform: (message: string, operation: () => Promise<void>) => Promise<void>;
}) {
  const featureColumns = useMemo(() => workspace.profile.columns.filter((column) => column.name !== run.target), [run.target, workspace]);
  const example = workspace.preview.rows[0] ?? {};
  const [row, setRow] = useState<Record<string, string>>(() => Object.fromEntries(featureColumns.map((column) => [column.name, example[column.name] === null || example[column.name] === undefined ? "" : String(example[column.name])])));
  const [batchOpen, setBatchOpen] = useState(false);
  const [batch, setBatch] = useState(`[
  {
    "feature": "value"
  }
]`);
  const [result, setResult] = useState<InferenceResult | null>(null);

  function typedRow(): Record<string, unknown> {
    return Object.fromEntries(featureColumns.map((column) => {
      const value = row[column.name];
      if (value === "") return [column.name, null];
      if (["integer", "continuous"].includes(column.semantic_type)) return [column.name, Number(value)];
      if (column.semantic_type === "boolean") return [column.name, ["true", "1", "yes"].includes(value.toLowerCase())];
      return [column.name, value];
    }));
  }
  async function predict(rows: Record<string, unknown>[]) {
    await perform(`Running ${rows.length} row${rows.length === 1 ? "" : "s"} through the fitted pipeline…`, async () => setResult(await api.infer(workspace.session_id, run.run_id, rows)));
  }
  async function predictBatch() {
    await perform("Validating and running the prediction batch…", async () => {
      const parsed: unknown = JSON.parse(batch);
      if (!Array.isArray(parsed) || !parsed.every((item) => item && typeof item === "object" && !Array.isArray(item))) throw new Error("Batch JSON must be an array of row objects.");
      setResult(await api.infer(workspace.session_id, run.run_id, parsed as Record<string, unknown>[]));
    });
  }

  return <div className="view-stack">
    <section className="panel">
      <header className="panel-heading"><div><p className="eyebrow">LIVE INFERENCE</p><h2>Try the fitted pipeline on new cases</h2><p>Inputs receive exactly the same date handling, imputation, encoding, and scaling learned during training.</p></div><HelpButton title="Predictions are estimates"><p>Use predictions only where the training data, target definition, and error tolerance match the real decision. Review uncertainty, subgroup behavior, and domain constraints before deployment.</p></HelpButton></header>
      <div className="inference-toolbar"><div className="segmented-control"><button type="button" className={!batchOpen ? "active" : ""} onClick={() => setBatchOpen(false)}><Plus size={16} /> One row</button><button type="button" className={batchOpen ? "active" : ""} onClick={() => setBatchOpen(true)}><FileJson size={16} /> JSON batch</button></div><a className="button ghost" href={api.artifactUrl(workspace.session_id, run.run_id)} download><Download size={16} /> Download model artifact</a></div>
      {!batchOpen ? <>
        <div className="prediction-form">{featureColumns.map((column) => <label key={column.name}><span>{column.name}<small>{column.semantic_type}</small></span><input type={["integer", "continuous"].includes(column.semantic_type) ? "number" : "text"} value={row[column.name] ?? ""} placeholder={column.sample_values.map(String).slice(0, 2).join(" · ")} onChange={(event) => setRow({ ...row, [column.name]: event.target.value })} /></label>)}</div>
        <div className="action-row"><button type="button" className="button primary large" onClick={() => void predict([typedRow()])}><Send size={17} /> Predict {run.target}</button><button type="button" className="button ghost" onClick={() => setRow(Object.fromEntries(featureColumns.map((column) => [column.name, ""]))) }><Trash2 size={16} /> Clear</button></div>
      </> : <><label className="json-editor">Prediction rows<textarea spellCheck={false} rows={15} value={batch} onChange={(event) => setBatch(event.target.value)} /><small>Provide 1–1,000 objects with every training feature. Extra keys are ignored.</small></label><button type="button" className="button primary" onClick={() => void predictBatch()}><Send size={17} /> Run batch</button></>}
    </section>
    {result ? <section className="prediction-results">
      <header><div><p className="eyebrow">PREDICTION RESULT</p><h2>{result.rows.length === 1 ? String(result.rows[0]?.prediction) : `${result.rows.length} predictions`}</h2></div><span>{result.target} · {result.task}</span></header>
      <div className="table-scroll"><table><thead><tr><th>Row</th><th>Prediction</th>{probabilityLabels(result).map((label) => <th key={label}>P({label})</th>)}</tr></thead><tbody>{result.rows.map((item, index) => { const probabilities = item.probabilities as Record<string, number> | undefined; return <tr key={index}><td>{index + 1}</td><td><strong>{String(item.prediction)}</strong></td>{probabilityLabels(result).map((label) => <td key={label}>{probabilities ? `${(probabilities[label] * 100).toFixed(1)}%` : "—"}</td>)}</tr>; })}</tbody></table></div>
    </section> : <section className="empty-state compact"><Send /><h2>Ready for a new case</h2><p>Review the values above, then run the complete fitted pipeline.</p></section>}
    <section className="artifact-warning"><strong>Model artifact safety</strong><p>The download is a Python joblib bundle. Load only artifacts you created and trust because Python deserialization can execute code.</p></section>
  </div>;
}

function probabilityLabels(result: InferenceResult): string[] { const probabilities = result.rows[0]?.probabilities; return probabilities && typeof probabilities === "object" ? Object.keys(probabilities as object) : []; }
