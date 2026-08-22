import { ArrowRight, Check, GitCompareArrows, Gauge, Play, Scale } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { HelpButton } from "../components/HelpButton";
import { api } from "../lib/api";
import type { Algorithm, ParameterSpec, Task, TrainingRun, Workspace } from "../types";
import type { PreparationState } from "./PrepareView";

interface Comparison {
  leaderboard: Array<Record<string, unknown>>;
  failures: Array<Record<string, unknown>>;
}

export function TrainView({ workspace, target, task, preparation, algorithms, perform, onRun }: {
  workspace: Workspace;
  target: string;
  task: Task;
  preparation: PreparationState;
  algorithms: Algorithm[];
  perform: (message: string, operation: () => Promise<void>) => Promise<void>;
  onRun: (run: TrainingRun) => void;
}) {
  const [algorithmId, setAlgorithmId] = useState(algorithms[0]?.id ?? "");
  const [parameters, setParameters] = useState<Record<string, Record<string, unknown>>>({});
  const [selected, setSelected] = useState<string[]>([]);
  const [mode, setMode] = useState<"single" | "compare">("single");
  const [options, setOptions] = useState({ test_size: 0.2, cv_folds: 5, random_seed: 42, permutation_importance: true });
  const [comparison, setComparison] = useState<Comparison | null>(null);

  useEffect(() => {
    if (!algorithms.some((item) => item.id === algorithmId)) setAlgorithmId(algorithms[0]?.id ?? "");
    setSelected((current) => current.filter((id) => algorithms.some((item) => item.id === id)));
  }, [algorithmId, algorithms]);

  const current = useMemo(() => algorithms.find((item) => item.id === algorithmId), [algorithmId, algorithms]);
  function valuesFor(algorithm: Algorithm) {
    return Object.fromEntries(algorithm.parameters.map((parameter) => [parameter.name, parameters[algorithm.id]?.[parameter.name] ?? parameter.default]));
  }
  function updateParameter(name: string, value: unknown) {
    setParameters({ ...parameters, [algorithmId]: { ...parameters[algorithmId], [name]: value } });
  }
  function trainingPayload(algorithm: string, params: Record<string, unknown>) {
    return { target, algorithm, parameters: params, preparation, options };
  }
  async function train() {
    if (!current) return;
    await perform(`Training ${current.label} with leakage-safe validation…`, async () => {
      onRun(await api.train(workspace.session_id, trainingPayload(current.id, valuesFor(current))));
    });
  }
  async function compare() {
    await perform("Training and ranking the selected model families…", async () => {
      setComparison(await api.compare(workspace.session_id, { target, algorithms: selected, parameters: Object.fromEntries(selected.map((id) => { const algorithm = algorithms.find((item) => item.id === id); return [id, algorithm ? valuesFor(algorithm) : {}]; })), preparation, options }));
    });
  }

  return <div className="view-stack">
    <section className="panel">
      <header className="panel-heading">
        <div><p className="eyebrow">MODEL LAB</p><h2>Choose a model family intentionally</h2><p>Start with an interpretable baseline, then compare complexity only when validation supports it.</p></div>
        <HelpButton title="How model selection works"><p>The test split remains untouched until final evaluation. Cross-validation scores come only from the training portion, and every preprocessing step is refitted inside each fold.</p></HelpButton>
      </header>
      <div className="segmented-control" aria-label="Training mode">
        <button type="button" className={mode === "single" ? "active" : ""} onClick={() => setMode("single")}><Play size={16} /> Train one</button>
        <button type="button" className={mode === "compare" ? "active" : ""} onClick={() => setMode("compare")}><GitCompareArrows size={16} /> Compare 2–6</button>
      </div>
      <div className="algorithm-grid">
        {algorithms.map((algorithm) => {
          const active = mode === "single" ? algorithmId === algorithm.id : selected.includes(algorithm.id);
          return <button type="button" className={`algorithm-card ${active ? "active" : ""}`} key={algorithm.id} onClick={() => {
            if (mode === "single") setAlgorithmId(algorithm.id);
            else setSelected(active ? selected.filter((id) => id !== algorithm.id) : selected.length < 6 ? [...selected, algorithm.id] : selected);
          }}>
            <span className="algorithm-check">{active ? <Check size={15} /> : null}</span>
            <span className="algorithm-family">{algorithm.family}</span><strong>{algorithm.label}</strong><p>{algorithm.summary}</p>
            <span className="algorithm-traits"><small><Gauge size={13} /> {algorithm.speed}</small><small><Scale size={13} /> {algorithm.interpretability} clarity</small></span>
          </button>;
        })}
      </div>
    </section>

    {mode === "single" && current ? <section className="panel parameter-panel">
      <header className="panel-heading"><div><p className="eyebrow">TUNE WITH RESTRAINT</p><h2>{current.label} settings</h2><p>{current.strengths}</p></div><span className="caution-note">{current.cautions}</span></header>
      <div className="control-grid">{current.parameters.map((parameter) => <ParameterControl key={parameter.name} parameter={parameter} value={valuesFor(current)[parameter.name]} onChange={(value) => updateParameter(parameter.name, value)} />)}</div>
    </section> : null}

    <section className="panel validation-panel">
      <header className="panel-heading"><div><p className="eyebrow">VALIDATION DESIGN</p><h2>Keep the final check honest</h2></div></header>
      <div className="control-grid compact">
        <label>Holdout fraction<input type="number" min="0.1" max="0.4" step="0.05" value={options.test_size} onChange={(event) => setOptions({ ...options, test_size: Number(event.target.value) })} /><small>Reserved once for final metrics.</small></label>
        <label>Cross-validation folds<input type="number" min="2" max="10" value={options.cv_folds} onChange={(event) => setOptions({ ...options, cv_folds: Number(event.target.value) })} /><small>Repeated training estimates stability.</small></label>
        <label>Random seed<input type="number" min="0" value={options.random_seed} onChange={(event) => setOptions({ ...options, random_seed: Number(event.target.value) })} /><small>Makes splits and stochastic models reproducible.</small></label>
      </div>
      <label className="switch-row prominent-switch"><input type="checkbox" checked={options.permutation_importance} onChange={(event) => setOptions({ ...options, permutation_importance: event.target.checked })} /><span><strong>Compute permutation importance</strong><small>Measures performance loss after shuffling each original feature.</small></span></label>
      <div className="action-row"><button type="button" className="button primary large" disabled={!target || (mode === "compare" && selected.length < 2)} onClick={() => void (mode === "single" ? train() : compare())}>{mode === "single" ? <><Play size={18} /> Train and evaluate</> : <><GitCompareArrows size={18} /> Compare {selected.length} models</>}</button><span className="split-note">Target: <strong>{target}</strong> · {task}</span></div>
    </section>

    {comparison ? <section className="panel leaderboard-panel">
      <header className="panel-heading"><div><p className="eyebrow">COMPARISON</p><h2>Validation leaderboard</h2><p>Select any completed run to inspect its full diagnostic report.</p></div></header>
      <div className="table-scroll"><table><thead><tr><th>Rank</th><th>Model</th><th>Metric</th><th>Holdout</th><th>CV mean</th><th /></tr></thead><tbody>{comparison.leaderboard.map((row, index) => { const algorithm = row.algorithm as Algorithm; return <tr key={String(row.run_id)}><td className="rank-cell">#{index + 1}</td><td><strong>{algorithm.label}</strong><small>{algorithm.family}</small></td><td>{String(row.primary_metric).replaceAll("_", " ")}</td><td>{formatMetric(row.test_score)}</td><td>{formatMetric(row.cv_score)}</td><td><button className="button ghost" onClick={() => void perform("Opening the full model report…", async () => onRun(await api.run(workspace.session_id, String(row.run_id))))}>Inspect <ArrowRight size={15} /></button></td></tr>; })}</tbody></table></div>
      {comparison.failures.length ? <p className="caution-note">{comparison.failures.length} selected model(s) could not be fitted. Successful runs are shown above.</p> : null}
    </section> : null}
  </div>;
}

function ParameterControl({ parameter, value, onChange }: { parameter: ParameterSpec; value: unknown; onChange: (value: unknown) => void }) {
  if (parameter.kind === "boolean") return <label className="switch-row"><input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} /><span><strong>{parameter.label}</strong><small>{parameter.help}</small></span></label>;
  if (parameter.kind === "choice") return <label>{parameter.label}<select value={value === null ? "__null" : String(value)} onChange={(event) => { const raw = event.target.value; const choice = raw === "__null" ? null : parameter.choices.find((item) => String(item) === raw) ?? raw; onChange(choice); }}>{parameter.choices.map((choice) => <option key={String(choice)} value={choice === null ? "__null" : String(choice)}>{choice === null ? "Automatic" : String(choice)}</option>)}</select><small>{parameter.help}</small></label>;
  return <label>{parameter.label}<input type="number" value={value === null ? "" : Number(value)} min={parameter.minimum} max={parameter.maximum} step={parameter.kind === "integer" ? 1 : "any"} onChange={(event) => onChange(event.target.value === "" ? null : Number(event.target.value))} /><small>{parameter.help}</small></label>;
}

function formatMetric(value: unknown) { const numeric = Number(value); return Number.isFinite(numeric) ? numeric.toFixed(4) : "—"; }
