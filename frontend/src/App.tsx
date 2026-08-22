import {
  BarChart3,
  BrainCircuit,
  Database,
  FlaskConical,
  LockKeyhole,
  Menu,
  RotateCcw,
  Sparkles,
  Target,
  WandSparkles,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "./lib/api";
import type {
  Algorithm,
  DatasetSummary,
  Task,
  TrainingRun,
  Workspace,
} from "./types";
import { DatasetLauncher } from "./features/DatasetLauncher";
import { DataView } from "./features/DataView";
import { PrepareView, type PreparationState } from "./features/PrepareView";
import { VisualizeView } from "./features/VisualizeView";
import { TrainView } from "./features/TrainView";
import { EvaluationView } from "./features/EvaluationView";
import { PredictView } from "./features/PredictView";

type View = "data" | "prepare" | "visualize" | "train" | "evaluate" | "predict";

const NAV: Array<{ id: View; label: string; icon: typeof Database }> = [
  { id: "data", label: "Understand", icon: Database },
  { id: "prepare", label: "Prepare", icon: WandSparkles },
  { id: "visualize", label: "Visualize", icon: BarChart3 },
  { id: "train", label: "Train", icon: BrainCircuit },
  { id: "evaluate", label: "Evaluate", icon: FlaskConical },
  { id: "predict", label: "Predict", icon: Target },
];

const DEFAULT_PREPARATION: PreparationState = {
  numeric_imputer: "median",
  categorical_imputer: "most_frequent",
  scaling: "standard",
  encoding: "onehot",
  add_missing_indicators: true,
  date_strategy: "components",
  rare_category_threshold: 0.01,
  exclude_columns: [],
};

export function App() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [view, setView] = useState<View>("data");
  const [target, setTarget] = useState("");
  const [task, setTask] = useState<Task>("classification");
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [preparation, setPreparation] =
    useState<PreparationState>(DEFAULT_PREPARATION);
  const [run, setRun] = useState<TrainingRun | null>(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [navOpen, setNavOpen] = useState(false);

  useEffect(() => {
    api.datasets().then(setDatasets).catch((caught: Error) => setError(caught.message));
  }, []);

  useEffect(() => {
    if (!workspace) return;
    api.algorithms(task).then(setAlgorithms).catch((caught: Error) => setError(caught.message));
  }, [task, workspace]);

  const candidateMap = useMemo(
    () =>
      new Map(
        workspace?.profile.target_candidates.map((candidate) => [
          candidate.column,
          candidate.task,
        ]) ?? [],
      ),
    [workspace],
  );

  function installWorkspace(next: Workspace) {
    window.scrollTo({ top: 0, behavior: "auto" });
    setWorkspace(next);
    const candidate = next.profile.target_candidates[0];
    setTarget(candidate?.column ?? "");
    setTask(candidate?.task ?? "classification");
    const recommended = next.dataset.metadata.recommended_exclusions ?? [];
    const automatic = next.profile.columns
      .filter((column) => column.suggested_role !== "feature")
      .map((column) => column.name);
    setPreparation({
      ...DEFAULT_PREPARATION,
      exclude_columns: Array.from(new Set([...recommended, ...automatic])),
    });
    setRun(null);
    setView("data");
  }

  async function perform(message: string, operation: () => Promise<void>) {
    setBusy(message);
    setError("");
    try {
      await operation();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The operation failed.");
    } finally {
      setBusy("");
    }
  }

  function chooseTarget(column: string) {
    setTarget(column);
    setTask(candidateMap.get(column) ?? "regression");
    setRun(null);
  }

  if (!workspace) {
    return (
      <>
        <DatasetLauncher
          datasets={datasets}
          onLoad={(key) =>
            void perform("Reading the dataset and profiling every column…", async () => {
              installWorkspace(await api.loadDataset(key));
            })
          }
          onUpload={(file) =>
            void perform("Parsing your table and detecting feature types…", async () => {
              installWorkspace(await api.uploadDataset(file));
            })
          }
        />
        <StatusOverlay busy={busy} error={error} onClose={() => setError("")} />
      </>
    );
  }

  const activeIndex = NAV.findIndex((item) => item.id === view);
  return (
    <div className="workspace-shell">
      <header className="workspace-header">
        <button
          type="button"
          className="icon-button mobile-menu"
          aria-label="Open navigation"
          onClick={() => setNavOpen(true)}
        >
          <Menu aria-hidden="true" />
        </button>
        <button
          type="button"
          className="brand brand-button"
          onClick={() => setWorkspace(null)}
        >
          <span className="brand-mark">T</span>
          <span><strong>Tabular ML</strong><small>Local workbench</small></span>
        </button>
        <div className="dataset-context">
          <span>ACTIVE DATASET</span>
          <strong>{workspace.dataset.name}</strong>
          <small>{workspace.dataset.rows.toLocaleString()} rows · {workspace.dataset.columns} columns</small>
        </div>
        <div className="header-actions">
          <span className="privacy-pill"><LockKeyhole size={14} /> In-memory only</span>
          <button
            type="button"
            className="button ghost"
            onClick={() => void perform("Restoring the original table…", async () => {
              installWorkspace(await api.reset(workspace.session_id));
            })}
          >
            <RotateCcw size={16} /> Reset data
          </button>
        </div>
      </header>

      <aside className={`workspace-nav ${navOpen ? "open" : ""}`}>
        <div className="nav-mobile-header">
          <strong>Workflow</strong>
          <button className="icon-button" onClick={() => setNavOpen(false)}><X /></button>
        </div>
        <p className="nav-label">MODEL WORKFLOW</p>
        <nav aria-label="Machine learning workflow">
          {NAV.map((item, index) => {
            const Icon = item.icon;
            return (
              <button
                type="button"
                key={item.id}
                className={view === item.id ? "active" : ""}
                disabled={(item.id === "evaluate" || item.id === "predict") && !run}
                onClick={() => {
                  setView(item.id);
                  setNavOpen(false);
                }}
              >
                <span className="nav-index">{String(index + 1).padStart(2, "0")}</span>
                <Icon size={19} aria-hidden="true" />
                <span>{item.label}</span>
                {index < activeIndex ? <span className="nav-done">✓</span> : null}
              </button>
            );
          })}
        </nav>
        <div className="nav-tip">
          <Sparkles aria-hidden="true" />
          <strong>Guided, not hidden</strong>
          <p>Recommended defaults are always visible and editable.</p>
        </div>
      </aside>

      <main className="workspace-main">
        <header className="view-header">
          <div>
            <p className="eyebrow">STEP {activeIndex + 1} OF 6</p>
            <h1>{NAV[activeIndex].label}</h1>
          </div>
          {target ? (
            <div className="target-context">
              <span>Target</span><strong>{target}</strong>
              <span className={`task-badge ${task}`}>{task}</span>
            </div>
          ) : null}
        </header>
        {view === "data" ? <DataView workspace={workspace} /> : null}
        {view === "prepare" ? (
          <PrepareView
            workspace={workspace}
            target={target}
            task={task}
            preparation={preparation}
            onTarget={chooseTarget}
            onPreparation={setPreparation}
            onWorkspace={setWorkspace}
            perform={perform}
          />
        ) : null}
        {view === "visualize" ? <VisualizeView workspace={workspace} perform={perform} /> : null}
        {view === "train" ? (
          <TrainView
            workspace={workspace}
            target={target}
            task={task}
            preparation={preparation}
            algorithms={algorithms}
            perform={perform}
            onRun={(next) => {
              setRun(next);
              setView("evaluate");
            }}
          />
        ) : null}
        {view === "evaluate" && run ? <EvaluationView run={run} /> : null}
        {view === "predict" && run ? (
          <PredictView workspace={workspace} run={run} perform={perform} />
        ) : null}
      </main>
      <StatusOverlay busy={busy} error={error} onClose={() => setError("")} />
    </div>
  );
}

function StatusOverlay({
  busy,
  error,
  onClose,
}: {
  busy: string;
  error: string;
  onClose: () => void;
}) {
  if (!busy && !error) return null;
  if (busy) {
    return (
      <div className="status-overlay" role="status" aria-live="polite">
        <div className="status-dialog">
          <span className="loader" />
          <p className="eyebrow">WORKING LOCALLY</p>
          <h2>{busy}</h2>
          <p>Nothing is being sent to an external service.</p>
        </div>
      </div>
    );
  }
  return (
    <div className="toast error-toast" role="alert">
      <div><strong>Couldn’t complete that step</strong><p>{error}</p></div>
      <button className="icon-button" aria-label="Dismiss error" onClick={onClose}><X /></button>
    </div>
  );
}
