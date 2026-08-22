import {
  ArrowRight,
  Binary,
  FileSpreadsheet,
  FlaskConical,
  LockKeyhole,
  Upload,
} from "lucide-react";
import { useRef, useState } from "react";
import type { DatasetSummary } from "../types";

interface DatasetLauncherProps {
  datasets: DatasetSummary[];
  onLoad: (key: string) => void;
  onUpload: (file: File) => void;
}

export function DatasetLauncher({
  datasets,
  onLoad,
  onUpload,
}: DatasetLauncherProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function accept(files: FileList | null) {
    if (files?.[0]) onUpload(files[0]);
  }

  return (
    <div className="landing-page">
      <header className="landing-header">
        <a className="brand" href="/tabular_ml/">
          <span className="brand-mark">T</span>
          <span>
            <strong>Tabular ML</strong>
            <small>Local workbench</small>
          </span>
        </a>
        <div className="privacy-pill">
          <LockKeyhole size={15} aria-hidden="true" />
          Your data stays in this process
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">FROM TABLE TO EVIDENCE</p>
            <h1>
              Build a model
              <em> you can defend.</em>
            </h1>
            <p className="hero-lede">
              Inspect the data, make preparation choices explicit, compare serious
              baselines, and understand every validation figure—without writing code
              or sending a row anywhere.
            </p>
            <div className="hero-proof">
              <span><Binary aria-hidden="true" /> Classification + regression</span>
              <span><FlaskConical aria-hidden="true" /> 14 model families</span>
              <span><FileSpreadsheet aria-hidden="true" /> CSV + Excel</span>
            </div>
          </div>

          <div
            className={`upload-stage ${dragging ? "dragging" : ""}`}
            onDragEnter={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              accept(event.dataTransfer.files);
            }}
          >
            <span className="upload-icon"><Upload aria-hidden="true" /></span>
            <p className="eyebrow">START WITH YOUR DATA</p>
            <h2>Drop a table here</h2>
            <p>CSV or Excel · up to 25 MB · processed in memory</p>
            <button
              type="button"
              className="button primary wide"
              onClick={() => inputRef.current?.click()}
            >
              Choose a file <ArrowRight size={17} aria-hidden="true" />
            </button>
            <input
              ref={inputRef}
              className="visually-hidden"
              type="file"
              accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
              onChange={(event) => accept(event.target.files)}
            />
            <p className="privacy-note">
              No telemetry, cloud upload, account, or persistent copy.
            </p>
          </div>
        </section>

        <section className="dataset-section" aria-labelledby="demo-title">
          <div className="section-heading">
            <div>
              <p className="eyebrow">CURATED STARTING POINTS</p>
              <h2 id="demo-title">Learn with a dataset that has a lesson.</h2>
            </div>
            <p>
              Each includes provenance, a declared license, recommended targets,
              workflow prompts, and cautions about responsible interpretation.
            </p>
          </div>
          <div className="dataset-grid">
            {datasets.map((dataset, index) => (
              <article className="dataset-card" key={dataset.key}>
                <header>
                  <span className="dataset-number">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <span className={`task-badge ${dataset.task}`}>
                    {dataset.task}
                  </span>
                </header>
                <h3>{dataset.name}</h3>
                <p>{dataset.summary}</p>
                <div className="dataset-stats">
                  <span><strong>{dataset.rows.toLocaleString()}</strong> rows</span>
                  <span><strong>{dataset.columns}</strong> columns</span>
                  <span><strong>{dataset.target}</strong> target</span>
                </div>
                <div className="tag-row">
                  {dataset.tags.slice(0, 3).map((tag) => (
                    <span key={tag}>{tag.replaceAll("-", " ")}</span>
                  ))}
                </div>
                <button
                  type="button"
                  className="text-button"
                  onClick={() => onLoad(dataset.key)}
                >
                  Open dataset <ArrowRight size={16} aria-hidden="true" />
                </button>
              </article>
            ))}
          </div>
        </section>
      </main>

      <footer className="landing-footer">
        <span>Tabular ML Workbench</span>
        <span>Offline-capable · open source · publication-ready figures</span>
      </footer>
    </div>
  );
}

