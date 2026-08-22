import { AlertTriangle, CheckCircle2, Database, ShieldAlert } from "lucide-react";
import type { Workspace } from "../types";
import { HelpButton } from "../components/HelpButton";

interface DataViewProps {
  workspace: Workspace;
}

function value(value: unknown) {
  if (value === null || value === undefined || value === "") {
    return <span className="missing-value">missing</span>;
  }
  return String(value);
}

export function DataView({ workspace }: DataViewProps) {
  const metadata = workspace.dataset.metadata;
  return (
    <div className="view-stack">
      <section className="summary-strip">
        <div>
          <span>Rows</span>
          <strong>{workspace.profile.overview.rows.toLocaleString()}</strong>
        </div>
        <div>
          <span>Columns</span>
          <strong>{workspace.profile.overview.columns}</strong>
        </div>
        <div>
          <span>Missing cells</span>
          <strong>{workspace.profile.overview.missing_cells.toLocaleString()}</strong>
        </div>
        <div>
          <span>Duplicates</span>
          <strong>{workspace.profile.overview.duplicate_rows.toLocaleString()}</strong>
        </div>
        <div className="health-summary">
          <span>Data health <HelpButton title="Data health score">
            <p>
              A quick diagnostic based on missing cells, duplicate rows, constants,
              and identifier-like columns. It is a triage signal, not a judgment of
              scientific validity.
            </p>
          </HelpButton></span>
          <strong>{workspace.profile.overview.health_score}<small>/100</small></strong>
        </div>
      </section>

      {workspace.profile.warnings.length ? (
        <section className="notice-list" aria-label="Data quality findings">
          {workspace.profile.warnings.map((warning) => (
            <article className={`notice ${warning.severity}`} key={warning.code}>
              <AlertTriangle size={18} aria-hidden="true" />
              <div><strong>{warning.code.replaceAll("_", " ")}</strong><p>{warning.message}</p></div>
            </article>
          ))}
        </section>
      ) : (
        <div className="notice success">
          <CheckCircle2 aria-hidden="true" /> No immediate structural issues detected.
        </div>
      )}

      {metadata.learning_goals?.length ? (
        <section className="learning-panel">
          <div>
            <p className="eyebrow">WHY THIS DATASET</p>
            <h2>{metadata.name}</h2>
            <p>{metadata.description}</p>
          </div>
          <ul>
            {metadata.learning_goals.map((goal) => <li key={goal}>{goal}</li>)}
          </ul>
          <aside>
            <ShieldAlert aria-hidden="true" />
            <div><strong>Use responsibly</strong><p>{metadata.ethical_notes}</p></div>
          </aside>
        </section>
      ) : null}

      <section className="panel">
        <header className="panel-heading">
          <div>
            <p className="eyebrow">SCHEMA MAP</p>
            <h2>What each column appears to mean</h2>
            <p>Automatic roles are editable in Prepare. Review them before fitting.</p>
          </div>
          <Database aria-hidden="true" />
        </header>
        <div className="table-scroll">
          <table className="schema-table">
            <thead><tr><th>Column</th><th>Semantic type</th><th>Suggested role</th><th>Missing</th><th>Distinct</th><th>Examples</th></tr></thead>
            <tbody>
              {workspace.profile.columns.map((column) => (
                <tr key={column.name}>
                  <th scope="row">{column.name}</th>
                  <td><span className={`type-chip ${column.semantic_type}`}>{column.semantic_type}</span></td>
                  <td>{column.suggested_role}</td>
                  <td>{column.missing ? `${column.missing} (${(column.missing_ratio * 100).toFixed(1)}%)` : "—"}</td>
                  <td>{column.unique.toLocaleString()}</td>
                  <td className="examples">{column.sample_values.slice(0, 3).map(String).join(" · ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <header className="panel-heading">
          <div><p className="eyebrow">DATA PREVIEW</p><h2>First {workspace.preview.shown} rows</h2></div>
        </header>
        <div className="table-scroll preview-scroll">
          <table>
            <thead><tr>{workspace.preview.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
            <tbody>{workspace.preview.rows.map((row, index) => (
              <tr key={index}>{workspace.preview.columns.map((column) => <td key={column}>{value(row[column])}</td>)}</tr>
            ))}</tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

