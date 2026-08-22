import { BarChart3, ScatterChart } from "lucide-react";
import { useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js";
import { ChartPanel } from "../components/ChartPanel";
import { HelpButton } from "../components/HelpButton";
import { api } from "../lib/api";
import type { Workspace } from "../types";

type ChartKind = "distribution" | "correlation" | "scatter" | "box" | "missingness";

interface ChartResult {
  type: string;
  data: Record<string, unknown>;
}

interface ChartDefinition {
  title: string;
  subtitle: string;
  traces: Data[];
  layout: Partial<Layout>;
}

export function VisualizeView({
  workspace,
  perform,
}: {
  workspace: Workspace;
  perform: (message: string, operation: () => Promise<void>) => Promise<void>;
}) {
  const numeric = useMemo(
    () => workspace.profile.columns.filter((column) => ["integer", "continuous"].includes(column.semantic_type)),
    [workspace],
  );
  const columns = workspace.profile.columns;
  const [kind, setKind] = useState<ChartKind>("distribution");
  const [column, setColumn] = useState(columns[0]?.name ?? "");
  const [x, setX] = useState(numeric[0]?.name ?? "");
  const [y, setY] = useState(numeric[1]?.name ?? numeric[0]?.name ?? "");
  const [group, setGroup] = useState("");
  const [method, setMethod] = useState("pearson");
  const [bins, setBins] = useState(24);
  const [result, setResult] = useState<ChartResult | null>(null);

  async function build() {
    const config: Record<string, unknown> = { type: kind };
    if (kind === "distribution") Object.assign(config, { column, bins, kde: true });
    if (kind === "correlation") Object.assign(config, { method, columns: numeric.map((item) => item.name) });
    if (kind === "scatter") Object.assign(config, { x, y, color: group || undefined, trendline: true });
    if (kind === "box") Object.assign(config, { column: x, group: group || undefined });
    await perform("Computing a chart-ready view of the current data…", async () => {
      setResult(await api.visualize(workspace.session_id, config));
    });
  }

  const definition = result ? chartDefinition(result) : null;
  return (
    <div className="view-stack">
      <section className="panel chart-builder">
        <header className="panel-heading">
          <div>
            <p className="eyebrow">EXPLORATORY ANALYSIS</p>
            <h2>Ask one clear visual question</h2>
            <p>Build bounded, reproducible figures from the working copy of your table.</p>
          </div>
          <HelpButton title="Choose a useful view">
            <p>Distributions expose skew and unusual values. Correlations summarize linear or rank relationships, but do not establish causality. Scatter and grouped box plots help inspect interactions.</p>
          </HelpButton>
        </header>
        <div className="chart-type-grid" role="radiogroup" aria-label="Chart type">
          {([
            ["distribution", "Distribution", "Shape, range, and density"],
            ["correlation", "Correlation", "Pairwise numeric relationships"],
            ["scatter", "Scatter", "Two features and a trend"],
            ["box", "Grouped box", "Spread across categories"],
            ["missingness", "Missingness", "Absence by column and row"],
          ] as const).map(([id, label, description]) => (
            <button type="button" role="radio" aria-checked={kind === id} className={kind === id ? "active" : ""} key={id} onClick={() => { setKind(id); setResult(null); }}>
              <BarChart3 size={18} /><strong>{label}</strong><small>{description}</small>
            </button>
          ))}
        </div>
        <div className="control-grid chart-controls">
          {kind === "distribution" ? <>
            <label>Column<select value={column} onChange={(event) => setColumn(event.target.value)}>{columns.map((item) => <option key={item.name}>{item.name}</option>)}</select></label>
            <label>Histogram bins<input type="number" min="5" max="100" value={bins} onChange={(event) => setBins(Number(event.target.value))} /></label>
          </> : null}
          {kind === "correlation" ? <label>Association method<select value={method} onChange={(event) => setMethod(event.target.value)}><option value="pearson">Pearson · linear</option><option value="spearman">Spearman · ranked</option><option value="kendall">Kendall · ranked pairs</option></select></label> : null}
          {kind === "scatter" || kind === "box" ? <>
            <label>{kind === "scatter" ? "X axis" : "Numeric column"}<select value={x} onChange={(event) => setX(event.target.value)}>{numeric.map((item) => <option key={item.name}>{item.name}</option>)}</select></label>
            {kind === "scatter" ? <label>Y axis<select value={y} onChange={(event) => setY(event.target.value)}>{numeric.map((item) => <option key={item.name}>{item.name}</option>)}</select></label> : null}
            <label>Group / color (optional)<select value={group} onChange={(event) => setGroup(event.target.value)}><option value="">None</option>{columns.map((item) => <option key={item.name}>{item.name}</option>)}</select></label>
          </> : null}
          {kind === "missingness" ? <div className="info-card"><strong>No setup needed</strong><p>Every column in the current table will be included.</p></div> : null}
        </div>
        <button type="button" className="button primary" onClick={() => void build()}><ScatterChart size={17} /> Generate figure</button>
      </section>
      {definition ? <ChartPanel title={definition.title} subtitle={definition.subtitle} traces={definition.traces} layout={definition.layout} settingsKey={`${workspace.session_id}:${result?.type}`} /> : (
        <section className="empty-state"><BarChart3 /><h2>Your figure will appear here</h2><p>Choose a chart, set its variables, then generate it. Every finished figure has its own export and configuration controls.</p></section>
      )}
    </div>
  );
}

function numbers(value: unknown): number[] { return Array.isArray(value) ? value.map(Number) : []; }
function strings(value: unknown): string[] { return Array.isArray(value) ? value.map(String) : []; }

function chartDefinition(result: ChartResult): ChartDefinition {
  const data = result.data;
  if (result.type === "correlation") {
    const labels = strings(data.labels);
    return { title: `${String(data.method)} correlation matrix`, subtitle: "Values range from −1 to +1. Hover for exact coefficients.", traces: [{ type: "heatmap", x: labels, y: labels, z: data.matrix as number[][], zmin: -1, zmax: 1, colorscale: [[0, "#b4473d"], [0.5, "#f7f4ec"], [1, "#0f766e"]], colorbar: { title: { text: "r" } }, hovertemplate: "%{x} × %{y}<br>r=%{z:.3f}<extra></extra>" }], layout: { xaxis: { side: "bottom", tickangle: -35 }, yaxis: { autorange: "reversed" } } };
  }
  if (result.type === "scatter") {
    const color = Array.isArray(data.color) ? strings(data.color) : null;
    const traces: Data[] = [{ type: "scatter", mode: "markers", name: "Observations", x: numbers(data.x), y: numbers(data.y), text: color ?? undefined, marker: { size: 7, opacity: 0.7 }, hovertemplate: color ? `%{x}, %{y}<br>${String(data.color_label)}: %{text}<extra></extra>` : "%{x}, %{y}<extra></extra>" }];
    const trend = data.trendline as Record<string, unknown> | null;
    if (trend) traces.push({ type: "scatter", mode: "lines", name: `Linear trend · R² ${Number(trend.r2).toFixed(3)}`, x: numbers(trend.x), y: numbers(trend.y), line: { width: 3, dash: "dash" } });
    return { title: `${String(data.y_label)} vs ${String(data.x_label)}`, subtitle: data.sampled ? `A reproducible sample of ${numbers(data.x).length} from ${String(data.total_points)} usable rows.` : `${String(data.total_points)} usable observations.`, traces, layout: { xaxis: { title: { text: String(data.x_label) } }, yaxis: { title: { text: String(data.y_label) } } } };
  }
  if (result.type === "box") {
    const groups = Array.isArray(data.groups) ? data.groups as Array<Record<string, unknown>> : [];
    return { title: `${String(data.column)} distribution${data.group ? ` by ${String(data.group)}` : ""}`, subtitle: data.truncated_groups ? "Showing the first 20 groups." : "Boxes show quartiles; points expose individual observations.", traces: groups.map((item) => ({ type: "box", name: String(item.label), y: numbers(item.values), boxpoints: "outliers", jitter: 0.25, pointpos: 0 })), layout: { yaxis: { title: { text: String(data.column) } } } };
  }
  if (result.type === "missingness") {
    return { title: "Missing values by column", subtitle: "Columns are ordered by missing-cell count.", traces: [{ type: "bar", orientation: "h", y: strings(data.columns).reverse(), x: numbers(data.counts).reverse(), customdata: numbers(data.ratios).reverse().map((item) => `${(item * 100).toFixed(1)}%`), hovertemplate: "%{y}<br>%{x} missing · %{customdata}<extra></extra>" }], layout: { xaxis: { title: { text: "Missing cells" } }, margin: { l: 120 } } };
  }
  if (data.kind === "categorical") {
    return { title: `${String(data.column)} frequency`, subtitle: data.truncated ? "Showing the 30 most frequent levels." : "Counts include an explicit missing-value level.", traces: [{ type: "bar", x: strings(data.labels), y: numbers(data.counts), hovertemplate: "%{x}<br>%{y} rows<extra></extra>" }], layout: { xaxis: { tickangle: -30 }, yaxis: { title: { text: "Rows" } } } };
  }
  const edges = numbers(data.edges); const counts = numbers(data.counts);
  const centers = counts.map((_, index) => (edges[index] + edges[index + 1]) / 2);
  const widths = counts.map((_, index) => edges[index + 1] - edges[index]);
  const traces: Data[] = [{ type: "bar", name: "Frequency", x: centers, y: counts, width: widths, opacity: 0.78, hovertemplate: "%{x:.3g}<br>%{y} rows<extra></extra>" }];
  const kde = data.kde as Record<string, unknown> | null;
  if (kde) { const scale = counts.reduce((sum, item) => sum + item, 0) * (widths[0] || 1); traces.push({ type: "scatter", mode: "lines", name: "Kernel density", x: numbers(kde.x), y: numbers(kde.y).map((item) => item * scale), line: { width: 3 } }); }
  return { title: `${String(data.column)} distribution`, subtitle: `${String(data.missing)} missing values. Histogram with optional density estimate.`, traces, layout: { xaxis: { title: { text: String(data.column) } }, yaxis: { title: { text: "Rows" } }, barmode: "overlay" } };
}
