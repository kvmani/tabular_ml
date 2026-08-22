import { Download, Settings2, X } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import type { Config, Data, Layout } from "plotly.js";
import type { ChartSettings } from "../types";

const PALETTES: Record<string, string[]> = {
  Verdigris: ["#0f766e", "#d97706", "#2563eb", "#be123c", "#7c3aed"],
  Scientific: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
  Monochrome: ["#102a2e", "#476568", "#819598", "#b9c4c5", "#dfe5e5"],
  Colorblind: ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#d55e00"],
};

const DEFAULTS: ChartSettings = {
  title: "",
  palette: "Verdigris",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 13,
  background: "#ffffff",
  showGrid: true,
  showLegend: true,
  showAnnotations: true,
  exportScale: 2,
};

export function chartLayout(
  title: string,
  layout: Partial<Layout>,
  settings: ChartSettings,
): Partial<Layout> {
  return {
    autosize: true,
    margin: { l: 64, r: 28, t: 58, b: 58 },
    paper_bgcolor: settings.background,
    plot_bgcolor: settings.background,
    font: { family: settings.fontFamily, size: settings.fontSize, color: "#163033" },
    title: {
      text: settings.title || title,
      x: 0.02,
      xanchor: "left",
      font: { size: settings.fontSize + 4 },
    },
    showlegend: settings.showLegend,
    legend: { orientation: "h", y: -0.2 },
    ...layout,
    xaxis: {
      gridcolor: settings.showGrid ? "#dfe7e2" : "rgba(0,0,0,0)",
      zerolinecolor: "#b8c9c2",
      ...(layout.xaxis ?? {}),
    },
    yaxis: {
      gridcolor: settings.showGrid ? "#dfe7e2" : "rgba(0,0,0,0)",
      zerolinecolor: "#b8c9c2",
      ...(layout.yaxis ?? {}),
    },
  };
}

interface ChartPanelProps {
  title: string;
  subtitle?: string;
  traces: Data[];
  layout?: Partial<Layout>;
  settingsKey: string;
  emptyMessage?: string;
}

export function ChartPanel({
  title,
  subtitle,
  traces,
  layout = {},
  settingsKey,
  emptyMessage = "No chart data is available for this selection.",
}: ChartPanelProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const plotlyRef = useRef<typeof import("plotly.js") | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<ChartSettings>(() => {
    const stored = sessionStorage.getItem(`chart:${settingsKey}`);
    return stored ? { ...DEFAULTS, ...JSON.parse(stored) } : DEFAULTS;
  });
  const dialogId = useId();

  useEffect(() => {
    let active = true;
    const node = chartRef.current;
    if (!node || traces.length === 0) return;
    void import("plotly.js-dist-min").then((module) => {
      if (!active || !chartRef.current) return;
      const Plotly = module.default;
      plotlyRef.current = Plotly;
      const colors = PALETTES[settings.palette] ?? PALETTES.Verdigris;
      const colored = traces.map((trace, index) => {
        const styled = trace as Data & {
          marker?: Record<string, unknown>;
          line?: Record<string, unknown>;
        };
        return ({
        ...styled,
        marker: {
          ...(typeof styled.marker === "object" ? styled.marker : {}),
          color:
            typeof styled.marker === "object" && "color" in styled.marker
              ? styled.marker.color
              : colors[index % colors.length],
        },
        line: {
          ...(typeof styled.line === "object" ? styled.line : {}),
          color:
            typeof styled.line === "object" && "color" in styled.line
              ? styled.line.color
              : colors[index % colors.length],
        },
      });}) as Data[];
      const config: Partial<Config> = {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
        toImageButtonOptions: {
          format: "svg",
          filename: settings.title || title.toLowerCase().replaceAll(" ", "-"),
          scale: settings.exportScale,
        },
      };
      void Plotly.react(
        chartRef.current,
        colored,
        chartLayout(title, layout, settings),
        config,
      );
    });
    return () => {
      active = false;
      if (node && plotlyRef.current) plotlyRef.current.purge(node);
    };
  }, [layout, settings, title, traces]);

  function update<K extends keyof ChartSettings>(key: K, value: ChartSettings[K]) {
    const next = { ...settings, [key]: value };
    setSettings(next);
    sessionStorage.setItem(`chart:${settingsKey}`, JSON.stringify(next));
  }

  function download() {
    if (!chartRef.current || !plotlyRef.current) return;
    const options = {
      format: "svg",
      filename: settings.title || title.toLowerCase().replaceAll(" ", "-"),
      scale: settings.exportScale,
    } as unknown as Parameters<typeof plotlyRef.current.downloadImage>[1];
    void plotlyRef.current.downloadImage(chartRef.current, options);
  }

  return (
    <section className="chart-panel">
      <header className="panel-heading">
        <div>
          <h3>{title}</h3>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        <div className="panel-actions">
          <button
            type="button"
            className="icon-button"
            aria-label={`Export ${title} as SVG`}
            onClick={download}
          >
            <Download size={18} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="icon-button gear-button"
            aria-label={`Configure ${title}`}
            aria-haspopup="dialog"
            onClick={() => setSettingsOpen(true)}
          >
            <Settings2 size={18} aria-hidden="true" />
          </button>
        </div>
      </header>
      {traces.length ? (
        <div
          ref={chartRef}
          className="plot"
          role="img"
          aria-label={`${title}. ${subtitle ?? ""}`}
        />
      ) : (
        <div className="empty-chart">{emptyMessage}</div>
      )}
      {settingsOpen ? (
        <div className="modal-backdrop" role="presentation">
          <section
            id={dialogId}
            className="modal settings-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby={`${dialogId}-title`}
          >
            <header className="modal-header">
              <div>
                <p className="eyebrow">FIGURE SETTINGS</p>
                <h2 id={`${dialogId}-title`}>{title}</h2>
              </div>
              <button
                type="button"
                className="icon-button"
                aria-label="Close figure settings"
                onClick={() => setSettingsOpen(false)}
              >
                <X aria-hidden="true" />
              </button>
            </header>
            <div className="settings-grid">
              <label>
                Figure title
                <input
                  value={settings.title}
                  placeholder={title}
                  onChange={(event) => update("title", event.target.value)}
                />
              </label>
              <label>
                Color palette
                <select
                  value={settings.palette}
                  onChange={(event) => update("palette", event.target.value)}
                >
                  {Object.keys(PALETTES).map((name) => (
                    <option key={name}>{name}</option>
                  ))}
                </select>
              </label>
              <label>
                Font family
                <select
                  value={settings.fontFamily}
                  onChange={(event) => update("fontFamily", event.target.value)}
                >
                  <option value="Inter, system-ui, sans-serif">System sans</option>
                  <option value="Arial, sans-serif">Arial</option>
                  <option value="Georgia, serif">Georgia</option>
                </select>
              </label>
              <label>
                Font size
                <input
                  type="range"
                  min="10"
                  max="22"
                  value={settings.fontSize}
                  onChange={(event) => update("fontSize", Number(event.target.value))}
                />
                <span className="range-value">{settings.fontSize} px</span>
              </label>
              <label>
                Background
                <input
                  type="color"
                  value={settings.background}
                  onChange={(event) => update("background", event.target.value)}
                />
              </label>
              <label>
                Export scale
                <select
                  value={settings.exportScale}
                  onChange={(event) => update("exportScale", Number(event.target.value))}
                >
                  <option value="1">1×</option>
                  <option value="2">2×</option>
                  <option value="3">3×</option>
                  <option value="4">4×</option>
                </select>
              </label>
              <label className="switch-row">
                <input
                  type="checkbox"
                  checked={settings.showGrid}
                  onChange={(event) => update("showGrid", event.target.checked)}
                />
                Show grid lines
              </label>
              <label className="switch-row">
                <input
                  type="checkbox"
                  checked={settings.showLegend}
                  onChange={(event) => update("showLegend", event.target.checked)}
                />
                Show legend
              </label>
            </div>
            <footer className="modal-actions">
              <button
                type="button"
                className="button secondary"
                onClick={() => setSettings(DEFAULTS)}
              >
                Reset
              </button>
              <button
                type="button"
                className="button primary"
                onClick={() => setSettingsOpen(false)}
              >
                Apply
              </button>
            </footer>
          </section>
        </div>
      ) : null}
    </section>
  );
}
