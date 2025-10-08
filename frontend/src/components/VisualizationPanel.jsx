import React, { useEffect, useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import { getHistogram, getScatter } from '../api/client.js';
import ConfigurablePlotSection from './visualization/ConfigurablePlotSection.jsx';
import HistogramOptionsForm from './visualization/HistogramOptionsForm.jsx';
import ScatterOptionsForm from './visualization/ScatterOptionsForm.jsx';
import {
  applyHistogramOptions,
  applyScatterOptions,
  createDefaultHistogramOptions,
  createDefaultScatterOptions
} from '../utils/plotCustomization.js';

export default function VisualizationPanel({ datasetId, columns, onNotify, disabled }) {
  const [histColumn, setHistColumn] = useState('');
  const [scatterX, setScatterX] = useState('');
  const [scatterY, setScatterY] = useState('');
  const [scatterColor, setScatterColor] = useState('');
  const [loading, setLoading] = useState(false);
  const [histFigure, setHistFigure] = useState(null);
  const [scatterFigure, setScatterFigure] = useState(null);
  const [histBaseFigure, setHistBaseFigure] = useState(null);
  const [scatterBaseFigure, setScatterBaseFigure] = useState(null);
  const [histOptions, setHistOptions] = useState(() => createDefaultHistogramOptions());
  const [scatterOptions, setScatterOptions] = useState(() => createDefaultScatterOptions());
  const [histOptionsOpen, setHistOptionsOpen] = useState(false);
  const [scatterOptionsOpen, setScatterOptionsOpen] = useState(false);

  const columnSignature = useMemo(() => columns.join(','), [columns]);

  useEffect(() => {
    setHistColumn('');
    setScatterX('');
    setScatterY('');
    setScatterColor('');
    setLoading(false);
    setHistBaseFigure(null);
    setScatterBaseFigure(null);
    setHistFigure(null);
    setScatterFigure(null);
    setHistOptions(createDefaultHistogramOptions());
    setScatterOptions(createDefaultScatterOptions());
    setHistOptionsOpen(false);
    setScatterOptionsOpen(false);
  }, [datasetId, columnSignature]);

  useEffect(() => {
    if (!histBaseFigure) {
      setHistFigure(null);
      return;
    }
    setHistFigure(applyHistogramOptions(histBaseFigure, histOptions));
  }, [histBaseFigure, histOptions]);

  useEffect(() => {
    if (!scatterBaseFigure) {
      setScatterFigure(null);
      return;
    }
    setScatterFigure(applyScatterOptions(scatterBaseFigure, scatterOptions));
  }, [scatterBaseFigure, scatterOptions]);

  const updateHistogramOption = (key, value) => {
    setHistOptions((previous) => ({ ...previous, [key]: value }));
  };

  const updateScatterOption = (key, value) => {
    setScatterOptions((previous) => ({ ...previous, [key]: value }));
  };

  const handleHistogram = async () => {
    if (!datasetId || !histColumn) {
      onNotify('Select a dataset and column for the histogram.');
      return;
    }
    setLoading(true);
    try {
      const response = await getHistogram(datasetId, histColumn);
      setHistBaseFigure(response.figure);
    } catch (error) {
      onNotify(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleScatter = async () => {
    if (!datasetId || !scatterX || !scatterY) {
      onNotify('Select X and Y axes for the scatter plot.');
      return;
    }
    setLoading(true);
    try {
      const response = await getScatter(datasetId, {
        x: scatterX,
        y: scatterY,
        color: scatterColor || undefined
      });
      setScatterBaseFigure(response.figure);
    } catch (error) {
      onNotify(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>3. Explore visually</h2>
      </div>
      <div className="card-body viz-grid">
        <ConfigurablePlotSection
          title="Histogram"
          disabled={disabled}
          isConfigOpen={histOptionsOpen}
          onToggleConfig={() => setHistOptionsOpen((open) => !open)}
          configContent={
            <HistogramOptionsForm
              options={histOptions}
              onChange={updateHistogramOption}
              disabled={disabled}
            />
          }
        >
          <label htmlFor="histogram-column">Column</label>
          <select
            id="histogram-column"
            value={histColumn}
            onChange={(event) => setHistColumn(event.target.value)}
            disabled={disabled}
          >
            <option value="">-- choose column --</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
          <button type="button" onClick={handleHistogram} disabled={disabled || loading}>
            Generate histogram
          </button>
          {histFigure && (
            <Plot
              data={histFigure.data}
              layout={histFigure.layout}
              style={{ width: '100%', height: '100%' }}
              data-testid="histogram-plot"
            />
          )}
        </ConfigurablePlotSection>
        <ConfigurablePlotSection
          title="Scatter plot"
          disabled={disabled}
          isConfigOpen={scatterOptionsOpen}
          onToggleConfig={() => setScatterOptionsOpen((open) => !open)}
          configContent={
            <ScatterOptionsForm
              options={scatterOptions}
              onChange={updateScatterOption}
              disabled={disabled}
            />
          }
        >
          <label htmlFor="scatter-x">X axis</label>
          <select
            id="scatter-x"
            value={scatterX}
            onChange={(event) => setScatterX(event.target.value)}
            disabled={disabled}
          >
            <option value="">-- choose X --</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
          <label htmlFor="scatter-y">Y axis</label>
          <select
            id="scatter-y"
            value={scatterY}
            onChange={(event) => setScatterY(event.target.value)}
            disabled={disabled}
          >
            <option value="">-- choose Y --</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
          <label htmlFor="scatter-color">Color grouping (optional)</label>
          <select
            id="scatter-color"
            value={scatterColor}
            onChange={(event) => setScatterColor(event.target.value)}
            disabled={disabled}
          >
            <option value="">-- none --</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
          <button type="button" onClick={handleScatter} disabled={disabled || loading}>
            Generate scatter
          </button>
          {scatterFigure && (
            <Plot
              data={scatterFigure.data}
              layout={scatterFigure.layout}
              style={{ width: '100%', height: '100%' }}
              data-testid="scatter-plot"
            />
          )}
        </ConfigurablePlotSection>
      </div>
    </div>
  );
}
