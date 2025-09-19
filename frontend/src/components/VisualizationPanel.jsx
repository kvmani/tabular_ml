import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { getHistogram, getScatter } from '../api/client.js';

export default function VisualizationPanel({ datasetId, columns, onNotify, disabled }) {
  const [histColumn, setHistColumn] = useState('');
  const [histFigure, setHistFigure] = useState(null);
  const [scatterX, setScatterX] = useState('');
  const [scatterY, setScatterY] = useState('');
  const [scatterColor, setScatterColor] = useState('');
  const [scatterFigure, setScatterFigure] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setHistColumn('');
    setHistFigure(null);
    setScatterX('');
    setScatterY('');
    setScatterColor('');
    setScatterFigure(null);
  }, [datasetId, columns.join(',')]);

  const handleHistogram = async () => {
    if (!datasetId || !histColumn) {
      onNotify('Select a dataset and column for the histogram.');
      return;
    }
    setLoading(true);
    try {
      const response = await getHistogram(datasetId, histColumn);
      setHistFigure(response.figure);
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
      setScatterFigure(response.figure);
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
        <div>
          <h3>Histogram</h3>
          <label>Column</label>
          <select value={histColumn} onChange={(event) => setHistColumn(event.target.value)} disabled={disabled}>
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
          {histFigure && <Plot data={histFigure.data} layout={histFigure.layout} style={{ width: '100%', height: '100%' }} />}
        </div>
        <div>
          <h3>Scatter plot</h3>
          <label>X axis</label>
          <select value={scatterX} onChange={(event) => setScatterX(event.target.value)} disabled={disabled}>
            <option value="">-- choose X --</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
          <label>Y axis</label>
          <select value={scatterY} onChange={(event) => setScatterY(event.target.value)} disabled={disabled}>
            <option value="">-- choose Y --</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
          <label>Color grouping (optional)</label>
          <select value={scatterColor} onChange={(event) => setScatterColor(event.target.value)} disabled={disabled}>
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
          {scatterFigure && <Plot data={scatterFigure.data} layout={scatterFigure.layout} style={{ width: '100%', height: '100%' }} />}
        </div>
      </div>
    </div>
  );
}
