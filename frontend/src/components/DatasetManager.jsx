import React, { useState } from 'react';

function PreviewTable({ data }) {
  if (!data || data.length === 0) {
    return <p className="muted">No preview available.</p>;
  }
  const columns = Object.keys(data[0]);
  return (
    <div className="table-wrapper" data-testid="dataset-preview">
      <table>
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr key={idx}>
              {columns.map((col) => (
                <td key={col}>{row[col] !== null && row[col] !== undefined ? row[col].toString() : ''}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SummaryTable({ summary }) {
  const columns = Object.keys(summary || {});
  if (columns.length === 0) {
    return <p className="muted">Run summary to see statistics.</p>;
  }
  const stats = Object.keys(summary[columns[0]] || {});
  return (
    <div className="table-wrapper" data-testid="summary-table">
      <table>
        <thead>
          <tr>
            <th>Column</th>
            {stats.map((stat) => (
              <th key={stat}>{stat}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {columns.map((col) => (
            <tr key={col}>
              <td>{col}</td>
              {stats.map((stat) => (
                <td key={stat}>{summary[col][stat] !== null ? summary[col][stat] : ''}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function DatasetManager({
  datasets,
  currentDatasetId,
  onDatasetSelect,
  onUpload,
  sampleDatasets,
  onSampleLoad,
  preview,
  summary,
  columnsInfo,
  loading,
  onRefresh,
  allowUploads
}) {
  const [file, setFile] = useState(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  const handleUpload = async (event) => {
    event.preventDefault();
    if (!allowUploads || !file) return;
    await onUpload(file, name, description);
    setFile(null);
    setName('');
    setDescription('');
    event.target.reset();
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>1. Dataset</h2>
        <button type="button" onClick={onRefresh} data-testid="refresh-datasets" className="secondary" disabled={loading}>
          Refresh list
        </button>
      </div>
      <div className="card-body">
        <div className="dataset-selection">
          <label htmlFor="dataset-select">Select dataset</label>
          <select
            id="dataset-select" data-testid="dataset-selector"
            value={currentDatasetId || ''}
            onChange={(event) => onDatasetSelect(event.target.value)}
          >
            <option value="">-- Choose a dataset --</option>
            {datasets.map((dataset) => (
              <option key={dataset.dataset_id} value={dataset.dataset_id}>
                {dataset.name} ({dataset.num_rows} rows)
              </option>
            ))}
          </select>
        </div>
        <div className="dataset-actions">
          <form className="upload-form" onSubmit={handleUpload}>
            <h3>Upload CSV/XLSX</h3>
            {!allowUploads && (
              <p className="muted">Uploads are disabled in this deployment.</p>
            )}
            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={(event) => setFile(event.target.files[0])}
              disabled={!allowUploads || loading}
            />
            <input
              type="text"
              placeholder="Display name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              disabled={!allowUploads || loading}
            />
            <textarea
              placeholder="Description"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              disabled={!allowUploads || loading}
            />
            <button type="submit" disabled={!allowUploads || !file || loading}>
              Upload dataset
            </button>
          </form>
          <div className="sample-datasets">
            <h3>Sample datasets</h3>
            <p className="muted">Instantly load curated offline datasets.</p>
            <div className="sample-list">
              {sampleDatasets.map((sample) => (
                <button
                  key={sample.key}
                  type="button"
                  onClick={() => onSampleLoad(sample.key)}
                  data-testid={`load-sample-${sample.key}`}
                  disabled={loading}
                >
                  {sample.name} ({sample.task})
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="dataset-details">
          <div>
            <h3>Column summary</h3>
            <SummaryTable summary={summary} />
          </div>
          <div>
            <h3>Preview</h3>
            <PreviewTable data={preview.slice(0, 15)} />
          </div>
          <div>
            <h3>Columns ({columnsInfo.columns.length})</h3>
            <ul className="column-list" data-testid="column-list">
              {columnsInfo.columns.map((col) => (
                <li key={col}>
                  <span>{col}</span>
                  <span className="muted">{columnsInfo.dtypes[col]}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
