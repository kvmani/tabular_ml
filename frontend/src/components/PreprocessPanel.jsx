import React, { useState } from 'react';
import {
  detectOutliers,
  removeOutliers,
  imputeDataset,
  filterDataset,
  splitDataset
} from '../api/client.js';

function parseColumns(value) {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
}

export default function PreprocessPanel({
  datasetId,
  columns,
  onDatasetCreated,
  onSplitCreated,
  onNotify,
  disabled
}) {
  const [outlierColumns, setOutlierColumns] = useState('');
  const [zThreshold, setZThreshold] = useState(3);
  const [outlierInfo, setOutlierInfo] = useState(null);
  const [imputeStrategy, setImputeStrategy] = useState('mean');
  const [imputeColumns, setImputeColumns] = useState('');
  const [imputeValue, setImputeValue] = useState('');
  const [filterColumn, setFilterColumn] = useState('');
  const [filterOperator, setFilterOperator] = useState('eq');
  const [filterValue, setFilterValue] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [taskType, setTaskType] = useState('classification');
  const [testSize, setTestSize] = useState(0.2);
  const [valSize, setValSize] = useState(0.2);
  const [stratify, setStratify] = useState(true);
  const [loading, setLoading] = useState(false);

  const ensureDataset = () => {
    if (!datasetId) {
      onNotify('Select or upload a dataset first.');
      return false;
    }
    return true;
  };

  const handleDetect = async () => {
    if (!ensureDataset()) return;
    setLoading(true);
    try {
      const response = await detectOutliers(datasetId, {
        columns: outlierColumns ? parseColumns(outlierColumns) : undefined,
        z_threshold: Number(zThreshold)
      });
      setOutlierInfo(response);
      onNotify(`Detected ${response.total_outliers} potential outliers.`);
    } catch (error) {
      onNotify(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRemove = async () => {
    if (!ensureDataset()) return;
    setLoading(true);
    try {
      const response = await removeOutliers(datasetId, {
        columns: outlierColumns ? parseColumns(outlierColumns) : undefined,
        z_threshold: Number(zThreshold)
      });
      setOutlierInfo(response.report);
      onDatasetCreated(response.dataset);
      onNotify('Created dataset with outliers removed.');
    } catch (error) {
      onNotify(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleImpute = async () => {
    if (!ensureDataset()) return;
    setLoading(true);
    try {
      const payload = {
        strategy: imputeStrategy,
        columns: imputeColumns ? parseColumns(imputeColumns) : undefined,
        fill_value: imputeValue ? Number(imputeValue) : undefined
      };
      const response = await imputeDataset(datasetId, payload);
      onDatasetCreated(response.dataset);
      onNotify('Created dataset with imputed values.');
    } catch (error) {
      onNotify(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFilter = async () => {
    if (!ensureDataset()) return;
    if (!filterColumn) {
      onNotify('Choose a column to filter.');
      return;
    }
    setLoading(true);
    try {
      const response = await filterDataset(datasetId, {
        rules: [
          {
            column: filterColumn,
            operator: filterOperator,
            value: filterValue
          }
        ]
      });
      onDatasetCreated(response.dataset);
      onNotify('Created filtered dataset.');
    } catch (error) {
      onNotify(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSplit = async () => {
    if (!ensureDataset()) return;
    if (!targetColumn) {
      onNotify('Select a target column for training.');
      return;
    }
    setLoading(true);
    try {
      const response = await splitDataset(datasetId, {
        target_column: targetColumn,
        task_type: taskType,
        test_size: Number(testSize),
        val_size: Number(valSize),
        stratify
      });
      onSplitCreated(response);
      onNotify(`Created split (${response.split_id}).`);
    } catch (error) {
      onNotify(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>2. Preprocess</h2>
      </div>
      <div className="card-body preprocess-grid">
        <div>
          <h3>Outliers</h3>
          <label>Columns (comma-separated)</label>
          <input
            type="text"
            placeholder="Leave empty for numeric columns"
            value={outlierColumns}
            onChange={(event) => setOutlierColumns(event.target.value)}
            disabled={disabled}
          />
          <label>Z-score threshold</label>
          <input
            type="number"
            step="0.1"
            min="0"
            value={zThreshold}
            onChange={(event) => setZThreshold(event.target.value)}
            disabled={disabled}
          />
          <div className="button-row">
            <button type="button" onClick={handleDetect} disabled={disabled || loading}>
              Detect outliers
            </button>
            <button type="button" onClick={handleRemove} disabled={disabled || loading}>
              Remove outliers
            </button>
          </div>
          {outlierInfo && (
            <p className="muted">
              {outlierInfo.total_outliers} rows flagged across {outlierInfo.inspected_columns.length} columns.
            </p>
          )}
        </div>
        <div>
          <h3>Impute</h3>
          <label>Strategy</label>
          <select value={imputeStrategy} onChange={(event) => setImputeStrategy(event.target.value)} disabled={disabled}>
            <option value="mean">Mean</option>
            <option value="median">Median</option>
            <option value="most_frequent">Most frequent</option>
            <option value="constant">Constant</option>
          </select>
          <label>Columns (comma-separated)</label>
          <input
            type="text"
            placeholder="All columns"
            value={imputeColumns}
            onChange={(event) => setImputeColumns(event.target.value)}
            disabled={disabled}
          />
          <label>Fill value (for constant)</label>
          <input
            type="number"
            value={imputeValue}
            onChange={(event) => setImputeValue(event.target.value)}
            disabled={disabled || imputeStrategy !== 'constant'}
          />
          <button type="button" onClick={handleImpute} disabled={disabled || loading}>
            Impute missing values
          </button>
        </div>
        <div>
          <h3>Filter</h3>
          <label>Column</label>
          <select value={filterColumn} onChange={(event) => setFilterColumn(event.target.value)} disabled={disabled}>
            <option value="">-- select column --</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
          <label>Operator</label>
          <select value={filterOperator} onChange={(event) => setFilterOperator(event.target.value)} disabled={disabled}>
            <option value="eq">equals</option>
            <option value="ne">not equals</option>
            <option value="gt">greater than</option>
            <option value="gte">greater or equal</option>
            <option value="lt">less than</option>
            <option value="lte">less or equal</option>
            <option value="contains">contains</option>
          </select>
          <label>Value</label>
          <input type="text" value={filterValue} onChange={(event) => setFilterValue(event.target.value)} disabled={disabled} />
          <button type="button" onClick={handleFilter} disabled={disabled || loading}>
            Apply filter
          </button>
        </div>
        <div>
          <h3>Split data</h3>
          <label>Target column</label>
          <select value={targetColumn} onChange={(event) => setTargetColumn(event.target.value)} disabled={disabled}>
            <option value="">-- select target --</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
          <label>Task type</label>
          <select value={taskType} onChange={(event) => setTaskType(event.target.value)} disabled={disabled}>
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
          </select>
          <label>Test size</label>
          <input
            type="number"
            step="0.05"
            min="0.05"
            max="0.8"
            value={testSize}
            onChange={(event) => setTestSize(event.target.value)}
            disabled={disabled}
          />
          <label>Validation size</label>
          <input
            type="number"
            step="0.05"
            min="0.05"
            max="0.8"
            value={valSize}
            onChange={(event) => setValSize(event.target.value)}
            disabled={disabled}
          />
          <label className="checkbox">
            <input
              type="checkbox"
              checked={stratify}
              onChange={(event) => setStratify(event.target.checked)}
              disabled={disabled}
            />
            Stratify splits (classification)
          </label>
          <button type="button" onClick={handleSplit} disabled={disabled || loading}>
            Create train/validation/test split
          </button>
        </div>
      </div>
    </div>
  );
}
