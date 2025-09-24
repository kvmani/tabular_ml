import React, { useState, useEffect } from 'react';
import { trainModel } from '../api/client.js';

export default function ModelTrainer({
  datasetId,
  splitId,
  columns,
  algorithms,
  onTrainingComplete,
  onNotify,
  disabled
}) {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('logistic_regression');
  const [targetColumn, setTargetColumn] = useState('');
  const [taskType, setTaskType] = useState('classification');
  const [hyperparams, setHyperparams] = useState('{"random_state": 42}');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (columns.length > 0 && !columns.includes(targetColumn)) {
      setTargetColumn(columns[columns.length - 1]);
    }
  }, [columns, targetColumn]);

  useEffect(() => {
    if (algorithms.length > 0) {
      setSelectedAlgorithm(algorithms[0].key);
    }
  }, [algorithms]);

  const handleTrain = async () => {
    if (!datasetId) {
      onNotify('Select a dataset before training.');
      return;
    }
    if (!targetColumn) {
      onNotify('Choose a target column for training.');
      return;
    }
    let parsedHyperparams = {};
    if (hyperparams.trim()) {
      try {
        parsedHyperparams = JSON.parse(hyperparams);
      } catch (error) {
        onNotify('Hyperparameters must be valid JSON.');
        return;
      }
    }
    setLoading(true);
    try {
      const response = await trainModel({
        dataset_id: datasetId,
        split_id: splitId || undefined,
        target_column: targetColumn,
        task_type: taskType,
        algorithm: selectedAlgorithm,
        hyperparameters: parsedHyperparams
      });
      onTrainingComplete(response);
      onNotify(`Model trained successfully (ID: ${response.model_id}).`);
    } catch (error) {
      onNotify(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>4. Train models</h2>
      </div>
      <div className="card-body model-grid">
        <div>
          <h3>Configuration</h3>
          <label htmlFor="algorithm-select">Algorithm</label>
          <select
            id="algorithm-select"
            value={selectedAlgorithm}
            onChange={(event) => setSelectedAlgorithm(event.target.value)}
            disabled={disabled}
          >
            {algorithms.map((algo) => (
              <option key={algo.key} value={algo.key}>
                {algo.label}
              </option>
            ))}
          </select>
          <label htmlFor="task-type-select">Task type</label>
          <select
            id="task-type-select"
            value={taskType}
            onChange={(event) => setTaskType(event.target.value)}
            disabled={disabled}
          >
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
          </select>
          <label htmlFor="target-column-select">Target column</label>
          <select
            id="target-column-select"
            value={targetColumn}
            data-testid="train-target"
            onChange={(event) => setTargetColumn(event.target.value)}
            disabled={disabled}
          >
            <option value="">-- select target --</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
          <label>Hyperparameters (JSON)</label>
          <textarea
            rows={6}
            value={hyperparams}
            onChange={(event) => setHyperparams(event.target.value)}
            disabled={disabled}
          />
          <button type="button" onClick={handleTrain} data-testid="train-button" disabled={disabled || loading}>
            Start training
          </button>
          {splitId && <p className="muted">Using split: {splitId}</p>}
        </div>
        <div>
          <h3>Guidance</h3>
          <ul className="muted">
            <li>Random Forest and XGBoost work well on most datasets.</li>
            <li>For regression tasks choose a numeric target column.</li>
            <li>
              Provide hyperparameters as JSON (e.g. <code>{'{'}"n_estimators": 300{'}'}</code>).
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
