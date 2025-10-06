import React, { useEffect, useMemo, useRef, useState } from 'react';
import Plot from 'react-plotly.js';

import { openTrainingStream, trainModel } from '../api/client.js';
import { buildTrainingHistoryFigure, latestHistoryLabel } from '../utils/trainingHistory.js';

export default function ModelTrainer({
  datasetId,
  splitId,
  columns,
  algorithms,
  onTrainingComplete,
  onNotify,
  onProgress,
  disabled
}) {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('logistic_regression');
  const [targetColumn, setTargetColumn] = useState('');
  const [taskType, setTaskType] = useState('classification');
  const [hyperparams, setHyperparams] = useState('{"random_state": 42}');
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Idle');
  const [history, setHistory] = useState([]);
  const eventSourceRef = useRef(null);

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

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    setHistory([]);
    setStatusMessage('Idle');
    onProgress?.([]);
  }, [datasetId, onProgress]);

  const historyFigure = useMemo(() => buildTrainingHistoryFigure(history), [history]);
  const latestLabel = useMemo(() => latestHistoryLabel(history), [history]);

  const closeStream = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  };

  const startStreamingTraining = (payload) =>
    new Promise((resolve, reject) => {
      let settled = false;
      const stream = openTrainingStream(payload);
      eventSourceRef.current = stream;

      const cleanup = () => {
        if (stream) {
          stream.removeEventListener('history', handleHistory);
          stream.removeEventListener('status', handleStatus);
          stream.removeEventListener('completed', handleCompleted);
          stream.removeEventListener('result', handleResult);
          stream.removeEventListener('error', handleError);
          stream.onerror = null;
          stream.close();
        }
        if (eventSourceRef.current === stream) {
          eventSourceRef.current = null;
        }
      };

      const safeResolve = (value) => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve(value);
      };

      const safeReject = (error) => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(error);
      };

      const parseEvent = (event) => {
        if (!event || !event.data) {
          return {};
        }
        try {
          return JSON.parse(event.data);
        } catch (error) {
          return {};
        }
      };

      const handleHistory = (event) => {
        const payload = parseEvent(event);
        if (!payload.entry) {
          return;
        }
        setHistory((prev) => {
          const next = [...prev, payload.entry];
          onProgress?.(next);
          return next;
        });
      };

      const handleStatus = (event) => {
        const payload = parseEvent(event);
        if (payload.stage === 'preprocessing') {
          setStatusMessage('Preprocessing features…');
        } else if (payload.stage === 'starting') {
          setStatusMessage('Starting training…');
        } else if (payload.stage === 'finished') {
          setStatusMessage('Wrapping up…');
        } else if (payload.stage) {
          setStatusMessage(payload.stage);
        }
      };

      const handleCompleted = (event) => {
        const payload = parseEvent(event);
        if (payload.model_id) {
          setStatusMessage(`Completed training for ${payload.model_id}`);
        }
      };

      const handleResult = (event) => {
        const payload = parseEvent(event);
        if (payload.payload) {
          const result = payload.payload;
          if (Array.isArray(result.history)) {
            setHistory(result.history);
            onProgress?.(result.history);
          }
          setStatusMessage('Training complete');
          safeResolve(result);
        }
      };

      const handleError = (event) => {
        const payload = parseEvent(event);
        const message = payload.message || 'Live updates failed';
        setStatusMessage(message);
        safeReject(new Error(message));
      };

      stream.addEventListener('history', handleHistory);
      stream.addEventListener('status', handleStatus);
      stream.addEventListener('completed', handleCompleted);
      stream.addEventListener('result', handleResult);
      stream.addEventListener('error', handleError);
      stream.onerror = handleError;
    });

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
    setHistory([]);
    onProgress?.([]);
    setStatusMessage('Connecting to trainer…');
    closeStream();
    try {
      const payload = {
        dataset_id: datasetId,
        split_id: splitId || undefined,
        target_column: targetColumn,
        task_type: taskType,
        algorithm: selectedAlgorithm,
        hyperparameters: parsedHyperparams
      };

      try {
        const streamed = await startStreamingTraining(payload);
        onTrainingComplete(streamed);
        onNotify(`Model trained successfully (ID: ${streamed.model_id}).`);
      } catch (streamError) {
        setStatusMessage('Falling back to synchronous training…');
        onNotify('Live updates unavailable, running synchronous training.');
        const response = await trainModel(payload);
        setHistory(response.history || []);
        onProgress?.(response.history || []);
        setStatusMessage('Training complete');
        onTrainingComplete(response);
        onNotify(`Model trained successfully (ID: ${response.model_id}).`);
      }
    } catch (error) {
      onNotify(error.message);
      setStatusMessage('Training failed');
    } finally {
      closeStream();
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
        <div>
          <h3>Live progress</h3>
          {historyFigure ? (
            <div data-testid="training-history-plot" className="training-progress-plot">
              <Plot
                data={historyFigure.data}
                layout={{ ...historyFigure.layout, autosize: true }}
                style={{ width: '100%', height: '100%' }}
              />
            </div>
          ) : (
            <p className="muted">Start training to view live metrics.</p>
          )}
          <p className="muted" data-testid="training-status" aria-live="polite">
            {statusMessage}
          </p>
          {latestLabel && (
            <p className="muted" data-testid="latest-epoch">
              Latest update: {latestLabel}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
