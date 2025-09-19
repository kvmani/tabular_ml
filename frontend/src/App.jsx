import React, { useEffect, useState } from 'react';
import DatasetManager from './components/DatasetManager.jsx';
import PreprocessPanel from './components/PreprocessPanel.jsx';
import VisualizationPanel from './components/VisualizationPanel.jsx';
import ModelTrainer from './components/ModelTrainer.jsx';
import EvaluationPanel from './components/EvaluationPanel.jsx';
import {
  listDatasets,
  listSampleDatasets,
  loadSampleDataset,
  uploadDataset,
  getDatasetPreview,
  getDatasetSummary,
  getDatasetColumns,
  listAlgorithms,
  evaluateModel
} from './api/client.js';

function useNotification() {
  const [message, setMessage] = useState(null);
  const [variant, setVariant] = useState('info');

  const notify = (text, type = 'info') => {
    setMessage(text);
    setVariant(type);
    if (text) {
      setTimeout(() => setMessage(null), 4000);
    }
  };

  return { message, variant, notify };
}

export default function App() {
  const [datasets, setDatasets] = useState([]);
  const [sampleDatasets, setSampleDatasets] = useState([]);
  const [algorithms, setAlgorithms] = useState([]);
  const [currentDatasetId, setCurrentDatasetId] = useState('');
  const [columnsInfo, setColumnsInfo] = useState({ columns: [], dtypes: {} });
  const [preview, setPreview] = useState([]);
  const [summary, setSummary] = useState({});
  const [splitId, setSplitId] = useState('');
  const [modelId, setModelId] = useState('');
  const [metrics, setMetrics] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [loading, setLoading] = useState(false);

  const { message, variant, notify } = useNotification();

  const refreshDatasets = async (selectId) => {
    const response = await listDatasets();
    setDatasets(response.datasets || []);
    if (selectId) {
      setCurrentDatasetId(selectId);
    }
  };

  const refreshDatasetDetails = async (datasetId) => {
    if (!datasetId) {
      setPreview([]);
      setSummary({});
      setColumnsInfo({ columns: [], dtypes: {} });
      return;
    }
    setLoading(true);
    try {
      const [previewResponse, summaryResponse, columnsResponse] = await Promise.all([
        getDatasetPreview(datasetId),
        getDatasetSummary(datasetId),
        getDatasetColumns(datasetId)
      ]);
      setPreview(previewResponse.data || []);
      setSummary(summaryResponse.summary || {});
      setColumnsInfo(columnsResponse || { columns: [], dtypes: {} });
    } catch (error) {
      notify(error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    (async () => {
      try {
        const [datasetsResponse, samplesResponse, algorithmsResponse] = await Promise.all([
          listDatasets(),
          listSampleDatasets(),
          listAlgorithms()
        ]);
        setDatasets(datasetsResponse.datasets || []);
        setSampleDatasets(samplesResponse.samples || []);
        setAlgorithms(algorithmsResponse.algorithms || []);
      } catch (error) {
        notify(error.message, 'error');
      }
    })();
  }, []);

  useEffect(() => {
    refreshDatasetDetails(currentDatasetId);
    setSplitId('');
    setModelId('');
    setMetrics(null);
    setEvaluation(null);
  }, [currentDatasetId]);

  const handleDatasetSelect = async (datasetId) => {
    setCurrentDatasetId(datasetId);
  };

  const handleUpload = async (file, name, description) => {
    setLoading(true);
    try {
      const response = await uploadDataset(file, name, description);
      await refreshDatasets(response.dataset.dataset_id);
      notify(`Uploaded ${response.dataset.name}.`);
    } catch (error) {
      notify(error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleSampleLoad = async (key) => {
    setLoading(true);
    try {
      const response = await loadSampleDataset(key);
      await refreshDatasets(response.dataset.dataset_id);
      notify(`Loaded sample dataset: ${response.dataset.name}.`);
    } catch (error) {
      notify(error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleDatasetCreated = async (dataset) => {
    await refreshDatasets(dataset.dataset_id);
    notify(`Dataset created: ${dataset.name}.`);
  };

  const handleSplitCreated = (split) => {
    setSplitId(split.split_id);
  };

  const handleTrainingComplete = (result) => {
    setModelId(result.model_id);
    setSplitId(result.split_id);
    setMetrics(result.metrics);
    setEvaluation(null);
  };

  const handleEvaluate = async () => {
    if (!modelId) {
      notify('Train a model before evaluation.', 'warning');
      return;
    }
    setLoading(true);
    try {
      const response = await evaluateModel(modelId);
      setEvaluation(response);
      setMetrics(response.metrics);
      notify('Evaluation completed.');
    } catch (error) {
      notify(error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header>
        <h1>Intranet Tabular ML Studio</h1>
        <p className="muted">Offline-first machine learning workflow for tabular data.</p>
      </header>
      {message && <div className={`notification ${variant}`}>{message}</div>}
      <DatasetManager
        datasets={datasets}
        currentDatasetId={currentDatasetId}
        onDatasetSelect={handleDatasetSelect}
        onUpload={handleUpload}
        sampleDatasets={sampleDatasets}
        onSampleLoad={handleSampleLoad}
        preview={preview}
        summary={summary}
        columnsInfo={columnsInfo}
        loading={loading}
        onRefresh={() => refreshDatasets(currentDatasetId)}
      />
      <PreprocessPanel
        datasetId={currentDatasetId}
        columns={columnsInfo.columns}
        onDatasetCreated={handleDatasetCreated}
        onSplitCreated={handleSplitCreated}
        onNotify={notify}
        disabled={!currentDatasetId || loading}
      />
      <VisualizationPanel
        datasetId={currentDatasetId}
        columns={columnsInfo.columns}
        onNotify={notify}
        disabled={!currentDatasetId || loading}
      />
      <ModelTrainer
        datasetId={currentDatasetId}
        splitId={splitId}
        columns={columnsInfo.columns}
        algorithms={algorithms}
        onTrainingComplete={handleTrainingComplete}
        onNotify={notify}
        disabled={!currentDatasetId || loading}
      />
      <EvaluationPanel
        modelId={modelId}
        metrics={metrics}
        evaluation={evaluation}
        onEvaluate={handleEvaluate}
        disabled={loading}
      />
      <footer>
        <p className="muted">Ready for air-gapped deployments. All processing happens on your infrastructure.</p>
      </footer>
    </div>
  );
}
