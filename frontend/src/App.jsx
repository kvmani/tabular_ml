import React, { useEffect, useRef, useState } from 'react';
import DatasetManager from './components/DatasetManager.jsx';
import PreprocessPanel from './components/PreprocessPanel.jsx';
import VisualizationPanel from './components/VisualizationPanel.jsx';
import ModelTrainer from './components/ModelTrainer.jsx';
import EvaluationPanel from './components/EvaluationPanel.jsx';
import SystemConfigPanel from './components/SystemConfigPanel.jsx';
import {
  listDatasets,
  listSampleDatasets,
  loadSampleDataset,
  uploadDataset,
  getDatasetPreview,
  getDatasetSummary,
  getDatasetColumns,
  listAlgorithms,
  evaluateModel,
  getSystemConfig
} from './api/client.js';

function useNotification() {
  const [message, setMessage] = useState(null);
  const [variant, setVariant] = useState('info');
  const timeoutRef = useRef(null);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const notify = (text, type = 'info', options = {}) => {
    const { persist = false } = options;
    setMessage(text);
    setVariant(type);
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (text && !persist) {
      timeoutRef.current = setTimeout(() => {
        setMessage(null);
        timeoutRef.current = null;
      }, 4000);
    }
  };

  const clear = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    setMessage(null);
  };

  return { message, variant, notify, clear };
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
  const [systemConfig, setSystemConfig] = useState(null);
  const [allowUploads, setAllowUploads] = useState(true);
  const [liveHistory, setLiveHistory] = useState([]);
  const prefetchedDatasetId = useRef(null);

  const { message, variant, notify } = useNotification();

  const refreshDatasets = async (selectId, options = {}) => {
    const { allowFallback = false } = options;

    const attemptTitanicFallback = async () => {
      setLoading(true);
      try {
        const sample = await loadSampleDataset('titanic');
        notify(`Loaded sample dataset: ${sample.dataset.name}.`);
        return sample.dataset.dataset_id;
      } catch (error) {
        notify(
          `Automatic Titanic preload failed: ${error.message}`,
          'error',
          { persist: true }
        );
        throw error;
      } finally {
        setLoading(false);
      }
    };

    let response;
    try {
      response = await listDatasets();
    } catch (error) {
      if (allowFallback) {
        const datasetId = await attemptTitanicFallback();
        return refreshDatasets(datasetId, { allowFallback: false });
      }
      throw error;
    }

    const nextDatasets = response.datasets || [];
    setDatasets(nextDatasets);
    setCurrentDatasetId((prevId) => {
      if (selectId) {
        return selectId;
      }
      const prevStillExists = nextDatasets.some(
        (dataset) => dataset.dataset_id === prevId
      );
      if (prevId && prevStillExists) {
        return prevId;
      }
      if (response.default_dataset_id) {
        return response.default_dataset_id;
      }
      return nextDatasets.length > 0 ? nextDatasets[0].dataset_id : '';
    });

    if (allowFallback && nextDatasets.length === 0 && !response.default_dataset_id) {
      const datasetId = await attemptTitanicFallback();
      return refreshDatasets(datasetId, { allowFallback: false });
    }

    return response;
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
        const [samplesResponse, algorithmsResponse, configResponse] = await Promise.all([
          listSampleDatasets(),
          listAlgorithms(),
          getSystemConfig()
        ]);
        setSampleDatasets(samplesResponse.samples || []);
        setAlgorithms(algorithmsResponse.algorithms || []);
        setSystemConfig(configResponse);
        if (configResponse?.settings?.app?.allow_file_uploads === false) {
          setAllowUploads(false);
        } else {
          setAllowUploads(true);
        }
      } catch (error) {
        notify(error.message, 'error');
      }
      try {
        const datasetsResponse = await refreshDatasets(undefined, { allowFallback: true });
        const datasetIdFromResponse =
          datasetsResponse?.default_dataset_id ||
          datasetsResponse?.datasets?.[0]?.dataset_id ||
          '';

        if (datasetIdFromResponse) {
          prefetchedDatasetId.current = datasetIdFromResponse;
          setCurrentDatasetId(datasetIdFromResponse);
          await refreshDatasetDetails(datasetIdFromResponse);
        }
      } catch (error) {
        // Fallback notifications are handled in refreshDatasets(); avoid duplicate toasts.
      }
    })();
  }, []);

  useEffect(() => {
    if (!currentDatasetId) {
      prefetchedDatasetId.current = null;
      setPreview([]);
      setSummary({});
      setColumnsInfo({ columns: [], dtypes: {} });
    } else if (prefetchedDatasetId.current === currentDatasetId) {
      prefetchedDatasetId.current = null;
    } else {
      refreshDatasetDetails(currentDatasetId);
    }
    setSplitId('');
    setModelId('');
    setMetrics(null);
    setEvaluation(null);
    setLiveHistory([]);
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
    setLiveHistory(result.history || []);
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
      {!allowUploads && (
        <div className="banner warning" role="status">File uploads are disabled; use the dataset registry instead.</div>
      )}
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
        allowUploads={allowUploads}
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
        onProgress={setLiveHistory}
        disabled={!currentDatasetId || loading}
      />
      <EvaluationPanel
        modelId={modelId}
        metrics={metrics}
        evaluation={evaluation}
        onEvaluate={handleEvaluate}
        streamedHistory={liveHistory}
        disabled={loading}
      />
      <SystemConfigPanel config={systemConfig} />
      <footer>
        <p className="muted">Ready for air-gapped deployments. All processing happens on your infrastructure.</p>
      </footer>
    </div>
  );
}
