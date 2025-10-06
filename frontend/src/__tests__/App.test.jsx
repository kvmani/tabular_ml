import { render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('../api/client.js', () => {
  const listDatasets = vi.fn();
  const loadSampleDataset = vi.fn();
  return {
    listDatasets,
    listSampleDatasets: vi.fn().mockResolvedValue({ samples: [] }),
    loadSampleDataset,
    uploadDataset: vi.fn().mockResolvedValue({}),
    getDatasetPreview: vi.fn().mockResolvedValue({ data: [] }),
    getDatasetSummary: vi.fn().mockResolvedValue({ summary: {} }),
    getDatasetColumns: vi.fn().mockResolvedValue({ columns: [], dtypes: {} }),
    listAlgorithms: vi.fn().mockResolvedValue({ algorithms: [] }),
    evaluateModel: vi.fn(),
    getSystemConfig: vi.fn().mockResolvedValue({}),
    getHealth: vi.fn(),
    trainModel: vi.fn(),
    detectOutliers: vi.fn(),
    removeOutliers: vi.fn(),
    imputeDataset: vi.fn(),
    filterDataset: vi.fn(),
    splitDataset: vi.fn()
  };
});

vi.mock('../components/DatasetManager.jsx', () => ({
  default: () => <div data-testid="dataset-manager" />
}));

vi.mock('../components/PreprocessPanel.jsx', () => ({
  default: () => <div data-testid="preprocess-panel" />
}));

vi.mock('../components/VisualizationPanel.jsx', () => ({
  default: () => <div data-testid="visualization-panel" />
}));

vi.mock('../components/ModelTrainer.jsx', () => ({
  default: () => <div data-testid="model-trainer" />
}));

vi.mock('../components/EvaluationPanel.jsx', () => ({
  default: () => <div data-testid="evaluation-panel" />
}));

vi.mock('../components/SystemConfigPanel.jsx', () => ({
  default: () => <div data-testid="config-panel" />
}));

import App from '../App.jsx';
import { listDatasets, loadSampleDataset } from '../api/client.js';

describe('App fallback behaviour', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.resetAllMocks();
  });

  it('shows a blocking notification when the Titanic fallback fails', async () => {
    listDatasets.mockRejectedValue(new Error('Service unavailable'));
    loadSampleDataset.mockRejectedValue(new Error('Bundle missing'));

    render(<App />);

    await waitFor(() => {
      expect(loadSampleDataset).toHaveBeenCalledWith('titanic');
    });

    await waitFor(() => {
      expect(
        screen.getByText(/Automatic Titanic preload failed/i)
      ).toBeInTheDocument();
    });

    const notification = screen.getByText(/Automatic Titanic preload failed/i);

    vi.useFakeTimers();
    vi.advanceTimersByTime(8000);
    expect(notification).toBeInTheDocument();
  });
});
