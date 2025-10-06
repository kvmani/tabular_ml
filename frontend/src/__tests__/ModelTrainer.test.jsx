import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('react-plotly.js', () => ({
  __esModule: true,
  default: (props) => <div data-testid={props['data-testid'] || 'plotly'} {...props} />
}));

const trainModelMock = vi.fn();
const openTrainingStreamMock = vi.fn();

vi.mock('../api/client.js', () => ({
  trainModel: trainModelMock,
  openTrainingStream: openTrainingStreamMock
}));

import ModelTrainer from '../components/ModelTrainer.jsx';

const createMockStream = () => {
  const listeners = new Map();
  let errorHandler = null;
  return {
    addEventListener(type, handler) {
      listeners.set(type, handler);
    },
    removeEventListener(type, handler) {
      const current = listeners.get(type);
      if (current === handler) {
        listeners.delete(type);
      }
    },
    close: vi.fn(),
    emit(type, payload) {
      const handler = listeners.get(type);
      if (handler) {
        handler({ data: JSON.stringify(payload) });
      }
    },
    set onerror(handler) {
      errorHandler = handler;
    },
    get onerror() {
      return errorHandler;
    },
    triggerError(message) {
      if (errorHandler) {
        errorHandler({ data: JSON.stringify({ message }) });
      }
    }
  };
};

describe('ModelTrainer live updates', () => {
  beforeEach(() => {
    trainModelMock.mockReset();
    openTrainingStreamMock.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('renders streaming updates and resolves with final payload', async () => {
    const stream = createMockStream();
    openTrainingStreamMock.mockReturnValue(stream);

    const onTrainingComplete = vi.fn();
    const onProgress = vi.fn();
    const onNotify = vi.fn();

    render(
      <ModelTrainer
        datasetId="dataset-1"
        splitId="split-1"
        columns={['a', 'target']}
        algorithms={[{ key: 'logistic_regression', label: 'Logistic Regression' }]}
        onTrainingComplete={onTrainingComplete}
        onNotify={onNotify}
        onProgress={onProgress}
        disabled={false}
      />
    );

    await act(async () => {
      fireEvent.click(screen.getByTestId('train-button'));
    });

    act(() => {
      stream.emit('history', {
        type: 'history',
        entry: { epoch: 1, train_loss: 0.5, val_loss: 0.4, metrics: { accuracy: 0.8 } }
      });
    });

    await waitFor(() => {
      expect(screen.getByTestId('latest-epoch')).toHaveTextContent(/Epoch 1/i);
    });
    expect(onProgress).toHaveBeenCalled();

    act(() => {
      stream.emit('result', {
        type: 'result',
        payload: {
          model_id: 'model-123',
          metrics: { validation: { accuracy: 0.81 }, test: { accuracy: 0.79 } },
          history: [{ epoch: 1, train_loss: 0.5 }],
          split_id: 'split-1'
        }
      });
    });

    await waitFor(() => {
      expect(onTrainingComplete).toHaveBeenCalledWith(
        expect.objectContaining({ model_id: 'model-123' })
      );
    });

    expect(onNotify).toHaveBeenCalledWith(expect.stringMatching(/model trained successfully/i));
    expect(screen.getByTestId('training-status')).toHaveTextContent(/Training complete/i);
  });
});
