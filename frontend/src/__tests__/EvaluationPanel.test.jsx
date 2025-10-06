import { render, screen } from '@testing-library/react';
import React from 'react';
import { describe, expect, it, vi } from 'vitest';

vi.mock('react-plotly.js', () => ({
  __esModule: true,
  default: (props) => <div data-testid={props['data-testid'] || 'plotly'} {...props} />
}));

import EvaluationPanel from '../components/EvaluationPanel.jsx';

describe('EvaluationPanel training history fallback', () => {
  it('renders streamed history when evaluation history is missing', () => {
    const streamedHistory = [
      { epoch: 1, train_loss: 0.6, val_loss: 0.5, metrics: { accuracy: 0.75 } },
      { epoch: 2, train_loss: 0.5, val_loss: 0.45, metrics: { accuracy: 0.78 } }
    ];

    render(
      <EvaluationPanel
        modelId="model-1"
        metrics={{ validation: { accuracy: 0.78 }, test: { accuracy: 0.76 } }}
        evaluation={{}}
        onEvaluate={vi.fn()}
        streamedHistory={streamedHistory}
        disabled={false}
      />
    );

    expect(screen.getByTestId('evaluation-history-plot')).toBeInTheDocument();
  });
});
