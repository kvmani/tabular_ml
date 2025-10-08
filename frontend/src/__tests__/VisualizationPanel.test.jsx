import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import VisualizationPanel from '../components/VisualizationPanel.jsx';

const plotRenderMock = vi.hoisted(() => vi.fn());
const getHistogramMock = vi.hoisted(() => vi.fn());
const getScatterMock = vi.hoisted(() => vi.fn());

vi.mock('react-plotly.js', () => ({
  __esModule: true,
  default: (props) => {
    plotRenderMock(props);
    return <div data-testid={props['data-testid'] || 'plotly'} />;
  }
}));

vi.mock('../api/client.js', () => ({
  __esModule: true,
  getHistogram: getHistogramMock,
  getScatter: getScatterMock
}));

describe('VisualizationPanel customization', () => {
  beforeEach(() => {
    plotRenderMock.mockReset();
    getHistogramMock.mockReset();
    getScatterMock.mockReset();
  });

  it('exposes histogram customization controls that update the plot', async () => {
    getHistogramMock.mockResolvedValue({
      figure: {
        data: [
          {
            type: 'histogram',
            marker: {},
            nbinsx: 10
          }
        ],
        layout: {
          title: { text: 'Base histogram' },
          xaxis: { showgrid: false },
          yaxis: { showgrid: false }
        }
      }
    });

    render(
      <VisualizationPanel
        datasetId="demo"
        columns={['age', 'fare']}
        onNotify={vi.fn()}
        disabled={false}
      />
    );

    fireEvent.change(screen.getByLabelText(/column/i), { target: { value: 'age' } });
    fireEvent.click(screen.getByRole('button', { name: /generate histogram/i }));

    await waitFor(() => expect(getHistogramMock).toHaveBeenCalled());
    await screen.findByTestId('histogram-plot');

    fireEvent.click(screen.getByRole('button', { name: /customize histogram options/i }));
    const binInput = await screen.findByLabelText(/bin count/i);
    fireEvent.change(binInput, { target: { value: '15' } });
    fireEvent.change(screen.getByLabelText(/opacity/i), { target: { value: '0.5' } });

    await waitFor(() => {
      const lastCall = plotRenderMock.mock.calls.at(-1)?.[0];
      expect(lastCall?.data?.[0]?.nbinsx).toBe(15);
      expect(lastCall?.data?.[0]?.marker?.opacity).toBeCloseTo(0.5);
      expect(lastCall?.layout?.xaxis?.showgrid).toBe(true);
    });
  });

  it('applies scatter customization updates to layout and markers', async () => {
    getScatterMock.mockResolvedValue({
      figure: {
        data: [
          {
            type: 'scatter',
            mode: 'markers',
            marker: { size: 6, opacity: 0.6, line: { width: 0 } }
          }
        ],
        layout: {
          title: { text: 'Base scatter' },
          xaxis: { title: { text: 'X' } },
          yaxis: { title: { text: 'Y' } },
          showlegend: true
        }
      }
    });

    render(
      <VisualizationPanel
        datasetId="demo"
        columns={['age', 'fare', 'pclass']}
        onNotify={vi.fn()}
        disabled={false}
      />
    );

    fireEvent.change(screen.getByLabelText(/x axis/i), { target: { value: 'age' } });
    fireEvent.change(screen.getByLabelText(/y axis/i), { target: { value: 'fare' } });
    const scatterButtons = screen.getAllByRole('button', { name: /generate scatter/i });
    fireEvent.click(scatterButtons[0]);

    await waitFor(() => expect(getScatterMock).toHaveBeenCalled());
    await screen.findByTestId('scatter-plot');

    const scatterGearButtons = screen.getAllByRole('button', { name: /customize scatter plot options/i });
    fireEvent.click(scatterGearButtons[0]);
    fireEvent.change(await screen.findByLabelText(/marker size/i), { target: { value: '12' } });
    fireEvent.change(screen.getByLabelText(/background color/i), { target: { value: '#222222' } });
    fireEvent.change(screen.getByLabelText(/x axis title/i), { target: { value: 'Age (years)' } });

    await waitFor(() => {
      const lastCall = plotRenderMock.mock.calls.at(-1)?.[0];
      expect(lastCall?.data?.[0]?.marker?.size).toBe(12);
      expect(lastCall?.layout?.plot_bgcolor).toBe('#222222');
      expect(lastCall?.layout?.xaxis?.title?.text).toBe('Age (years)');
    });
  });
});
