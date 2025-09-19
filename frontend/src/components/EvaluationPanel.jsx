import React from 'react';
import Plot from 'react-plotly.js';

function MetricTable({ title, metrics }) {
  if (!metrics) {
    return null;
  }
  const rows = Object.entries(metrics);
  return (
    <div className="metric-block">
      <h4>{title}</h4>
      <table>
        <tbody>
          {rows.map(([key, value]) => (
            <tr key={key}>
              <td>{key}</td>
              <td>{typeof value === 'number' ? value.toFixed(4) : value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function EvaluationPanel({ modelId, metrics, evaluation, onEvaluate, disabled }) {
  return (
    <div className="card">
      <div className="card-header">
        <h2>5. Evaluate &amp; share</h2>
        <button type="button" onClick={onEvaluate} data-testid="evaluate-button" disabled={!modelId || disabled}>
          Evaluate model
        </button>
      </div>
      <div className="card-body evaluation-grid">
        <div>
          <h3>Metrics</h3>
          {modelId ? (
            <div className="metric-grid" data-testid="metrics-summary">
              <MetricTable title="Validation" metrics={metrics?.validation} />
              <MetricTable title="Test" metrics={metrics?.test} />
            </div>
          ) : (
            <p className="muted">Train a model to see metrics.</p>
          )}
        </div>
        <div>
          <h3>Training history</h3>
          {evaluation?.training_history ? (
            <Plot
              data={evaluation.training_history.data}
              layout={{ ...evaluation.training_history.layout, autosize: true }}
              style={{ width: '100%', height: '100%' }}
            />
            </div>
          ) : (
            <p className="muted">Evaluate the model to visualise training progress.</p>
          )}
        </div>
        <div>
          <h3>Classification insights</h3>
          {evaluation?.confusion_matrix ? (
            <div data-testid="confusion-matrix">
              <Plot
                data={evaluation.confusion_matrix.data}
                layout={{ ...evaluation.confusion_matrix.layout, autosize: true }}
                style={{ width: '100%', height: '100%' }}
              />
            </div>
          ) : (
            <p className="muted">Confusion matrix appears for classification models.</p>
          )}
        </div>
        <div>
          <h3>ROC curve</h3>
          {evaluation?.roc_curve ? (
            <Plot
              data={evaluation.roc_curve.data}
              layout={{ ...evaluation.roc_curve.layout, autosize: true }}
              style={{ width: '100%', height: '100%' }}
            />\r\n            </div>\r\n          ) : (
            <p className="muted">Available for binary classifiers with probability outputs.</p>
          )}
        </div>
        <div>
          <h3>Regression diagnostics</h3>
          {evaluation?.regression_diagnostics ? (
            <div className="diagnostic-plots">
              <Plot
                data={evaluation.regression_diagnostics.comparison.data}
                layout={{ ...evaluation.regression_diagnostics.comparison.layout, autosize: true }}
                style={{ width: '100%', height: '100%' }}
              />
              <Plot
                data={evaluation.regression_diagnostics.residuals.data}
                layout={{ ...evaluation.regression_diagnostics.residuals.layout, autosize: true }}
                style={{ width: '100%', height: '100%' }}
              />
            </div>
          ) : (
            <p className="muted">Actual/predicted plots appear for regression models.</p>
          )}
        </div>
      </div>
    </div>
  );
}

