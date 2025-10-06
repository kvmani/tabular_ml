const isNumber = (value) => typeof value === 'number' && Number.isFinite(value);

const normaliseNumeric = (value) => (isNumber(value) ? value : null);

export function buildTrainingHistoryFigure(history = []) {
  if (!Array.isArray(history) || history.length === 0) {
    return null;
  }

  const epochs = history.map((entry, index) => {
    if (entry == null || typeof entry !== 'object') {
      return index + 1;
    }
    if (isNumber(entry.epoch)) {
      return entry.epoch;
    }
    if (isNumber(entry.iteration)) {
      return entry.iteration + 1;
    }
    return index + 1;
  });

  const data = [];

  if (history.some((entry) => isNumber(entry?.train_loss))) {
    data.push({
      x: epochs,
      y: history.map((entry) => normaliseNumeric(entry?.train_loss)),
      mode: 'lines+markers',
      name: 'Train Loss'
    });
  }

  if (history.some((entry) => isNumber(entry?.val_loss))) {
    data.push({
      x: epochs,
      y: history.map((entry) => normaliseNumeric(entry?.val_loss)),
      mode: 'lines+markers',
      name: 'Validation Loss'
    });
  }

  if (history.some((entry) => isNumber(entry?.train_metric))) {
    const metricName = history.find((entry) => entry?.metric)?.metric || 'Metric';
    data.push({
      x: epochs,
      y: history.map((entry) => normaliseNumeric(entry?.train_metric)),
      mode: 'lines+markers',
      name: `Train ${metricName}`
    });
  }

  if (history.some((entry) => isNumber(entry?.validation_metric))) {
    const metricName = history.find((entry) => entry?.metric)?.metric || 'Metric';
    data.push({
      x: epochs,
      y: history.map((entry) => normaliseNumeric(entry?.validation_metric)),
      mode: 'lines+markers',
      name: `Validation ${metricName}`,
      yaxis: 'y2'
    });
  }

  const metricsEntries = history.filter((entry) => entry?.metrics && typeof entry.metrics === 'object');
  if (metricsEntries.length > 0) {
    const latestMetrics = metricsEntries[metricsEntries.length - 1].metrics;
    const metricKeys = Object.keys(latestMetrics);
    if (metricKeys.length > 0) {
      const preferredKeys = ['accuracy', 'f1', 'r2', 'mae'];
      const selectedKey = preferredKeys.find((key) => key in latestMetrics) || metricKeys[0];
      data.push({
        x: epochs,
        y: history.map((entry) => normaliseNumeric(entry?.metrics?.[selectedKey])),
        mode: 'lines+markers',
        name: `Validation ${selectedKey}`,
        yaxis: 'y2'
      });
    }
  }

  const layout = {
    title: 'Training History',
    xaxis: { title: 'Epoch' },
    yaxis: { title: 'Loss' },
    template: 'plotly_white'
  };

  if (data.some((trace) => trace.yaxis === 'y2')) {
    layout.yaxis2 = {
      title: 'Validation Metric',
      overlaying: 'y',
      side: 'right'
    };
  }

  if (data.length === 0) {
    data.push({
      x: epochs,
      y: history.map(() => 0),
      mode: 'lines+markers',
      name: 'Progress'
    });
  }

  return { data, layout };
}

export function latestHistoryLabel(history = []) {
  if (!Array.isArray(history) || history.length === 0) {
    return null;
  }
  const latest = history[history.length - 1];
  if (isNumber(latest?.epoch)) {
    return `Epoch ${latest.epoch}`;
  }
  if (isNumber(latest?.iteration)) {
    return `Iteration ${latest.iteration + 1}`;
  }
  if (latest?.stage) {
    return latest.stage;
  }
  return `Update ${history.length}`;
}
