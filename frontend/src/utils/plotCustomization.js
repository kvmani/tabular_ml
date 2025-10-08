const cloneFigure = (figure) => JSON.parse(JSON.stringify(figure));

export const createDefaultHistogramOptions = () => ({
  binCount: 30,
  orientation: 'v',
  color: '#2563eb',
  opacity: 0.75,
  barGap: 0.05,
  showGrid: true,
  legendOrientation: 'v',
  title: ''
});

export const createDefaultScatterOptions = () => ({
  markerSize: 8,
  markerOpacity: 0.8,
  lineWidth: 0,
  showLegend: true,
  title: '',
  xTitle: '',
  yTitle: '',
  background: '#ffffff'
});

export const applyHistogramOptions = (figure, options) => {
  if (!figure) {
    return null;
  }

  const next = cloneFigure(figure);
  next.data = (next.data || []).map((trace) => {
    const orientation = options.orientation;
    const marker = { ...(trace.marker || {}), color: options.color, opacity: options.opacity };
    return {
      ...trace,
      marker,
      orientation,
      nbinsx: orientation === 'v' ? options.binCount : undefined,
      nbinsy: orientation === 'h' ? options.binCount : undefined
    };
  });

  const baseTitle = next.layout?.title?.text || '';
  next.layout = {
    ...next.layout,
    bargap: options.barGap,
    legend: {
      ...(next.layout?.legend || {}),
      orientation: options.legendOrientation
    },
    title: {
      ...(next.layout?.title || {}),
      text: options.title || baseTitle
    },
    xaxis: {
      ...(next.layout?.xaxis || {}),
      showgrid: options.showGrid
    },
    yaxis: {
      ...(next.layout?.yaxis || {}),
      showgrid: options.showGrid
    }
  };

  return next;
};

export const applyScatterOptions = (figure, options) => {
  if (!figure) {
    return null;
  }

  const next = cloneFigure(figure);
  next.data = (next.data || []).map((trace) => ({
    ...trace,
    marker: {
      ...(trace.marker || {}),
      size: options.markerSize,
      opacity: options.markerOpacity,
      line: {
        ...(trace.marker?.line || {}),
        width: options.lineWidth
      }
    }
  }));

  const baseTitle = next.layout?.title?.text || '';
  const baseX = next.layout?.xaxis?.title?.text || '';
  const baseY = next.layout?.yaxis?.title?.text || '';

  next.layout = {
    ...next.layout,
    showlegend: options.showLegend,
    title: {
      ...(next.layout?.title || {}),
      text: options.title || baseTitle
    },
    xaxis: {
      ...(next.layout?.xaxis || {}),
      title: {
        ...(next.layout?.xaxis?.title || {}),
        text: options.xTitle || baseX
      }
    },
    yaxis: {
      ...(next.layout?.yaxis || {}),
      title: {
        ...(next.layout?.yaxis?.title || {}),
        text: options.yTitle || baseY
      }
    },
    plot_bgcolor: options.background,
    paper_bgcolor: options.background
  };

  return next;
};
