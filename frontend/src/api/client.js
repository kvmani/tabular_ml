const trimTrailingSlash = (value) => (typeof value === 'string' ? value.replace(/\/$/, '') : value);

const resolveBaseUrl = () => {
  const metaEnv = typeof import.meta !== 'undefined' && import.meta.env ? import.meta.env : {};
  const configured = metaEnv.VITE_API_BASE_URL;
  const isDev = Boolean(metaEnv && metaEnv.DEV);
  if (configured && configured.trim().length > 0) {
    return trimTrailingSlash(configured.trim());
  }

  if (isDev) {
    return '';
  }

  if (typeof window === 'undefined' || !window.location) {
    return '';
  }

  const { protocol, hostname, host, port } = window.location;

  const codespacesMatch = hostname.match(/^(.*)-(\d+)(\..*)$/);
  if (codespacesMatch) {
    const [, prefix, portSuffix, suffix] = codespacesMatch;
    if (portSuffix !== '8000') {
      return `${protocol}//${trimTrailingSlash(`${prefix}-8000${suffix}`)}`;
    }
  }

  if (port && port !== '8000') {
    return `${protocol}//${hostname}:8000`;
  }

  if (host) {
    return trimTrailingSlash(`${protocol}//${host}`);
  }

  return '';
};

const API_BASE_URL = resolveBaseUrl();

const buildUrl = (path) => {
  if (/^https?:\/\//i.test(path)) {
    return path;
  }
  const normalisedPath = path.startsWith('/') ? path.slice(1) : path;
  if (!API_BASE_URL) {
    return `/${normalisedPath}`;
  }
  return `${API_BASE_URL}/${normalisedPath}`;
};

let csrfToken = null;

function updateCsrfToken(response) {
  const headerToken = response.headers.get('x-csrf-token') || response.headers.get('X-CSRF-Token');
  if (headerToken) {
    csrfToken = headerToken;
  }
}

async function request(path, options = {}) {
  const method = (options.method || 'GET').toUpperCase();
  const headers = {
    ...(options.headers || {})
  };
  const hasContentType = headers['Content-Type'] || headers['content-type'];
  if (!hasContentType && !['GET', 'HEAD', 'OPTIONS', 'TRACE'].includes(method)) {
    headers['Content-Type'] = 'application/json';
  }
  if (!['GET', 'HEAD', 'OPTIONS', 'TRACE'].includes(method) && csrfToken) {
    headers['X-CSRF-Token'] = csrfToken;
  }

  let response;
  try {
    response = await fetch(buildUrl(path), {
      ...options,
      headers,
      credentials: 'include'
    });
  } catch (error) {
    throw new Error(`Network request failed: ${error.message}`);
  }

  updateCsrfToken(response);

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || 'Request failed');
  }

  const contentType = response.headers.get('content-type');
  if (contentType && contentType.includes('application/json')) {
    return response.json();
  }
  return response.text();
}

export async function listDatasets() {
  return request('/data/datasets');
}

export async function listSampleDatasets() {
  return request('/data/samples');
}

export async function loadSampleDataset(key) {
  return request(`/data/samples/${key}`, { method: 'POST' });
}

export async function uploadDataset(file, name, description) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('name', name || file.name);
  formData.append('description', description || 'Uploaded dataset');
  const headers = {};
  if (csrfToken) {
    headers['X-CSRF-Token'] = csrfToken;
  }
  let response;
  try {
    response = await fetch(buildUrl('/data/upload'), {
      method: 'POST',
      body: formData,
      headers,
      credentials: 'include'
    });
  } catch (error) {
    throw new Error(`Network request failed: ${error.message}`);
  }
  updateCsrfToken(response);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || 'Upload failed');
  }
  return response.json();
}

export async function getDatasetPreview(datasetId) {
  return request(`/data/${datasetId}/preview`);
}

export async function getDatasetSummary(datasetId) {
  return request(`/data/${datasetId}/summary`);
}

export async function getDatasetColumns(datasetId) {
  return request(`/data/${datasetId}/columns`);
}

export async function detectOutliers(datasetId, payload) {
  return request(`/preprocess/${datasetId}/detect-outliers`, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export async function removeOutliers(datasetId, payload) {
  return request(`/preprocess/${datasetId}/remove-outliers`, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export async function imputeDataset(datasetId, payload) {
  return request(`/preprocess/${datasetId}/impute`, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export async function filterDataset(datasetId, payload) {
  return request(`/preprocess/${datasetId}/filter`, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export async function splitDataset(datasetId, payload) {
  return request(`/preprocess/${datasetId}/split`, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export async function listAlgorithms() {
  return request('/model/algorithms');
}

export async function getSystemConfig() {
  return request('/system/config');
}

export async function getHealth() {
  return request('/health');
}

export async function trainModel(payload) {
  return request('/model/train', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export async function evaluateModel(modelId) {
  return request('/model/evaluate', {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId })
  });
}

export async function getHistogram(datasetId, column) {
  return request(`/visualization/${datasetId}/histogram`, {
    method: 'POST',
    body: JSON.stringify({ column })
  });
}

export async function getScatter(datasetId, payload) {
  return request(`/visualization/${datasetId}/scatter`, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}
