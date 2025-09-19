const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

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

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
    credentials: 'include'
  });

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
  const response = await fetch(`${API_BASE_URL}/data/upload`, {
    method: 'POST',
    body: formData,
    headers,
    credentials: 'include'
  });
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
