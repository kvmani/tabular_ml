import React from 'react';

export default function SystemConfigPanel({ config }) {
  if (!config) {
    return null;
  }
  const datasets = Object.entries(config.dataset_registry || {});
  const appSettings = config.settings?.app || {};
  const limits = config.settings?.limits || {};
  const security = config.settings?.security || {};

  return (
    <div className="card" data-testid="system-config-panel">
      <div className="card-header">
        <h2>6. Configuration</h2>
      </div>
      <div className="card-body config-panel">
        <div>
          <h3>Runtime</h3>
          <ul className="muted-list">
            <li>
              <strong>Environment:</strong> {appSettings.environment}
            </li>
            <li>
              <strong>Host:</strong> {appSettings.host}:{appSettings.port}
            </li>
            <li>
              <strong>Debug:</strong> {appSettings.debug ? 'enabled' : 'disabled'}
            </li>
            <li>
              <strong>Uploads:</strong> {appSettings.allow_file_uploads ? 'enabled' : 'disabled'}
            </li>
            <li>
              <strong>Random seed:</strong> {appSettings.random_seed}
            </li>
          </ul>
          <h3>Security</h3>
          <ul className="muted-list">
            <li>CSP: {security.csp_enabled ? 'enabled' : 'disabled'}</li>
            <li>CSRF protection: {security.csrf_protect ? 'enabled' : 'disabled'}</li>
            <li>
              Policy: <code>{security.csp_policy}</code>
            </li>
          </ul>
          <h3>Limits</h3>
          <ul className="muted-list">
            <li>Preview rows: {limits.max_rows_preview}</li>
            <li>Training rows: {limits.max_rows_train}</li>
            <li>Columns: {limits.max_cols}</li>
          </ul>
        </div>
        <div>
          <h3>Dataset registry ({datasets.length})</h3>
          <ul className="registry-list">
            {datasets.map(([key, entry]) => (
              <li key={key}>
                <strong>{entry.name || key}</strong> - {entry.task} (target: {entry.target})
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
