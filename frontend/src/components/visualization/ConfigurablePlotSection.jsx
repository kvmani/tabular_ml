import React from 'react';

export default function ConfigurablePlotSection({
  title,
  children,
  configContent,
  disabled,
  isConfigOpen,
  onToggleConfig
}) {
  return (
    <div className="viz-section">
      <div className="viz-section-heading">
        <h3>{title}</h3>
        <button
          type="button"
          className="viz-config-button"
          onClick={onToggleConfig}
          aria-label={`Customize ${title.toLowerCase()} options`}
          aria-expanded={isConfigOpen}
          disabled={disabled}
        >
          <span aria-hidden="true">⚙</span>
        </button>
      </div>
      {isConfigOpen && <div className="viz-config-panel">{configContent}</div>}
      <div className="viz-section-body">{children}</div>
    </div>
  );
}

