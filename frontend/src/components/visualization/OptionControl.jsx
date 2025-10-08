import React from 'react';

export default function OptionControl({ id, label, description = '', children }) {
  return (
    <div className="option-control">
      <div className="option-control-labels">
        <label htmlFor={id}>{label}</label>
        {description && <span className="option-control-description">{description}</span>}
      </div>
      {children}
    </div>
  );
}

