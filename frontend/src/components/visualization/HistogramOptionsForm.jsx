import React from 'react';
import OptionControl from './OptionControl.jsx';

export default function HistogramOptionsForm({ options, onChange, disabled }) {
  const handleNumberChange = (key) => (event) => {
    const value = Number(event.target.value);
    onChange(key, Number.isNaN(value) ? options[key] : value);
  };

  const handleSelectChange = (key) => (event) => {
    onChange(key, event.target.value);
  };

  const handleCheckboxChange = (key) => (event) => {
    onChange(key, event.target.checked);
  };

  const handleTextChange = (key) => (event) => {
    onChange(key, event.target.value);
  };

  return (
    <div className="options-grid">
      <OptionControl id="hist-bin-count" label="Bin count" description="Adjust resolution of the bars.">
        <input
          id="hist-bin-count"
          type="number"
          min={1}
          max={200}
          value={options.binCount}
          onChange={handleNumberChange('binCount')}
          disabled={disabled}
        />
      </OptionControl>
      <OptionControl id="hist-orientation" label="Orientation">
        <select
          id="hist-orientation"
          value={options.orientation}
          onChange={handleSelectChange('orientation')}
          disabled={disabled}
        >
          <option value="v">Vertical</option>
          <option value="h">Horizontal</option>
        </select>
      </OptionControl>
      <OptionControl id="hist-color" label="Bar color">
        <input
          id="hist-color"
          type="color"
          value={options.color}
          onChange={handleTextChange('color')}
          disabled={disabled}
        />
      </OptionControl>
      <OptionControl id="hist-opacity" label="Opacity" description="0 is transparent, 1 is solid.">
        <input
          id="hist-opacity"
          type="range"
          min={0.1}
          max={1}
          step={0.05}
          value={options.opacity}
          onChange={handleNumberChange('opacity')}
          disabled={disabled}
        />
        <span className="option-value-indicator">{options.opacity.toFixed(2)}</span>
      </OptionControl>
      <OptionControl id="hist-bar-gap" label="Bar gap" description="Spacing between bars.">
        <input
          id="hist-bar-gap"
          type="range"
          min={0}
          max={0.5}
          step={0.05}
          value={options.barGap}
          onChange={handleNumberChange('barGap')}
          disabled={disabled}
        />
        <span className="option-value-indicator">{options.barGap.toFixed(2)}</span>
      </OptionControl>
      <OptionControl id="hist-show-grid" label="Grid lines">
        <div className="option-inline-checkbox">
          <input
            id="hist-show-grid"
            type="checkbox"
            checked={options.showGrid}
            onChange={handleCheckboxChange('showGrid')}
            disabled={disabled}
          />
          <span>Show axis grid</span>
        </div>
      </OptionControl>
      <OptionControl id="hist-legend" label="Legend orientation">
        <select
          id="hist-legend"
          value={options.legendOrientation}
          onChange={handleSelectChange('legendOrientation')}
          disabled={disabled}
        >
          <option value="v">Vertical</option>
          <option value="h">Horizontal</option>
        </select>
      </OptionControl>
      <OptionControl id="hist-title" label="Custom title">
        <input
          id="hist-title"
          type="text"
          placeholder="Override plot title"
          value={options.title}
          onChange={handleTextChange('title')}
          disabled={disabled}
        />
      </OptionControl>
    </div>
  );
}

