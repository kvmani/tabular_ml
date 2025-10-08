import React from 'react';
import OptionControl from './OptionControl.jsx';

export default function ScatterOptionsForm({ options, onChange, disabled }) {
  const handleNumberChange = (key) => (event) => {
    const value = Number(event.target.value);
    onChange(key, Number.isNaN(value) ? options[key] : value);
  };

  const handleCheckboxChange = (key) => (event) => {
    onChange(key, event.target.checked);
  };

  const handleTextChange = (key) => (event) => {
    onChange(key, event.target.value);
  };

  return (
    <div className="options-grid">
      <OptionControl id="scatter-marker-size" label="Marker size">
        <input
          id="scatter-marker-size"
          type="number"
          min={1}
          max={40}
          value={options.markerSize}
          onChange={handleNumberChange('markerSize')}
          disabled={disabled}
        />
      </OptionControl>
      <OptionControl id="scatter-marker-opacity" label="Marker opacity">
        <input
          id="scatter-marker-opacity"
          type="range"
          min={0.1}
          max={1}
          step={0.05}
          value={options.markerOpacity}
          onChange={handleNumberChange('markerOpacity')}
          disabled={disabled}
        />
        <span className="option-value-indicator">{options.markerOpacity.toFixed(2)}</span>
      </OptionControl>
      <OptionControl id="scatter-line-width" label="Outline width" description="Adds a border around markers.">
        <input
          id="scatter-line-width"
          type="range"
          min={0}
          max={5}
          step={0.5}
          value={options.lineWidth}
          onChange={handleNumberChange('lineWidth')}
          disabled={disabled}
        />
        <span className="option-value-indicator">{options.lineWidth.toFixed(1)}</span>
      </OptionControl>
      <OptionControl id="scatter-show-legend" label="Legend">
        <div className="option-inline-checkbox">
          <input
            id="scatter-show-legend"
            type="checkbox"
            checked={options.showLegend}
            onChange={handleCheckboxChange('showLegend')}
            disabled={disabled}
          />
          <span>Show legend</span>
        </div>
      </OptionControl>
      <OptionControl id="scatter-title" label="Custom title">
        <input
          id="scatter-title"
          type="text"
          placeholder="Override plot title"
          value={options.title}
          onChange={handleTextChange('title')}
          disabled={disabled}
        />
      </OptionControl>
      <OptionControl id="scatter-x-title" label="X axis title">
        <input
          id="scatter-x-title"
          type="text"
          placeholder="Override X axis"
          value={options.xTitle}
          onChange={handleTextChange('xTitle')}
          disabled={disabled}
        />
      </OptionControl>
      <OptionControl id="scatter-y-title" label="Y axis title">
        <input
          id="scatter-y-title"
          type="text"
          placeholder="Override Y axis"
          value={options.yTitle}
          onChange={handleTextChange('yTitle')}
          disabled={disabled}
        />
      </OptionControl>
      <OptionControl id="scatter-background" label="Background color">
        <input
          id="scatter-background"
          type="color"
          value={options.background}
          onChange={handleTextChange('background')}
          disabled={disabled}
        />
      </OptionControl>
    </div>
  );
}

