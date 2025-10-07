import React, { useEffect, useMemo, useRef, useState } from 'react';
import useLogStream from '../hooks/useLogStream.js';

const LEVEL_METADATA = {
  ERROR: { label: 'Error', icon: '❌', className: 'log-error' },
  WARNING: { label: 'Warning', icon: '⚠️', className: 'log-warn' },
  INFO: { label: 'Info', icon: 'ℹ️', className: 'log-info' },
  DEBUG: { label: 'Debug', icon: '🐞', className: 'log-debug' }
};

const LEVEL_ORDER = ['ERROR', 'WARNING', 'INFO', 'DEBUG'];

function formatTime(value) {
  try {
    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toLocaleTimeString([], { hour12: false });
    }
  } catch (error) {
    // Ignore and fall back to raw value.
  }
  return value;
}

export default function LogConsole({ stream }) {
  const logStream = stream ?? useLogStream();
  const {
    logs,
    connectionState,
    lastError,
    lastConnectionError,
    backlogWarning,
    isPaused,
    autoScroll,
    setAutoScroll,
    includeDebug,
    setIncludeDebug,
    clear,
    pause,
    resume,
    acknowledgeBacklogWarning
  } = logStream;

  const [levelFilters, setLevelFilters] = useState(() => ({
    ERROR: true,
    WARNING: true,
    INFO: true,
    DEBUG: includeDebug
  }));

  useEffect(() => {
    setLevelFilters((prev) => ({
      ...prev,
      DEBUG: includeDebug ? prev.DEBUG || true : false
    }));
  }, [includeDebug]);

  const visibleLogs = useMemo(
    () => logs.filter((entry) => levelFilters[entry.level] !== false),
    [logs, levelFilters]
  );

  const bodyRef = useRef(null);

  useEffect(() => {
    if (!autoScroll) {
      return;
    }
    const node = bodyRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [visibleLogs, autoScroll]);

  const handleLevelToggle = (level) => {
    setLevelFilters((prev) => {
      const next = { ...prev, [level]: !prev[level] };
      if (level === 'DEBUG') {
        setIncludeDebug(next[level]);
      }
      return next;
    });
  };

  const handlePauseToggle = () => {
    if (isPaused) {
      resume();
    } else {
      pause();
    }
  };

  const handleAutoScrollToggle = () => {
    setAutoScroll(!autoScroll);
  };

  const handleBacklogDismiss = () => {
    if (acknowledgeBacklogWarning) {
      acknowledgeBacklogWarning();
    }
  };

  return (
    <div className="log-console">
      <div className="log-console__header">
        <div className="log-console__title">
          <h3>Live Logs</h3>
          <span className={`log-console__status log-console__status--${connectionState}`}>
            {connectionState === 'open' ? 'Connected' : connectionState}
          </span>
        </div>
        <div className="log-console__actions">
          <button type="button" onClick={handlePauseToggle} className="log-console__button">
            {isPaused ? 'Resume' : 'Pause'}
          </button>
          <button type="button" onClick={clear} className="log-console__button">
            Clear
          </button>
        </div>
      </div>
      {lastConnectionError && (
        <div className="log-console__notice log-console__notice--connection" role="alert">
          <span aria-hidden="true">🔌</span>
          <div>
            <strong>Connection issue:</strong> {lastConnectionError}
          </div>
        </div>
      )}
      {backlogWarning && (
        <div className="log-console__notice log-console__notice--warning" role="status">
          <span aria-hidden="true">⚠️</span>
          <div>
            <strong>Backpressure detected:</strong> {backlogWarning.message}
            <div className="log-console__notice-actions">
              <button type="button" className="log-console__button" onClick={handleBacklogDismiss}>
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}
      <div className="log-console__controls">
        <fieldset className="log-console__levels">
          <legend className="sr-only">Severity filters</legend>
          {LEVEL_ORDER.map((level) => {
            const meta = LEVEL_METADATA[level];
            return (
              <label key={level} className="log-console__level">
                <input
                  type="checkbox"
                  checked={Boolean(levelFilters[level])}
                  onChange={() => handleLevelToggle(level)}
                />
                <span>
                  {meta.icon} {meta.label}
                </span>
              </label>
            );
          })}
        </fieldset>
        <label className="log-console__level">
          <input type="checkbox" checked={autoScroll} onChange={handleAutoScrollToggle} />
          <span>Auto-scroll</span>
        </label>
      </div>
      <div
        className="log-console__body"
        role="log"
        aria-live="polite"
        aria-relevant="additions"
        tabIndex={0}
        ref={bodyRef}
      >
        {visibleLogs.length === 0 ? (
          <div className="log-console__empty">Logs will appear here as the system runs.</div>
        ) : (
          visibleLogs.map((entry) => {
            const meta = LEVEL_METADATA[entry.level] || LEVEL_METADATA.INFO;
            return (
              <div key={entry.id} className={`log-entry ${meta.className}`}>
                <div className="log-entry__icon" aria-hidden="true">
                  {meta.icon}
                </div>
                <div className="log-entry__content">
                  <div className="log-entry__meta">
                    <time dateTime={entry.timestamp}>{formatTime(entry.timestamp)}</time>
                    {entry.logger && <span className="log-entry__logger">{entry.logger}</span>}
                    {entry.module && <span className="log-entry__module">{entry.module}</span>}
                  </div>
                  <div className="log-entry__message">{entry.message}</div>
                </div>
              </div>
            );
          })
        )}
      </div>
      {lastError && (
        <div className="log-console__footer" role="alert">
          {lastError}
        </div>
      )}
    </div>
  );
}
