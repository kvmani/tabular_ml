import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { openLogStream } from '../api/client.js';

const DEFAULT_CAPACITY = 500;

const LEVELS = new Set(['DEBUG', 'INFO', 'WARNING', 'ERROR']);

export default function useLogStream(options = {}) {
  const { capacity = DEFAULT_CAPACITY, includeDebug: includeDebugInitial = false } = options;
  const [logs, setLogs] = useState([]);
  const [connectionState, setConnectionState] = useState('idle');
  const [lastError, setLastError] = useState(null);
  const [lastConnectionError, setLastConnectionError] = useState(null);
  const [backlogWarning, setBacklogWarning] = useState(null);
  const [isPaused, setIsPaused] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [includeDebug, setIncludeDebug] = useState(includeDebugInitial);

  const eventSourceRef = useRef(null);
  const pausedBufferRef = useRef([]);
  const isPausedRef = useRef(false);
  const counterRef = useRef(0);

  const normaliseEntry = useCallback((entry) => {
    const level = typeof entry.level === 'string' ? entry.level.toUpperCase() : 'INFO';
    const id = `log-${++counterRef.current}`;
    return {
      id,
      level: LEVELS.has(level) ? level : 'INFO',
      message: entry.message ?? '',
      timestamp: entry.timestamp ?? new Date().toISOString(),
      logger: entry.logger ?? '',
      module: entry.module ?? '',
    };
  }, []);

  const appendEntries = useCallback(
    (entries) => {
      if (!entries.length) {
        return;
      }
      setLogs((prev) => {
        let next = prev.concat(entries);
        if (next.length > capacity) {
          next = next.slice(next.length - capacity);
        }
        return next;
      });
    },
    [capacity]
  );

  const clear = useCallback(() => {
    pausedBufferRef.current = [];
    setLogs([]);
  }, []);

  const flushPaused = useCallback(() => {
    if (pausedBufferRef.current.length === 0) {
      return;
    }
    const buffered = pausedBufferRef.current;
    pausedBufferRef.current = [];
    appendEntries(buffered);
  }, [appendEntries]);

  const pause = useCallback(() => {
    isPausedRef.current = true;
    setIsPaused(true);
  }, []);

  const resume = useCallback(() => {
    isPausedRef.current = false;
    setIsPaused(false);
    flushPaused();
  }, [flushPaused]);

  const pushLog = useCallback(
    (entry) => {
      const normalised = normaliseEntry(entry);
      if (
        normalised.logger === 'log_stream' &&
        typeof normalised.message === 'string' &&
        /backlog/i.test(normalised.message)
      ) {
        setBacklogWarning({
          message: normalised.message,
          timestamp: normalised.timestamp,
        });
      }
      if (isPausedRef.current) {
        const next = pausedBufferRef.current.concat(normalised);
        if (next.length > capacity) {
          pausedBufferRef.current = next.slice(next.length - capacity);
        } else {
          pausedBufferRef.current = next;
        }
        return;
      }
      appendEntries([normalised]);
    },
    [appendEntries, capacity, normaliseEntry]
  );

  useEffect(() => {
    if (!isPaused) {
      flushPaused();
    }
  }, [isPaused, flushPaused]);

  useEffect(() => {
    setConnectionState('connecting');
    setLastError(null);
    setLastConnectionError(null);

    const source = openLogStream(includeDebug ? 'DEBUG' : 'INFO');
    eventSourceRef.current = source;

    source.onopen = () => {
      setConnectionState('open');
      setLastConnectionError(null);
    };

    source.onerror = () => {
      setConnectionState('reconnecting');
      const message = 'Connection interrupted. Attempting to reconnect…';
      setLastConnectionError(message);
    };

    const handleLog = (event) => {
      try {
        const payload = JSON.parse(event.data);
        pushLog({
          level: payload.level,
          message: payload.message,
          timestamp: payload.timestamp,
          logger: payload.logger,
          module: payload.module,
        });
      } catch (error) {
        setLastError(`Failed to parse log event: ${error.message}`);
      }
    };

    source.addEventListener('log', handleLog);

    return () => {
      source.removeEventListener('log', handleLog);
      source.close();
      eventSourceRef.current = null;
      setConnectionState('closed');
    };
  }, [includeDebug, pushLog]);

  const state = useMemo(
    () => ({
      logs,
      connectionState,
      lastError,
      lastConnectionError,
      backlogWarning,
      isPaused,
      autoScroll,
      includeDebug,
    }),
    [
      logs,
      connectionState,
      lastError,
      lastConnectionError,
      backlogWarning,
      isPaused,
      autoScroll,
      includeDebug
    ]
  );

  return {
    ...state,
    setAutoScroll,
    setIncludeDebug,
    clear,
    pause,
    resume,
    pushLog,
    acknowledgeBacklogWarning: () => setBacklogWarning(null)
  };
}
