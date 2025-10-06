import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';
import useLogStream from '../hooks/useLogStream.js';
import { openLogStream } from '../api/client.js';

vi.mock('../api/client.js', () => ({
  openLogStream: vi.fn()
}));

class MockEventSource {
  static instances = [];

  constructor() {
    this.url = null;
    this.withCredentials = true;
    this.readyState = 0;
    this.listeners = new Map();
    this.onopen = null;
    this.onerror = null;
    MockEventSource.instances.push(this);
  }

  addEventListener(type, handler) {
    const existing = this.listeners.get(type) || new Set();
    existing.add(handler);
    this.listeners.set(type, existing);
  }

  removeEventListener(type, handler) {
    const existing = this.listeners.get(type);
    if (existing) {
      existing.delete(handler);
    }
  }

  emit(type, data) {
    const handlers = this.listeners.get(type) || [];
    handlers.forEach((handler) => handler({ data }));
  }

  dispatchOpen() {
    if (this.onopen) {
      this.onopen();
    }
  }

  dispatchError() {
    if (this.onerror) {
      this.onerror(new Event('error'));
    }
  }

  close() {
    this.readyState = 2;
  }
}

describe('useLogStream', () => {
  beforeEach(() => {
    MockEventSource.instances = [];
    openLogStream.mockImplementation(() => new MockEventSource());
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('appends log entries from the stream and respects the capacity cap', () => {
    const { result } = renderHook(() => useLogStream({ capacity: 2 }));
    const source = MockEventSource.instances[0];

    act(() => {
      source.emit('log', JSON.stringify({
        level: 'INFO',
        message: 'First',
        timestamp: '2024-01-01T00:00:00Z',
        logger: 'backend',
        module: 'main'
      }));
      source.emit('log', JSON.stringify({
        level: 'WARNING',
        message: 'Second',
        timestamp: '2024-01-01T00:00:01Z',
        logger: 'backend',
        module: 'main'
      }));
      source.emit('log', JSON.stringify({
        level: 'ERROR',
        message: 'Third',
        timestamp: '2024-01-01T00:00:02Z',
        logger: 'backend',
        module: 'main'
      }));
    });

    expect(result.current.logs).toHaveLength(2);
    expect(result.current.logs[0].message).toBe('Second');
    expect(result.current.logs[1].level).toBe('ERROR');
  });

  it('buffers logs while paused and flushes when resumed', () => {
    const { result } = renderHook(() => useLogStream({ capacity: 5 }));
    const source = MockEventSource.instances[0];

    act(() => {
      source.emit('log', JSON.stringify({
        level: 'INFO',
        message: 'Before pause',
        timestamp: '2024-01-01T00:00:00Z'
      }));
    });

    act(() => {
      result.current.pause();
    });

    act(() => {
      source.emit('log', JSON.stringify({
        level: 'WARNING',
        message: 'During pause',
        timestamp: '2024-01-01T00:00:01Z'
      }));
    });

    expect(result.current.logs).toHaveLength(1);

    act(() => {
      result.current.resume();
    });

    expect(result.current.logs).toHaveLength(2);
    expect(result.current.logs[1].message).toBe('During pause');
  });

  it('exposes manual push and debug toggling', () => {
    const { result, rerender } = renderHook(({ includeDebug }) =>
      useLogStream({ includeDebug })
    , {
      initialProps: { includeDebug: false }
    });

    expect(openLogStream).toHaveBeenCalledWith('INFO');

    act(() => {
      result.current.pushLog({ message: 'Manual error', level: 'ERROR' });
    });

    expect(result.current.logs).toHaveLength(1);

    act(() => {
      result.current.setIncludeDebug(true);
    });

    rerender({ includeDebug: true });

    expect(openLogStream).toHaveBeenLastCalledWith('DEBUG');
  });
});
