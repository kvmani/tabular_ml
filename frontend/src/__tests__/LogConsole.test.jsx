import React from 'react';
import { render } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import LogConsole from '../components/LogConsole.jsx';

const baseStream = {
  logs: [
    {
      id: '1',
      level: 'INFO',
      message: 'Application boot complete',
      timestamp: '2024-01-01T00:00:00Z',
      logger: 'backend.main',
      module: 'main'
    },
    {
      id: '2',
      level: 'WARNING',
      message: 'Cache miss for dataset preview',
      timestamp: '2024-01-01T00:00:01Z',
      logger: 'services.cache',
      module: 'cache'
    },
    {
      id: '3',
      level: 'ERROR',
      message: 'Failed to load dataset metadata',
      timestamp: '2024-01-01T00:00:02Z',
      logger: 'services.data',
      module: 'data_manager'
    },
    {
      id: '4',
      level: 'DEBUG',
      message: 'Trainer heartbeat received',
      timestamp: '2024-01-01T00:00:03Z',
      logger: 'services.training',
      module: 'trainer'
    }
  ],
  connectionState: 'open',
  lastError: null,
  isPaused: false,
  autoScroll: true,
  includeDebug: true,
  setAutoScroll: vi.fn(),
  setIncludeDebug: vi.fn(),
  clear: vi.fn(),
  pause: vi.fn(),
  resume: vi.fn()
};

describe('LogConsole', () => {
  it('renders mixed severity entries', () => {
    const { container } = render(<LogConsole stream={baseStream} />);
    expect(container).toMatchSnapshot();
  });
});
