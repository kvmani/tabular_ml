"""Tests for the log streaming infrastructure."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Tuple

import pytest

from backend.app.api.routes.system import system_log_stream
from backend.app.log_stream import LogEvent, LogStreamHandler, LogStreamManager, log_stream_manager


async def _drain(queue: asyncio.Queue[LogEvent]) -> LogEvent:
    return await asyncio.wait_for(queue.get(), timeout=1)


def test_log_handler_serialises_records() -> None:
    manager = LogStreamManager(maxsize=5)

    async def _run() -> None:
        queue = manager.register()
        handler = LogStreamHandler(manager)

        logger = logging.getLogger('test.log.handler')
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(handler)

        logger.warning('Hello %s', 'world')

        event = await _drain(queue)

        assert event.level == 'WARNING'
        assert event.message == 'Hello world'
        assert event.timestamp.endswith('+00:00')
        assert event.logger == 'test.log.handler'

        logger.removeHandler(handler)
        manager.unregister(queue)

    asyncio.run(_run())


def test_manager_emits_backlog_warning_when_queues_overflow() -> None:
    manager = LogStreamManager(maxsize=1)

    async def _run() -> None:
        queue = manager.register()
        handler = LogStreamHandler(manager)

        logger = logging.getLogger('test.backpressure')
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(handler)

        try:
            logger.info('first event')
            logger.info('second event triggers drop')
            warning_event = None
            for _ in range(3):
                candidate = await _drain(queue)
                if candidate.level == 'WARNING':
                    warning_event = candidate
                    break
            assert warning_event is not None, 'Expected backlog warning event'
        finally:
            logger.removeHandler(handler)
            manager.unregister(queue)

        assert warning_event.levelno == logging.WARNING
        assert 'backlog' in warning_event.message

    asyncio.run(_run())
@pytest.mark.asyncio
async def test_system_log_stream_emits_events() -> None:
    received: dict[str, dict] = {}

    response = await system_log_stream('INFO')
    iterator = response.body_iterator

    async def _consume() -> None:
        buffer = ''
        async for chunk in iterator:
            buffer += chunk
            while '\n\n' in buffer:
                block, buffer = buffer.split('\n\n', 1)
                event_type, data = _parse_sse_block(block)
                if event_type != 'log':
                    continue
                received['payload'] = json.loads(data)
                return

    consumer = asyncio.create_task(_consume())

    try:
        await asyncio.sleep(0.2)
        log_event = LogEvent(
            level='WARNING',
            message='Stream verification',
            timestamp=datetime.now(timezone.utc).isoformat(),
            logger='test.system.logstream',
            module='tests',
            levelno=logging.WARNING,
        )
        log_stream_manager.publish(log_event)

        await asyncio.wait_for(consumer, timeout=5)
    finally:
        await iterator.aclose()

    assert 'payload' in received
    payload = received['payload']
    assert payload['message'] == 'Stream verification'
    assert payload['level'] == 'WARNING'
    assert payload['logger'] == 'test.system.logstream'
    assert 'timestamp' in payload


def _parse_sse_block(block: str) -> Tuple[str, str]:
    event_type = 'message'
    data_lines: list[str] = []
    for line in block.splitlines():
        if line.startswith(':'):
            continue
        if line.startswith('event:'):
            event_type = line[len('event:') :].strip()
        elif line.startswith('data:'):
            data_lines.append(line[len('data:') :].strip())
    return event_type, '\n'.join(data_lines)
