"""Utilities for streaming application logs to server-sent event clients."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Set

_LOG_LEVEL_NAMES = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "ERROR",
}


@dataclass(slots=True)
class LogEvent:
    """Structured payload sent to frontend clients."""

    level: str
    message: str
    timestamp: str
    logger: str
    module: str
    levelno: int

    def to_json(self) -> str:
        payload = {
            "level": self.level,
            "message": self.message,
            "timestamp": self.timestamp,
            "logger": self.logger,
            "module": self.module,
        }
        return json.dumps(payload)


class LogStreamManager:
    """Broadcasts log events to registered asyncio queues."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._maxsize = maxsize
        self._queues: Set[asyncio.Queue[LogEvent]] = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_drop_notice: float = 0.0

    def register(self) -> asyncio.Queue[LogEvent]:
        queue: asyncio.Queue[LogEvent] = asyncio.Queue(maxsize=self._maxsize)
        loop = asyncio.get_running_loop()
        with self._lock:
            self._queues.add(queue)
            self._loop = loop
        return queue

    def unregister(self, queue: asyncio.Queue[LogEvent]) -> None:
        with self._lock:
            self._queues.discard(queue)

    def _broadcast(self, event: LogEvent) -> None:
        queues = list(self._queues)
        now = time.monotonic()
        notice_needed = False
        for queue in queues:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                else:
                    notice_needed = True
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Extremely defensive: if dropping still leaves queue full we skip.
                continue

        if notice_needed and now - self._last_drop_notice > 60:
            self._last_drop_notice = now
            warning_event = LogEvent(
                level="WARNING",
                message=(
                    "Log stream backlog reached capacity; oldest entries were discarded."
                ),
                timestamp=datetime.now(timezone.utc).isoformat(),
                logger="log_stream", 
                module="log_stream",
                levelno=logging.WARNING,
            )
            for queue in queues:
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        continue
                try:
                    queue.put_nowait(warning_event)
                except asyncio.QueueFull:
                    continue

    def publish(self, event: LogEvent) -> None:
        with self._lock:
            loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._broadcast, event)


def build_event(record: logging.LogRecord) -> Optional[LogEvent]:
    levelno = record.levelno
    level = _LOG_LEVEL_NAMES.get(levelno)
    if level is None:
        # Ignore custom levels; the frontend only understands the canonical set.
        return None
    timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
    message = record.getMessage()
    return LogEvent(
        level=level,
        message=message,
        timestamp=timestamp,
        logger=record.name,
        module=getattr(record, "module", ""),
        levelno=levelno,
    )


class LogStreamHandler(logging.Handler):
    """Logging handler that forwards records to the shared stream manager."""

    def __init__(self, manager: LogStreamManager) -> None:
        super().__init__(logging.DEBUG)
        self.manager = manager

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - exercised in tests
        try:
            event = build_event(record)
        except Exception:
            # We deliberately swallow formatting errors to avoid interfering with
            # the application's main logging pipeline.
            self.handleError(record)
            return
        if event is None:
            return
        self.manager.publish(event)


log_stream_manager = LogStreamManager(maxsize=1000)
