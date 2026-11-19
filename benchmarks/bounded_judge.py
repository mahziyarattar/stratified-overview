"""A judge client for BenchmarkQED that respects a shared endpoint's limits.

BenchmarkQED launches every judge call at once (``asyncio.gather``), and its
``concurrent_requests`` setting only limits how many *start* per second. When
the endpoint slows down, open requests pile up (rate x latency): a run with
2 starts/s reached 500 open connections. This class enforces two limits:

- ``JUDGE_MAX_IN_FLIGHT`` (default 14): requests open at once, retries included.
  Kept below the agreed 16 connections: the HTTP client also holds idle
  connections for reuse, which run_judge.sh counts against its hard stop.
- ``JUDGE_MAX_RPM`` (default 25; the key allows 30): HTTP attempts started per minute, retries
  included. Attempts are spaced evenly (60 / RPM seconds apart), so no
  60-second window exceeds it. Pacing wraps ``litellm.acompletion``, the call
  graphrag-llm's retry middleware repeats, so every attempt is counted.

It also reports what BenchmarkQED does not: progress (a line on stderr every
50 finished calls, ``judge_progress.json`` in the working directory every 10)
and failed attempts by error type, each type logged the first time it occurs.
Set ``JUDGE_EXPECTED_CALLS`` to get a percentage and an ETA.

Used through the configs' ``custom_providers`` entry; run BenchmarkQED with
this folder on PYTHONPATH (benchmarks/run_judge.sh does both).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
from graphrag_llm.completion.lite_llm_completion import LiteLLMCompletion

MAX_IN_FLIGHT = int(os.environ.get("JUDGE_MAX_IN_FLIGHT", "14"))
MAX_RPM = float(os.environ.get("JUDGE_MAX_RPM", "25"))  # endpoint key limit: 30/min
EXPECTED_CALLS = int(os.environ.get("JUDGE_EXPECTED_CALLS", "0")) or None
PROGRESS_FILE = Path("judge_progress.json")
LOG_EVERY, WRITE_EVERY = 50, 10


@dataclass
class _Limits:
    """Per-event-loop limits: asyncio primitives are bound to the loop that uses them."""

    in_flight: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(MAX_IN_FLIGHT))
    spacing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    next_start: float = 0.0

    async def wait_for_start_slot(self) -> None:
        async with self.spacing_lock:
            now = time.monotonic()
            if self.next_start > now:
                await asyncio.sleep(self.next_start - now)
            self.next_start = max(now, self.next_start) + 60.0 / MAX_RPM


class _Progress:
    """Counts finished calls across the whole process (all question sets)."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.in_flight = self.succeeded = self.failed = self.attempts = 0
        self.total_seconds = 0.0
        self.attempt_errors: Counter[str] = Counter()

    def record_attempt_error(self, error: BaseException) -> None:
        status = getattr(error, "status_code", None)
        kind = f"{type(error).__name__}{f' ({status})' if status else ''}"
        if kind not in self.attempt_errors:
            print(f"[judge] first {kind}: {str(error)[:300]}", file=sys.stderr, flush=True)
        self.attempt_errors[kind] += 1

    @property
    def finished(self) -> int:
        return self.succeeded + self.failed

    def record(self, seconds: float, ok: bool) -> None:
        self.total_seconds += seconds
        if ok:
            self.succeeded += 1
        else:
            self.failed += 1
        # Also save whenever nothing is in flight, so the file is exact at the end of a run.
        if self.finished % WRITE_EVERY == 0 or self.in_flight == 0:
            self._write()
        if self.finished % LOG_EVERY == 0:
            print(f"[judge progress] {self.summary()}", file=sys.stderr, flush=True)

    def snapshot(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started_at
        rate = self.finished / elapsed if elapsed else 0.0
        snap: dict[str, Any] = {
            "finished": self.finished,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "in_flight": self.in_flight,
            "attempts": self.attempts,
            "attempt_errors": dict(self.attempt_errors),
            "avg_seconds_per_call": round(self.total_seconds / self.finished, 1) if self.finished else None,
            "calls_per_minute": round(rate * 60, 1),
            "elapsed_minutes": round(elapsed / 60, 1),
            "expected_calls": EXPECTED_CALLS,
        }
        if EXPECTED_CALLS:
            snap["percent"] = round(100 * self.finished / EXPECTED_CALLS, 1)
            remaining = max(EXPECTED_CALLS - self.finished, 0)
            snap["eta_minutes"] = round(remaining / rate / 60, 1) if rate else None
        return snap

    def summary(self) -> str:
        s = self.snapshot()
        parts = [f"{s['finished']} done ({s['failed']} failed)"]
        if EXPECTED_CALLS:
            parts.append(f"{s['percent']}% of {EXPECTED_CALLS}, ETA {s['eta_minutes']} min")
        parts.append(f"avg {s['avg_seconds_per_call']} s/call, {s['in_flight']} in flight")
        if s["attempt_errors"]:
            parts.append(f"{s['attempts']} attempts, errors {s['attempt_errors']}")
        return " | ".join(parts)

    def _write(self) -> None:
        try:
            PROGRESS_FILE.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        except OSError:
            pass  # progress reporting must never break a judging run


_limits: dict[asyncio.AbstractEventLoop, _Limits] = {}
_progress = _Progress()


def _loop_limits() -> _Limits:
    loop = asyncio.get_running_loop()
    if loop not in _limits:
        _limits[loop] = _Limits()
    return _limits[loop]


_original_acompletion = litellm.acompletion


async def _paced_acompletion(*args: Any, **kwargs: Any) -> Any:
    """Every HTTP attempt, retries included: paced to the RPM limit and error-counted."""
    await _loop_limits().wait_for_start_slot()
    _progress.attempts += 1
    try:
        return await _original_acompletion(*args, **kwargs)
    except Exception as error:
        _progress.record_attempt_error(error)
        raise


litellm.acompletion = _paced_acompletion


class BoundedLiteLLMCompletion(LiteLLMCompletion):
    """LiteLLM completion with caps on requests in flight and per minute, plus progress."""

    async def completion_async(self, /, **kwargs: Any) -> Any:
        limits = _loop_limits()
        async with limits.in_flight:
            _progress.in_flight += 1
            started, ok = time.monotonic(), False
            try:
                result = await super().completion_async(**kwargs)
                ok = True
                return result
            finally:
                _progress.in_flight -= 1
                _progress.record(time.monotonic() - started, ok)
