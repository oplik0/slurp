"""Server-Sent Events (SSE) stream for live job updates.

Implements a custom ``JobStream`` that polls ``SyncClient`` and yields
SSE-formatted deltas over an async generator.  Since ``SyncClient`` is
synchronous, blocking calls are offloaded to a thread pool via
``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from slurp.client import SyncClient
from slurp.domain import Job
from slurp.errors import SSHError

# Polling intervals (seconds)
_POLL_INTERVAL = 5.0
_HEARTBEAT_INTERVAL = 15.0


def _sse_event(event: str, data: dict[str, Any]) -> str:
    """Format a single SSE event.

    Args:
        event: The event name (e.g. ``job_update``).
        data: JSON-serialisable payload.

    Returns:
        A string ready to be yielded by a streaming response.
    """
    lines = f"event: {event}\n"
    for chunk in json.dumps(data, separators=(",", ":")).splitlines():
        lines += f"data: {chunk}\n"
    lines += "\n"
    return lines


def _job_to_dict(job: Job) -> dict[str, Any]:
    """Serialise a :class:`slurp.domain.Job` to a plain dict."""
    return {
        "job_id": job.job_id,
        "name": job.name,
        "status": job.status.value,
        "profile": job.profile,
        "experiment": job.experiment,
        "command": job.command,
        "resources": job.resources.model_dump(mode="json"),
        "working_dir": job.working_dir,
    }


class JobStream:
    """Async SSE generator that polls SLURM state and emits deltas.

    Parameters:
        client: The synchronous client used to query SLURM.
        experiment: Optional experiment filter passed to ``list_jobs``.
    """

    def __init__(self, client: SyncClient, experiment: str | None = None) -> None:
        self._client = client
        self._experiment = experiment
        self._last_jobs: dict[str, Job] = {}
        self._log_offsets: dict[str, int] = {}
        self._running = True

    async def _poll_jobs(self) -> list[Job]:
        """Return the current job list, offloaded to a thread."""
        return await asyncio.to_thread(
            self._client.list_jobs,
            experiment=self._experiment,
            limit=200,
        )

    async def _poll_logs(self, job: Job) -> str:
        """Return new log text for *job* since the last offset.

        Uses ``job_logs`` with ``follow=False`` and a large tail, then
        discards everything before the stored byte offset.  The offset is
        updated after a successful read.
        """
        offset = self._log_offsets.get(job.job_id, 0)
        try:
            # Grab the last 500 lines; we will slice from the offset.
            lines: list[str] = []
            for chunk in await asyncio.to_thread(
                self._client.job_logs, job, follow=False, tail=500
            ):
                lines.append(chunk)
            full_text = "\n".join(lines)
        except Exception:
            return ""
        if len(full_text) <= offset:
            return ""
        new_text = full_text[offset:]
        self._log_offsets[job.job_id] = len(full_text)
        return new_text

    async def _refresh_job(self, job: Job) -> Job:
        """Reconcile a single job with SLURM via ``refresh_job``."""
        return await asyncio.to_thread(self._client.refresh_job, job)

    async def events(self) -> AsyncIterator[str]:
        """Yield SSE-formatted strings indefinitely (or until ``close()``)."""
        last_heartbeat = time.monotonic()

        while self._running:
            try:
                jobs = await self._poll_jobs()
            except SSHError as exc:
                yield _sse_event(
                    "server_error",
                    {"message": exc.message, "hint": exc.hint},
                )
                self._running = False
                return
            except Exception as exc:
                yield _sse_event(
                    "server_error",
                    {"message": str(exc), "hint": "Unexpected polling error."},
                )
                self._running = False
                return

            # Reconcile with SLURM and detect deltas
            current: dict[str, Job] = {}
            for job in jobs:
                refreshed = await self._refresh_job(job)
                current[refreshed.job_id] = refreshed

                prev = self._last_jobs.get(refreshed.job_id)
                if prev is None or prev.status != refreshed.status:
                    yield _sse_event(
                        "job_update",
                        {"job": _job_to_dict(refreshed)},
                    )

                # Logs
                new_log_text = await self._poll_logs(refreshed)
                if new_log_text:
                    yield _sse_event(
                        "log_append",
                        {
                            "job_id": refreshed.job_id,
                            "text": new_log_text,
                        },
                    )

            # Detect removed jobs
            for job_id in self._last_jobs:
                if job_id not in current:
                    yield _sse_event(
                        "job_update",
                        {
                            "job": {
                                "job_id": job_id,
                                "status": "REMOVED",
                            }
                        },
                    )

            self._last_jobs = current

            # Heartbeat
            now = time.monotonic()
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                yield _sse_event("heartbeat", {"time": now})
                last_heartbeat = now

            await asyncio.sleep(_POLL_INTERVAL)

    def close(self) -> None:
        """Signal the stream to stop on the next iteration."""
        self._running = False
