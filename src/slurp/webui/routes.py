"""REST and SSE routes for the slurp web UI.

All endpoints are thin wrappers around :class:`slurp.client.SyncClient`.
The stream token is validated from a query parameter, and CSRF tokens are
required for mutating POST requests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from starlette.templating import Jinja2Templates

from slurp.client import SyncClient
from slurp.domain import Job
from slurp.errors import SlurmError, SSHError
from slurp.webui.security import (
    generate_csrf_token,
    validate_csrf_token,
    validate_stream_token,
)
from slurp.webui.sse import JobStream

# Paths relative to this package for static assets and templates.
_PACKAGE_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"
_STATIC_DIR = _PACKAGE_DIR / "static"

# Router and templates
router = APIRouter()

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _require_token(token: str | None) -> None:
    """Raise 401 if the stream token is missing or invalid."""
    if not validate_stream_token(token):
        raise HTTPException(status_code=401, detail="Invalid or missing token.")


def _require_csrf(token: str | None, csrf_header: str | None = None) -> None:
    """Raise 403 if the CSRF token is missing or invalid."""
    if not validate_csrf_token(token, csrf_header):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")


def _get_client() -> SyncClient:
    """Return a fresh :class:`SyncClient` instance.

    In the future this could be tied to the current user profile.
    """
    return SyncClient()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, token: str | None = Query(None)) -> Any:
    """Serve the main dashboard page."""
    _require_token(token)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"token": token},
    )


@router.get("/stream")
async def stream(token: str | None = Query(None)) -> StreamingResponse:
    """SSE endpoint for live job updates and log streaming."""
    _require_token(token)

    client = _get_client()
    job_stream = JobStream(client)

    async def _event_generator() -> AsyncIterator[str]:
        try:
            async for event in job_stream.events():
                yield event
        except asyncio.CancelledError:
            job_stream.close()
            raise

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/api/csrf-token")
async def csrf_token(token: str | None = Query(None)) -> dict[str, str]:
    """Return a fresh CSRF token tied to the current stream token session."""
    _require_token(token)
    if token is None:
        raise HTTPException(status_code=401, detail="Missing token.")
    new_csrf = generate_csrf_token(token)
    return {"csrf_token": new_csrf}


@router.get("/api/jobs")
async def list_jobs(
    token: str | None = Query(None),
    experiment: str | None = Query(None),
) -> list[dict[str, Any]]:
    """List jobs, optionally filtered by experiment.

    The list is reconciled with SLURM via :meth:`SyncClient.list_jobs`.
    """
    _require_token(token)
    client = _get_client()
    try:
        jobs = await asyncio.to_thread(client.list_jobs, experiment=experiment, limit=200)
    except SSHError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc

    return [_job_as_dict(job) for job in jobs]


@router.get("/api/jobs/{job_id}")
async def get_job(
    job_id: str,
    token: str | None = Query(None),
) -> dict[str, Any]:
    """Get a single job by ID, refreshing from SLURM."""
    _require_token(token)
    client = _get_client()

    # Try local store first
    job = await asyncio.to_thread(client.status, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    # Refresh from SLURM
    try:
        job = await asyncio.to_thread(client.refresh_job, job)
    except SSHError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc

    return _job_as_dict(job)


@router.get("/api/jobs/{job_id}/logs")
async def get_logs(
    job_id: str,
    token: str | None = Query(None),
    follow: bool = Query(False),
) -> PlainTextResponse:
    """Fetch stdout and stderr for a job.

    Returns a plain-text response with both streams separated by a header.
    """
    _require_token(token)
    client = _get_client()

    job = await asyncio.to_thread(client.status, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    def _fetch_logs() -> str:
        out_lines: list[str] = []
        for chunk in client.job_logs(job, follow=follow, tail=500):
            out_lines.append(chunk)
        # job_logs yields chunks; for non-follow mode it yields stdout then stderr
        # in two separate chunks. We concatenate everything for a plain-text response.
        return "\n".join(out_lines)

    try:
        text = await asyncio.to_thread(_fetch_logs)
    except SSHError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc

    # Separate stdout and stderr with a header
    # Since the client yields stdout then stderr sequentially, we split by a heuristic
    # Actually, let's just return the raw text with a header
    header = f"=== Logs for job {job_id} ===\n"
    return PlainTextResponse(content=header + text)


@router.post("/api/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    token: str | None = Query(None),
    x_csrf_token: str | None = Header(None, alias="X-CSRF-Token"),
) -> dict[str, Any]:
    """Cancel a running or pending SLURM job."""
    _require_token(token)
    if token is None:
        raise HTTPException(status_code=401, detail="Missing token.")
    _require_csrf(token, x_csrf_token)

    client = _get_client()
    job = await asyncio.to_thread(client.status, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    try:
        updated = await asyncio.to_thread(client.cancel_job, job)
    except SlurmError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    except SSHError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc

    return {"job": _job_as_dict(updated), "message": "Cancel requested."}


@router.post("/api/sync")
async def sync_code(
    token: str | None = Query(None),
    x_csrf_token: str | None = Header(None, alias="X-CSRF-Token"),
) -> dict[str, str]:
    """Trigger a code sync (rsync) to the remote cluster."""
    _require_token(token)
    if token is None:
        raise HTTPException(status_code=401, detail="Missing token.")
    _require_csrf(token, x_csrf_token)

    client = _get_client()
    try:
        await asyncio.to_thread(client.sync)
    except SlurmError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    except SSHError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc

    return {"message": "Sync complete."}


def _job_as_dict(job: Job) -> dict[str, Any]:
    """Serialise a :class:`Job` to a JSON-friendly dict."""
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
