"""Atomic JSON job store with file locking."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from datetime import UTC
from pathlib import Path
from typing import Any

from slurp.domain import JobRecord, JobStatus, LogOffsetRecord
from slurp.errors import ConfigError, SlurpError

DEFAULT_STORE_DIR = Path.home() / ".local" / "share" / "slurp"
DEFAULT_STORE_PATH = DEFAULT_STORE_DIR / "jobs.json"
DEFAULT_OFFSET_PATH = DEFAULT_STORE_DIR / "log_offsets.json"


class JobStore:
    """Atomic read-modify-write of the local job cache."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_STORE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.parent / ".store.lock"

    def _acquire_lock(self) -> int:
        """Acquire an exclusive file lock. Returns the fd."""
        fd = os.open(str(self._lock_path), os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as exc:
            os.close(fd)
            raise SlurpError(
                f"Failed to acquire store lock: {exc}",
                hint="Another slurp process may be holding the job store. Wait and retry.",
            )
        return fd

    def _release_lock(self, fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def read(self) -> dict[str, Any]:
        """Read the store, returning a dict with jobs, log_offsets, idempotency."""
        if not self.path.exists():
            return {"jobs": {}, "log_offsets": {}, "idempotency": {}}
        try:
            with open(self.path) as f:
                data: dict[str, Any] = json.load(f)
        except json.JSONDecodeError:
            # Corruption recovery: move corrupt file and start fresh
            corrupt_path = self.path.with_suffix(f".json.corrupt.{os.getpid()}")
            self.path.rename(corrupt_path)
            return {"jobs": {}, "log_offsets": {}, "idempotency": {}}
        except FileNotFoundError:
            return {"jobs": {}, "log_offsets": {}, "idempotency": {}}
        return data

    def _atomic_write(self, data: dict[str, Any]) -> None:
        """Write data atomically using temp file + rename."""
        # Clean up stale temp files
        for tmp in self.path.parent.glob("*.tmp.*"):
            try:
                tmp.unlink()
            except OSError:
                pass

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.path.parent, prefix="jobs.json.tmp.", suffix=f".{os.getpid()}"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, sort_keys=True, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, self.path)
        except OSError as exc:
            raise ConfigError(
                f"Failed to write job store: {exc}",
                hint="Check disk space and permissions for ~/.local/share/slurp/",
            )

    def write_jobs(self, jobs: dict[str, JobRecord]) -> None:
        """Replace the entire jobs section."""
        fd = self._acquire_lock()
        try:
            data = self.read()
            data["jobs"] = {
                jid: record.model_dump(mode="json") for jid, record in jobs.items()
            }
            self._atomic_write(data)
        finally:
            self._release_lock(fd)

    def append_job(self, record: JobRecord) -> None:
        """Add or update a single job record."""
        fd = self._acquire_lock()
        try:
            data = self.read()
            data["jobs"][record.job_id] = record.model_dump(mode="json")
            if record.idempotency_hash:
                data["idempotency"][record.idempotency_hash] = {
                    "job_id": record.job_id,
                    "submitted_at": record.submitted_at.isoformat(),
                }
            self._prune_idempotency(data)
            self._atomic_write(data)
        finally:
            self._release_lock(fd)

    def update_job_status(self, job_id: str, status: JobStatus) -> None:
        """Update just the status of a job."""
        fd = self._acquire_lock()
        try:
            data = self.read()
            if job_id in data.get("jobs", {}):
                data["jobs"][job_id]["status"] = status.value
                self._atomic_write(data)
        finally:
            self._release_lock(fd)

    def get_job(self, job_id: str) -> JobRecord | None:
        """Get a single job record."""
        data = self.read()
        raw = data.get("jobs", {}).get(job_id)
        if raw is None:
            return None
        return JobRecord.model_validate(raw)

    def list_jobs(self) -> dict[str, JobRecord]:
        """Return all job records."""
        data = self.read()
        return {
            jid: JobRecord.model_validate(raw)
            for jid, raw in data.get("jobs", {}).items()
        }

    def check_idempotency(self, hash_key: str, window_seconds: float = 30.0) -> str | None:
        """Return job_id if a matching hash exists within the window."""
        data = self.read()
        entry: dict[str, Any] | None = data.get("idempotency", {}).get(hash_key)
        if entry is None:
            return None
        from datetime import datetime

        submitted = datetime.fromisoformat(entry["submitted_at"])
        age = (datetime.now(UTC) - submitted).total_seconds()
        if age > window_seconds:
            return None
        # Also check if the job is still pending
        job = data.get("jobs", {}).get(entry["job_id"])
        if job and job.get("status") == "PENDING":
            job_id: str | None = entry.get("job_id")
            return job_id
        return None

    def _prune_idempotency(self, data: dict[str, Any]) -> None:
        """Remove idempotency entries older than 30 seconds."""
        from datetime import datetime

        now = datetime.now(UTC)
        stale = []
        for hash_key, entry in list(data.get("idempotency", {}).items()):
            try:
                submitted = datetime.fromisoformat(entry["submitted_at"])
                if (now - submitted).total_seconds() > 30.0:
                    stale.append(hash_key)
            except (ValueError, KeyError):
                stale.append(hash_key)
        for k in stale:
            data["idempotency"].pop(k, None)


class LogOffsetStore:
    """Best-effort store for log byte offsets."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_OFFSET_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict[str, LogOffsetRecord]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
        return {k: LogOffsetRecord.model_validate(v) for k, v in raw.items()}

    def write(self, offsets: dict[str, LogOffsetRecord]) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.path.parent, prefix="log_offsets.json.tmp.", suffix=f".{os.getpid()}"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(
                    {k: v.model_dump(mode="json") for k, v in offsets.items()},
                    f,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, self.path)
        except OSError:
            pass  # Best-effort; do not crash on log offset write failure

    def get_offset(self, job_id: str) -> LogOffsetRecord:
        return self.read().get(job_id, LogOffsetRecord())

    def set_offset(self, job_id: str, out: int, err: int) -> None:
        offsets = self.read()
        from datetime import datetime

        offsets[job_id] = LogOffsetRecord(
            out=out, err=err, last_read=datetime.now(UTC)
        )
        self.write(offsets)
