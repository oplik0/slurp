"""Domain models for slurp — pure data, no I/O."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, field_validator


class JobStatus(StrEnum):
    """SLURM job status enumeration."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        }


class ResourceRequest(BaseModel):
    """Normalized resource specification for a SLURM job."""

    gpus: int = 0
    nodes: int = 1
    time: str = "2:00:00"
    mem: str | None = None
    cpus: int = 8
    partition: str | None = None
    account: str | None = None
    constraint: str | None = None
    qos: str | None = None
    mail_type: str | None = None
    job_name: str | None = None
    slurm_kwargs: dict[str, str] = Field(default_factory=dict)

    @field_validator("time")
    @classmethod
    def _validate_time(cls, v: str) -> str:
        # Accept [[HH:]MM:]SS or D-HH:MM:SS
        if not re.match(r"^(\d+-)?\d*\d:\d{2}:\d{2}$|^\d+$|^\d+:\d{2}$", v):
            raise ValueError(f"Invalid SLURM time format: {v}")
        return v

    @field_validator("job_name")
    @classmethod
    def _validate_job_name(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            v = re.sub(r"[^a-zA-Z0-9_-]+", "-", v).strip("-")
        return v

    @field_validator("gpus")
    @classmethod
    def _validate_gpus(cls, v: int) -> int:
        if v < 0:
            raise ValueError("gpus must be non-negative")
        return v

    @field_validator("nodes")
    @classmethod
    def _validate_nodes(cls, v: int) -> int:
        if v < 1:
            raise ValueError("nodes must be positive")
        return v

    @field_validator("cpus")
    @classmethod
    def _validate_cpus(cls, v: int) -> int:
        if v < 1:
            raise ValueError("cpus must be positive")
        return v


class Profile(BaseModel):
    """Parsed representation of a TOML profile section."""

    name: str
    hostname: str
    username: str | None = None
    key_file: str | None = None
    proxy_jump: str | None = None
    partition: str | None = None
    account: str | None = None
    prologue: str = ""
    mpi_mode: str = "pmi2"
    cpu_bind: str = "cores"
    gpu_flag_style: str = "gres"  # "gres" or "gpus"
    sync: SyncConfig | None = None

    class SyncConfig(BaseModel):
        local: str
        remote: str

    def format_prologue(self) -> str:
        """Return prologue with {account} placeholder substituted."""
        return self.prologue.format(account=self.account or "")


class Job(BaseModel):
    """Immutable handle representing a single SLURM job."""

    job_id: str
    name: str
    status: JobStatus
    profile: str
    experiment: str | None = None
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    command: str = ""
    resources: ResourceRequest = Field(default_factory=ResourceRequest)
    working_dir: str = ""

    # Internal cache for JobResult (keyed by job_id)
    _result_cache: dict[str, JobResult] = PrivateAttr(default_factory=dict)

    def refresh(self) -> Job:
        """Query sacct for the current state and return a new Job."""
        # Implemented in client.py to avoid circular imports
        from slurp.client import SyncClient

        return SyncClient(profile=self.profile).refresh_job(self)

    def wait(
        self,
        *,
        timeout: str | float | None = None,
        follow_logs: bool = False,
        poll_interval: float = 5.0,
    ) -> JobResult:
        from slurp.client import SyncClient

        return SyncClient(profile=self.profile).wait_job(
            self, timeout=timeout, follow_logs=follow_logs, poll_interval=poll_interval
        )

    def logs(self, *, follow: bool = False, tail: int = 100) -> Any:
        from slurp.client import SyncClient

        return SyncClient(profile=self.profile).job_logs(self, follow=follow, tail=tail)

    def cancel(self) -> Job:
        from slurp.client import SyncClient

        return SyncClient(profile=self.profile).cancel_job(self)

    def result(self) -> JobResult:
        from slurp.client import SyncClient

        return SyncClient(profile=self.profile).job_result(self)


class JobResult(BaseModel):
    """Terminal state capture for a job."""

    job_id: str
    status: JobStatus
    exit_code: int | None = None
    stdout: str = ""  # Capped at 1MB
    stderr: str = ""  # Capped at 1MB
    max_rss_mb: float | None = None
    wall_time: float = 0.0  # seconds


class ArrayJob(BaseModel):
    """Handle for a SLURM job array."""

    array_job_id: str
    name: str
    profile: str
    experiment: str | None = None
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    task_count: int
    throttle: int = 20

    def watch(self) -> None:
        from slurp.client import SyncClient

        return SyncClient(profile=self.profile).watch_array(self)

    def logs(
        self,
        *,
        task_id: int | None = None,
        follow: bool = False,
        tail: int = 100,
    ) -> Any:
        from slurp.client import SyncClient

        return SyncClient(profile=self.profile).array_logs(
            self, task_id=task_id, follow=follow, tail=tail
        )

    def cancel(self) -> ArrayJob:
        from slurp.client import SyncClient

        return SyncClient(profile=self.profile).cancel_array(self)

    def cancel_task(self, task_id: int) -> ArrayJob:
        from slurp.client import SyncClient

        return SyncClient(profile=self.profile).cancel_array_task(self, task_id)

    def results(
        self,
        *,
        timeout: str | float | None = None,
        poll_interval: float = 5.0,
    ) -> list[JobResult]:
        from slurp.client import SyncClient

        return SyncClient(profile=self.profile).array_results(
            self, timeout=timeout, poll_interval=poll_interval
        )

    def tasks(self) -> list[Job]:
        from slurp.client import SyncClient

        return SyncClient(profile=self.profile).array_tasks(self)


class JobRecord(BaseModel):
    """Record stored in the local job cache."""

    job_id: str
    name: str
    status: JobStatus
    profile: str
    experiment: str | None = None
    submitted_at: datetime
    command: str
    resources: ResourceRequest
    working_dir: str
    idempotency_hash: str | None = None
    idempotency_time: datetime | None = None


class LogOffsetRecord(BaseModel):
    """Record stored in log_offsets.json."""

    out: int = 0
    err: int = 0
    last_read: datetime | None = None


class SlurmJobInfo(BaseModel):
    """Parsed sacct output for a single job."""

    job_id: str
    state: str
    exit_code: str  # "0:0", "1:0", etc.
    max_rss: str | None = None
    elapsed: str | None = None

    @property
    def status(self) -> JobStatus:
        state_upper = self.state.upper()
        mapping = {
            "PENDING": JobStatus.PENDING,
            "RUNNING": JobStatus.RUNNING,
            "COMPLETED": JobStatus.COMPLETED,
            "FAILED": JobStatus.FAILED,
            "CANCELLED": JobStatus.CANCELLED,
            "CANCELLED+": JobStatus.CANCELLED,
            "TIMEOUT": JobStatus.TIMEOUT,
            "TIMEOUT+": JobStatus.TIMEOUT,
            "NODE_FAIL": JobStatus.FAILED,
            "OUT_OF_MEMORY": JobStatus.FAILED,
        }
        return mapping.get(state_upper, JobStatus.UNKNOWN)

    @property
    def exit_code_int(self) -> int | None:
        """Return the job exit code (first part of 'exit_code:signal')."""
        if not self.exit_code:
            return None
        try:
            return int(self.exit_code.split(":")[0])
        except (ValueError, IndexError):
            return None


def slugify_command(command: str) -> str:
    """Derive a job name slug from a command string."""
    # Remove path prefix and arguments, keep base name
    base = command.split()[0] if command else "job"
    base = Path(base).name
    base = re.sub(r"[^a-zA-Z0-9_-]+", "-", base).strip("-")
    return base or "job"
