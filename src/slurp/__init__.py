"""slurp — A Python library and CLI for running ML jobs on SLURM clusters.

Public API:
    slurp.submit(...) -> Job
    slurp.submit_array(...) -> ArrayJob
    slurp.Experiment(name) -> experiment helper
    slurp.log_progress(...) -> write progress.jsonl

Exception classes:
    slurp.SlurpError
    slurp.SSHError
    slurp.SlurmError
    slurp.JobFailedError
    slurp.SyncError
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from slurp.client import SyncClient
from slurp.domain import (
    ArrayJob,
    Job,
    JobResult,
    JobStatus,
    Profile,
    ResourceRequest,
)
from slurp.errors import (
    JobFailedError,
    SlurmError,
    SlurpError,
    SSHError,
    SyncError,
)

__all__ = [
    "submit",
    "submit_array",
    "Experiment",
    "log_progress",
    # Types
    "Job",
    "JobResult",
    "JobStatus",
    "ArrayJob",
    "Profile",
    "ResourceRequest",
    # Exceptions
    "SlurpError",
    "SSHError",
    "SlurmError",
    "JobFailedError",
    "SyncError",
]


def submit(
    command: str,
    *,
    profile: str | None = None,
    experiment: str | None = None,
    name: str | None = None,
    gpus: int | None = None,
    nodes: int = 1,
    cpus: int | None = None,
    mem: str | None = None,
    time: str | None = None,
    partition: str | None = None,
    account: str | None = None,
    constraint: str | None = None,
    qos: str | None = None,
    depends_on: list[Job] | None = None,
    depends_on_type: str = "afterok",
    slurm_kwargs: dict[str, str] | None = None,
    working_dir: str | None = None,
    sync: bool = True,
    snapshot: bool = False,
) -> Job:
    """Submit a job to SLURM.

    Example:
        job = slurp.submit("python train.py --lr 0.01", gpus=4, time="2:00:00")
    """
    client = SyncClient(profile=profile)
    return client.submit(
        command,
        name=name,
        gpus=gpus,
        nodes=nodes,
        cpus=cpus,
        mem=mem,
        time=time,
        partition=partition,
        account=account,
        constraint=constraint,
        qos=qos,
        depends_on=depends_on,
        depends_on_type=depends_on_type,
        slurm_kwargs=slurm_kwargs,
        working_dir=working_dir,
        experiment=experiment,
        sync=sync,
        snapshot=snapshot,
    )


def submit_array(
    template: str,
    *,
    configs: list[dict[str, str]],
    throttle: int = 20,
    profile: str | None = None,
    experiment: str | None = None,
    **kwargs: Any,
) -> ArrayJob:
    """Submit a SLURM job array.

    Example:
        array = slurp.submit_array(
            "python train.py --seed {seed}",
            configs=[{"seed": str(i)} for i in range(5)],
            gpus=4,
        )
    """
    client = SyncClient(profile=profile)
    return client.submit_array(
        template,
        configs=configs,
        throttle=throttle,
        experiment=experiment,
        **kwargs,
    )


class Experiment:
    """Convenience wrapper that sets a default experiment tag on every job."""

    def __init__(self, name: str) -> None:
        self.name = name

    def submit(self, command: str, **kwargs: Any) -> Job:
        """Submit with experiment tag."""
        kwargs.setdefault("experiment", self.name)
        return submit(command, **kwargs)

    def submit_array(self, template: str, **kwargs: Any) -> ArrayJob:
        """Submit array with experiment tag."""
        kwargs.setdefault("experiment", self.name)
        return submit_array(template, **kwargs)

    def watch(self) -> None:
        """Watch jobs in this experiment."""
        SyncClient().watch(experiment=self.name)

    def cancel_all(self) -> None:
        """Cancel all jobs in this experiment."""
        client = SyncClient()
        jobs = client.list_jobs(experiment=self.name)
        for job in jobs:
            client.cancel_job(job)


def log_progress(
    step: int,
    *,
    total_steps: int | None = None,
    metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    path: str = "progress.jsonl",
) -> None:
    """Append a progress record to progress.jsonl.

    Example:
        slurp.log_progress(
            step=epoch,
            total_steps=100,
            metrics={"loss": loss.item(), "accuracy": acc},
        )
    """
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "step": step,
    }
    if total_steps is not None:
        record["total_steps"] = total_steps
    if metrics:
        record["metrics"] = metrics
    if metadata:
        record["metadata"] = metadata

    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


