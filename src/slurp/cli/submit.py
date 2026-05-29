"""CLI submit commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.syntax import Syntax

from slurp.client import SyncClient

console = Console()
app = typer.Typer()


def _common_opts(
    profile: str = typer.Option(None, "--profile"),
    gpus: int = typer.Option(0, "--gpus"),
    nodes: int = typer.Option(1, "--nodes"),
    cpus: int = typer.Option(8, "--cpus"),
    mem: str = typer.Option(None, "--mem"),
    time: str = typer.Option(None, "--time"),
    partition: str = typer.Option(None, "--partition"),
    account: str = typer.Option(None, "--account"),
    constraint: str = typer.Option(None, "--constraint"),
    qos: str = typer.Option(None, "--qos"),
    mail_type: str = typer.Option(None, "--mail-type"),
    job_name: str = typer.Option(None, "--job-name"),
    experiment: str = typer.Option(None, "--experiment"),
    snapshot: bool = typer.Option(False, "--snapshot"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    slurm_kwargs: list[str] = typer.Option([], "--slurm-kwargs"),  # noqa: B008
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "profile": profile,
        "gpus": gpus,
        "nodes": nodes,
        "cpus": cpus,
        "mem": mem,
        "time": time,
        "partition": partition,
        "account": account,
        "constraint": constraint,
        "qos": qos,
        "mail_type": mail_type,
        "name": job_name,
        "experiment": experiment,
        "snapshot": snapshot,
        "slurm_kwargs": {},
    }
    for kw in slurm_kwargs:
        if "=" in kw:
            k, v = kw.split("=", 1)
            kwargs["slurm_kwargs"][k] = v
    return kwargs


@app.command(name="submit", help="Fire-and-forget job submission")
def submit_cmd(
    ctx: typer.Context,
    command: list[str] = typer.Argument(..., help="Command to run"),  # noqa: B008
    profile: str = typer.Option(None, "--profile"),  # noqa: B008
    gpus: int = typer.Option(0, "--gpus"),  # noqa: B008
    nodes: int = typer.Option(1, "--nodes"),  # noqa: B008
    cpus: int = typer.Option(8, "--cpus"),  # noqa: B008
    mem: str = typer.Option(None, "--mem"),  # noqa: B008
    time: str = typer.Option(None, "--time"),  # noqa: B008
    partition: str = typer.Option(None, "--partition"),  # noqa: B008
    account: str = typer.Option(None, "--account"),  # noqa: B008
    constraint: str = typer.Option(None, "--constraint"),  # noqa: B008
    qos: str = typer.Option(None, "--qos"),  # noqa: B008
    mail_type: str = typer.Option(None, "--mail-type"),  # noqa: B008
    job_name: str = typer.Option(None, "--job-name"),  # noqa: B008
    experiment: str = typer.Option(None, "--experiment"),  # noqa: B008
    snapshot: bool = typer.Option(False, "--snapshot"),  # noqa: B008
    dry_run: bool = typer.Option(False, "--dry-run"),  # noqa: B008
    slurm_kwargs: list[str] = typer.Option([], "--slurm-kwargs"),  # noqa: B008
    sync: bool = typer.Option(True, "--sync/--no-sync"),  # noqa: B008
) -> None:
    cmd_str = " ".join(command)
    client = SyncClient(profile=profile)
    kwargs: dict[str, Any] = {
        "gpus": gpus,
        "nodes": nodes,
        "cpus": cpus,
        "mem": mem,
        "time": time,
        "partition": partition,
        "account": account,
        "constraint": constraint,
        "qos": qos,
        "mail_type": mail_type,
        "name": job_name,
        "experiment": experiment,
        "snapshot": snapshot,
        "sync": sync,
        "slurm_kwargs": {},
    }
    for kw in slurm_kwargs:
        if "=" in kw:
            k, v = kw.split("=", 1)
            kwargs["slurm_kwargs"][k] = v

    if dry_run:
        from slurp.core.slurm import generate_sbatch_script
        from slurp.domain import ResourceRequest

        profile_obj = client.profile
        resources = ResourceRequest(
            gpus=gpus,
            nodes=nodes,
            time=time or "2:00:00",
            cpus=cpus,
            partition=partition or profile_obj.partition,
            account=account or profile_obj.account,
            job_name=job_name,
            slurm_kwargs=kwargs["slurm_kwargs"],
        )
        script = generate_sbatch_script(
            resources=resources,
            profile=profile_obj,
            command=cmd_str,
            working_dir=str(Path.cwd()),
        )
        console.print(Syntax(script, "bash", theme="monokai"))
        return

    job = client.submit(cmd_str, **kwargs)
    console.print(f"Job {job.job_id} submitted.")


@app.command(name="submit-array", help="Submit a SLURM job array")
def submit_array_cmd(
    ctx: typer.Context,
    template: str = typer.Argument(..., help="Command template or command"),
    profile: str = typer.Option(None, "--profile"),
    gpus: int = typer.Option(0, "--gpus"),
    nodes: int = typer.Option(1, "--nodes"),
    time: str = typer.Option(None, "--time"),
    partition: str = typer.Option(None, "--partition"),
    account: str = typer.Option(None, "--account"),
    experiment: str = typer.Option(None, "--experiment"),  # noqa: B008
    throttle: int = typer.Option(20, "--throttle"),  # noqa: B008
    slurm_kwargs: list[str] = typer.Option([], "--slurm-kwargs"),  # noqa: B008
) -> None:
    # Parse template: if no {placeholders}, treat as a command with trailing options as sweep params
    # For simplicity, require explicit --key value1,value2,... flags
    console.print("Array submission requires explicit configs via Python API for now.")
    console.print("Example: slurp.submit_array('python train.py --seed {seed}', configs=[...])")
    raise typer.Exit(1)
