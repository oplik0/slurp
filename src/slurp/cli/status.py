"""CLI status and list commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from slurp.client import SyncClient

console = Console()
app = typer.Typer()


@app.command(name="status", help="Show job status")
def status_cmd(
    job_id: str = typer.Argument(..., help="Job ID"),
    profile: str = typer.Option(None, "--profile"),
) -> None:
    client = SyncClient(profile=profile)
    job = client.status(job_id)
    if not job:
        console.print(f"[red]Job {job_id} not found.[/red]")
        raise typer.Exit(1)
    console.print(f"Job ID: {job.job_id}")
    console.print(f"Name:   {job.name}")
    console.print(f"Status: {job.status.value}")
    console.print(f"Profile: {job.profile}")


@app.command(name="list", help="List tracked jobs")
def list_cmd(
    profile: str = typer.Option(None, "--profile"),
    experiment: str = typer.Option(None, "--experiment"),
    status: str = typer.Option(None, "--status"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    client = SyncClient(profile=profile)
    jobs = client.list_jobs(experiment=experiment, status=status, limit=limit)
    table = Table(title="Jobs")
    table.add_column("Job ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Experiment", style="yellow")
    for job in jobs:
        table.add_row(job.job_id, job.name, job.status.value, job.experiment or "")
    console.print(table)
