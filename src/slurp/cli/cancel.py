"""CLI cancel command."""

from __future__ import annotations

import typer
from rich.console import Console

from slurp.client import SyncClient

console = Console()
app = typer.Typer()


@app.command(name="cancel", help="Cancel one or more jobs")
def cancel_cmd(
    job_ids: list[str] = typer.Argument(..., help="Job IDs to cancel"),
    profile: str = typer.Option(None, "--profile"),
) -> None:
    client = SyncClient(profile=profile)
    for job_id in job_ids:
        job = client.status(job_id)
        if job:
            client.cancel_job(job)
            console.print(f"Cancelled job {job_id}.")
        else:
            console.print(f"[yellow]Job {job_id} not found.[/yellow]")
