"""CLI pull command."""

from __future__ import annotations

import typer
from rich.console import Console

from slurp.client import SyncClient

console = Console()
app = typer.Typer()


@app.command(name="pull", help="Download job results")
def pull_cmd(
    job_id: str = typer.Argument(..., help="Job ID"),
    local_dir: str = typer.Option(None, "--local"),
    profile: str = typer.Option(None, "--profile"),
) -> None:
    with SyncClient(profile=profile) as client:
        client.pull(job_id, local_dir=local_dir)
        console.print(f"Pulled job {job_id} to {local_dir or './outputs/' + job_id}.")
