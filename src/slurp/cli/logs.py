"""CLI logs command."""

from __future__ import annotations

import sys

import typer
from rich.console import Console

from slurp.client import SyncClient

console = Console()
app = typer.Typer()


@app.command(name="logs", help="Show job logs")
def logs_cmd(
    job_id: str = typer.Argument(..., help="Job ID"),
    profile: str = typer.Option(None, "--profile"),
    follow: bool = typer.Option(False, "--follow", "-f"),
    tail: int = typer.Option(100, "--tail"),
    stderr: bool = typer.Option(False, "--stderr"),
    stdout: bool = typer.Option(False, "--stdout"),
) -> None:
    with SyncClient(profile=profile) as client:
        job = client.status(job_id)
        if not job:
            console.print(f"[red]Job {job_id} not found.[/red]")
            raise typer.Exit(1)
        stream = "both"
        if stderr and not stdout:
            stream = "stderr"
        elif stdout and not stderr:
            stream = "stdout"
        for line in client.job_logs(job, follow=follow, tail=tail, stream=stream):
            sys.stdout.write(line)
            sys.stdout.flush()
