"""CLI watch command."""

from __future__ import annotations

import typer
from rich.console import Console

from slurp.client import SyncClient

console = Console()
app = typer.Typer()


@app.command(name="watch", help="Live table of all jobs")
def watch_cmd(
    profile: str = typer.Option(None, "--profile"),
    experiment: str = typer.Option(None, "--experiment"),
    refresh: float = typer.Option(5.0, "--refresh"),
) -> None:
    with SyncClient(profile=profile) as client:
        client.watch(experiment=experiment, refresh=refresh)
