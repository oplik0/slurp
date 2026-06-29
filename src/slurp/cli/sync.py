"""CLI sync command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from slurp.client import SyncClient

console = Console()
app = typer.Typer()


@app.command(name="sync", help="Sync code to remote without submitting")
def sync_cmd(
    profile: str = typer.Option(None, "--profile"),
    local_dir: str = typer.Option(None, "--local"),
    remote_dir: str = typer.Option(None, "--remote"),
) -> None:
    with SyncClient(profile=profile) as client:
        client.sync(
            local_dir=Path(local_dir) if local_dir else None,
            remote_dir=remote_dir,
        )
        console.print("Sync complete.")
