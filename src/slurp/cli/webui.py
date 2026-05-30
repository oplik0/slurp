"""CLI command for launching the slurp web UI."""

from __future__ import annotations

import sys

import typer
from rich.console import Console

console = Console()


def webui_cmd(
    port: int = typer.Option(8745, "--port", help="Port to bind the web UI server"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind the web UI server"),
    profile: str = typer.Option(None, "--profile", help="Profile name"),
) -> None:
    """Launch the slurp web UI dashboard.

    Opens a local FastAPI server that provides a real-time dashboard for
    monitoring SLURM jobs. The server URL includes a random security token.
    """
    # Check that optional web dependencies are installed
    try:
        import uvicorn  # noqa: F401
        from fastapi import FastAPI  # noqa: F401
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] Web UI dependencies not installed.\n"
            "Run: [bold]pip install slurp[web][/bold]"
        )
        sys.exit(1)

    try:
        from slurp.webui import create_app
        from slurp.webui.security import STREAM_TOKEN
    except ImportError as exc:
        console.print(
            f"[bold red]Error:[/bold red] Failed to import web UI module: {exc}\n"
            "Run: [bold]pip install slurp[web][/bold]"
        )
        sys.exit(1)

    if create_app is None:
        console.print(
            "[bold red]Error:[/bold red] Web UI dependencies not installed.\n"
            "Run: [bold]pip install slurp[web][/bold]"
        )
        sys.exit(1)

    url = f"http://{host}:{port}/?token={STREAM_TOKEN}"
    console.print(f"\n[bold]slurp web UI[/bold] starting at [link={url}]{url}[/link]\n")

    try:
        app = create_app()
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] Failed to create web UI app: {exc}")
        sys.exit(1)

    import uvicorn

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        console.print("\n[dim]Web UI stopped.[/dim]")
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] Server failed: {exc}")
        sys.exit(1)

