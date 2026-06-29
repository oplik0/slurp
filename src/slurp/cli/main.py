"""CLI entry point and top-level exception handler."""

from __future__ import annotations

import sys
from typing import Any

import structlog
import typer
from rich.console import Console
from rich.panel import Panel

from slurp.errors import (
    ConfigError,
    IdempotencyError,
    ProfileError,
    SlurmError,
    SlurpError,
    SSHError,
    SyncError,
)

from .cancel import cancel_cmd
from .config import app as config_app
from .logs import logs_cmd
from .pull import pull_cmd
from .status import list_cmd, status_cmd
from .submit import _split_command_args, submit_array_cmd, submit_cmd
from .sync import sync_cmd
from .watch import watch_cmd
from .webui import webui_cmd

console = Console()
app = typer.Typer(
    name="slurp",
    help="Run ML jobs on SLURM clusters. Simpler than sbatch.",
    no_args_is_help=True,
    add_completion=True,
)

# Global options
def _global_opts(
    profile: str = typer.Option(None, "--profile", help="Profile name"),
    experiment: str = typer.Option(None, "--experiment", help="Experiment tag"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Verbose output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show script and exit"),
) -> dict[str, Any]:
    return {
        "profile": profile,
        "experiment": experiment,
        "verbose": verbose,
        "dry_run": dry_run,
    }


# Register top-level commands
app.command("submit")(submit_cmd)
app.command(
    "submit-array",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)(submit_array_cmd)
app.command("cancel")(cancel_cmd)
app.command("logs")(logs_cmd)
app.command("status")(status_cmd)
app.command("list")(list_cmd)
app.command("watch")(watch_cmd)
app.command("sync")(sync_cmd)
app.command("pull")(pull_cmd)
app.command("webui")(webui_cmd)
app.add_typer(config_app, name="config")


# Aliases: slurp run = slurp submit --wait
@app.command(name="run", help="Blocking submit with live log streaming")
def run_cmd(
    command: list[str] = typer.Argument(..., help="Command to run"),
    profile: str = typer.Option(None, "--profile"),
    gpus: int = typer.Option(0, "--gpus"),
    nodes: int = typer.Option(1, "--nodes"),
    time: str = typer.Option(None, "--time"),
    partition: str = typer.Option(None, "--partition"),
    account: str = typer.Option(None, "--account"),
    experiment: str = typer.Option(None, "--experiment"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    from slurp.client import SyncClient

    with SyncClient(profile=profile) as client:
        command = _split_command_args(command)
        cmd_str = " ".join(command)
        job = client.submit(
            cmd_str,
            gpus=gpus,
            nodes=nodes,
            time=time or "2:00:00",
            partition=partition,
            account=account,
            experiment=experiment,
        )
        console.print(f"Job {job.job_id} submitted.")
        try:
            result = client.wait_job(job, follow_logs=True)
            console.print(f"Job {job.job_id} completed with exit code {result.exit_code}.")
            sys.exit(result.exit_code or 0)
        except Exception as exc:
            _handle_error(exc, verbose)


# Error handling
_EXIT_CODES: dict[type, int] = {
    SlurmError: 1,
    SSHError: 2,
    SyncError: 3,
    IdempotencyError: 4,
    ProfileError: 10,
    ConfigError: 11,
}


def _handle_error(exc: Exception, verbose: int) -> None:
    if isinstance(exc, SlurpError):
        title = type(exc).__name__
        content = f"[bold red]{exc.message}[/bold red]"
        if exc.hint:
            content += f"\n\n[yellow]Hint: {exc.hint}[/yellow]"
        if getattr(exc, "stderr_fragment", None):
            content += f"\n\n[dim]stderr:[/dim]\n{exc.stderr_fragment}"
        if exc.retryable:
            content += "\n\n[dim]This error may resolve if you retry.[/dim]"
        panel = Panel(content, title=title, border_style="red")
        console.print(panel)
        if verbose >= 2:
            console.print_exception()
        code = _EXIT_CODES.get(type(exc), 125)
        sys.exit(code)
    else:
        console.print(f"[bold red]Unexpected error: {exc}[/bold red]")
        if verbose >= 2:
            console.print_exception()
        sys.exit(125)


# Hook into Typer's exception handling
@app.callback()
def main(
    ctx: typer.Context,
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            10 if verbose >= 2 else 20 if verbose >= 1 else 30
        ),
    )


def main_entry() -> None:
    try:
        app()
    except Exception as exc:
        _handle_error(exc, verbose=0)


if __name__ == "__main__":
    main_entry()
