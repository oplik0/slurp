"""CLI submit commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import questionary
import typer
from rich.console import Console
from rich.syntax import Syntax

from slurp.client import SyncClient
from slurp.errors import ProfileError

console = Console()
app = typer.Typer()

CONFIG_DIR = Path.home() / ".config" / "slurp"
CONFIG_FILE = CONFIG_DIR / "profiles.toml"


def _ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _split_command_args(args: list[str]) -> list[str]:
    """Return the user command portion after an optional '--' separator.

    Typer already strips the separator in most cases, but some shells and
    invocation paths leave it in. Dropping it makes command joining safe.
    """
    if args and args[0] == "--":
        return args[1:]
    return args


def _interactive_create_profile() -> str:
    """Prompt the user for basic profile info and write it to profiles.toml."""
    _ensure_config_dir()
    console.print("[yellow]No profile found. Let's create one quickly.[/yellow]")
    name = str(questionary.text("Profile name (e.g., default):").ask())
    hostname = str(questionary.text("Hostname (e.g., cluster.example.com):").ask())
    user = str(questionary.text("Username:").ask())
    partition = str(questionary.text("Default partition (optional):").ask() or "")
    account = str(questionary.text("Default account (optional):").ask() or "")

    lines = [
        "",
        f"[profiles.{name}]",
        f'hostname = "{hostname}"',
        f'username = "{user}"',
    ]
    if partition:
        lines.append(f'partition = "{partition}"')
    if account:
        lines.append(f'account = "{account}"')

    with open(CONFIG_FILE, "a") as f:
        f.write("\n".join(lines) + "\n")
    console.print(f"[green]Profile '{name}' saved to {CONFIG_FILE}[/green]")
    return name


def _parse_sweep_params(raw_args: list[str]) -> dict[str, list[str]]:
    """Parse trailing flags like --seed 1,2,3 into a mapping."""
    params: dict[str, list[str]] = {}
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        if arg.startswith("--") and i + 1 < len(raw_args):
            key = arg.lstrip("-")
            value = raw_args[i + 1]
            params[key] = [v.strip() for v in value.split(",")]
            i += 2
        else:
            i += 1
    return params


def _build_configs(params: dict[str, list[str]]) -> list[dict[str, str]]:
    """Build config dicts from sweep params. Uses the longest list length; shorter lists repeat their last value."""
    if not params:
        return []
    max_len = max(len(v) for v in params.values())
    configs: list[dict[str, str]] = []
    for i in range(max_len):
        cfg: dict[str, str] = {}
        for key, values in params.items():
            cfg[key] = values[min(i, len(values) - 1)]
        configs.append(cfg)
    return configs


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
    slurm_kwargs: list[str] = typer.Option([], "--slurm-kwargs"),
) -> dict[str, Any]:
    slurm_kwargs_dict: dict[str, str] = {}
    kwargs = {
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
        "slurm_kwargs": slurm_kwargs_dict,
    }
    for kw in slurm_kwargs:
        if "=" in kw:
            k, v = kw.split("=", 1)
            slurm_kwargs_dict[k] = v
    return kwargs


@app.command(name="submit", help="Fire-and-forget job submission")
def submit_cmd(
    ctx: typer.Context,
    command: list[str] = typer.Argument(..., help="Command to run"),
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
    slurm_kwargs: list[str] = typer.Option([], "--slurm-kwargs"),
    sync: bool = typer.Option(True, "--sync/--no-sync"),
) -> None:
    command = _split_command_args(command)
    cmd_str = " ".join(command)
    try:
        client = SyncClient(profile=profile)
    except ProfileError:
        if sys.stdin.isatty():
            profile = _interactive_create_profile()
            client = SyncClient(profile=profile)
        else:
            raise

    try:
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
                slurm_kwargs_dict: dict[str, str] = kwargs["slurm_kwargs"]
                slurm_kwargs_dict[k] = v

        if dry_run:
            from slurp.core.slurm import generate_sbatch_script
            from slurp.domain import ResourceRequest

            profile_obj = client.profile
            # Mirror client.submit()'s remote working-dir resolution so the
            # dry-run shows the paths a real submission would use, not the
            # local cwd. This expands $PROJECT/$SCRATCH via the login shell;
            # if the cluster is unreachable, fall back to the raw template
            # path so --dry-run still works offline.
            if profile_obj.sync and profile_obj.sync.remote:
                try:
                    remote_dir, _ = client._resolve_working_dir()  # noqa: SLF001
                except Exception as exc:
                    remote_dir = profile_obj.format_remote()
                    console.print(
                        f"[yellow]Could not resolve $VAR in remote path "
                        f"({exc}); showing raw template path.[/yellow]"
                    )
            else:
                remote_dir = str(Path.cwd())
            resources = ResourceRequest(
                gpus=gpus,
                nodes=nodes,
                time=time or "2:00:00",
                mem=mem,
                cpus=cpus,
                partition=partition or profile_obj.partition,
                account=account or profile_obj.account,
                constraint=constraint,
                qos=qos,
                mail_type=mail_type,
                job_name=job_name,
                slurm_kwargs=kwargs["slurm_kwargs"],
            )
            script = generate_sbatch_script(
                resources=resources,
                profile=profile_obj,
                command=cmd_str,
                working_dir=remote_dir,
            )
            console.print(Syntax(script, "bash", theme="monokai"))
            return

        job = client.submit(cmd_str, **kwargs)
        console.print(f"Job {job.job_id} submitted.")
    finally:
        client.close()


@app.command(
    name="submit-array",
    help="Submit a SLURM job array",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
def submit_array_cmd(
    ctx: typer.Context,
    template: str = typer.Argument(..., help="Command template or command"),
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
    sync: bool = typer.Option(True, "--sync/--no-sync"),
    throttle: int = typer.Option(20, "--throttle"),
    slurm_kwargs: list[str] = typer.Option([], "--slurm-kwargs"),
) -> None:
    # Parse trailing flags as sweep parameters, e.g. --seed 1,2,3
    raw_args = ctx.args
    sweep_params = _parse_sweep_params(raw_args)
    configs = _build_configs(sweep_params)

    if not configs:
        console.print("[red]No sweep parameters provided.[/red]")
        console.print("Usage: slurp submit-array 'python train.py --seed {seed}' --seed 1,2,3")
        raise typer.Exit(1)

    try:
        client = SyncClient(profile=profile)
    except ProfileError:
        if sys.stdin.isatty():
            profile = _interactive_create_profile()
            client = SyncClient(profile=profile)
        else:
            raise

    try:
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
        }
        for kw in slurm_kwargs:
            if "=" in kw:
                k, v = kw.split("=", 1)
                kwargs.setdefault("slurm_kwargs", {})[k] = v

        array = client.submit_array(
            template,
            configs=configs,
            throttle=throttle,
            **kwargs,
        )
        console.print(f"Array job {array.array_job_id} submitted ({array.task_count} tasks).")
    finally:
        client.close()
