"""CLI config commands."""

from __future__ import annotations

import os
from pathlib import Path

import questionary
import typer
from rich.console import Console

console = Console()
app = typer.Typer()

CONFIG_DIR = Path.home() / ".config" / "slurp"
CONFIG_FILE = CONFIG_DIR / "profiles.toml"


def _ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


@app.command(name="add-profile", help="Add a new profile")
def add_profile_cmd(
    name: str = typer.Argument(..., help="Profile name"),
    hostname: str = typer.Option(None, "--hostname"),
    user: str = typer.Option(None, "--user"),
    key_file: str = typer.Option(None, "--key-file"),
    proxy_jump: str = typer.Option(None, "--proxy-jump"),
    partition: str = typer.Option(None, "--partition"),
    account: str = typer.Option(None, "--account"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    _ensure_config_dir()

    if CONFIG_FILE.exists() and not force:
        content = CONFIG_FILE.read_text()
        if f"[profiles.{name}]" in content:
            console.print(
                f"[yellow]Profile '{name}' already exists. Use --force to overwrite.[/yellow]"
            )
            raise typer.Exit(10)

    # Interactive fallback
    if not hostname:
        hostname = questionary.text("Hostname:").ask()
    if not user:
        user = questionary.text("Username:").ask()
    if not partition:
        partition = questionary.text("Default partition (optional):").ask() or ""
    if not account:
        account = questionary.text("Default account (optional):").ask() or ""

    lines = [
        "",
        f"[profiles.{name}]",
        f'hostname = "{hostname}"',
        f'username = "{user}"',
    ]
    if key_file:
        lines.append(f'key_file = "{key_file}"')
    if proxy_jump:
        lines.append(f'proxy_jump = "{proxy_jump}"')
    if partition:
        lines.append(f'partition = "{partition}"')
    if account:
        lines.append(f'account = "{account}"')

    with open(CONFIG_FILE, "a") as f:
        f.write("\n".join(lines) + "\n")
    console.print(f"[green]Profile '{name}' saved to {CONFIG_FILE}[/green]")


@app.command(name="list-profiles", help="List all profiles")
def list_profiles_cmd() -> None:
    if not CONFIG_FILE.exists():
        console.print("No profiles configured.")
        return
    content = CONFIG_FILE.read_text()
    for line in content.splitlines():
        if line.startswith("[profiles."):
            name = line[10:-1]
            console.print(f"  {name}")


@app.command(name="show-profile", help="Show a profile")
def show_profile_cmd(name: str = typer.Argument(..., help="Profile name")) -> None:
    if not CONFIG_FILE.exists():
        console.print("No profiles configured.")
        raise typer.Exit(10)
    content = CONFIG_FILE.read_text()
    in_profile = False
    for line in content.splitlines():
        if line.startswith(f"[profiles.{name}]"):
            in_profile = True
        elif in_profile and line.startswith("[profiles."):
            break
        if in_profile:
            console.print(line)


@app.command(name="edit-profile", help="Open profiles.toml in $EDITOR")
def edit_profile_cmd() -> None:
    editor = os.environ.get("EDITOR", "nano")
    os.system(f"{editor} {CONFIG_FILE}")
