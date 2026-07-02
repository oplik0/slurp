"""Interactive ``slurp setup`` wizard.

Creates ``~/.config/slurp/profiles.toml`` by asking the user a sequence of
questions covering every field that the :class:`~slurp.domain.Profile` model
understands — connection details, SLURM defaults, cluster tuning knobs, and
the optional ``sync``/``venv`` sub-tables.

Uses :mod:`questionary` for prompts and :mod:`rich` for presentation. The
wizard is intentionally interactive: if stdin is not a TTY (so ``.ask()``
returns ``None``) it aborts cleanly rather than writing a half-empty file.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import questionary
import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

CONFIG_DIR = Path.home() / ".config" / "slurp"
CONFIG_FILE = CONFIG_DIR / "profiles.toml"

# Defaults mirror slurp.domain.Profile so the generated file only records
# values that differ from the model's built-in defaults — keeping it clean.
_DEFAULT_MPI_MODE = "pmi2"
_DEFAULT_CPU_BIND = "cores"
_DEFAULT_GPU_FLAG_STYLE = "gres"
_DEFAULT_VENV_PATH = "$PROJECT/.venv"

_PROFILE_NAME_RE = re.compile(r"[a-zA-Z0-9_-]+")


# --- small helpers ---------------------------------------------------------


def _ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _ask(question: questionary.Question) -> Any:
    """Run a questionary prompt, aborting cleanly on cancel / no TTY.

    ``questionary``'s ``.ask()`` returns ``None`` when the user presses Ctrl-C
    or when stdin is not an interactive terminal. Writing a file from ``None``
    answers would silently produce garbage, so we treat that as an explicit
    cancellation.
    """
    answer = question.ask()
    if answer is None:
        console.print("[yellow]Setup cancelled.[/yellow]")
        raise typer.Exit(1)
    return answer


def _required(value: str) -> bool | str:
    """questionary validator: reject empty input."""
    if not value.strip():
        return "This field is required."
    return True


def _valid_profile_name(value: str) -> bool | str:
    if not value.strip():
        return "Profile name is required."
    if not _PROFILE_NAME_RE.fullmatch(value):
        return "Use only letters, numbers, hyphens, and underscores."
    return True


def _toml_escape(value: str) -> str:
    """Escape a string for a TOML basic (double-quoted) string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


# --- TOML rendering --------------------------------------------------------


def _build_toml(a: dict[str, Any]) -> str:
    """Render profile answers as a TOML document.

    Fields equal to the model's built-in defaults are omitted so the file
    stays readable; they take effect via :class:`~slurp.domain.Profile`'s
    defaults when the file is loaded.
    """
    name = a["name"]
    lines: list[str] = [f"[profiles.{name}]", f'hostname = "{_toml_escape(a["hostname"])}"']

    if a.get("username"):
        lines.append(f'username = "{_toml_escape(a["username"])}"')
    if a.get("key_file"):
        lines.append(f'key_file = "{_toml_escape(a["key_file"])}"')
    if a.get("proxy_jump"):
        lines.append(f'proxy_jump = "{_toml_escape(a["proxy_jump"])}"')
    if a.get("partition"):
        lines.append(f'partition = "{_toml_escape(a["partition"])}"')
    if a.get("account"):
        lines.append(f'account = "{_toml_escape(a["account"])}"')

    # Tuning knobs — only emit when they differ from the model defaults.
    prologue = (a.get("prologue") or "").strip()
    if prologue:
        if "\n" in prologue:
            lines.append(f'prologue = """\n{prologue}\n"""')
        else:
            lines.append(f'prologue = "{_toml_escape(prologue)}"')
    if a.get("mpi_mode") and a["mpi_mode"] != _DEFAULT_MPI_MODE:
        lines.append(f'mpi_mode = "{_toml_escape(a["mpi_mode"])}"')
    if a.get("cpu_bind") and a["cpu_bind"] != _DEFAULT_CPU_BIND:
        lines.append(f'cpu_bind = "{_toml_escape(a["cpu_bind"])}"')
    if a.get("gpu_flag_style") and a["gpu_flag_style"] != _DEFAULT_GPU_FLAG_STYLE:
        lines.append(f'gpu_flag_style = "{_toml_escape(a["gpu_flag_style"])}"')

    if a.get("sync"):
        lines.append("")
        lines.append(f"[profiles.{name}.sync]")
        lines.append(f'local = "{_toml_escape(a["sync_local"])}"')
        lines.append(f'remote = "{_toml_escape(a["sync_remote"])}"')

    if a.get("venv"):
        lines.append("")
        lines.append(f"[profiles.{name}.venv]")
        lines.append('strategy = "uv-sync"')
        path = a.get("venv_path") or _DEFAULT_VENV_PATH
        lines.append(f'path = "{_toml_escape(path)}"')
        if a.get("venv_all_extras"):
            lines.append("all_extras = true")
        else:
            raw_extras = a.get("venv_extras") or ""
            extras = [e.strip() for e in raw_extras.split(",") if e.strip()]
            if extras:
                rendered = ", ".join(f'"{_toml_escape(e)}"' for e in extras)
                lines.append(f"extras = [{rendered}]")

    lines.append("")
    return "\n".join(lines)


# --- the wizard ------------------------------------------------------------


def _prompt_answers() -> dict[str, Any]:
    """Run the full question sequence and return the collected answers."""
    console.print(
        Panel(
            "This creates [bold]~/.config/slurp/profiles.toml[/bold] with a single\n"
            "cluster profile. You can add more profiles later with\n"
            "[cyan]slurp config add-profile[/cyan].",
            title="slurp setup",
            border_style="cyan",
        )
    )

    answers: dict[str, Any] = {}

    # --- identity & connection -----------------------------------------
    console.rule("[bold]Connection[/bold]")
    answers["name"] = _ask(
        questionary.text(
            "Profile name:",
            default="default",
            validate=_valid_profile_name,
        )
    )
    answers["hostname"] = _ask(
        questionary.text(
            "Cluster hostname (e.g. cluster.example.edu):",
            validate=_required,
        )
    )
    answers["username"] = _ask(
        questionary.text("SSH username (leave blank to use your local user):")
    )
    answers["key_file"] = _ask(
        questionary.text("SSH private key path (optional, e.g. ~/.ssh/id_ed25519):")
    )
    answers["proxy_jump"] = _ask(
        questionary.text("Bastion / proxy jump host (optional):")
    )

    # --- SLURM defaults ------------------------------------------------
    console.rule("[bold]SLURM defaults[/bold]")
    answers["partition"] = _ask(
        questionary.text("Default partition (optional):")
    )
    answers["account"] = _ask(
        questionary.text("Default account (optional):")
    )

    # --- advanced tuning ----------------------------------------------
    if _ask(
        questionary.confirm(
            "Configure advanced cluster options (prologue, MPI, GPU flags)?",
            default=False,
        )
    ):
        answers["prologue"] = _ask(
            questionary.text(
                "Prologue shell commands "
                "(optional, e.g. 'module load Python; module load CUDA'):"
            )
        )
        answers["mpi_mode"] = _ask(
            questionary.select(
                "MPI mode for multi-node jobs:",
                choices=["pmi2", "pmix", "none"],
                default=_DEFAULT_MPI_MODE,
            )
        )
        answers["cpu_bind"] = _ask(
            questionary.select(
                "CPU binding strategy:",
                choices=["cores", "threads", "none"],
                default=_DEFAULT_CPU_BIND,
            )
        )
        answers["gpu_flag_style"] = _ask(
            questionary.select(
                "GPU request flag style (gres for new SLURM, gpus for older):",
                choices=["gres", "gpus"],
                default=_DEFAULT_GPU_FLAG_STYLE,
            )
        )

    # --- code sync -----------------------------------------------------
    if _ask(
        questionary.confirm("Configure code sync (rsync local -> remote)?", default=True)
    ):
        answers["sync"] = True
        answers["sync_local"] = _ask(
            questionary.text("Local path to sync from:", default=".")
        )
        answers["sync_remote"] = _ask(
            questionary.text(
                "Remote path to sync to "
                "({username}/{account} are auto-substituted):",
                validate=_required,
            )
        )

    # --- remote venv ---------------------------------------------------
    if _ask(
        questionary.confirm(
            "Configure remote venv management (uv-sync)?", default=False
        )
    ):
        answers["venv"] = True
        answers["venv_path"] = _ask(
            questionary.text("Remote venv path:", default=_DEFAULT_VENV_PATH)
        )
        answers["venv_all_extras"] = _ask(
            questionary.confirm("Install all optional extras?", default=False)
        )
        if not answers["venv_all_extras"]:
            answers["venv_extras"] = _ask(
                questionary.text(
                    "Specific extras (comma-separated, e.g. cu121,dev):"
                )
            )

    return answers


def _backup_existing() -> Path | None:
    """If profiles.toml exists, ask whether to back it up and rename it.

    Returns the backup path (or ``None`` if nothing existed), or aborts the
    process if the user declines. A pre-existing ``.bak`` is preserved by
    suffixing a timestamp rather than being overwritten.
    """
    if not CONFIG_FILE.exists():
        return None

    console.print(f"[yellow]Found existing config: {CONFIG_FILE}[/yellow]")
    if not _ask(
        questionary.confirm("Back it up and overwrite?", default=False)
    ):
        console.print("[yellow]Setup aborted — existing config left untouched.[/yellow]")
        raise typer.Exit(0)

    backup = CONFIG_FILE.with_name(CONFIG_FILE.name + ".bak")
    if backup.exists():
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        backup = CONFIG_FILE.with_name(f"{CONFIG_FILE.name}.bak.{ts}")
    CONFIG_FILE.rename(backup)
    console.print(f"[dim]Backed up existing config to {backup}[/dim]")
    return backup


def setup_cmd() -> None:
    """Interactively create ~/.config/slurp/profiles.toml."""
    _ensure_config_dir()
    _backup_existing()

    answers = _prompt_answers()
    toml = _build_toml(answers)

    # Review
    console.print()
    console.print(
        Panel(
            Syntax(toml, "toml", theme="ansi_dark", word_wrap=True),
            title=f"Preview — {CONFIG_FILE}",
            border_style="green",
        )
    )
    if not _ask(questionary.confirm("Write this file?", default=True)):
        console.print("[yellow]Setup cancelled — nothing was written.[/yellow]")
        raise typer.Exit(0)

    CONFIG_FILE.write_text(toml)
    console.print(f"[green]Wrote {CONFIG_FILE}[/green]")
    console.print(
        "Next: [cyan]slurp submit python train.py[/cyan] "
        "(or [cyan]slurp config edit-profile[/cyan] to tweak)."
    )


# Typer registration ---------------------------------------------------------
# Exposed as ``app.command`` is done by the caller (cli/main.py), but we also
# provide a Typer app so the module is self-contained and importable.
app = typer.Typer()
app.command(name="setup")(setup_cmd)
