"""Multi-node launcher: auto-generate PyTorch Distributed commands."""

from __future__ import annotations

from slurp.core.slurm import build_torchrun_command as _build_torchrun_command

__all__ = ["build_torchrun_command"]

# Re-export from slurm.py for modularity
build_torchrun_command = _build_torchrun_command
