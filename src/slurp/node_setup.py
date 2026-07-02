"""Node setup: per-node initialization functions.

A function decorated with ``@slurp.node_setup`` runs on the compute node
before any ``@task`` function.  The worker calls all registered setup
functions once on startup.

This is the Python-level equivalent of the profile prologue — use it for
things like configuring thread counts, setting up logging, or warming up
imports::

    @slurp.node_setup
    def setup():
        import torch
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Registry of setup functions, keyed by name.
# The worker picks these up from the payload (they are cloudpickled alongside
# the task function and called before it).
_setup_functions: dict[str, Callable[..., Any]] = {}


def node_setup(func: Callable[..., Any]) -> Callable[..., Any]:
    """Register a function to run on every compute node before tasks.

    The function takes no arguments and its return value is ignored.
    Multiple setup functions are called in registration order.
    """
    _setup_functions[func.__name__] = func
    return func


def get_setup_functions() -> dict[str, Callable[..., Any]]:
    """Return all registered setup functions (for the worker)."""
    return dict(_setup_functions)


def run_setup_functions() -> None:
    """Call all registered setup functions. Called by the worker."""
    for func in _setup_functions.values():
        func()


__all__ = ["node_setup", "get_setup_functions", "run_setup_functions"]
