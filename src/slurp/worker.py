"""Worker: the compute-node entry point for @task functions.

Invoked by SLURM as::

    python -m slurp.worker <payload_path> [result_path]

The payload is a cloudpickle file containing ``(func, args, kwargs)``.
If ``result_path`` is given, the return value is pickled there; otherwise
the function is fire-and-forget (``.distribute()`` mode).

On exception, the worker writes ``("error", exc, traceback)`` to the
result file so ``slurp.get()`` can re-raise it on the caller side.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

import cloudpickle


def main() -> None:
    """Entry point for python -m slurp.worker."""
    if len(sys.argv) < 2:
        print("Usage: python -m slurp.worker <payload> [result]", file=sys.stderr)
        sys.exit(1)

    payload_path = sys.argv[1]
    result_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Load payload
    with open(payload_path, "rb") as f:
        payload: tuple[Any, tuple[Any, ...], dict[str, Any]] = cloudpickle.load(f)

    func, args, kwargs = payload

    # Resolve any ObjectRef arguments (from slurp.put)
    args = tuple(_resolve_arg(a) for a in args)
    kwargs = {k: _resolve_arg(v) for k, v in kwargs.items()}

    # Run node_setup functions if any are registered in the payload's
    # environment.  These are loaded by cloudpickle as part of the import
    # chain.
    try:
        from slurp.node_setup import run_setup_functions

        run_setup_functions()
    except Exception:
        # Setup is optional — don't fail the job if no setup is registered
        pass

    # Execute
    if result_path is None:
        # Fire-and-forget mode: just run the function.
        # Exceptions go to stderr (visible via slurp logs).
        func(*args, **kwargs)
        return

    # Result-collecting mode: write ("ok", value) or ("error", exc, tb)
    try:
        result = func(*args, **kwargs)
        _write_result(result_path, ("ok", result))
    except Exception as exc:
        tb_str = traceback.format_exc()
        _write_result(result_path, ("error", exc, tb_str))
        # Exit non-zero so SLURM marks the job as FAILED
        sys.exit(1)


def _resolve_arg(obj: Any) -> Any:
    """If obj is an ObjectRef, load and return the referenced object."""
    # Duck-type: ObjectRef has _slurp_object_path
    path = getattr(obj, "_slurp_object_path", None)
    if path is not None:
        with open(path, "rb") as f:
            return cloudpickle.load(f)
    return obj


def _write_result(path: str, data: Any) -> None:
    """Write result data to a file, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        cloudpickle.dump(data, f)


if __name__ == "__main__":
    main()
