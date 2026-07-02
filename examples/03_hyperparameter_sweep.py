"""Hyperparameter sweep with a SLURM job array.

submit_array() generates a native SLURM array (--array=0-N%M) and a wrapper
script that maps SLURM_ARRAY_TASK_ID to a config dict. The template uses
Python brace-formatting: {seed}, {lr}, etc.

This is cheaper than submitting N independent jobs: one sbatch call, one
queue entry, and SLURM's own throttling handles concurrency.

Prerequisites:
    - A profile in ~/.config/slurp/profiles.toml
    - train.py in the local directory (see examples/train.py)

Run:
    python examples/03_hyperparameter_sweep.py
"""

from __future__ import annotations

import slurp
from slurp.errors import JobFailedError


def main() -> None:
    # Each dict fills the {placeholder} slots in the template. The list
    # length determines the array size. Keep the values as strings —
    # they're substituted via str.format(), not type-coerced.
    configs = [
        {"seed": str(s), "lr": f"0.{lr:02d}"}
        for s in range(5)
        for lr in (1, 10, 100)
    ]
    print(f"Sweeping {len(configs)} configurations")

    array = slurp.submit_array(
        "python train.py --seed {seed} --lr {lr} --epochs 20",
        configs=configs,
        gpus=1,
        time="2:00:00",
        throttle=10,  # max 10 concurrent tasks (--array=0-N%10)
        experiment="lr-sweep",
    )
    print(f"Array job {array.array_job_id} submitted ({array.task_count} tasks)")

    # results() blocks until every task reaches a terminal state.
    # If any task fails, JobFailedError is raised for that task and the
    # loop below catches it — other tasks are NOT cancelled by slurp.
    # They keep running on the cluster; you just don't get their results
    # in the same call.
    try:
        results = array.results()
    except JobFailedError as exc:
        print(f"Task {exc.job_id} failed (exit={exc.exit_code})")
        print(f"  stderr tail:\n{exc.stderr_tail[-500:]}")
        return

    # Aggregate results. The order matches the configs list above.
    best_idx, best_loss = -1, float("inf")
    for i, result in enumerate(results):
        cfg = configs[i]
        # Parse the last loss from stdout. Your training script should
        # print structured output (see train.py for progress.jsonl).
        print(f"Task {i}: seed={cfg['seed']}, lr={cfg['lr']}, exit={result.exit_code}")
        # In a real sweep, parse metrics from result.stdout or progress.jsonl.

    if best_idx >= 0:
        print(f"\nBest config: {configs[best_idx]} (loss={best_loss})")


if __name__ == "__main__":
    main()
