"""Quickstart: submit a job and wait for it.

This is the smallest useful slurp program. It submits a training command
to the default cluster profile, blocks until the job finishes, and prints
the captured stdout.

Prerequisites:
    - A profile configured in ~/.config/slurp/profiles.toml
      (run `slurp config add-profile default` to create one interactively)
    - train.py in the local directory (see examples/train.py)

Run:
    python examples/01_quickstart.py
"""

from __future__ import annotations

import slurp


def main() -> None:
    # submit() returns immediately with a Job handle. The job is PENDING;
    # the handle is a snapshot, not a live view. Call wait() to block.
    job = slurp.submit(
        "python train.py --lr 0.001 --epochs 10",
        gpus=1,
        time="1:00:00",
        experiment="quickstart",
    )
    print(f"Submitted job {job.job_id} (status={job.status})")

    # wait() polls sacct until the job reaches a terminal state.
    # follow_logs=True streams stdout/stderr to the terminal while waiting.
    result = job.wait(follow_logs=True)

    print(f"Job finished with exit code {result.exit_code}")
    print(f"Status: {result.status}")
    # stdout is capped at 1 MB to keep memory bounded.
    print(f"stdout (first 500 chars):\n{result.stdout[:500]}")


if __name__ == "__main__":
    main()
