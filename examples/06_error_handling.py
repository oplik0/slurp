"""Error handling patterns for slurp.

slurp's exception hierarchy is flat and actionable. Every SlurpError has:
  - .message   — what went wrong
  - .hint       — what to do about it (often a copy-pasteable command)
  - .retryable — whether re-running the same call might succeed

This matters for automation: you don't want to retry a ProfileError
(config is wrong), but you do want to retry an SSHError (network blip).

Prerequisites:
    - A profile in ~/.config/slurp/profiles.toml (or not, to see ProfileError)

Run:
    python examples/06_error_handling.py
"""

from __future__ import annotations

import slurp
from slurp import SlurpError, SyncClient
from slurp.errors import (
    JobFailedError,
    ProfileError,
    SSHError,
    SyncError,
)


def main() -> None:
    # --- Pattern 1: Profile errors are never retryable. ---------------
    # The config is wrong; retrying won't help. Surface the hint to the
    # user and exit. This is what the CLI does internally.
    try:
        SyncClient(profile="nonexistent-profile")
    except ProfileError as exc:
        print(f"[ProfileError] {exc.message}")
        print(f"  hint: {exc.hint}")
        print(f"  retryable: {exc.retryable}")
        # Don't retry. The user needs to create the profile first.

    # --- Pattern 2: Job failure carries the log tail. ------------------
    # JobFailedError is a subclass of SlurmError. It includes the last
    # 1KB of stdout and stderr so you can diagnose without a separate
    # `slurp logs` call. This is the most common error in practice.
    try:
        job = slurp.submit(
            "python train.py --bad-flag",
            gpus=1,
            time="0:10:00",
            experiment="error-demo",
        )
        job.wait(follow_logs=True)
    except JobFailedError as exc:
        print(f"\n[JobFailedError] job={exc.job_id}, exit={exc.exit_code}")
        print(f"  stderr tail:\n{exc.stderr_tail[-300:]}")
        # The job has already terminated on the cluster. Don't retry —
        # fix the command and resubmit.

    # --- Pattern 3: SSH errors may be transient. ------------------------
    # A network blip or login node overload can cause SSHError. The
    # control master auto-retries internally, but if it gives up, the
    # error is flagged retryable. This is where exponential backoff helps.
    try:
        job = slurp.submit("python train.py", gpus=1)
    except SSHError as exc:
        print(f"\n[SSHError] {exc.message}")
        if exc.retryable:
            print("  This might resolve on retry. Waiting 30s...")
            import time
            time.sleep(30)
            # In production, wrap this in a retry loop with backoff.

    # --- Pattern 4: Sync errors happen before submission. --------------
    # rsync runs before sbatch. If it fails (disk quota, permissions),
    # no job is submitted. This is a design choice: a partially-synced
    # code tree on the compute node is worse than no job at all.
    try:
        slurp.submit("python train.py", gpus=1, sync=True)
    except SyncError as exc:
        print(f"\n[SyncError] {exc.message}")
        print(f"  hint: {exc.hint}")
        # Check remote disk quota: the most common cause.

    # --- Pattern 5: Catch-all for automation. -------------------------
    # If you're running slurp in a CI pipeline and want to fail loudly
    # on any slurp-specific error but let unexpected exceptions bubble:
    try:
        job = slurp.submit("python train.py", gpus=1)
        job.wait()
    except SlurpError as exc:
        # Every slurp exception inherits from SlurpError. The .retryable
        # flag is the key decision point for retry logic.
        print(f"\n[SlurpError] {type(exc).__name__}: {exc.message}")
        if exc.retryable:
            print("  -> retry with backoff")
        else:
            print("  -> fix and resubmit")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
