"""SyncClient lifecycle: submit, poll, stream, pull.

Using the top-level slurp.submit() is convenient for one-shots, but each
call opens and closes an SSH connection. When you need multiple operations
on the same cluster (submit, then poll, then pull), use SyncClient as a
context manager. It reuses one SSH control master for the whole block.

Prerequisites:
    - A profile in ~/.config/slurp/profiles.toml
    - train.py in the local directory

Run:
    python examples/02_sync_client_lifecycle.py
"""

from __future__ import annotations

import time

from slurp import SyncClient
from slurp.domain import JobStatus


def main() -> None:
    # The context manager closes the SSH connection on exit. If you forget
    # the `with` block, call client.close() explicitly — a leaked control
    # master socket will linger until the OS reaps it.
    with SyncClient(profile="default") as client:
        job = client.submit(
            "python train.py --lr 0.01 --epochs 5",
            gpus=1,
            time="0:30:00",
            experiment="lifecycle-demo",
        )
        print(f"Submitted job {job.job_id}")

        # Poll once to show the non-blocking refresh path.
        job = client.refresh_job(job)
        print(f"Initial status: {job.status}")

        # While the job runs, you can do other work. Here we just sleep,
        # but in a real script you might submit more jobs, query others,
        # or prepare the next experiment. The local job cache is updated
        # on every refresh, so list_jobs() stays consistent.
        while not job.status.is_terminal:
            time.sleep(10)
            job = client.refresh_job(job)
            print(f"  status={job.status}")

        # If the job is still running and you want to bail out:
        #   client.cancel_job(job)
        # This calls scancel on the remote side. The job is NOT killed
        # locally — it keeps running on the cluster until scancel takes effect.

        # Pull outputs back to ./outputs/<job_id>/. This is an rsync from
        # the remote working directory, so it grabs everything: checkpoints,
        # logs, progress.jsonl, and the slurm-*.out / .err files.
        if job.status == JobStatus.COMPLETED:
            client.pull(job.job_id)
            print(f"Pulled results to ./outputs/{job.job_id}/")
        else:
            print(f"Job ended with status {job.status}, skipping pull.")

        # List all jobs tagged with this experiment. The store is reconciled
        # against sacct first, so statuses reflect the real cluster state.
        jobs = client.list_jobs(experiment="lifecycle-demo")
        print(f"Jobs in experiment: {len(jobs)}")
        for j in jobs:
            print(f"  {j.job_id}  {j.status}  {j.name}")


if __name__ == "__main__":
    main()
