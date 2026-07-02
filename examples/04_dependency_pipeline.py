"""Dependency pipeline: preprocess -> train -> evaluate.

SLURM dependencies let you express a DAG as a chain of submit() calls.
Each job gets a --dependency=afterok:<job_id> directive. The default
type is afterok (start only if dependencies exit 0), which is what you
want for pipelines where a failed preprocess step should not launch
training.

This is fire-and-forget: all three jobs are submitted immediately and
SLURM handles the scheduling. You can shut down your laptop; the
dependency chain is stored in SLURM's own state, not slurp's.

Prerequisites:
    - A profile in ~/.config/slurp/profiles.toml
    - preprocess.py, train.py, evaluate.py in the local directory

Run:
    python examples/04_dependency_pipeline.py
"""

from __future__ import annotations

import slurp


def main() -> None:
    # Step 1: preprocess (CPU-only, cheap)
    preprocess = slurp.submit(
        "python preprocess.py --dataset cifar10",
        gpus=0,
        cpus=8,
        time="0:30:00",
        name="preprocess",
        experiment="pipeline",
    )
    print(f"Preprocess job: {preprocess.job_id}")

    # Step 2: train (depends on preprocess). depends_on takes Job objects,
    # not job IDs — slurp validates that the job has been submitted before
    # constructing the directive.
    train = slurp.submit(
        "python train.py --data preprocessed/ --epochs 100",
        gpus=4,
        time="8:00:00",
        name="train",
        experiment="pipeline",
        depends_on=[preprocess],
        # depends_on_type="afterok" is the default. Other options:
        #   "afterany"  — run regardless of dependency exit code
        #   "after"     — start when dependencies *begin* running
        #   "afternotok" — run only if dependencies fail (for cleanup/alerts)
    )
    print(f"Train job: {train.job_id} (depends on {preprocess.job_id})")

    # Step 3: evaluate (depends on train). Multiple dependencies are OR'd
    # into a single directive: --dependency=afterok:12345,12346
    evaluate = slurp.submit(
        "python evaluate.py --checkpoint best.pt",
        gpus=1,
        time="1:00:00",
        name="evaluate",
        experiment="pipeline",
        depends_on=[train],
    )
    print(f"Evaluate job: {evaluate.job_id} (depends on {train.job_id})")

    print("\nAll jobs submitted. Pipeline will execute on the cluster.")
    print("Monitor with:  slurp watch --experiment pipeline")
    print("Pull results:  slurp pull <evaluate_job_id>")

    # If you want to block until the whole pipeline finishes:
    #   result = evaluate.wait(follow_logs=True, timeout="12h")
    # But the point of fire-and-forget is that you don't have to.


if __name__ == "__main__":
    main()
