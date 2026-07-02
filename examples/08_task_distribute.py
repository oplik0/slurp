"""@task fire-and-forget: distribute many calls and join.

This is the slurminade pattern — decorate a function, call .distribute()
in a loop, and slurp.join() to wait for all. No return values; the
function writes results to files.

JobBundling batches calls into SLURM job arrays to avoid spamming the
scheduler with hundreds of individual jobs.

Prerequisites:
    - A profile in ~/.config/slurp/profiles.toml
    - slurp installed on the remote cluster

Run:
    python examples/08_task_distribute.py
"""

from __future__ import annotations

from pathlib import Path

import slurp


@slurp.task(gpus=1, time="0:30:00")
def evaluate_instance(instance_path: str, seed: int = 0) -> None:
    """Evaluate one problem instance. Writes results to a file.

    This function runs on the compute node. Its return value is
    discarded (.distribute() mode). Write results to files instead.
    """
    import json
    import random

    random.seed(seed)

    # Simulate solving
    result = {
        "instance": Path(instance_path).name,
        "seed": seed,
        "cost": random.uniform(10, 100),
        "iterations": random.randint(100, 1000),
    }

    out_path = Path(f"results/{Path(instance_path).stem}_s{seed}.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))


def main() -> None:
    instances = [f"data/instance_{i}.txt" for i in range(50)]
    seeds = [0, 1, 2]

    # --- Without bundling: one sbatch per call (slow, spams scheduler) ---
    # for inst in instances:
    #     for seed in seeds:
    #         evaluate_instance.distribute(instance_path=inst, seed=seed)
    # slurp.join()

    # --- With JobBundling: batches into job arrays ---
    # 150 calls / max_size=20 = 8 job arrays of ~20 tasks each.
    # Only 8 sbatch submissions instead of 150.
    with slurp.JobBundling(max_size=20):
        for inst in instances:
            for seed in seeds:
                evaluate_instance.distribute(instance_path=inst, seed=seed)
    print("All tasks submitted. Waiting...")

    slurp.join()
    print("All tasks complete. Results in ./results/")

    # --- Local mode (testing without SLURM) ---
    # Set LocalDispatcher to run everything in-process:
    #   slurp.set_dispatcher(slurp.LocalDispatcher())
    # Then .distribute() just calls the function directly.


if __name__ == "__main__":
    main()
