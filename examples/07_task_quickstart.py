"""@task quickstart: submit a function and get the result.

This is the Ray-style pattern — decorate a function, call .remote(),
and slurp.get() the result. No separate script file needed.

Unlike the string-based API (slurp.submit("python train.py")), the
function is serialized via cloudpickle and shipped to the cluster.
The return value is pickled back.

Prerequisites:
    - A profile in ~/.config/slurp/profiles.toml
    - slurp installed on the remote cluster (for the worker module)

Run:
    python examples/07_task_quickstart.py
"""

from __future__ import annotations

import slurp


@slurp.task(gpus=1, time="1:00:00")
def train(lr: float = 0.001, epochs: int = 10) -> dict[str, float]:
    """A training function. Runs locally when called directly,
    on SLURM when called via .remote()."""
    import random

    random.seed(42)
    loss = 2.0
    for epoch in range(epochs):
        loss = loss * 0.9 + random.uniform(-0.05, 0.05)
        slurp.log_progress(
            step=epoch,
            total_steps=epochs,
            metrics={"loss": round(loss, 4), "lr": lr},
        )

    return {"final_loss": round(loss, 4), "lr": lr, "epochs": epochs}


def main() -> None:
    # --- Local call (testing) ---
    # The decorated function is still directly callable.
    result = train(lr=0.01, epochs=3)
    print(f"Local result: {result}")

    # --- Remote submit (Ray-style) ---
    # .remote() serializes the function + args via cloudpickle, rsyncs
    # the working directory (including the payload), and submits
    # `python -m slurp.worker <payload> <result>` to SLURM.
    ref = train.remote(lr=0.001, epochs=50)
    print(f"Submitted job {ref.job_id}")

    # slurp.get() blocks until the job finishes, pulls the result file,
    # and deserializes the return value.
    result = slurp.get(ref)
    print(f"Remote result: {result}")


if __name__ == "__main__":
    main()
