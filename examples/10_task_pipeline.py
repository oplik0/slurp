"""@task pipeline: preprocess -> train -> evaluate with dependencies.

Shows the full ML lifecycle using the @task API. Each stage is a
decorated function. Dependencies are expressed via depends_on.

The pipeline is fire-and-forget: all three stages are submitted
immediately, and SLURM handles the scheduling. The --dependency
directive ensures train only starts after preprocess succeeds.

Prerequisites:
    - A profile in ~/.config/slurp/profiles.toml
    - slurp installed on the remote cluster

Run:
    python examples/10_task_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import slurp


@slurp.node_setup
def setup() -> None:
    """Runs on every compute node before any task function."""
    import os

    # Limit threads to avoid contention on shared nodes
    os.environ["OMP_NUM_THREADS"] = "4"


@slurp.task(gpus=0, cpus=8, time="0:30:00")
def preprocess(dataset: str) -> str:
    """Preprocess data. Returns the path to processed data."""
    print(f"Preprocessing {dataset}...")
    out_dir = Path("data/processed")
    out_dir.mkdir(exist_ok=True)
    # In production: actual preprocessing logic
    (out_dir / "train.pt").write_text("processed train data")
    (out_dir / "val.pt").write_text("processed val data")
    return str(out_dir)


@slurp.task(gpus=4, time="8:00:00")
def train(data_dir: str, lr: float = 0.001, epochs: int = 100) -> dict[str, float]:
    """Train a model. Returns metrics."""
    import random

    random.seed(42)
    loss = 2.0
    for epoch in range(epochs):
        loss = loss * 0.95 + random.uniform(-0.02, 0.02)
        slurp.log_progress(
            step=epoch,
            total_steps=epochs,
            metrics={"loss": round(loss, 4), "lr": lr},
        )

    # Save checkpoint
    Path("checkpoints").mkdir(exist_ok=True)
    Path("checkpoints/best.pt").write_text(f"model lr={lr} loss={loss:.4f}")

    return {"final_loss": round(loss, 4), "lr": lr, "epochs": epochs}


@slurp.task(gpus=1, time="1:00:00")
def evaluate(checkpoint_path: str) -> dict[str, float]:
    """Evaluate the model. Returns metrics."""
    import random

    random.seed(0)
    return {
        "accuracy": round(random.uniform(0.85, 0.99), 4),
        "f1": round(random.uniform(0.80, 0.95), 4),
        "checkpoint": checkpoint_path,
    }


def main() -> None:
    # --- Option A: Fire-and-forget pipeline ---
    # All three stages submitted immediately. SLURM handles dependencies.
    # The return values are discarded; results are written to files.

    # prep_ref = preprocess.remote(dataset="cifar10")
    # train_ref = train.remote(data_dir="data/processed", lr=0.001, epochs=100,
    #                          depends_on=[prep_ref])
    # eval_ref = evaluate.remote(checkpoint_path="checkpoints/best.pt",
    #                            depends_on=[train_ref])
    # slurp.join()

    # --- Option B: Blocking pipeline (Ray-style with results) ---
    # Each stage blocks until complete, then passes the result forward.
    # This is simpler for scripting but requires the full pipeline to
    # finish before you get any output.

    print("Stage 1: Preprocessing")
    prep_result = slurp.get(preprocess.remote(dataset="cifar10"))
    print(f"  Processed data: {prep_result}")

    print("Stage 2: Training")
    train_result = slurp.get(
        train.remote(data_dir=prep_result, lr=0.001, epochs=50)
    )
    print(f"  Training result: {train_result}")

    print("Stage 3: Evaluation")
    eval_result = slurp.get(
        evaluate.remote(checkpoint_path="checkpoints/best.pt")
    )
    print(f"  Evaluation result: {eval_result}")

    print("\nPipeline complete!")
    print(f"  Final accuracy: {eval_result['accuracy']}")


if __name__ == "__main__":
    # For local testing (no SLURM):
    # slurp.set_dispatcher(slurp.LocalDispatcher())
    main()
