"""@task batch prediction: shared model + job array.

Translates the Ray batch prediction pattern to SLURM. The model is
serialized once via slurp.put() and shared across all prediction tasks.
Each task loads a different data shard.

Key differences from Ray:
- slurp.put() writes to the shared filesystem (not an in-memory store).
- remote_batch() submits as a single SLURM job array (one sbatch call).
- Results are returned via slurp.get(), but large outputs should be
  written to files at the remote side (as shown here).

Prerequisites:
    - A profile in ~/.config/slurp/profiles.toml
    - slurp installed on the remote cluster
    - Shards in data/shards/ (or adjust the paths)

Run:
    python examples/09_task_batch_prediction.py
"""

from __future__ import annotations

from pathlib import Path

import slurp


def load_model() -> dict[str, float]:
    """Load or construct a model. In production, use torch.load() etc."""
    return {"weights": [0.1, 0.2, 0.3], "bias": 0.05}


@slurp.task(gpus=1, time="1:00:00")
def make_prediction(
    model: dict[str, float],
    shard_path: str,
) -> int:
    """Run prediction on one shard. Returns the count (metadata only).

    Large prediction outputs should be written to files at the remote
    side, not returned — returning them via slurp.get() pickles the
    entire result through SSH, which is slow for big data.
    """
    import json
    import random

    # In production: model = slurp.get(model)  (resolved automatically)
    # The ObjectRef is resolved by the worker before calling this function,
    # so `model` is already the actual dict here.

    # Simulate processing the shard
    shard_size = random.randint(100, 1000)

    # Write predictions to a file (don't return large data)
    shard_name = Path(shard_path).stem
    out_path = Path(f"predictions/{shard_name}_pred.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "shard": shard_name,
        "count": shard_size,
        "model_bias": model["bias"],
    }))

    return shard_size  # return only metadata


def main() -> None:
    model = load_model()

    # Serialize the model once. The file is synced to the shared
    # filesystem and loaded by each worker task.
    model_ref = slurp.put(model)

    shards = [f"data/shards/shard_{i:03d}.parquet" for i in range(12)]

    # Submit as a job array: one sbatch call, 12 tasks.
    # SLURM handles throttling via --array=0-11%20 (max 20 concurrent).
    refs = make_prediction.remote_batch(
        [{"model": model_ref, "shard_path": s} for s in shards],
        # model_ref is an ObjectRef — the worker resolves it to the
        # actual model dict before calling make_prediction.
    )
    print(f"Submitted {len(refs)} prediction tasks")

    # Block and collect metadata results
    results = slurp.get(refs)
    total = sum(results)
    print(f"Total predictions: {total}")
    for i, count in enumerate(results):
        print(f"  Shard {i}: {count} predictions")


if __name__ == "__main__":
    main()
