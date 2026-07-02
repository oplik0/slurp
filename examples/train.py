"""Sample training script for slurp.

This is the script you submit to the cluster:
    slurp submit python examples/train.py --lr 0.001 --epochs 10 --gpus 1

It uses slurp.log_progress() to write a progress.jsonl file in the
working directory. slurp's watch and list commands can pick this up
to show live metrics in the dashboard.

The training loop is fake (random loss), but the structure is real:
parse args, set up the model, loop epochs, log progress, save checkpoints.
Replace the dummy logic with your actual training code.

This script is also runnable locally without SLURM — log_progress()
just appends a JSON line to a file, no cluster required.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import slurp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model (demo)")
    parser.add_argument("--lr", type=str, default="0.001")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ddp", action="store_true", help="Enable DDP (multi-node)")
    parser.add_argument("--data", type=str, default="./data", help="Data directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))

    print(f"Training with lr={args.lr}, seed={args.seed}, epochs={args.epochs}")
    print(f"Data dir: {args.data}")
    print(f"DDP enabled: {args.ddp}")

    # If --ddp was passed, initialize the process group. In a real script
    # you'd use torch.distributed.init_process_group(backend="nccl").
    # slurp sets MASTER_ADDR and MASTER_PORT for you when nodes > 1.
    if args.ddp:
        import os
        print(f"  MASTER_ADDR={os.environ.get('MASTER_ADDR', 'unset')}")
        print(f"  MASTER_PORT={os.environ.get('MASTER_PORT', 'unset')}")
        print(f"  SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'unset')}")

    output_dir = Path("checkpoints")
    output_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        # --- Fake training step ---
        loss = random.uniform(2.0, 2.0) / (epoch + 1) + random.uniform(0, 0.1)
        accuracy = min(0.99, 0.1 + epoch * 0.08 + random.uniform(-0.02, 0.02))

        # Log structured progress. This appends a JSON line to progress.jsonl
        # in the current directory. slurp watch reads this file (if present)
        # and shows the metrics in its live table.
        slurp.log_progress(
            step=epoch,
            total_steps=args.epochs,
            metrics={
                "loss": round(loss, 4),
                "accuracy": round(accuracy, 4),
                "lr": args.lr,
            },
            metadata={
                "seed": args.seed,
                "epoch": str(epoch),
            },
        )

        # Also print to stdout — this goes to slurm-<name>-<job_id>.out
        # and is what `slurp logs <job_id>` reads.
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={loss:.4f}  acc={accuracy:.4f}")

        # Save a checkpoint every 5 epochs (or final).
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            ckpt_path = output_dir / f"checkpoint-epoch{epoch + 1}.json"
            ckpt_path.write_text(json.dumps({
                "epoch": epoch,
                "loss": loss,
                "accuracy": accuracy,
                "lr": args.lr,
                "seed": args.seed,
            }, indent=2))
            print(f"  Saved checkpoint: {ckpt_path}")

    print("\nTraining complete.")
    print(f"Progress log: {Path('progress.jsonl').resolve()}")


if __name__ == "__main__":
    main()
