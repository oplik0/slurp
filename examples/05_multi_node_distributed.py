"""Multi-node distributed training with torchrun.

When nodes > 1, slurp generates a wrapper command that sets up the
PyTorch Distributed environment (MASTER_ADDR, MASTER_PORT, NCCL_DEBUG)
and launches via `srun --mpi=<mpi_mode> torchrun ...`.

The generated wrapper is cluster-agnostic: it reads mpi_mode and
cpu_bind from the profile, not from hard-coded defaults. This means
the same Python code works on JURECA (pmi2) and a PMIx cluster without
changes — the profile carries the cluster-specific bits.

Prerequisites:
    - A profile with mpi_mode and cpu_bind set (see templates/jureca.toml)
    - A DDP-aware train.py (uses torch.distributed.init_process_group)

Run:
    python examples/05_multi_node_distributed.py
"""

from __future__ import annotations

import slurp


def main() -> None:
    # 2 nodes, 4 GPUs per node = 8 total ranks. torchrun gets
    # --nnodes=$SLURM_JOB_NUM_NODES --nproc-per-node=$SLURM_GPUS_PER_NODE
    # so the per-node GPU count is derived from the allocation, not passed
    # separately. This prevents the classic mismatch bug where --nproc
    # doesn't match --gres.
    job = slurp.submit(
        "python train.py --ddp --batch-size 256",
        nodes=2,
        gpus=4,  # per node
        cpus=8,  # per task
        time="4:00:00",
        partition="gpu",
        experiment="multinode",
    )
    print(f"Submitted multi-node job {job.job_id}")
    print(f"  nodes={job.resources.nodes}, gpus={job.resources.gpus}")
    print(f"  partition={job.resources.partition}")

    # What slurp generates internally (you'd see this with --dry-run):
    #
    #   export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    #   export MASTER_PORT=29500
    #   export OMP_NUM_THREADS=1
    #   export NCCL_DEBUG=INFO
    #   export NCCL_DEBUG_FILE=$PROJECT/.slurp/nccl_logs/$SLURM_JOB_ID-%h.log
    #   srun --mpi=pmi2 --cpu-bind=cores --distribution=block:cyclic \
    #     torchrun --nnodes=$SLURM_JOB_NUM_NODES \
    #              --nproc-per-node=$SLURM_GPUS_PER_NODE \
    #              --rdzv-backend=c10d \
    #              --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    #              --rdzv-id=$SLURM_JOB_ID \
    #              train.py --ddp --batch-size 256
    #
    # MASTER_PORT is hard-coded to 29500. If you need a different port
    # (e.g., multiple jobs sharing a node), override via slurm_kwargs —
    # but this is rarely necessary since each job gets its own node set.

    # Block with live logs. Multi-node jobs are long; follow_logs=True
    # lets you watch NCCL initialization and catch topology errors early.
    # The NCCL_DEBUG_FILE output goes to a separate file, not stdout,
    # so the main log stays readable.
    result = job.wait(follow_logs=True, timeout="4h")
    print(f"Job {job.job_id} finished: exit={result.exit_code}, status={result.status}")

    # Common multi-node failure: NCCL IB errors. These appear in stderr
    # as "NCCL WARN NET/IB : Got completion with error". If you see them,
    # check that the profile's prologue sets NCCL_IB_HCA correctly for
    # the cluster's InfiniBand devices. This is a profile concern, not
    # a slurp concern — the core code is cluster-agnostic by design.


if __name__ == "__main__":
    main()
