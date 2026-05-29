# slurp — Research Synthesis

This document synthesizes findings from the exploration team into a single overview of the problem space, prior art, technical constraints, and open design questions.

## 1. Problem Statement

We want a Python library + CLI called **slurp** that makes running ML training jobs on SLURM clusters (specifically Jülich JURECA) as simple as possible. The target user knows Python/PyTorch but does not want to learn SLURM internals or write SBATCH scripts. The workflow is:

1. **Local development** — write code on laptop
2. **SSH jump host** — sync code to JURECA login node
3. **Submit** — run the job on SLURM without manual SBATCH writing
4. **Watch** — see real-time progress for single or multiple jobs
5. **Debug** — attach a debugger to a running job when needed
6. **Retrieve** — pull results back locally

**Key non-goals:** No workflow DAG engine, no web UI, no container orchestration, no cloud abstraction, no pickle-based function submission. This is SLURM-only, bare-metal, SSH-centric.

## 2. Target Environment: JURECA

### Access
- SSH to `jrlogin` nodes (jump host). Key-based auth only (Ed25519). Agent forwarding disabled.
- Direct SSH to compute nodes is restricted.

### Compute
- **No internet on compute nodes** (dc-gpu, dc-cpu, dc-hwai). Internet only on login/devel nodes.
- **No containers** — no Pyxis, no Singularity. Bare-metal modules only.
- Partitions:
  - `dc-gpu`: 4x A100 per node, up to 24 nodes, 24h limit
  - `dc-hwai`: 4x H100, restricted access
  - `dc-cpu`: up to 128 nodes
  - `dc-gpu-devel`: 2h, internet allowed (good for debugging)
- **No node sharing** — jobs are exclusive per node.
- `jutil env activate -p <project>` sets up environment variables.
- `mpiexec` NOT supported; use `srun` for MPI/multi-node.
- Auto-injected defaults: `--gres=gpu:4` on GPU partitions, `--gres=mem512` if omitted.

### File Systems
- `$HOME` (small, permanent)
- `$PROJECT` / `$SCRATCH` (large, project-specific)
- Code should live under `$PROJECT`; use `$SCRATCH` for temp data.

## 3. Prior Art: What Works vs What Overcomplicates

### srunx
- **Borrow:** SSH profile + mount auto-sync, transport resolution (`--profile` on every command), incremental log tailing with byte offsets (`LogChunk`), `sbatch --wrap` semantics, SQLite-backed local history.
- **Avoid:** FastAPI+React web UI, MCP server, notification outbox, Jinja2 workflow DAGs, sweep orchestrator with connection pools and atomic DB counters, per-mount file locking + SHA256 verification.

### submitit
- **Borrow:** Minimum viable abstraction (`executor.submit(fn, *args) -> Job`, `job.result()`), `InfoWatcher` batched `sacct` polling with adaptive backoff, concurrent.futures-like API.
- **Avoid:** Pickle-based function submission (ML workloads are command-line driven, not function-based), no SSH transport at all.

### NeMo Run
- **Borrow:** `SSHTunnel` concept, `GitArchivePackager` for clean-commit sync, experiment grouping.
- **Avoid:** Fiddle/Config/Partial abstraction tower, TorchX backend, Pyxis-first design, rigid `Experiment` context manager, ~8 layers from `Config` to `sbatch`, blocking `tail_logs` in Experiment context.

### CISPA Hackathon (Custom DUCI Code)
- **Borrow:** Pragmatic thin SSH wrapping (`subprocess.run(["ssh", host, cmd])`), Rich `Live` table for multi-job watching, `progress.jsonl` file-based progress reporting, adaptive pipeline with simple JSON state file.
- **Avoid:** Hardcoded paths for finding progress files, naive `time.sleep(60)` polling with no backoff, ad-hoc per-project progress parsing.

## 4. Key Technical Decisions

### 4.1 SSH Transport
- **Recommendation:** Use `asyncssh` as primary driver. It is ~15x faster than Paramiko in multi-host benchmarks, natively supports asyncio, ProxyJump/tunneling, and non-blocking streams. Fallback to `subprocess ssh` for file transfers if license (EPL-2.0) is a concern.
- **Pool:** Bounded async pool of 4-8 connections. Respect `MaxSessions` (~10). Use keepalives (30s) and health checks.

### 4.2 Code Sync
- **Recommendation:** `rsync -avz --delete --filter=':- .gitignore' ./ user@host:remote_proj_dir/`
- Sync **synchronously before each submission**. No background daemons, no per-mount locks.
- Warn on dirty git repo, but don't block.
- Opt-in `--sync-strategy=git-archive` for clean-commit reproducibility.
- Preserve user's `#SBATCH` directives by running in-place on the remote when script is under synced mount.

### 4.3 Job Submission
- **Recommendation:** Thin wrapper around `sbatch`. Auto-generate minimal SBATCH script or use `--wrap`.
- JURECA-aware defaults: auto-inject `--gres=gpu:4` on GPU partitions, set `--account` from profile, include `jutil env activate` and venv activation in setup lines.
- Return a `Job` handle with `job_id`, `name`, `status`.
- Support `--experiment=my_exp` tags for grouping.

### 4.4 Log Streaming
- **Primary:** Incremental byte-offset polling (`LogChunk`-style). Poll `tail -c +<offset> <logfile>` every 2-5s. Reuse pooled SSH connection. Track offsets locally; only transfer new bytes.
- **Opt-in:** Blocking `tail -F` for single-job real-time streaming.
- **Avoid:** WebSocket/HTTP daemons on login nodes (likely against policy).

### 4.5 Progress Reporting
- **Primary:** Standardize `progress.jsonl` file-based approach. Provide `slurp.log_progress(epoch, loss, ...)` helper for training scripts. Poll every 5-10s for RUNNING jobs.
- **Secondary:** Batched `sacct` queries for job state and resource usage (MaxRSS, GPU TRES). Use adaptive backoff (30-60s interval). One query for all tracked jobs.
- **Avoid:** stdout-parsing as primary progress channel (fragile). Socket-based streaming impossible on JURECA.

### 4.6 Multi-Job Tracking
- **Data store:** SQLite at `$XDG_DATA_HOME/slurp/jobs.db` for structured job metadata. JSONL optional for append-only audit.
- **Polling strategy:**
  - Status loop: `squeue`/`sacct` every 30-60s for all tracked jobs (batched).
  - Progress loop: `cat progress.jsonl` every 5-10s for RUNNING jobs only.
  - Log loop: `tail_log_incremental` every 2-5s only for actively watched jobs.
- **UI:** `slurp watch` using `rich.live.Live` with a table. Compact one-line-per-job for 100 jobs.

### 4.7 Remote Debugging
- **Level 1 (must):** Document pattern + provide `slurp.debug.setup()` helper that inserts `debugpy.listen()` boilerplate on rank 0.
- **Level 2 (should):** `slurp debug tunnel <job_id>` CLI helper that queries `squeue` for the node, prints/executes the correct `ssh -J ... -L ...` command.
- **Level 3 (could):** Auto-poll `squeue` after submission and establish tunnel automatically when job starts.
- Multi-node: debug on rank 0 only by default.

### 4.8 PyTorch Integration
- Provide `TorchrunLauncher` that auto-generates:
  ```bash
  export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
  export MASTER_PORT=29500
  srun torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc-per-node=$SLURM_GPUS_PER_NODE \
    --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv-id=$SLURM_JOB_ID \
    train.py
  ```
- Use `--ntasks-per-node=1` and let torchrun spawn per-GPU processes.
- Inject deterministic checkpoint path via env var.

## 5. Suggested Core Verbs / API

### Python API
```python
import slurp

# Configure profile once
slurp.config.add_profile("jureca", hostname="jrlogin", username="user", key="~/.ssh/id_ed25519")

# Submit a job
job = slurp.submit(
    "python train.py --lr 0.01",
    profile="jureca",
    gpus=4,
    nodes=1,
    time="2:00:00",
    experiment="exp_v1",
)

# Wait for completion with live logs
job.wait(follow_logs=True)

# Or poll status
print(job.status())  # PENDING, RUNNING, COMPLETED, FAILED, etc.

# Stream logs
for line in job.logs(follow=True):
    print(line)

# Cancel
job.cancel()

# Submit multiple jobs
jobs = [slurp.submit(f"python train.py --seed {s}", ...) for s in range(5)]
slurp.watch(jobs)  # Rich live table

# Download results
slurp.download(job, local_dir="./outputs")
```

### CLI
```bash
# One-time setup
slurp config add-profile jureca --hostname jrlogin --user user --key ~/.ssh/id_ed25519

# Submit
slurp submit --profile jureca --gpus 4 --time 2:00:00 --experiment exp_v1 -- python train.py --lr 0.01

# Watch all jobs in experiment
slurp watch --experiment exp_v1 --profile jureca

# Stream logs for a job
slurp logs <job_id> --follow --profile jureca

# Status
slurp status --experiment exp_v1

# Cancel
slurp cancel <job_id>

# Download outputs
slurp download <job_id> --local ./outputs

# Debug tunnel
slurp debug tunnel <job_id>
```

## 6. Architecture Sketch

```
src/slurp/
├── __init__.py
├── domain/          # Pydantic models: Job, JobStatus, Profile, ResourceRequest
├── core/
│   ├── ssh.py       # asyncssh transport, connection pool, ProxyJump
│   ├── slurm.py     # sbatch/squeue/sacct/scancel wrappers
│   ├── sync.py      # rsync code sync
│   └── launcher.py  # DirectLauncher, TorchrunLauncher
├── cli/
│   ├── main.py      # Typer app
│   ├── submit.py    # slurp submit
│   ├── watch.py     # slurp watch (rich.live)
│   ├── logs.py      # slurp logs
│   └── debug.py     # slurp debug tunnel
├── config/
│   ├── profiles.py  # TOML-based profile storage
│   └── settings.py  # Global defaults
├── monitoring/
│   ├── poller.py    # Batched squeue/sacct with adaptive backoff
│   ├── logs.py      # Incremental log streaming
│   └── progress.py  # progress.jsonl polling
├── db/
│   └── store.py     # SQLite metadata store
└── helpers/
    └── debug.py     # debugpy.setup() helper
```

## 7. Error Handling Strategy

- **SSH transient failures:** Exponential backoff (1s base, 30s cap, 3-5 max attempts).
- **Auth failures:** Fail fast, never retry.
- **Stateful commands (sbatch, scancel):** Do NOT blindly retry. A dropped transport does not mean the command failed.
- **Idempotent commands (squeue, sacct):** Safe to retry.
- **SLURM submission failures:** Parse stderr for known strings (`invalid partition`, `time limit`, etc.) and provide actionable hints.
- **Job failures:** Do NOT auto-retry training jobs. Retrying is opt-in only.
- **Timeouts:** Layered — connect (10s), auth (30s), command exec (600s), watch idle (5min).
- **Cancellation:** `scancel <job_id>` + verify with `squeue`. Guaranteed cleanup via context managers.

## 8. Open Questions for Brainstorming

1. **Should we support job arrays?** SLURM arrays are efficient for parameter sweeps, but complicate log/progress tracking. Is a Python `for` loop calling `submit()` sufficient?

2. **How should experiments be grouped?** By `--experiment` tag? By a Python context manager? By a YAML file?

3. **Should we provide a Python `Experiment` class or keep it purely functional?** NeMo Run's context manager is rigid; srunx has no experiment grouping; CISPA uses ad-hoc strings.

4. **Sync strategy: rsync every time, or only when dirty?** What about large datasets / model checkpoints that shouldn't be synced?

5. **Progress format: standardize `progress.jsonl`, or allow pluggable parsers?** What schema should we enforce?

6. **Should `watch()` block the Python process, or run in a background thread?** How does this integrate with Jupyter notebooks?

7. **Multi-node debugging: is Level 2 (CLI tunnel helper) enough, or do we need Level 3 (auto-tunnel)?**

8. **Should we support `slurp run` that blocks until completion (like submitit's `job.result()`), or only `submit` + separate `watch`?**

9. **How do we handle `jutil env activate` and venv paths?** Auto-detect? Explicit in profile? Per-job?

10. **What is the minimal viable release?** Which verbs are essential for a 0.1.0 vs which can be deferred?

