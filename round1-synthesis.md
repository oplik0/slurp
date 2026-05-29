# slurp — Round 1 Brainstorming Synthesis

This document synthesizes feedback from 7 specialized agents (architect, ml-researcher, api-designer, gpu-engineer, python-engineer, slurm-expert, scripter) into a revised design for slurp.

## Executive Summary

The core vision is validated: a thin, SSH-centric wrapper around SLURM that eliminates NeMo Run's abstraction tower while keeping srunx's best ideas. The key tension is between **simplicity for scripters** (4 concepts, CLI-primary, zero-config) and **power for power users** (job arrays, dependencies, structured APIs). The consensus design resolves this with a **tiered API**: zero-config defaults for beginners, explicit opt-in for advanced features.

## 1. Rejected Ideas (Ruthlessly)

Based on scripter + architect + slurm-expert consensus:

1. **Pickle-based function submission** — Rejected. ML workloads are command-line driven. Keep slurm-only, command-string based.
2. **Jinja2 template rendering for SBATCH** — Rejected. Users write shell scripts with `#SBATCH` or use `--wrap`. No template engine.
3. **Web UI / React frontend** — Rejected. Not needed for a CLI-first tool.
4. **MCP server** — Rejected. Out of scope for v0.1.
5. **Notification outbox / background pollers** — Rejected. Poll on demand only.
6. **TorchX / Kubernetes abstractions** — Rejected. JURECA is bare-metal SLURM.
7. **Fiddle / Config configuration system** — Rejected. YAML/TOML/CLI args are sufficient.
8. **Per-mount file locking and SHA256 verification** — Rejected. Sync before submit, warn on dirty repo.
9. **Auto-resubmit on TIMEOUT** — Rejected per slurm-expert. Explicit `slurp resume <job_id>` only.
10. **SQLite for v0.1** — Rejected per architect. JSONL for v0.1, migrate to SQLite later.
11. **loguru** — Rejected per python-engineer. Use `structlog` instead.
12. **Python 3.10 support** — Rejected per python-engineer. Target 3.11+.

## 2. Core Philosophy: "Simpler than sbatch, not simpler than thinking"

From the scripter's brutal critique:
- **Only 4 required concepts:** `submit`, `watch`, `logs`, `cancel`
- **CLI is primary**, Python API is secondary
- **Zero-config path:** `slurp submit jrlogin python train.py --gpus 4` should work on first run
- **Profiles learned, not required:** Read `~/.ssh/config`, ask interactively for missing info, save for next time
- **Everything else is optional metadata**, not a required abstraction

## 3. Architecture (Revised)

```
src/slurp/
├── __init__.py          # Thin re-exports
├── domain/              # Pydantic models: Job, JobStatus, Profile, ResourceRequest, etc.
├── client.py            # Public API: SyncClient, AsyncClient, Experiment (convenience)
├── core/
│   ├── backend.py       # Backend Protocol + SlurmSSHBackend + LocalSlurmBackend
│   ├── ssh.py           # asyncssh transport + lazy pool
│   ├── slurm.py         # sbatch/squeue/sacct/scancel wrappers
│   ├── sync.py          # rsync code sync
│   ├── launcher.py      # DirectLauncher, TorchrunLauncher
│   └── store.py         # JSONL metadata store (v0.1)
├── cli/                 # Typer apps (thin wrappers around core)
│   ├── main.py
│   ├── submit.py        # slurp submit / slurp run
│   ├── watch.py         # slurp watch (rich.live)
│   ├── logs.py          # slurp logs
│   ├── status.py        # slurp status / slurp list
│   ├── cancel.py        # slurp cancel
│   ├── pull.py          # slurp pull (was download)
│   ├── sync.py          # slurp sync
│   ├── debug.py         # slurp debug tunnel
│   └── config.py        # slurp config
├── config/
│   ├── profiles.py      # TOML profile storage
│   └── settings.py      # Global defaults + per-project .slurprc.toml
├── monitoring/
│   ├── poller.py        # Batched squeue/sacct with adaptive backoff
│   ├── logs.py          # Incremental log streaming
│   └── progress.py      # progress.jsonl / tfevents parsing
└── debug.py             # debugpy.setup() helper
```

## 4. Key Technical Decisions (Revised)

### 4.1 SSH Transport
- **Primary:** `asyncssh` + asyncio internally.
- **Sync API:** `SyncClient` manages event loop internally (handles Jupyter compatibility).
- **Pool:** Lazy-open `WeakValueDictionary[profile, SSHClientConnection]` with keepalive (30s). NOT a hard pool of 4-8.
- **Fallback:** Subprocess `ssh` for file transfers and "robust mode" (`ssh -MNf` control master).
- **Jupyter:** Detect running event loop. If present, use threaded sync fallback or `nest_asyncio`.

### 4.2 Code Sync
- **Default:** `rsync -avz --delete --filter=':- .gitignore' ./ remote_proj_dir/`
- **Sync before each submit** (synchronous, blocking).
- **Copy-on-submit for reproducibility** (ml-researcher requirement): Copy rsynced tree into job-specific dir on remote (e.g. `$PROJECT/.slurp/runs/<job_id>/`). Opt-out with `--in-place` for rapid iteration.
- **Warn if >50 MB** or if dirty git repo. Provide `.slurpignore`.
- **Git provenance:** Capture `git rev-parse HEAD`, `git diff --stat`, `git status` at submit time, store in metadata.

### 4.3 Job Submission
- **Default:** Generate a temp SBATCH script (slurm-expert requirement). Inject JURECA-aware defaults as header lines, then user command.
- **`--wrap` is opt-in** for truly trivial commands.
- **JURECA defaults:**
  - Auto-inject `--gres=gpu:4` on GPU partitions (explicit, overridable)
  - Auto-inject `--gres=mem512` if memory omitted (visible, warn)
  - Include `jutil env activate -p <project>` and venv activation in prologue
  - `module load CUDA/12.2` and `module load PyTorch/2.1` in prologue (per gpu-engineer)
- **Tiered resource API:**
  - Simple: `gpus`, `nodes`, `time`
  - Standard: `mem`, `cpus`, `partition`
  - Advanced: `slurm_kwargs={...}` passthrough for full power
- **Return:** `Job` frozen dataclass with `job_id`, `name`, `status`, `profile`, `experiment`, `submitted_at`.

### 4.4 Log Streaming
- **Primary:** Incremental byte-offset polling with persisted offsets to disk (`~/.local/share/slurp/log_offsets.jsonl`).
- **Batch tail commands:** One SSH exec per cycle for all watched jobs, demuxed with separators.
- **Tiered frequency:** 5s default, 2s for "focused" job, no logs for PENDING.
- **On laptop close:** Write offset checkpoint on SIGTERM/HUP. Resume from checkpoint on reconnect.
- **Opt-in blocking:** `tail -F` for single-job real-time streaming.

### 4.5 Progress Reporting
- **Primary:** `progress.jsonl` with `slurp.log_progress(epoch, loss, ...)` helper. Poll every 5-10s for RUNNING jobs.
- **Alternative:** TensorBoard event file parsing (zero instrumentation for PyTorch Lightning/HF users).
- **Secondary:** Batched `sacct` for resource usage (MaxRSS, GPU TRES). Adaptive backoff, 30-60s interval.
- **Jupyter:** Non-blocking `Watcher` class returning an IPython widget or display-able object.

### 4.6 Multi-Job Tracking
- **Data store:** JSONL at `~/.local/share/slurp/jobs.jsonl` (v0.1). Abstracted behind `core/store.py`.
- **SLURM as source of truth.** Local store is a cache. Reconcile on every `status` / `watch` start.
- **State machine:** `squeue` authoritative for active states, `sacct` for terminal. 60s grace period on conflict.
- **Polling:**
  - Status: batched `sacct` every 30-60s for all tracked jobs.
  - Progress: `cat progress.jsonl` every 5-10s for RUNNING jobs.
  - Logs: incremental tail every 2-5s for actively watched jobs.
- **Rate limiting:** Token-bucket for SSH commands. `squeue`/`sacct` max 1/min (batched). Warn if watching >20 jobs.

### 4.7 Job Arrays and Sweeps
- **Job arrays are first-class** (slurm-expert requirement). `slurp.submit_array(...)` maps to `--array=0-N%M`.
- Return `ArrayJob` handle with `.watch()`, `.logs(task_id=...)`, `.cancel()`.
- **Python `for` loop calling `submit()` is discouraged** past 20 jobs. Warn and suggest arrays.
- **Hydra integration:** Provide `SlurpLauncher` for Hydra's `BasicSweeper` (post-MVP).
- **Array throttle:** Always append `%M` with sensible default (e.g. 20).

### 4.8 Job Dependencies
- **Native support:** `slurp.submit(..., depends_on=[job1, job2])` → `--dependency=afterok:<id1>:<id2>`.
- **Also support:** `afterany`, `afternotok`.
- **Validation:** Detect circular dependencies client-side.

### 4.9 Remote Debugging
- **Level 1 (must):** `slurp.debug.setup()` helper. Document pattern.
- **Level 2 (should):** `slurp debug tunnel <job_id>` CLI helper. Queries `squeue` for node, prints/executes `ssh -J ... -L ...`.
- **Level 3 (could):** Auto-poll and establish tunnel (post-MVP).
- Multi-node: debug on rank 0 only by default.

### 4.10 PyTorch / Multi-Node Integration
- **TorchrunLauncher** auto-generates correct `srun torchrun ...`:
  ```bash
  export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
  export MASTER_PORT=29500
  export OMP_NUM_THREADS=1
  srun --mpi=pmi2 --cpu-bind=cores --distribution=block:cyclic \
    torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc-per-node=$SLURM_GPUS_PER_NODE \
    --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv-id=$SLURM_JOB_ID \
    train.py
  ```
- **NCCL debugging:** Auto-enable `NCCL_DEBUG=INFO` + `NCCL_DEBUG_FILE` for `nodes > 1`.
- **InfiniBand:** Auto-export `NCCL_IB_HCA` via profile setting.
- **GPU monitoring:** `--monitor-gpus` runs `nvidia-smi dmon` per node.
- **Pre-flight health check:** Small `all_reduce` micro-benchmark before training (post-MVP).

### 4.11 Error Handling & Retry
- **Exception hierarchy:**
  ```
  SlurpError
  ├── ConfigError
  │   ├── ProfileNotFoundError
  │   └── InvalidResourceError
  ├── SSHError
  │   ├── SSHAuthError (never retry)
  │   └── SSHConnectionError (retry with backoff)
  ├── SlurmError
  │   └── SlurmSubmissionError (actionable hints)
  └── JobError
      ├── JobFailedError
      ├── JobCancelledError
      └── JobTimeoutError
  ```
- **Retry rules:** Exponential backoff (1s base, 30s cap, 3-5 max) for transient network. Never retry auth or stateful commands.
- **Idempotency tokens:** Write `.slurp-submit-{token}.json` to remote before sbatch. On ambiguous outcome, query SLURM for token.

## 5. API Design (Revised)

### 5.1 Python API (Secondary, for Programmatic Use)

```python
import slurp

# Zero-config blocking run ( Tier 1 )
result = slurp.run(
    "python train.py --lr 0.01",
    profile="jureca",  # optional if only one profile
    gpus=4,
    experiment="exp_v1",
)
print(result.stdout[-500:])

# Fine-grained ( Tier 2 )
profile = slurp.get_profile("jureca")
experiment = slurp.Experiment("exp_v1", profile=profile)
job = experiment.submit("python train.py --lr 0.01", gpus=4)

# Job handle
job.refresh()
job.wait(timeout=3600, poll_interval=5.0)
for line in job.logs(follow=True, tail=100):
    print(line)
job.cancel()

# Arrays
array_job = slurp.submit_array(
    "python train.py --seed {seed}",
    configs=[{"seed": 1}, {"seed": 2}, {"seed": 3}],
    gpus=4,
)
array_job.watch()

# Dependencies
preprocess = slurp.submit("python preprocess.py", gpus=0)
train = slurp.submit("python train.py", gpus=4, depends_on=[preprocess])
eval = slurp.submit("python eval.py", gpus=0, depends_on=[train])
```

### 5.2 CLI (Primary)

```bash
# Zero-config first run (learns profile interactively)
slurp submit jrlogin python train.py --gpus 4

# After profile exists
slurp submit python train.py --gpus 4 --experiment exp_v1

# Blocking run (debug loop)
slurp run python train.py --gpus 4 --wait

# Array sweep
slurp submit-array python train.py --seed 1,2,3,4,5 --gpus 4

# Watch
slurp watch --experiment exp_v1

# Logs
slurp logs <job_id> --follow

# Status with diagnostics
slurp status <job_id>
slurp list --experiment exp_v1

# Cancel
slurp cancel <job_id>

# Sync only
slurp sync

# Pull results
slurp pull <job_id> --local ./outputs

# Debug tunnel
slurp debug tunnel <job_id>

# Config
slurp config add-profile jureca --hostname jrlogin --user user
slurp config list-profiles
```

### 5.3 Configuration Hierarchy

Priority (later overrides earlier):
1. Built-in defaults (JURECA-aware)
2. Global config (`~/.config/slurp/config.toml`)
3. Per-project config (`.slurprc.toml` in repo root)
4. Profile defaults
5. Environment variables (`SLURP_*`)
6. CLI flags / Python kwargs

## 6. Multi-Node & GPU Best Practices (gpu-engineer)

- **Explicit `--gres`:** Always emit `--gres=gpu:{gpus}` in SBATCH. Decouple from torchrun worker count.
- **CPU affinity:** `srun --mpi=pmi2 --cpu-bind=cores --distribution=block:cyclic`
- **NUMA:** Auto-compute `OMP_NUM_THREADS = floor(cores / gpus)`. Optional `numactl` binding.
- **Memory:** Set `PYTORCH_CUDA_ALLOC_CONF` via `--memory-optimize` flag.
- **Profiling:** `--profile=nsys` or `--profile=pytorch`. Auto-add `--disable-dcgm`.
- **`slurp doctor --profile jureca`:** Sanity check CUDA/PyTorch versions on devel node.

## 7. Fair Use & Scheduling (slurm-expert)

- **Promote arrays:** Documented pattern for any sweep > 20 jobs.
- **Array throttle:** Always `--array=0-N%20`.
- **Queue-depth guard:** If pending > 50, block and suggest arrays.
- **Rate limit:** Sleep 0.5s between `sbatch` calls in loops.
- **Sanity check:** Warn on extreme GPU-hour requests (24 nodes × 4 GPUs × 24h).

## 8. Integration with Existing Tools (ml-researcher)

- **WandB offline:** Auto-detect `wandb/` usage. `slurp sync-wandb <job_id>` runs `wandb sync` on login node.
- **TensorBoard tunnel:** `slurp tunnel <job_id> --tensorboard --port 6006`.
- **Hydra awareness:** Auto-set `hydra.run.dir=$SCRATCH/slurp-runs/${job_id}` when Hydra detected.
- **Reproduce command:** `slurp reproduce <job_id>` creates local dir with exact code state + runnable script.

## 9. MVP Scope

### Must-have (v0.1.0)
1. `slurp submit` with zero-config / learned profiles
2. `slurp run` (blocking submit + follow logs)
3. `slurp watch` with Rich live table
4. `slurp logs` with incremental tail
5. `slurp status` / `slurp list`
6. `slurp cancel`
7. `slurp sync` + copy-on-submit
8. `slurp config add-profile/list`
9. Job arrays (`slurp submit-array`)
10. Job dependencies
11. `debugpy.setup()` helper

### Should-have (v0.2.0)
12. `slurp debug tunnel`
13. `slurp pull`
14. `slurp doctor`
15. GPU monitoring (`--monitor-gpus`)
16. TensorBoard event parsing
17. Hydra launcher
18. `slurp reproduce`
19. Auto-resume (`slurp resume`)

### Could-defer (v0.3.0+)
20. Async public API (`AsyncClient`)
21. Local SLURM backend (for testing)
22. Pre-flight health check
23. VSCode launch.json generation
24. Plugin system (callbacks → entry points)

## 10. Open Questions for Round 2

1. Should `slurp submit` default to blocking or fire-and-forget? (scripter wants fire-and-forget CLI, ml-researcher wants blocking Python)
2. How should the `--wait` flag interact with `slurp run`? Is `slurp run` just `slurp submit --wait`?
3. Should job arrays use SLURM arrays under the hood, or should `slurp submit-array` be a Python-level loop with client-side throttling?
4. How do we handle the `jutil env activate` auto-injection without confusing users on non-JURECA clusters?
5. Should `slurp watch` show aggregate progress % across all jobs, or only per-job status?
6. What is the exact schema for `progress.jsonl`?
7. Should we support `slurm.submit` from inside a Jupyter cell with a running event loop?
