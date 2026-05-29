# slurp — Final Design Plan

> *A Python library + CLI for running ML jobs on SLURM clusters, specifically designed for the Jülich JURECA environment. Significantly simpler than NeMo Run, without unnecessary abstraction layers.*

## 1. Philosophy: Simpler Than sbatch

**Target user:** A researcher who knows Python/PyTorch but does not want to learn SLURM internals.

**Design principle:** The tool should feel like `sbatch` for people who don't know `sbatch` — not like NeMo Run for people who don't want to learn NeMo Run.

**User-facing concepts (v0.1):**
1. `slurp submit` — run my code on the cluster
2. `slurp watch` — show my jobs
3. `slurp logs` — show output
4. `slurp cancel` — stop a job

That's it. Everything else is optional or hidden.

## 2. Zero-Config Path

The first run must require zero pre-existing configuration files (slurp learns interactively, then remembers for next time):

```bash
# First run — interactive fallback for missing info
$ slurp submit jrlogin python train.py --gpus 4
? JURECA account: training2615
? Partition [dc-gpu]: 
? Time limit [2:00:00]: 
? Save as profile 'jureca'? [Y/n]: Y
✓ Profile 'jureca' saved to ~/.config/slurp/profiles.toml
Job 12345 submitted.

# Next run — one-liner
$ slurp submit python train.py --gpus 4
```

slurp reads `~/.ssh/config` for host/user/key. If a Host entry exists, no questions asked. If missing, it asks interactively and remembers.

## 3. Configuration

**Two layers only:**
1. Built-in defaults (cluster-agnostic)
2. CLI flags / Python kwargs

**Profiles** are stored in `~/.config/slurp/profiles.toml` but are NOT a "config layer." They are just SSH connection info + per-profile defaults (partition, account). The user never thinks about "layers."

**Profile TOML schema example (JURECA):**
```toml
[profiles.jureca]
hostname = "jrlogin"
username = "user"
key_file = "~/.ssh/id_ed25519"
proxy_jump = ""  # optional

# Per-profile SLURM defaults
partition = "dc-gpu"
account = "training2615"

# Cluster-specific prologue (injected into every SBATCH script)
prologue = """
jutil env activate -p {account}
module load Stages/2024
module load CUDA/12
module load PyTorch
source $PROJECT/.venv/bin/activate
"""

# Multi-node launcher extras
mpi_mode = "pmi2"
cpu_bind = "cores"

[profiles.jureca.sync]
local = "/home/user/projects"
remote = "/p/project1/training2615/user/projects"
```

The `prologue` field is the key to cluster-agnostic core: all JURECA-specific boilerplate lives in the profile, not in `core/slurm.py`. When a new cluster is added, the user just creates a new profile with different `prologue` and `partition` defaults. The core code remains unchanged.

**No `.slurprc.toml`** for v0.1. If teams want shared defaults, they can set env vars in a `.env` file.

## 4. CLI Design (Primary)

### Core verbs (v0.1)
```bash
# Fire-and-forget submit (default)
slurp submit python train.py --lr 0.01 --gpus 4 --time 2:00:00

# Blocking run with log streaming (sugar for submit --wait --follow-logs)
slurp run python train.py --lr 0.01 --gpus 4
# Internally: slurp run = slurp submit + job.wait(follow_logs=True)
# The run loop:
# 1. Submit the job
# 2. Start incremental log tailing (tail -c +offset, 2s interval)
# 3. Poll sacct every 5s for terminal state
# 4. If terminal state reached: print final status, exit with job's exit code
# 5. If SSH drops: reconnect using control master, resume from last log offset

# Job arrays / sweeps
# CLI generates a wrapper script that maps SLURM_ARRAY_TASK_ID to parameter values
slurp submit-array python train.py --seed 1,2,3,4,5 --gpus 4
# Internally: generates --array=0-4%20, writes a wrapper script that sets seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Alternative: explicit template syntax
slurp submit-array "python train.py --seed {seed}" --seed 1,2,3,4,5 --gpus 4

# Watch all jobs
slurp watch
slurp watch --experiment exp_v1

# Logs
slurp logs 12345
slurp logs 12345 --follow

# Status
slurp status 12345
slurp list --experiment exp_v1

# Cancel
slurp cancel 12345
slurp cancel-all --experiment exp_v1

# Sync code only
slurp sync

# Pull results
slurp pull 12345 --local ./outputs

# Debug tunnel
slurp debug tunnel 12345

# Profile management
slurp config add-profile jureca --hostname jrlogin --user user
slurp config list-profiles
```

### Argument parsing
Commands after `--` are the user's command. If no `--`, trailing positional args are treated as the command:
```bash
slurp submit --gpus 4 -- python train.py --lr 0.01 --time 10
slurp submit --gpus 4 python train.py --lr 0.01
```

### Resource flags (flat, no tiers)
```
--gpus N              # Maps to --gres=gpu:N (JURECA) or --gpus=N (other clusters)
--nodes N             # --nodes=N
--time T              # --time=T
--mem M               # --mem=M
--cpus N              # --cpus-per-task=N
--partition P         # --partition=P
--constraint C        # --constraint=C
--qos Q               # --qos=Q
--account A           # --account=A
--mail-type T         # --mail-type=T
--slurm-kwargs K=V    # Passthrough any SBATCH directive
```

No "Simple/Standard/Advanced" concept. All flags are just flags.

## 5. Python API (Secondary)

```python
import slurp

# Fire-and-forget
job = slurp.submit("python train.py --lr 0.01", gpus=4, time="2:00:00")

# Advanced: passthrough any SBATCH directive
job = slurp.submit("python train.py", gpus=4, slurm_kwargs={"constraint": "a100", "exclude": "node05"})

# Block until done, then get result
result = job.wait(follow_logs=True, timeout="2h")
print(result.exit_code, result.stdout)

# result() is idempotent — safe to call again
result2 = job.result()
assert result is result2  # Same object, cached

# Arrays
array = slurp.submit_array(
    "python train.py --seed {seed}",
    configs=[{"seed": i} for i in range(5)],
    gpus=4,
)
results = array.results(timeout="4h")

# Dependencies
preprocess = slurp.submit("python preprocess.py", gpus=0)
train = slurp.submit("python train.py", gpus=4, depends_on=[preprocess])

# Experiment grouping (optional, convenience wrapper)
# Experiment is just a container that sets a default experiment tag on every job.
# It does NOT manage state, orchestrate, or enforce anything. You can achieve the same
# with --experiment=exp_v1 on every command, but the class saves typing.
exp = slurp.Experiment("exp_v1")
job = exp.submit("python train.py", gpus=4)  # auto-sets experiment="exp_v1"
exp.watch()      # slurp watch --experiment exp_v1
exp.cancel_all() # slurp cancel-all --experiment exp_v1
```

**No `slurp.run()` in Python.** One `submit()` + `job.wait()` teaches the model once.

## 6. Job Object

```python
@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    name: str
    status: JobStatus  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT
    profile: str
    experiment: Optional[str]
    submitted_at: datetime

    def refresh(self) -> "Job": ...        # Returns NEW Job with updated status
    def wait(self, *, timeout=None, follow_logs=False, poll_interval=5.0) -> "JobResult": ...
    def logs(self, *, follow=False, tail=100) -> Iterator[str]: ...
    def cancel(self) -> "Job": ...
    def result(self) -> "JobResult": ...  # Idempotent; returns cached JobResult if wait() was already called

@dataclass(frozen=True, slots=True)
class JobResult:
    job_id: str
    status: JobStatus
    exit_code: Optional[int]
    stdout: str          # Capped at 1MB
    stderr: str
    max_rss_mb: Optional[float]
    wall_time: float

@dataclass(frozen=True, slots=True)
class ArrayJob:
    """Handle for a SLURM job array."""
    array_job_id: str      # Base job ID (e.g., "12345")
    name: str
    profile: str
    experiment: Optional[str]
    submitted_at: datetime
    task_count: int        # Number of array tasks
    throttle: int          # Max concurrent tasks

    def watch(self) -> None: ...
    def logs(self, *, task_id: Optional[int] = None, follow=False, tail=100) -> Iterator[str]: ...
    def cancel(self) -> "ArrayJob": ...
    def cancel_task(self, task_id: int) -> "ArrayJob": ...
    def results(self, *, timeout=None, poll_interval=5.0) -> list[JobResult]: ...
    def tasks(self) -> list[Job]: ...  # Return individual Job handles for each task
```

## 7. Code Sync

**Default: in-place rsync.** `rsync -avz --delete --filter=':- .gitignore' ./ remote_dir/`

**Rapid iteration is the default.** For debugging loops (fix bug → resubmit → repeat), in-place is correct.

**Reproducibility is opt-in:** `slurp submit --snapshot` copies the rsynced tree into `$PROJECT/.slurp/runs/<job_id>/` before running.

**Warn on dirty git repo** but do not block.

**Auto-sync on every submit** (synchronous, blocking). No separate `slurp sync` required for normal use.

## 8. Job Submission Details

### SBATCH generation

**Example generated script for a command:**
```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=2:00:00
#SBATCH --output=/p/project1/training2615/user/projects/slurm-train-12345.out
#SBATCH --error=/p/project1/training2615/user/projects/slurm-train-12345.err
#SBATCH --account=training2615

# Profile prologue (injected from profile.toml)
jutil env activate -p training2615
module load Stages/2024
module load CUDA/12
module load PyTorch
source $PROJECT/.venv/bin/activate

# User command
python train.py --lr 0.01
```

**Log file paths:** slurp explicitly sets `--output` and `--error` to deterministic paths in the remote working directory: `slurm-<job_name>-<job_id>.out` and `.err`. This ensures the client knows exactly where to tail, regardless of SLURM defaults.

**Script detection:**
- **Commands** (e.g. `python train.py`): Generate a temp SBATCH script with injected directives and setup lines.
- **Existing scripts** (e.g. `train.sh` with `#SBATCH` lines): Detect and submit directly, prepending any missing directives.
- **`--wrap`**: Available for truly trivial one-liners.

### JURECA-aware defaults (profile-specific)
Profiles with `hostname` matching `*.fz-juelich.de` or `jureca*` get:
- Auto `--gres=gpu:4` on GPU partitions (explicit, overridable)
- `jutil env activate -p <account>` in prologue
- `module load Stages/2024` + `module load CUDA/12` + `module load PyTorch` in prologue (configurable per profile)
- `srun --mpi=pmi2 --cpu-bind=cores` for multi-node

**These defaults are NOT built-in.** They live in the JURECA profile template. Other clusters have different templates.

### Job arrays
`slurp submit-array` uses native SLURM arrays (`--array=0-N%M`).
- Log files: `slurm-%A_%a.out`
- Progress files: `progress_<task_id>.jsonl`
- Throttle default: 20 concurrent tasks

Python `for` loops calling `submit()` work fine. Warn past 20 jobs, suggest arrays.

### Job dependencies
```python
slurp.submit(..., depends_on=[job1, job2])  # afterok default
slurp.submit(..., depends_on=[job1], depends_on_type="afterany")
```

### Idempotency (v0.1)

**Problem:** A dropped SSH connection during `sbatch` creates ambiguous state: "Did my job submit or not?" The natural user response is to retry, creating duplicate jobs.

**Minimal solution:**
- Store a hash of `(command + resources + working_dir)` in the local job store.
- If the user submits the exact same spec within 30 seconds and the previous job is still PENDING, warn: "Job 12345 with identical spec submitted 5s ago. Submit again? [y/N]"
- If the user Ctrl+C'd during the call, query `sacct -S now-5minutes` to see if it actually landed.
- No remote tokens, no distributed state machine — just client-side deduplication with a short memory.

## 9. Log Streaming & Progress

### Log streaming
- **Primary:** Incremental byte-offset polling. `tail -c +<offset> <logfile>` every 2-5s.
  - All tail commands execute over the **existing asyncssh connection** (via the control master socket) — no new SSH connection per poll.
  - `asyncssh` multiplexes multiple tail streams over one socket, so watching 20 jobs requires 20 concurrent `tail` processes on the login node but only 1 SSH connection.
- **Single job opt-in:** Blocking `tail -F` for real-time streaming.
- **Log offsets:** Persisted to `~/.local/share/slurp/log_offsets.json` for resume after disconnect.

### Progress reporting
- **Primary:** `progress.jsonl` with generic schema (not training-specific):
  ```json
  {"timestamp": "2024-01-15T10:30:00Z", "step": 12, "total_steps": 100, "metrics": {"loss": 0.42, "accuracy": 0.95, "perplexity": 3.1}, "metadata": {"task": "task2", "fold": 3}}
  ```
  Only `timestamp` and `step` are required. `metrics` is a free-form dict. No training-specific fields — handles any task type.
- **Helper:** `slurp.log_progress(step=12, total_steps=100, metrics={...})` — one-liner for scripts.
- **Alternative (v0.2):** TensorBoard event file parsing for zero-instrumentation users.
- **Polling:** Every 5-10s for RUNNING jobs.

### Multi-job watch
`slurp watch` uses `rich.live.Live` with a table showing job name, status, epoch, loss, ETA.

## 10. Error Handling

**Flattened exceptions for v0.1:**
```python
class SlurpError(Exception):
    """Base exception with .message, .hint, .retryable"""

class SSHError(SlurpError): pass
class SlurmError(SlurpError): pass
class JobFailedError(SlurmError): pass
```

Every error includes: what failed, stderr fragment, actionable hint.

## 11. SSH Transport

**Hybrid approach:**
- **Control master:** `subprocess ssh -MNf` to establish a persistent OpenSSH control master. This handles key renegotiation, host key verification, and jump host routing correctly.
- **Command execution:** `asyncssh` connects through the control master's Unix socket. This gives us OpenSSH's reliability for connection setup and Python's ergonomics for multiplexing.

**Why not Fabric/Paramiko?** Paramiko is single-threaded and GIL-bound (~15x slower than asyncssh in multi-host benchmarks). It's sysadmin-oriented, not library-oriented. For our use case — multiplexing many concurrent log streams and polls over one or two jump hosts — asyncssh is the right primitive.

**Why not pure subprocess?** Pure subprocess `ssh` is brittle for programmatic use: raw bytes on stdout, no structured error handling, and streaming requires manual pipe management. It's fine for one-shot commands (like setting up the control master), but terrible for `watch` running 50 incremental tail polls.

**Auto-reconnect:** If the control master dies (laptop sleep, login node maintenance), the next asyncssh command detects the dead socket and triggers a respawn:
1. `ssh -O check <profile>` — test if control master is alive (0.5s timeout)
2. If dead: `ssh -MNf <profile>` — respawn control master in background
3. Retry the failed command with exponential backoff (1s, 2s, 4s, max 30s)
4. If all retries fail, raise `SSHError` with actionable hint: "SSH connection to jrlogin failed. Check network and try again."

**Jupyter:** Sync API manages event loop internally. No `nest_asyncio` required for basic usage.

## 12. State Management

**SLURM is the source of truth.** Local state is a cache only.

**Local store:** Atomic JSON file at `~/.local/share/slurp/jobs.json`. Write to temp file, then `os.rename()` for atomicity.

**File locking for concurrent access:** Use `fcntl.flock` (Unix) or `portalocker` (cross-platform) for read-modify-write operations. Two concurrent `slurp submit` processes must serialize their access to the job store. The lock is held only for the duration of the read-write cycle (typically < 10ms), not across the entire SSH round-trip.

**Reconciliation:** On `slurp list` or `slurp watch` start, query `sacct` for all tracked job IDs and refresh local cache.

**No local state machine with grace periods.** If `squeue` says RUNNING and `sacct` says COMPLETED, `sacct` wins (terminal state is final).

## 13. Architecture

```
src/slurp/
├── __init__.py          # slurp.submit, slurp.Experiment, slurp.log_progress
├── domain.py            # Job, JobResult, JobStatus, Profile, ResourceRequest (Pydantic)
├── client.py            # SyncClient (public API)
├── core/
│   ├── ssh.py           # Control master + asyncssh transport
│   ├── slurm.py         # sbatch/squeue/sacct/scancel wrappers
│   ├── sync.py          # rsync code sync
│   ├── launcher.py      # TorchrunLauncher (auto-generates correct multi-node command)
│   └── store.py         # Atomic JSON job store
├── cli/
│   ├── main.py          # Typer app
│   ├── submit.py        # submit, run, submit-array
│   ├── watch.py         # watch (rich.live)
│   ├── logs.py          # logs
│   ├── status.py        # status, list
│   ├── cancel.py        # cancel, cancel-all
│   ├── pull.py          # pull results
│   ├── sync.py          # sync only
│   ├── debug.py         # debug tunnel
│   └── config.py        # profile management
└── helpers/
    └── debug.py         # debugpy.setup() helper
```

**Simplified from Round 1:** Merged `config/`, `monitoring/`, `db/` into `core/` or eliminated. CLI is 10 commands but each is a thin wrapper. `helpers/` is just one file for debugpy.

**TUI approach:** questionary for interactive prompts (first-run profile setup), Rich for all display. No full TUI framework (Textual, argenta, pytermGUI) in v0.1.

**Web UI (v0.2, `pip install slurp[web]`):**
```
src/slurp/webui/
├── app.py              # FastAPI app
├── routes.py           # API endpoints
├── sse.py              # Server-Sent Events for live updates
├── templates/          # Jinja2 HTML templates
│   └── index.html
└── static/             # Minimal CSS/JS
    └── dashboard.js
```

The web UI imports `slurp.client` directly — no `--json` passthrough. It is a first-party presentation layer, not an external package.

## 14. Multi-Node / GPU Best Practices

`TorchrunLauncher` auto-generates:
```bash
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=1
srun --mpi=pmi2 --cpu-bind=cores --distribution=block:cyclic \
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc-per-node=$SLURM_GPUS_PER_NODE \
  --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv-id=$SLURM_JOB_ID \
  train.py
```

**Auto-behaviors for multi-node (v0.1):**
- `NCCL_DEBUG=INFO` + `NCCL_DEBUG_FILE` for `nodes > 1`
- `NCCL_IB_HCA` auto-export via profile

**Deferred to v0.2:**
- `nvidia-smi dmon` with `--monitor-gpus`

## 15. MVP Scope (v0.1.0)

### Must-have (9 CLI commands)
1. `slurp submit` — zero-config / learned profiles
2. `slurp run` — blocking submit + log streaming
3. `slurp submit-array` — native SLURM array support
4. `slurp watch` — live Rich table
5. `slurp logs` — incremental tail with resume
6. `slurp status` / `slurp list` — job status and filtering
7. `slurp cancel` — stop jobs
8. `slurp sync` + `slurp pull` — code sync and result download
9. `slurp config` — interactive profile setup

**Core infrastructure:** SSH control master transport, atomic JSON job store, idempotency check.

### Deferred to v0.2
- `slurp doctor` — sanity-check environment on devel node
- `slurp reproduce` — recreate exact job state locally
- `slurp resume` — auto-resubmit timed-out jobs with checkpoint detection
- `slurp cancel-all` — bulk cancel by experiment
- `slurp debug tunnel` — SSH port forwarding for remote debugpy
- `slurp debug config` — generate VSCode launch.json
- `--monitor-gpus` — live GPU utilization parsing
- `progress.jsonl` / `slurp.log_progress()` — structured progress reporting
- Job dependencies — native SLURM dependency chains
- `AsyncClient` — async Python API
- Local SLURM backend — Docker-based testing

### Deferred to v0.3
- **Web UI** (`pip install slurp[web]`) — FastAPI + SSE + minimal HTML frontend. First-party integration importing `slurp.client` directly.
- Pre-flight health check — small distributed `all_reduce` micro-benchmark
- TensorBoard event file parsing
- WandB offline sync helper
- Hydra launcher (`SlurpLauncher` for `BasicSweeper`)

### Deferred to v0.4+
- Plugin system (callbacks → entry points)
- Jupyter notebook widget (`slurp.jupyter.WatchWidget`)
- Preemptive scheduling / fair-share optimization hints

## 16. Dependencies

```toml
[project]
requires-python = ">=3.11"
dependencies = [
  "asyncssh>=2.14",
  "pydantic>=2.0",
  "typer>=0.9",
  "rich>=13.0",
  "structlog>=24.0",
  "questionary>=2.0",
]

[project.optional-dependencies]
debug = ["debugpy"]
web = ["fastapi>=0.100", "uvicorn>=0.23", "jinja2>=3.0"]
dev = ["pytest", "pytest-asyncio", "mypy", "ruff", "pre-commit"]
docs = ["mkdocs-material", "mkdocstrings[python]"]
```

**questionary** is used for the zero-config first-run interactive prompts (profile setup). Rich handles all display. No full TUI framework in v0.1.

## 17. Testing Strategy

- **Unit (80%):** Mock asyncssh. Mock SLURM commands with captured stdout/stderr. `tmp_path` fixtures.
- **Integration (15%):** Docker container with fake SLURM (`giovtorres/slurm-docker-cluster`) + SSH server. Full `submit → poll → cancel` locally.
- **E2E/Smoke (5%):** Real JURECA tests marked `@pytest.mark.e2e`. Manual/nightly only.

## 18. Project-Killing Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| **State corruption from concurrent processes** | Atomic JSON writes (temp+rename). No multi-file transactions. SLURM as source of truth. |
| **JURECA specificity baked into core** | All cluster defaults are profile-specific templates. Core is cluster-agnostic. |
| **Fragile SSH lifecycle** | Hybrid: subprocess `ssh -MNf` control master + asyncssh. Auto-reconnect on disconnect. |
| **Copy-on-submit hurting iteration** | Default to in-place. `--snapshot` opt-in. |
| **Config hierarchy surprising users** | Only 2 layers: built-in + CLI flags. Profiles are connection info, not a config layer. |
| **Duplicate jobs from ambiguous submit** | Client-side idempotency (hash of command+resources+timestamp). Warn on duplicate within 30s. |
| **GPU monitoring gap** | Deferred to v0.2 with `--monitor-gpus` and `nvidia-smi dmon` parsing. |

## 19. User Feedback Resolution

1. **In-place default:** Confirmed — in-place is correct for rapid iteration. `--snapshot` opt-in for reproducibility.
2. **`slurp run` vs `slurp submit`:** Confirmed intuitive — matches `srun`/`sbatch` mental model.
3. **SLURM directives:** No missing directives identified.
4. **`progress.jsonl` schema:** Updated to generic schema with `step`/`total_steps`/`metrics`/`metadata`. Not training-specific.
5. **Jupyter/IPython:** Not a priority.
6. **TUI frameworks:** questionary for interactive prompts, Rich for display. No full TUI framework (Textual, argenta, pytermGUI) in v0.1.
7. **SSH transport:** Hybrid approach — subprocess `ssh -MNf` control master + asyncssh for multiplexing. Not Fabric/Paramiko.
8. **Web UI:** First-party integration via `pip install slurp[web]`. FastAPI + SSE + minimal HTML frontend. Imports `slurp.client` directly. No `--json` passthrough.
