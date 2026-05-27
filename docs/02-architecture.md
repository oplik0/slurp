# 02 — Architecture and Module Design

## 1. Directory Structure

```
src/slurp/
├── __init__.py              # Public API exports: submit, Experiment, log_progress
├── domain.py                # Pydantic models: Job, JobResult, Profile, ResourceRequest
├── client.py                # SyncClient: synchronous facade over core modules
├── core/
│   ├── __init__.py          # (empty — no public exports from core)
│   ├── ssh.py               # Control master lifecycle + asyncssh transport
│   ├── slurm.py             # SBATCH generation, sbatch/squeue/sacct/scancel wrappers
│   ├── sync.py              # rsync invocation, snapshot logic
│   ├── launcher.py          # TorchrunLauncher: multi-node command generation
│   └── store.py             # Atomic JSON job store with file locking
├── cli/
│   ├── __init__.py          # (empty)
│   ├── main.py              # Typer app entry point, top-level exception handler
│   ├── submit.py            # submit, run, submit-array command implementations
│   ├── watch.py             # watch command: rich.live.Live table
│   ├── logs.py              # logs command: incremental tail + follow
│   ├── status.py            # status and list commands
│   ├── cancel.py            # cancel and cancel-all commands
│   ├── pull.py              # pull results from remote to local
│   ├── sync.py              # sync-only command
│   ├── debug.py             # debug tunnel command (v0.2)
│   └── config.py            # profile add/list/edit commands
└── helpers/
    ├── __init__.py          # (empty)
    └── debug.py             # debugpy.setup() helper for remote debugging
```

**v0.2 additions (not present in v0.1 source tree):**

```
src/slurp/webui/
├── app.py                   # FastAPI application factory
├── routes.py                # REST endpoints (JSON + HTML)
├── sse.py                   # Server-Sent Events stream for live job updates
├── templates/
│   └── index.html           # Jinja2 dashboard template
└── static/
    └── dashboard.js         # Minimal vanilla-JS frontend
```

---

## 2. Module Boundaries and Interface Contracts

### 2.1 `domain.py` — Value Objects

`domain.py` contains **pure data** — no I/O, no side effects. All models are Pydantic v2 `BaseModel` subclasses (not dataclasses) to get free validation, JSON serialization, and schema documentation.

**Public types:**
- `Job` — immutable handle representing a single SLURM job
- `JobResult` — terminal state capture (exit code, stdout/stderr, resource usage)
- `ArrayJob` — handle for a SLURM job array
- `Profile` — parsed representation of a TOML profile section
- `ResourceRequest` — normalized resource specification (GPUs, nodes, time, memory, etc.)
- `JobStatus` — enum: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`, `TIMEOUT`

**Invariants enforced by Pydantic:**
- `time` must match SLURM duration format (`[[HH:]MM:]SS` or `D-HH:MM:SS`)
- `gpus` must be non-negative; `nodes` must be positive
- `name` is slugified to `[a-zA-Z0-9_-]+` to prevent SBATCH parsing issues

**Private / internal:** None. `domain.py` is fully public and safe to import by downstream code.

### 2.2 `core/` — Private Implementation

Nothing in `core/` is imported by end users. The public API (`client.py`) is the only authorized consumer. This boundary allows internal refactoring without breaking downstream code.

#### `core/ssh.py`

**Responsibility:** Establish and maintain a persistent SSH connection to the cluster login node.

**Public interface (within core only):**
```python
async def ensure_control_master(profile: Profile) -> Path:
    """Return path to control master socket; spawn if missing."""

async def run_remote(
    profile: Profile,
    command: str,
    *,
    timeout: float | None = None,
    stdin: bytes | None = None,
) -> tuple[int, str, str]:  # (exit_code, stdout, stderr)
```

**Internal mechanics:**
- Uses `subprocess` to spawn `ssh -MNf -S <socket>` for the control master
- Uses `asyncssh` connecting through the Unix socket for multiplexed command execution
- Auto-reconnect loop: `ssh -O check` → respawn on failure → retry with exponential backoff

**Failure modes handled internally:**
- Control master dies during long `watch` session → transparent respawn
- Jump host (`ProxyJump`) unreachable → `SSHError` with actionable hint
- Host key change → propagates OpenSSH's strict host-key error verbatim

#### `core/slurm.py`

**Responsibility:** Generate SBATCH scripts and wrap SLURM binaries.

**Public interface (within core only):**
```python
def generate_sbatch_script(
    resource_request: ResourceRequest,
    profile: Profile,
    command: str,
    *,
    job_name: str,
    working_dir: Path,
    snapshot: bool = False,
) -> str:
    """Return full SBATCH script as a string."""

async def sbatch_submit(
    profile: Profile,
    script: str,
    *,
    working_dir: Path,
) -> str:  # job_id

async def sacct_query(
    profile: Profile,
    job_ids: list[str],
) -> list[Job]:

async def scancel(profile: Profile, job_id: str) -> None:
```

**Design note:** `slurm.py` is **cluster-agnostic.** It does not know about JURECA, `jutil`, or `module load`. All cluster-specific strings come from `profile.prologue`. This is the single most important architectural invariant.

**Failure modes:**
- `sbatch` returns non-zero with stderr → `SlurmError` parsed from stderr fragment
- Job ID not found in `sacct` → returns `Job` with status `UNKNOWN` (reconciled later)
- `squeue` output format changes (rare) → sacct is preferred; squeue is fallback only

#### `core/sync.py`

**Responsibility:** Mirror local code to remote working directory.

**Public interface (within core only):**
```python
async def sync_to_remote(
    profile: Profile,
    local_dir: Path,
    remote_dir: Path,
    *,
    exclude_gitignore: bool = True,
) -> None:

async def snapshot_remote(
    profile: Profile,
    remote_dir: Path,
    job_id: str,
) -> Path:
```

**Implementation detail:** Uses `rsync -avz --delete --filter=':- .gitignore'` over the existing SSH control master. No separate SSH connection is opened. The `--filter` flag respects `.gitignore` automatically, preventing `__pycache__`, `.git/`, and virtual-env directories from polluting the remote tree.

**Failure mode:** If `rsync` exits non-zero (e.g., disk quota exceeded on remote), the error is surfaced before `sbatch` is called, preventing a job that would fail immediately due to missing code.

#### `core/launcher.py`

**Responsibility:** Generate multi-node PyTorch Distributed launch commands.

**Public interface (within core only):**
```python
def build_torchrun_command(
    user_command: str,
    profile: Profile,
    *,
    nodes: int,
    gpus_per_node: int,
) -> str:
    """Return shell command string with srun + torchrun prefix."""
```

**Invariants:**
- For `nodes == 1`, the launcher returns the user command unchanged (no `srun` wrapper needed)
- For `nodes > 1`, it exports `MASTER_ADDR`, `MASTER_PORT`, `OMP_NUM_THREADS=1`, and wraps with `srun --mpi=pmi2 --cpu-bind=cores`
- Profile fields `mpi_mode` and `cpu_bind` override defaults

**Failure mode:** If `nodes > 1` but `profile.mpi_mode` is missing, it falls back to `pmi2` and logs a warning.

#### `core/store.py`

**Responsibility:** Atomic read-modify-write of the local job cache.

**Public interface (within core only):**
```python
def read_jobs() -> dict[str, JobRecord]:
def write_jobs(jobs: dict[str, JobRecord]) -> None:
def append_job(job: JobRecord) -> None:
```

**Implementation detail:** Writes to a temporary file in the same directory, then `os.rename()` for atomic replacement. File locking via `fcntl.flock` (Unix) or `portalocker` (cross-platform) ensures two concurrent `slurp submit` processes do not corrupt the JSON. The lock is held only during the read-write cycle (< 10 ms), never across the SSH round-trip.

### 2.3 `cli/` — User Interface

Each file in `cli/` is a thin translation layer: parse CLI arguments → call `SyncClient` → format output with Rich. No CLI module contains business logic.

**Import rule:** `cli/` may import `client.py` and `domain.py`. It must not import `core/` directly.

### 2.4 `client.py` — Public API

`SyncClient` is the single public entry point for programmatic use. It exposes the same operations as the CLI but as typed Python functions.

```python
class SyncClient:
    def submit(self, command: str, *, gpus: int = 0, ...) -> Job: ...
    def submit_array(self, command: str, configs: list[dict], ...) -> ArrayJob: ...
    def watch(self, experiment: str | None = None) -> None: ...
    def logs(self, job_id: str, *, follow: bool = False) -> Iterator[str]: ...
    def cancel(self, job_id: str) -> Job: ...
    def sync(self) -> None: ...
```

**Design note:** There is intentionally no `AsyncClient` in v0.1. All async I/O is hidden inside `core/`. The sync API manages an internal `asyncio` event loop. This keeps the public surface simple for notebook users who do not want to think about `await`.

---

## 3. Data Flow Diagrams

### 3.1 Submit Flow

```
User CLI                    SyncClient                  core/slurm.py               SLURM
   |                            |                            |                          |
   | slurp submit train.py      |                            |                          |
   |--------------------------->|                            |                          |
   |                            | 1. resolve_profile()       |                          |
   |                            | 2. build ResourceRequest   |                          |
   |                            |---------------------------->|                          |
   |                            |                            | generate_sbatch_script() |
   |                            |                            |------------------------->|
   |                            |                            |    sbatch <script>       |
   |                            |                            |<-------------------------|
   |                            |<---------------------------|     job_id                |
   |                            | 3. sync_to_remote()        |                          |
   |                            |--------------------------->|                          |
   |                            |    (rsync over SSH)        |                          |
   |                            |<---------------------------|                          |
   |                            | 4. store.append_job()      |                          |
   |                            |--------------------------->|                          |
   |<---------------------------|     Job object               |                          |
   |  "Job 12345 submitted"     |                            |                          |
```

**Key invariant:** Sync happens *before* `sbatch`. If rsync fails, the job is never submitted. This prevents the common failure mode where a job starts on a node but the code directory is only partially transferred.

### 3.2 Watch Flow

```
User CLI                    SyncClient                  core/ssh.py                 SLURM
   |                            |                            |                          |
   | slurp watch                |                            |                          |
   |--------------------------->|                            |                          |
   |                            | 1. read_jobs()               |                          |
   |                            | 2. sacct_query(all_job_ids)  |                          |
   |                            |--------------------------->|                          |
   |                            |    run_remote("sacct ...") |                          |
   |                            |<---------------------------|                          |
   |                            | 3. For each RUNNING job:     |                          |
   |                            |    logs.tail(job_id, offset) |                          |
   |                            |--------------------------->|                          |
   |                            |    run_remote("tail -c +N")  |                          |
   |                            |<---------------------------|                          |
   |                            | 4. parse progress.jsonl      |                          |
   |                            | 5. render rich.live table    |                          |
   |<---------------------------|  (update every 2-5s)        |                          |
   |  Live table output         |                            |                          |
```

**Key invariant:** All concurrent tail commands execute over a single SSH connection (asyncssh multiplexing). Watching 20 jobs requires 20 `tail` processes on the login node but only 1 SSH connection from the laptop.

### 3.3 Log Streaming Flow

```
User CLI                    SyncClient                  core/ssh.py                 Remote FS
   |                            |                            |                          |
   | slurp logs 12345 --follow  |                            |                          |
   |--------------------------->|                            |                          |
   |                            | 1. resolve log path          |                          |
   |                            |    (deterministic: slurm-<name>-<id>.out)                |
   |                            | 2. open_remote_log()         |                          |
   |                            |--------------------------->|                          |
   |                            |    run_remote("tail -c +0 -f")                         |
   |                            |<---------------------------|  stream stdout lines      |
   |<---------------------------|  yield lines to CLI          |                          |
   |  Printed output            |                            |                          |
   |                            | 3. on disconnect:            |                          |
   |                            |     - persist offset         |                          |
   |                            |     - reconnect control master                          |
   |                            |     - resume from offset     |                          |
```

**Key invariant:** Log offsets are persisted to `~/.local/share/slurp/log_offsets.json`. If the SSH connection drops (laptop sleep, login node restart), the next `slurp logs --follow` resumes from the last known byte offset, not from the beginning.

---

## 4. Import Graph

```
                    +-------------------+
                    |   domain.py       |
                    | (no dependencies) |
                    +---------+---------+
                              ^
              +---------------+---------------+
              |                               |
      +-------+-------+               +-------+-------+
      |  client.py    |               |  cli/*.py     |
      |  (public API) |               |  (UI layer)   |
      +-------+-------+               +-------+-------+
              |                               |
              |         +---------------------+
              |         |
      +-------+---------+-------+
      |        core/              |
      |  +-----+-----+ +-----+    |
      |  | ssh.py    | | slurm.py| |
      |  +-----+-----+ +-----+    |
      |        |           |      |
      |  +-----+-----+ +---+----+ |
      |  | sync.py   | |store.py | |
      |  +-----------+ +--------+ |
      |        |                  |
      |  +-----+-----+            |
      |  |launcher.py|            |
      |  +-----------+            |
      +---------------------------+
```

**Rules:**
- `domain.py` has no imports from other slurp modules.
- `client.py` imports `domain.py` and all `core/` modules.
- `cli/` imports `client.py` and `domain.py` only.
- `core/` modules may import `domain.py` and each other freely, but must not import `client.py` or `cli/`.
- `helpers/` is leaf code imported by user scripts, not by slurp internals.

---

## 5. Public vs Private Surfaces

| Surface | Modules | Consumers | Stability Guarantee |
|---------|---------|-----------|---------------------|
| **Public API** | `slurp.__init__`, `client.py`, `domain.py` | End-user scripts, notebooks, web UI | SemVer-major stable |
| **CLI** | `cli/*.py` | Terminal users | Command syntax SemVer-minor stable |
| **Internal** | `core/*.py`, `helpers/*.py` | `client.py` and `cli/` only | No stability guarantee; may refactor any time |

The `core/` modules are not hidden by `__all__` or underscore prefixes — they are simply not imported by the public `__init__.py`. This is a social contract, not an enforcement mechanism. Documentation explicitly warns users not to import from `slurp.core`.
