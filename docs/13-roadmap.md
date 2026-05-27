# 13 — Version Roadmap

This document defines the scope, deliverables, and deferral rationale for each version of slurp. It serves as the contract between maintainers and users: what is promised, what is excluded, and why.

---

## 1. v0.1 — Core CLI and Infrastructure

**Goal:** A researcher can submit, watch, cancel, and sync jobs without learning SLURM syntax.

### Deliverables (9 CLI Commands)

| Command | Purpose |
|---------|---------|
| `slurp submit` | Fire-and-forget job submission with zero-config profile learning |
| `slurp run` | Blocking submit with live log tailing and exit-code propagation |
| `slurp submit-array` | Native SLURM array support with parameter sweep generation |
| `slurp watch` | Live Rich table of all jobs with incremental status updates |
| `slurp logs` | Incremental log tail with byte-offset resume after disconnect |
| `slurp status` / `slurp list` | Single-job detail and filtered job listings |
| `slurp cancel` | Stop one or more jobs by ID |
| `slurp sync` / `slurp pull` | Code sync via rsync and result download |
| `slurp config` | Interactive profile setup and management |

### Core Infrastructure

- **SSH transport:** Hybrid control master (`ssh -MNf`) + `asyncssh` multiplexing with auto-reconnect.
- **Atomic job store:** `~/.local/share/slurp/jobs.json` with `fcntl.flock` locking and SLURM-as-source-of-truth reconciliation.
- **Idempotency:** Client-side hash deduplication within a 30-second window.
- **SBATCH generation:** Automatic wrapper script creation for commands; direct submission for existing scripts.
- **Multi-node launcher:** `TorchrunLauncher` auto-generates `srun` + `torchrun` for `nodes > 1`.

### Explicitly Excluded from v0.1

| Feature | Exclusion Rationale |
|---------|-------------------|
| `slurp doctor` | Needs cluster-specific health checks; better deferred until profile system is battle-tested. |
| `slurp reproduce` | Requires snapshot-by-default, which conflicts with rapid-iteration default. Needs user feedback first. |
| `slurp resume` | Checkpoint detection is fragile and training-framework-specific. Wait for stable `progress.jsonl` first. |
| `slurp cancel-all` | Trivial to implement, but low priority compared to core submit/watch/cancel. |
| `slurp debug tunnel` | Debug workflows are advanced; first-time users won't need this. |
| `--monitor-gpus` | `nvidia-smi dmon` parsing is straightforward but adds CLI surface area. Defer to focus on stability. |
| `progress.jsonl` / `slurp.log_progress()` | Needs a stable schema and user validation before becoming a public API. |
| Job dependencies | SLURM dependency chains are powerful but easy to misuse. Defer until basic submit is solid. |
| `AsyncClient` | The sync API covers 95% of use cases; async adds complexity without proven demand. |
| Local SLURM backend | Docker-based fake SLURM is for testing only, not a user-facing feature. |

---

## 2. v0.2 — Developer Experience and Observability

**Goal:** Power users can debug, resume, and monitor jobs with minimal friction.

### Deliverables

| Feature | Description |
|---------|-------------|
| `slurp doctor` | SSH connectivity test, SBATCH syntax preflight, module availability check, disk-space warning. |
| `slurp reproduce` | Recreate exact job environment locally: clone snapshot, print SBATCH script, suggest local command. |
| `slurp resume` | Auto-resubmit timed-out jobs if checkpoint file exists (configurable path pattern). |
| `slurp cancel-all` | Bulk cancel filtered by experiment, status, or age. |
| `slurp debug tunnel` | SSH port forwarding for remote `debugpy` attachment. |
| `slurp debug config` | Generate VS Code `launch.json` snippet for remote debugging. |
| `--monitor-gpus` | Live GPU utilization parsing from `nvidia-smi dmon` during `slurp run`. |
| `progress.jsonl` / `slurp.log_progress()` | Structured progress reporting with generic schema (not training-specific). |
| Job dependencies | Native SLURM `afterok` / `afterany` chains via `depends_on=[job1, job2]`. |
| `AsyncClient` | Async Python API for use in Jupyter notebooks and async frameworks. |
| Local SLURM backend | Docker-based testing environment for CI and local development. |

### Scope Boundary

v0.2 does **not** include:
- Web UI (deferred to v0.3 to avoid splitting frontend/backend focus).
- TensorBoard or WandB integration (needs stable progress reporting first).
- Plugin system (not enough surface area to justify extensibility hooks).

---

## 3. v0.3 — Web UI and ML Ecosystem Integration

**Goal:** Researchers can monitor jobs in a browser and integrate with standard ML tooling without manual setup.

### Deliverables

| Feature | Description |
|---------|-------------|
| Web UI (`pip install slurp[web]`) | FastAPI + SSE + vanilla JS dashboard. First-party integration, imports `slurp.client` directly. |
| Pre-flight health check | Small distributed `all_reduce` micro-benchmark to verify NCCL/InfiniBand before long training runs. |
| TensorBoard event file parsing | Zero-instrumentation progress tracking by polling `events.out.tfevents.*` files. |
| WandB offline sync helper | `slurp wandb-sync <job_id>` downloads offline WandB run data and pushes to cloud. |
| Hydra launcher (`SlurpLauncher`) | Custom Hydra `BasicSweeper` plugin that submits each config variant as a SLURM array task. |

### Rationale for v0.3 Deferral

| Feature | Why Deferred |
|---------|-------------|
| Web UI | Requires stable `SyncClient` API and a clear separation between core and presentation. Building it on v0.1 would mean rewriting routes as the API changes. |
| Pre-flight health check | Needs multi-node infrastructure from v0.1 and GPU monitoring from v0.2. Only then can we validate the check results. |
| TensorBoard parsing | Requires a stable progress abstraction (`progress.jsonl` in v0.2). Parsing binary event files is more complex than line-delimited JSON. |
| WandB sync | User demand unclear in v0.1. Defer until we have evidence that manual `wandb sync` is a pain point. |
| Hydra launcher | Hydra's plugin API is stable, but we need the array-job infrastructure from v0.1 and dependency chains from v0.2 to implement a useful launcher. |

### Anti-Features in v0.3

- No local database for the web UI.
- No background pollers — all live updates via SSE.
- No workflow editor or visual DAG builder.
- No authentication or multi-user support.

---

## 4. v0.4+ — Extensibility and Advanced Scheduling

**Goal:** slurp becomes a platform that teams can extend without forking.

### Tentative Deliverables

| Feature | Description |
|---------|-------------|
| Plugin system | Entry-point based callbacks: `on_submit`, `on_status_change`, `on_log_line`. Teams can register custom behavior without modifying core. |
| Jupyter notebook widget | `slurp.jupyter.WatchWidget` — embedded Rich-like table for notebook environments. |
| Scheduling hints | Preemptive fair-share optimization: suggest partition, time limit, or node count based on historical queue data. |
| Multi-cluster profiles | Submit to different clusters from the same client; cluster selection via `--profile` or auto-routing by resource availability. |
| REST API server | Headless API for integration with external orchestration tools (separate from the v0.3 web UI). |

### Deferral Rationale

| Feature | Why Deferred to v0.4+ |
|---------|----------------------|
| Plugin system | Not needed until slurp has a large enough user base that custom integrations are requested. The core must be stable first. |
| Jupyter widget | Jupyter support is niche compared to CLI/web. Wait for explicit user demand. |
| Scheduling hints | Requires historical queue data collection, which introduces privacy and storage concerns. Needs design review. |
| Multi-cluster | Complexity is high: different SLURM versions, different prologue conventions, different authentication. Only justified after single-cluster is flawless. |
| REST API server | The v0.3 web UI's internal API is not versioned or documented for external consumers. A public REST API needs OpenAPI specs, auth, and rate limiting. |

---

## 5. Version Support Policy

- **Latest version only:** No LTS releases. The project is pre-1.0 and moves fast.
- **Breaking changes:** Allowed between minor versions (v0.1 → v0.2) with a deprecation notice in the previous version's final release.
- **Migration scripts:** If a breaking change affects the local store format (e.g., `jobs.json` schema change), provide an automatic migration on first run.

---

## 6. Decision Log

| Date | Decision | Context |
|------|----------|---------|
| v0.1 | In-place sync is default; `--snapshot` is opt-in | Rapid iteration is the primary use case; reproducibility is secondary. |
| v0.1 | No `slurp.run()` in Python API | `submit()` + `job.wait()` teaches the model once and avoids API bloat. |
| v0.1 | No full TUI framework | questionary + Rich cover all v0.1 interaction needs; Textual adds complexity. |
| v0.1 | Two config layers only (built-in + CLI) | Profiles are connection info, not a config layer. Eliminates surprise overrides. |
| v0.2 | `AsyncClient` added, but `SyncClient` remains primary | Jupyter and async frameworks need async; CLI users do not. |
| v0.3 | Web UI is first-party, not external package | Tight coupling to `slurp.client` makes an external package brittle. Optional extra keeps core lean. |
