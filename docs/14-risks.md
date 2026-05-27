# 14 — Project Risks and Mitigations

This document expands the risk matrix from the final design plan into detailed failure scenarios, early warning indicators, and contingency plans. Each risk is scored by probability (Low / Medium / High) and impact (Low / Medium / High) based on pre-release assumptions.

---

## 1. State Corruption from Concurrent Processes

**Probability:** Medium  
**Impact:** High

### Description

Multiple `slurp` CLI processes (or a CLI process and the web UI) may attempt to read and write `~/.local/share/slurp/jobs.json` simultaneously. Without proper serialization, one process may read a partially written file, overwrite another process's update, or leave the store in an unparseable state.

### Mitigation Details

1. **Atomic writes:** Every write creates a temporary file in the same directory, then calls `os.replace()` (atomic on POSIX). There is never a window where the store file is partially written.
2. **File locking:** `fcntl.flock` (Unix) or `portalocker` (cross-platform) is acquired in exclusive mode (`LOCK_EX`) before read-modify-write cycles. The lock is held for < 10ms — only for the JSON serialization and rename, not across the SSH round-trip.
3. **SLURM as source of truth:** The local store is a cache. If the file is corrupted or lost, `slurp list` or `slurp watch` reconciles against `sacct` and rebuilds the cache. No job is orphaned.
4. **Graceful degradation:** If `fcntl.flock` fails (e.g., on a network filesystem without lock support), the code falls back to a naive write with a warning: `Warning: file locking unavailable. Avoid concurrent slurp processes.`

### Early Warning Indicators

- Unit test `test_store_concurrent_writes` begins failing on CI.
- Bug reports mentioning "job disappeared after submit" or "duplicate job IDs in list."
- JSON decode errors in `~/.local/share/slurp/jobs.json` (detected via `structlog` error telemetry if opt-in logging is enabled).

### Contingency Plan

If atomic writes + locking prove insufficient on high-latency filesystems (e.g., Lustre), migrate to SQLite. SQLite handles concurrency natively and is still a single file. The migration would be transparent: on first read of the old `jobs.json`, import into `jobs.sqlite`, then use SQLite going forward.

---

## 2. JURECA Specificity Baked into Core Code

**Probability:** Medium  
**Impact:** Medium

### Description

slurp was designed for the Jülich JURECA environment. If cluster-specific defaults (e.g., `jutil env activate`, `--gres=gpu:`, `module load Stages/2024`) leak into `core/slurm.py`, the tool becomes unusable on other SLURM clusters.

### Mitigation Details

1. **Profile-specific templates:** All cluster-specific boilerplate lives in the profile `prologue` field. The core `core/slurm.py` generates directives from `ResourceRequest` and prepends the profile `prologue` verbatim. It does not know what JURECA is.
2. **No hard-coded cluster detection in core:** JURECA-aware defaults are in the default profile template shipped with the package (`slurp/templates/profiles/jureca.toml`), not in Python code.
3. **GPU resource abstraction:** `--gpus N` maps to `--gres=gpu:N` or `--gpus=N` based on a per-profile setting `gpu_flag_style`, defaulting to `gres`. Other clusters set `gpu_flag_style = "gpus"` in their profile.
4. **Testing on non-JURECA clusters:** Integration tests run against the generic `giovtorres/slurm-docker-cluster` image, which has no JURECA-specific setup. This forces the code to work without JURECA defaults.

### Early Warning Indicators

- A user reports slurp fails on their university cluster because it injects `jutil` or `module load` commands that do not exist.
- The `core/slurm.py` module imports anything from `jureca` or `fz-juelich`.
- Profile templates contain SLURM directives (e.g., `#SBATCH --partition`) — these belong in the profile's per-default section, not the prologue.

### Contingency Plan

If JURECA-specific code is found in `core/`, perform a hard refactor: extract every cluster-specific string into a `ClusterTemplate` dataclass and move all instances to `slurp/templates/profiles/`. Add a CI lint rule that blocks any PR adding hard-coded cluster strings to `core/`.

---

## 3. Fragile SSH Lifecycle

**Probability:** High  
**Impact:** High

### Description

Laptop sleep, login node maintenance, VPN drops, and idle timeouts all kill SSH connections. A killed connection during `sbatch` creates ambiguous state: "Did the job submit or not?" A killed connection during `watch` leaves the user with a stale table. A killed connection during `run` may lose log tail progress.

### Mitigation Details

1. **Hybrid transport:** `subprocess ssh -MNf` establishes a persistent OpenSSH control master. `asyncssh` connects through the control master's Unix socket. This gives OpenSSH's battle-tested rekeying, host verification, and jump-host routing, plus Python's async multiplexing.
2. **Auto-reconnect on disconnect:**
   - Before every command, `ssh -O check <profile>` tests the control master (0.5s timeout).
   - If dead: `ssh -MNf <profile>` respawns the control master in the background.
   - Failed commands retry with exponential backoff (1s, 2s, 4s, max 30s).
   - After 3 retries, raise `SSHError` with an actionable hint.
3. **Log offset persistence:** `tail -c +<offset>` offsets are written to `~/.local/share/slurp/log_offsets.json` after every successful read. On reconnect, the client resumes from the last offset. No log data is lost.
4. **Idempotency on submit:** A hash of `(command + resources + working_dir)` is stored locally. If the user retries within 30 seconds and the previous job is still `PENDING`, slurp warns and asks for confirmation. This prevents duplicate jobs from ambiguous `sbatch` state.

### Early Warning Indicators

- Frequent `SSHError` exceptions in user reports.
- Log gaps after reconnect (detected by comparing local offsets to remote file size).
- Duplicate job submissions after network blips.

### Contingency Plan

If the hybrid approach still proves brittle on Windows (where `ssh -MNf` behavior differs), provide a pure-`asyncssh` fallback mode. This sacrifices jump-host routing efficiency but eliminates the subprocess dependency. Document the trade-off and let users choose via `profile.transport = "pure_asyncssh"`.

---

## 4. Copy-on-Submit Hurting Iteration Speed

**Probability:** Medium  
**Impact:** Medium

### Description

If every `slurp submit` copies the entire working directory to a new run-specific folder, rapid iteration loops (fix bug → resubmit → repeat) become painful. A 2GB project with 50 re-submits wastes 100GB of storage and adds minutes of sync time per iteration.

### Mitigation Details

1. **In-place sync is the default:** `rsync -avz --delete --filter=':- .gitignore' ./ remote_dir/` syncs the local tree directly into the remote working directory. No copy is made on submit.
2. **Reproducibility is opt-in:** `slurp submit --snapshot` copies the rsynced tree into `$PROJECT/.slurp/runs/<job_id>/` before execution. This is a deliberate, explicit action.
3. **Dirty git warning:** If the repo has uncommitted changes, slurp prints a warning: `Warning: dirty git repo. Use --snapshot for reproducibility.` It does not block submission.
4. **Incremental rsync:** Subsequent submits sync only changed files. A typical bug-fix resubmit takes < 1 second for a Python-only project.

### Early Warning Indicators

- Users complain that `.slurp/runs/` consumes terabytes of disk space.
- Benchmarks show `slurp submit` taking > 10 seconds for small projects.
- `--snapshot` is used by > 50% of users in anonymous telemetry (if enabled).

### Contingency Plan

If in-place sync causes reproducibility disputes in published research, add a post-submit hook that automatically tags the current git commit and working-dir hash in the job metadata. This provides an audit trail without copying files.

---

## 5. Config Hierarchy Surprising Users

**Probability:** Low  
**Impact:** High

### Description

Complex configuration hierarchies (built-in defaults → `.slurprc.toml` → profile settings → env vars → CLI flags) create "where did this value come from?" bugs. A user sets `--gpus 2` but gets 4 because a forgotten `.slurprc.toml` overrides it.

### Mitigation Details

1. **Only two layers:**
   - **Layer 1:** Built-in defaults (cluster-agnostic, e.g., `time="1:00:00"`).
   - **Layer 2:** CLI flags / Python kwargs. What you type is what you get.
2. **Profiles are not a config layer:** Profiles store SSH connection info and per-profile defaults (`partition`, `account`), but these are explicitly connection-oriented. They do not override arbitrary resource flags.
3. **No `.slurprc.toml` in v0.1:** If teams want shared defaults, they use a `.env` file or a wrapper script. No implicit file is loaded.
4. **Verbose mode transparency:** `slurp submit --verbose` prints the final resolved SBATCH script before submission, showing every directive and its source.

### Early Warning Indicators

- Users report "I set `--gpus 2` but got 4" and the answer is a hidden config file.
- Support requests asking "what is my effective config?"
- Users creating `.slurprc.toml` workarounds despite the design intent.

### Contingency Plan

If demand for shared team defaults becomes overwhelming, introduce a single, explicit `--config FILE.toml` flag. The file is loaded only when pointed to, never implicitly. It is treated as CLI flags in bulk, not as a hidden override layer.

---

## 6. Duplicate Jobs from Ambiguous Submit State

**Probability:** Medium  
**Impact:** Medium

### Description

A dropped SSH connection during `sbatch` leaves the user unsure whether the job was submitted. The natural response is to retry, potentially creating two identical jobs that consume double the resources.

### Mitigation Details

1. **Client-side idempotency hash:** Before calling `sbatch`, slurp computes a SHA-256 hash of `(command + resources + working_dir + profile)`. It stores this hash and a timestamp in the local job store.
2. **Duplicate detection:** On subsequent submit, if a hash match is found within 30 seconds and the associated job is still `PENDING`, slurp prints:
   ```
   Warning: Job 12345 with identical spec submitted 5s ago.
   Submit again? [y/N]
   ```
   The default is `N` (no duplicate).
3. **Post-Ctrl+C recovery:** If the user interrupts the CLI during `sbatch`, slurp queries `sacct -S now-5minutes` on reconnect to see if the job actually landed. If found, it links the job ID to the local store instead of re-submitting.
4. **No remote tokens:** The solution is purely client-side. There is no distributed state machine, no remote lock file, and no SLURM-side metadata.

### Early Warning Indicators

- Users reporting accidental duplicate jobs in `slurp watch`.
- `sacct` showing multiple jobs with identical names and start times within seconds.
- The 30-second window is too short for slow networks; users cancel the warning and resubmit anyway.

### Contingency Plan

If 30 seconds is insufficient, extend to 120 seconds and add a `--force` flag to bypass the warning. If false positives become common (users legitimately want rapid identical submits), allow disabling per-profile: `profile.idempotency = false`.

---

## 7. GPU Monitoring Gap

**Probability:** Low  
**Impact:** Medium

### Description

Users cannot see GPU utilization (memory, compute, temperature) from slurp in v0.1. They must SSH into the compute node or parse `sacct` resource statistics after the job ends. For long-running training jobs, silent GPU underutilization (e.g., data-loading bottleneck, wrong device placement) is a common and expensive waste.

### Mitigation Details

1. **Deferred to v0.2:** GPU monitoring is explicitly out of scope for v0.1. The v0.1 design focuses on job lifecycle management, not runtime introspection.
2. **v0.2 implementation plan:**
   - `--monitor-gpus` flag on `slurp run` that launches `nvidia-smi dmon -s umc -d 5` in the background on the compute node.
   - Parse `dmon` output (CSV-like) into a live Rich panel showing GPU utilization, memory usage, and temperature.
   - Store a summary in `JobResult` (max GPU memory, average utilization).
3. **Alternative for v0.1 users:** Document how to add `nvidia-smi` to the job script prologue and redirect output to a known log file. `slurp logs` can tail this file manually.

### Early Warning Indicators

- Users asking "how do I know my GPUs are actually being used?"
- Training jobs with low `MaxRSS` but high GPU allocation — a sign of CPU-bound data loading.
- Comparison requests with `nvitop`, `gpustat`, or `weave`.

### Contingency Plan

If `nvidia-smi dmon` proves unreliable on JURECA (e.g., nodes lack `nvidia-smi` in PATH, or permissions restrict it), fall back to DCGM (NVIDIA Data Center GPU Manager) if available, or parse `sacct` `--format=MaxGPUUtil` if JURECA exposes it. If no programmatic source exists, drop the feature and document manual monitoring instead.

---

## 8. Risk Summary Matrix

| Risk | Prob. | Impact | Primary Mitigation | Contingency |
|------|-------|--------|-------------------|-------------|
| State corruption | Medium | High | Atomic writes + file locking + SLURM source of truth | Migrate to SQLite |
| JURECA specificity | Medium | Medium | Profile templates, no hard-coded cluster logic | Hard refactor + CI lint rule |
| SSH lifecycle fragility | High | High | Hybrid control master + asyncssh, auto-reconnect | Pure-asyncssh fallback mode |
| Copy-on-submit pain | Medium | Medium | In-place default, `--snapshot` opt-in | Git-commit audit trail |
| Config hierarchy surprise | Low | High | Two layers only, no `.slurprc.toml` | Explicit `--config FILE.toml` |
| Duplicate jobs | Medium | Medium | Client-side hash dedup, 30s window | Extend window, add `--force` |
| GPU monitoring gap | Low | Medium | Deferred to v0.2 with `nvidia-smi dmon` | DCGM or `sacct` fallback |

---

## 9. Risk Monitoring Process

1. **Every release:** Review the risk matrix. Update probabilities based on bug report volume.
2. **Post-mortem on every High-impact incident:** Write a one-page post-mortem within 48 hours. Update mitigations if the existing ones failed.
3. **Quarterly design review:** Re-evaluate deferred risks (e.g., GPU monitoring). If a deferred risk is causing user churn, pull it forward.
