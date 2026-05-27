# 08 — Code Sync and Log Streaming

This document specifies how slurp mirrors local code to the cluster, manages log file streaming, persists offsets for resume, and handles progress reporting.

---

## 1. Code Sync

### 1.1 Default: In-Place Rsync

On every `slurp submit` (and `slurp run`), slurp synchronizes the local working directory to the remote directory specified by the profile before calling `sbatch`. The default invocation is:

```bash
rsync -avz --delete --filter=':- .gitignore' \
    ./ \
    jrlogin:/p/project1/training2615/user/projects/
```

| Flag | Purpose |
|------|---------|
| `-a` | Archive mode (recursive, preserve permissions, times, symlinks) |
| `-v` | Verbose — list transferred files (useful for debugging sync issues) |
| `-z` | Compress during transfer |
| `--delete` | Remove remote files that no longer exist locally |
| `--filter=':- .gitignore'` | Exclude everything matched by the local `.gitignore` |

The `--filter=':- .gitignore'` rule is critical: it prevents `__pycache__/`, `.git/`, virtual environments, and editor swap files from being copied to the cluster. This is done automatically; the user does not need to maintain a separate exclude list.

**Sync happens before `sbatch`.** If rsync fails (e.g., disk quota exceeded, network timeout), the job is never submitted. This prevents the common failure mode where a job starts on a compute node but the code tree is only partially transferred.

### 1.2 `--snapshot` Opt-In

For reproducibility, the user can pass `--snapshot` on submit:

```bash
slurp submit python train.py --gpus 4 --snapshot
```

When `--snapshot` is enabled:

1. slurp performs the normal in-place rsync.
2. It then copies the synced tree into a run-specific directory on the remote node:
   ```bash
   cp -a /p/project1/training2615/user/projects \
         /p/project1/training2615/user/.slurp/runs/12345/
   ```
3. The SBATCH script's `cd` is set to the snapshot directory, not the original remote working directory.

**Tradeoff:** Snapshots consume additional storage and take an extra `cp -a` after rsync. Use `--snapshot` for final training runs or publication artifacts, not for rapid debugging loops.

### 1.3 Dirty Git Repo Warning

Before syncing, slurp checks whether the local directory is inside a Git repository and whether the working tree is dirty:

```bash
git status --porcelain
```

If the output is non-empty, slurp prints a warning but does **not** block:

```
⚠ Warning: Git working tree is dirty. Job will run with uncommitted changes.
   Consider committing before a reproducibility-critical run.
```

This warning is suppressed if `--snapshot` is used (the snapshot captures the exact state) or if the directory is not a Git repository.

### 1.4 Auto-Sync on Every Submit

Sync is **synchronous and blocking** on every submit. There is no background sync daemon. The typical latency for a small-to-medium project (< 10 MB, < 1,000 files) is 1–3 s on a typical university network. For large projects, the user can run `slurp sync` manually before a batch of submissions:

```bash
slurp sync   # rsync only, no job submission
```

---

## 2. Log Streaming

### 2.1 Incremental Byte-Offset Polling (Primary)

For `slurp watch` and `slurp logs` (without `--follow`), slurp polls log files using byte offsets:

```bash
tail -c +<offset> slurm-train-12345.out
```

- `<offset>` is the number of bytes already consumed.
- `tail -c +N` outputs everything from byte N to the end.
- The client reads the output, appends it to local buffers, and updates the offset.

**Polling intervals:**

| Context | Interval | Rationale |
|---------|----------|-----------|
| `slurp watch` (running jobs) | 2 s | Live table needs frequent updates |
| `slurp logs <job_id>` (no follow) | 1 s | User is waiting for output |
| `slurp logs <job_id> --follow` | blocking | See `tail -F` below |

All `tail` commands execute over the **existing asyncssh connection** via the control master socket. No new SSH connection is opened per poll. `asyncssh` multiplexes multiple concurrent `tail` streams over one socket, so watching 20 jobs requires 20 `tail` processes on the login node but only 1 SSH connection from the client.

### 2.2 Blocking `tail -F` for Single Job Follow

When `slurp logs 12345 --follow` is used, slurp opens a single blocking `tail -F` session:

```bash
tail -F -c +<offset> slurm-train-12345.out
```

- `-F` follows the file by name: if SLURM rotates or renames the log, `tail` reopens the new file.
- The session streams stdout lines back over the asyncssh channel.
- If the SSH connection drops, the `tail -F` process on the login node exits. The client catches the disconnect, reconnects, and resumes from the persisted offset.

**Difference from polling:** `--follow` gives true real-time streaming (no 2-second polling delay) but consumes one long-lived SSH channel. It is only used for single-job follow, not for multi-job watch.

### 2.3 Log Offset Persistence

Log offsets are stored in:

```
~/.local/share/slurp/log_offsets.json
```

Schema:

```json
{
  "12345": {
    "stdout_offset": 8192,
    "stderr_offset": 0,
    "last_read_at": "2024-01-15T10:30:00Z"
  },
  "12346": {
    "stdout_offset": 16384,
    "stderr_offset": 1024,
    "last_read_at": "2024-01-15T10:31:00Z"
  }
}
```

- Keys are `job_id` strings.
- `stdout_offset` and `stderr_offset` are byte counts.
- `last_read_at` is an ISO-8601 UTC timestamp.

**Write semantics:** Offsets are written atomically (temp file + `os.rename`) with `fcntl.flock` to prevent corruption from concurrent `slurp logs` and `slurp watch` processes. Writes happen every 5 s during active streaming, and immediately on disconnect.

**Pruning:** Entries older than 30 days are removed on every write to prevent unbounded growth.

### 2.4 Resume After Disconnect

If the SSH connection drops during `slurp logs --follow` or `slurp watch`, the recovery flow is:

1. Detect disconnect via `asyncssh` connection-lost exception.
2. Persist current offsets to `log_offsets.json` immediately.
3. Trigger auto-reconnect (see SSH Transport spec: `ssh -O check` → respawn → retry).
4. On reconnection, read the last known offset from `log_offsets.json`.
5. Resume `tail -c +<offset>` from that byte position.

**No data loss:** Because offsets are persisted before reconnect, the user sees a gap in output only for bytes written between the last offset write and the disconnect. The default 5-second offset write interval means the maximum loss is ~5 s of output.

### 2.5 Multi-Job Log Streaming

`slurp watch` streams logs for up to 20 concurrent jobs over a single SSH connection. For each `RUNNING` job, slurp opens an independent asyncssh channel running `tail -c +<offset>` every 2 s. The channels are multiplexed by asyncssh over the control master socket.

**Resource limits on the login node:**
- Each `tail` is a shell process.
- 20 `tail` processes + 1 `sacct` query every 5 s is well within typical `MaxSessions` limits.
- If the user monitors > 50 jobs, slurp batches jobs into groups of 50 and warns that watch granularity is reduced.

---

## 3. Progress Reporting

### 3.1 `progress.jsonl` Schema

User scripts can write structured progress to a file named `progress.jsonl` in the working directory:

```json
{"timestamp": "2024-01-15T10:30:00Z", "step": 12, "total_steps": 100, "metrics": {"loss": 0.42, "accuracy": 0.95}, "metadata": {"task": "task2"}}
```

Schema (only `timestamp` and `step` are required):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `timestamp` | ISO-8601 string | Yes | UTC time of the record |
| `step` | integer | Yes | Current iteration (epoch, batch, etc.) |
| `total_steps` | integer | No | Total expected iterations |
| `metrics` | object | No | Free-form key-value numbers/strings |
| `metadata` | object | No | Arbitrary tags (task, fold, seed, etc.) |

**Why generic?** The schema is intentionally not training-specific. It works for data preprocessing, hyperparameter sweeps, evaluation loops, or any stepped process.

### 3.2 Helper: `slurp.log_progress()`

A one-liner for scripts:

```python
import slurp

for epoch in range(100):
    # ... training loop ...
    slurp.log_progress(
        step=epoch,
        total_steps=100,
        metrics={"loss": loss.item(), "accuracy": acc},
        metadata={"task": "task2"},
    )
```

Implementation: appends a JSON line to `progress.jsonl` with an automatic `timestamp`.

### 3.3 TensorBoard Event Parsing (v0.2)

For zero-instrumentation users, slurp v0.2 will parse TensorBoard event files (`events.out.tfevents.*`) directly:

```python
from slurp.monitoring import read_tensorboard_scalars

scalars = read_tensorboard_scalars("./logs/events.out.tfevents.12345")
# Returns: {"loss": [(step, value), ...], "accuracy": [(step, value), ...]}
```

This is deferred to v0.2 because it requires a dependency on `tensorboard` or a lightweight event-file parser, which adds complexity for v0.1.

---

## 4. Polling Intervals Summary

| Operation | Interval | Scope |
|-----------|----------|-------|
| `rsync` on submit | blocking, once per submit | Working directory |
| `tail -c +offset` in `watch` | 2 s | Each running job |
| `sacct` in `watch` | 5 s | All tracked jobs (batched) |
| `progress.jsonl` read | 5 s | Each running job |
| `tail -F` in `logs --follow` | real-time (blocking) | Single job |
| Log offset persistence | 5 s (and on disconnect) | All streamed jobs |

---

## Summary

| Concern | Implementation | Failure handling |
|---------|----------------|------------------|
| Code sync | `rsync -avz --delete --filter=':- .gitignore'` | Fail before `sbatch` if rsync exits non-zero |
| Snapshot | `cp -a` to `~/.slurp/runs/<job_id>/` | Opt-in; extra storage cost |
| Dirty git | `git status --porcelain` | Warn, do not block |
| Log polling | `tail -c +<offset>` every 2 s | Resume from persisted offset after reconnect |
| Log follow | `tail -F` over asyncssh channel | Reconnect + resume on disconnect |
| Offset storage | `~/.local/share/slurp/log_offsets.json` | Atomic writes with file locking; prune > 30 days |
| Progress | `progress.jsonl` + `slurp.log_progress()` | No schema enforcement beyond required fields |
| TensorBoard | Event file parsing (v0.2) | Deferred |

