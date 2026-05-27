# State Management and Job Store

SLURM is the source of truth for all job state. The local store is a cache, not a ledger. This document specifies the local store format, write semantics, locking strategy, reconciliation, and conflict resolution.

---

## SLURM as Source of Truth

The local store (`~/.local/share/slurp/jobs.json`) records job metadata, log offsets, and idempotency hashes. It does **not** maintain a state machine. If the local file says a job is `RUNNING` but `sacct` says `COMPLETED`, `sacct` wins immediately. There is no grace period, no retry, and no "pending confirmation" state.

This design eliminates an entire class of bugs: stale local state, clock skew between client and cluster, and race conditions between multiple clients. The cost is an extra `sacct` query on every `list` or `watch` start, which is negligible (one RTT for up to hundreds of job IDs).

---

## Local Store: `~/.local/share/slurp/jobs.json`

The store is a single JSON file containing a top-level object with three keys:

```json
{
  "jobs": {
    "12345": {
      "job_id": "12345",
      "name": "train",
      "status": "RUNNING",
      "profile": "jureca",
      "experiment": "exp_v1",
      "submitted_at": "2024-01-15T10:30:00Z",
      "command": "python train.py --lr 0.01",
      "resources": {"gpus": 4, "time": "2:00:00"},
      "working_dir": "/p/project1/training2615/user/projects",
      "idempotency_hash": "sha256:abc123...",
      "idempotency_time": "2024-01-15T10:30:00Z"
    }
  },
  "log_offsets": {
    "12345": {
      "out": 4096,
      "err": 2048,
      "last_read": "2024-01-15T10:35:00Z"
    }
  },
  "idempotency": {
    "sha256:abc123...": {
      "job_id": "12345",
      "submitted_at": "2024-01-15T10:30:00Z"
    }
  }
}
```

### `jobs` section

Maps `job_id` to a record. The record contains all metadata needed to reconstruct a `Job` object and to resolve idempotency queries. The `status` field is the last known state from SLURM (or `PENDING` immediately after submission). It is updated on reconciliation.

### `log_offsets` section

Tracks byte offsets for incremental log tailing. Each job has `out` and `err` offsets (bytes read so far) and a `last_read` timestamp. When a log stream resumes after an SSH disconnect, slurp reads from the stored offset.

### `idempotency` section

Maps idempotency hashes to the job that produced them. Used to detect duplicate submissions within the 30-second window.

---

## Atomic Writes: Temp File + `os.rename()`

All writes to `jobs.json` follow this sequence:

1. Acquire an exclusive lock on the file (see File Locking below).
2. Read the current contents into memory.
3. Apply the update (add a job, update a status, merge reconciliation results).
4. Serialize to JSON with `sort_keys=True` and compact formatting.
5. Write to a temporary file in the same directory: `jobs.json.tmp.<pid>`.
6. Call `os.rename(tmp_path, jobs_json_path)`.
7. Release the lock.

`os.rename()` is atomic on POSIX (and on Windows when the destination is on the same filesystem). This ensures that no concurrent reader ever sees a partially written file. The temp file is created in the same directory as the target to guarantee same-filesystem behavior.

**Crash safety:** If the process crashes between steps 5 and 6, the temp file is left behind. On the next write, slurp deletes any stale `.tmp.*` files in the store directory before creating a new temp file. The stale file is harmless to readers.

**Disk full:** If the disk is full, the write to the temp file fails with `OSError`, and the lock is released. The original `jobs.json` remains intact. The user sees a clear error: `SlurpError: Failed to write job store (disk full?)`.

---

## File Locking: `fcntl.flock` for Unix, `portalocker` for Cross-Platform

The store is accessed by multiple concurrent processes: two `slurp submit` calls running in parallel, a `slurp watch` process and a `slurp cancel` process, or a Python script and a CLI invocation.

**Implementation:**

```python
import fcntl

with open(store_path, "r+") as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    # read, modify, write
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

On Windows, `portalocker` is used as a drop-in replacement:

```python
import portalocker

with open(store_path, "r+") as f:
    portalocker.lock(f, portalocker.LOCK_EX)
    # read, modify, write
    portalocker.unlock(f)
```

**Lock duration:** The lock is held only for the duration of the read-modify-write cycle. In normal operation this is **< 10 ms**. The lock is **never** held across an SSH round-trip. The sequence is:

1. Lock, read store, unlock.
2. Submit job to SLURM via SSH (this may take 1–5 seconds).
3. Lock, read store again, merge new job, write, unlock.

This two-phase pattern prevents long lock holds during network latency. The tradeoff is a small window where two processes could submit the same job before the store is updated. This is acceptable because:
- Idempotency is handled by the 30-second hash window, not by the store lock.
- Duplicate jobs are a minor inconvenience, not a correctness issue.
- SLURM allows duplicate submissions; slurp warns about them.

**Lock failure:** If the lock cannot be acquired within 5 seconds (e.g., another process is stuck holding the lock), slurp raises `SlurpError` with a hint: `Another slurp process may be holding the job store. Wait and retry.`

---

## Reconciliation: `sacct` Query on `list` / `watch` Start

When `slurp list`, `slurp watch`, or `job.refresh()` is called, slurp queries `sacct` for all tracked job IDs and refreshes the local cache.

**Query:**
```bash
sacct -j <job_id1>,<job_id2>,... --format=JobID,State,ExitCode,MaxRSS,Elapsed --noheader --parsable2
```

**Process:**
1. Read all `job_id` values from the local store.
2. Issue the `sacct` query in a single batch.
3. Parse the output.
4. For each job, update the `status` field in the local store.
5. If a job is not found in `sacct` (e.g., purged from the SLURM database), mark it as `UNKNOWN` and log a warning.

**Frequency:** Reconciliation runs once at the start of `list` and `watch`. It does not run continuously during `watch`; the watch loop polls `squeue` for live updates and reconciles only if a discrepancy is detected.

**Performance:** For 100 tracked jobs, the `sacct` query completes in ~200 ms on JURECA. The store write is < 10 ms. This is well within the 5-second `watch` poll interval.

---

## No Local State Machine with Grace Periods

slurp does not implement a state machine like `PENDING → RUNNING → COMPLETING → COMPLETED`. It does not use grace periods or "wait for confirmation" steps. The state transition is:

1. After submission: local status = `PENDING` (optimistic).
2. On `refresh()` / `list` / `watch` start: status = `sacct` output.
3. On `watch` loop: status = `squeue` output (faster, but may miss terminal states for recently completed jobs).

If `squeue` and `sacct` disagree, `sacct` wins for terminal states. `squeue` may report `RUNNING` for a job that just finished because the entry has not yet been removed from the queue. `sacct` is the authoritative source for terminal states.

**Example conflict:**
- `squeue` says `RUNNING`.
- `sacct` says `COMPLETED`.
- Resolution: `COMPLETED`.

**Example conflict:**
- `squeue` says `PENDING`.
- `sacct` says `RUNNING`.
- Resolution: `RUNNING` (non-terminal, `squeue` is fresher for pending/running).

**Example conflict:**
- `squeue` says `FAILED`.
- `sacct` says `CANCELLED`.
- Resolution: `CANCELLED` (both are terminal; `sacct` is more authoritative, but in practice either is acceptable because the job is done).

---

## Conflict Resolution: `sacct` Wins Over `squeue` for Terminal States

The conflict resolution rules are:

| `squeue` | `sacct` | Resolved State | Reason |
|----------|---------|----------------|--------|
| RUNNING | COMPLETED | COMPLETED | Terminal state is final |
| RUNNING | FAILED | FAILED | Terminal state is final |
| RUNNING | CANCELLED | CANCELLED | Terminal state is final |
| RUNNING | TIMEOUT | TIMEOUT | Terminal state is final |
| PENDING | RUNNING | RUNNING | Non-terminal, `sacct` is more authoritative |
| PENDING | COMPLETED | COMPLETED | Job finished before `squeue` updated |
| COMPLETED | FAILED | FAILED | `sacct` has more detail |
| COMPLETED | CANCELLED | CANCELLED | `sacct` has more detail |

If both sources report a terminal state but differ, `sacct` wins because it is the accounting database and its states are final. `squeue` is a snapshot and may report stale terminal states.

---

## Log Offsets: `~/.local/share/slurp/log_offsets.json`

Log offsets are stored in a separate file (`log_offsets.json`) within the same `~/.local/share/slurp/` directory. This separation allows log streaming to update offsets without reading the full job store.

**Format:**
```json
{
  "12345": {
    "out": 4096,
    "err": 2048,
    "last_read": "2024-01-15T10:35:00Z"
  }
}
```

**Update semantics:** Log offsets are updated asynchronously during `watch` and `wait(follow_logs=True)`. The write is atomic (temp file + rename) but does not use file locking. Log offsets are best-effort: if a write is lost due to a crash, the next log read will start from the beginning of the file, which is inefficient but correct.

**Pruning:** Offsets for jobs not present in the job store are pruned on the next write. This prevents unbounded growth.

---

## Idempotency Hash Storage

The idempotency hash is computed as SHA-256 of the canonical JSON representation of:

```json
{
  "command": "python train.py --lr 0.01",
  "resources": {"gpus": 4, "time": "2:00:00"},
  "working_dir": "/p/project1/training2615/user/projects",
  "profile": "jureca"
}
```

The hash is stored in two places:
1. Inside the job record (`jobs["<job_id>"]["idempotency_hash"]`).
2. In the top-level `idempotency` mapping (`idempotency["<hash>"] = {"job_id": ..., "submitted_at": ...}`).

The top-level mapping is used for O(1) duplicate detection. When a new submission arrives, slurp computes the hash and checks the mapping. If a matching entry exists and `submitted_at` is within the 30-second window, slurp warns the user.

**Expiry:** Entries older than 30 seconds are removed from the `idempotency` mapping during every store write. The hash remains in the job record for forensic purposes but no longer triggers deduplication.

---

## Concurrent Access Scenarios and Race Condition Prevention

### Scenario 1: Two `slurp submit` calls in parallel

**Process A:** Lock → read → unlock → submit to SLURM → lock → read → write → unlock.
**Process B:** Same sequence.

**Race:** Both processes read the store before either writes. Both submit jobs. Both write. Result: two jobs in the store, no data loss.

**Mitigation:** The 30-second idempotency window catches the duplicate if the commands are identical. If the commands differ, both jobs are valid and should be submitted.

### Scenario 2: `slurp cancel` while `slurp watch` is reconciling

**Process A (watch):** Lock → read → unlock → query `sacct` → lock → merge → write → unlock.
**Process B (cancel):** Lock → read → write `status=CANCELLED` → unlock.

**Race:** If B writes after A reads but before A writes, A's reconciliation may overwrite `CANCELLED` with `RUNNING` (if `sacct` is stale).

**Mitigation:** The merge step in reconciliation checks the existing status. If the local status is terminal (`COMPLETED`, `FAILED`, `CANCELLED`, `TIMEOUT`), it is **never** overwritten by a non-terminal `sacct` state. If `sacct` reports a different terminal state, it wins (see conflict resolution table above). This prevents `cancel` from being clobbered by a stale `squeue` poll.

### Scenario 3: `job.wait()` and `job.refresh()` in different threads

**Thread A:** `job.wait()` polls `sacct` every 5 seconds.
**Thread B:** `job.refresh()` queries `sacct` independently.

**Race:** Both threads may read the same `sacct` output and write to the store.

**Mitigation:** Store writes are atomic and the data is the same, so the order does not matter. The `Job` object is immutable; each thread gets its own `Job` instance. There is no shared mutable state.

### Scenario 4: SSH disconnect during store write

**Process:** Lock → read → write → `os.rename()` → unlock → SSH disconnect.

**Race:** The store write is complete before the SSH disconnect. The disconnect affects the job submission or log streaming, not the local store. The next operation will reconnect and reconcile.

**Mitigation:** Store operations are independent of network state. The only coupling is the two-phase write pattern: the store is updated after the SSH call succeeds, so a disconnect during submission leaves the store in a consistent state (no record of the unconfirmed job). The idempotency `sacct` fallback (Ctrl+C handling) handles this case.

### Scenario 5: Crash during `os.rename()`

**Process:** Lock → write to temp → `os.rename()` → crash before unlock.

**Race:** The lock is held by a dead process. On Linux, `flock` locks are advisory and are released when the file descriptor is closed (which happens on process exit). On Windows, `portalocker` locks are also released on process exit. The next process will acquire the lock normally.

**Mitigation:** The OS releases the lock automatically. The temp file is cleaned up on the next write.

### Scenario 6: Network filesystem (NFS) store directory

**Risk:** `os.rename()` is not atomic across NFS boundaries, and `flock` may not work correctly on all NFS implementations.

**Mitigation:** The store directory is `~/.local/share/slurp/`, which is on the local filesystem in virtually all deployments. slurp does not support placing the store on a network filesystem. If the user moves `~/.local` to NFS, they accept the risk of non-atomic renames and broken locking.

---

## Corruption Recovery

If `jobs.json` is corrupted (invalid JSON, truncated, or zero-length), slurp detects this on the next read and recovers gracefully.

**Detection:**
Every read attempt parses the file with `json.load()`. If parsing raises `JSONDecodeError` or `UnicodeDecodeError`, the file is flagged as corrupt.

**Recovery:**
1. Parse error is logged as a warning: `Job store corrupted; rebuilding from SLURM.`
2. The corrupt file is moved to `jobs.json.corrupt.<timestamp>` for forensic inspection.
3. A fresh empty store (`{"jobs": {}, "log_offsets": {}, "idempotency": {}}`) is created atomically.
4. `slurp list` triggers an immediate `sacct` reconciliation, repopulating the `jobs` section with all jobs still visible in SLURM accounting.
5. Log offsets and idempotency hashes are lost, but job metadata is restored. Log streaming resumes from offset 0 (inefficient but correct). Idempotency deduplication is reset, so a duplicate submission within the window may not be caught until the store is repopulated.

**Prevention:**
Atomic writes (temp file + `os.rename()`) make on-disk corruption extremely unlikely. The only corruption vectors are:
- Power loss or kernel panic between `os.rename()` and `fsync()` (extremely rare; modern filesystems journal metadata).
- A bug in slurp producing malformed JSON (caught by CI tests).
- User manually editing `jobs.json` and introducing syntax errors.

In all cases, recovery is automatic and non-destructive: the corrupt file is preserved, and operation continues after reconciliation.

---

## Summary of Guarantees

| Property | Guarantee | Implementation |
|----------|-----------|----------------|
| Atomic store writes | Yes | Temp file + `os.rename()` |
| Concurrent access safety | Yes | `fcntl.flock` / `portalocker` |
| Lock hold duration | < 10 ms | Two-phase write pattern |
| Source of truth | SLURM | `sacct` reconciliation on every `list`/`watch` |
| No grace periods | Yes | Terminal states are final immediately |
| Idempotency window | 30 seconds | SHA-256 hash + timestamp |
| Log offset durability | Best-effort | Separate file, no locking |
| Crash recovery | Yes | Lock release on process exit, temp file cleanup |
