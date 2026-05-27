# 05 — SSH Transport

This document specifies the SSH transport layer: the hybrid control-master + asyncssh architecture, alternative approaches considered, auto-reconnect behavior, connection pooling, jump host routing, key management, and event-loop handling for Jupyter.

---

## 1. Hybrid Architecture

slurp uses a **two-layer SSH transport**:

| Layer | Tool | Purpose |
|-------|------|---------|
| Control master | `subprocess` + OpenSSH (`ssh -MNf`) | Establishes and maintains the persistent connection, handles host keys, jump hosts, and key renegotiation |
| Command execution | `asyncssh` (via control-master Unix socket) | Multiplexes many concurrent commands over a single SSH connection |

**Why the split?** OpenSSH is the canonical implementation of the SSH protocol. It handles edge cases (host key rotation, KEX renegotiation, `ProxyJump` chaining, `~/.ssh/config` parsing) correctly. Pure-Python SSH libraries reimplement these and often get them wrong. By letting OpenSSH own the socket and using `asyncssh` only for command multiplexing, slurp gets OpenSSH's reliability with Python's ergonomics.

### Control master setup

```python
async def ensure_control_master(profile: Profile) -> Path:
    """Return path to control master socket; spawn if missing."""
```

Implementation:

```bash
# Spawn control master in background, no command execution, no TTY
ssh -MNf -S ~/.slurp/sockets/jureca-%r@%h:%p \
    -o ControlPersist=600 \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    jrlogin
```

| Flag | Meaning |
|------|---------|
| `-M` | Master mode — accept multiplex connections |
| `-N` | No remote command (just hold the connection) |
| `-f` | Fork to background after authentication |
| `-S <socket>` | Unix socket path for multiplexing |
| `-o ControlPersist=600` | Keep master alive for 600 s after last client disconnects |
| `-o ServerAliveInterval=30` | Send keepalive every 30 s |
| `-o ServerAliveCountMax=3` | Drop connection after 3 unanswered keepalives (90 s total) |

The socket path is deterministic per profile: `~/.slurp/sockets/<slug>.sock`. Deterministic paths allow multiple slurp processes to share the same control master without coordination.

---

## 2. Why Not Fabric, Paramiko, or Pure Subprocess?

### Fabric / Paramiko

- **Single-threaded / GIL-bound.** Paramiko's transport runs on one thread and spends significant time in Python-level crypto. In multi-host benchmarks it is ~15× slower than asyncssh for concurrent operations.
- **Sysadmin-oriented.** Fabric is designed for running ad-hoc commands across fleets of servers. Its connection caching and result APIs are not structured for long-running multiplexed streams.
- **Config gap.** Neither reads `~/.ssh/config` natively; users must re-specify host, user, key, and jump host in Python.

### Pure subprocess `ssh`

- **Brittle for programmatic use.** Raw bytes on stdout, no structured error handling, exit codes conflated with SSH protocol errors.
- **No multiplexing.** Each `ssh` command opens a new TCP connection. Watching 20 jobs with `tail -c +offset` every 2 s would create 20 new SSH connections per poll cycle — unacceptable overhead on shared login nodes.
- **Manual pipe management.** Streaming `tail -F` over subprocess requires threading or `select` on stdout/stderr pipes. `asyncssh` handles backpressure, cancellation, and buffer management automatically.

### asyncssh alone

- **Would require reimplementing OpenSSH features.** Host key verification, jump host routing, and `~/.ssh/config` parsing would need custom code. The hybrid approach delegates these to OpenSSH for free.

---

## 3. Control Master Lifecycle

### Startup

1. Check if socket file exists (`~/.slurp/sockets/<slug>.sock`).
2. If missing, spawn `ssh -MNf` via `asyncio.create_subprocess_exec`.
3. Wait up to 10 s for the socket to appear (poll every 100 ms).
4. If the socket never appears, raise `SSHError` with the `ssh` stderr.

### Health check

```bash
ssh -O check -S ~/.slurp/sockets/jureca.sock jrlogin
```

- Exit code 0: master is alive.
- Exit code 255: master is dead or socket is stale.

### Shutdown

slurp does **not** explicitly kill the control master on exit. OpenSSH's `ControlPersist=600` cleans it up automatically after the last client disconnects. This avoids race conditions where a background `slurp watch` is orphaned by a foreground `slurp submit` exiting and tearing down the socket.

If the user wants immediate cleanup, they can run:

```bash
ssh -O exit -S ~/.slurp/sockets/jureca.sock jrlogin
```

---

## 4. Auto-Reconnect Strategy

If the control master dies (laptop sleep, Wi-Fi dropout, login node restart), the next slurp command detects the dead socket and triggers a 4-step recovery:

### Step 1: Fast check

```bash
ssh -O check -S ~/.slurp/sockets/jureca.sock jrlogin
```

Timeout: **0.5 s**. This is a local Unix-socket operation; it should be instantaneous.

### Step 2: Respawn

If step 1 fails, spawn a new control master:

```bash
ssh -MNf -S ~/.slurp/sockets/jureca.sock jrlogin
```

Timeout: **10 s**. If this also fails, proceed to backoff.

### Step 3: Exponential backoff retry

The original asyncssh command is retried with the following schedule:

| Attempt | Delay | Max wait for this attempt |
|---------|-------|---------------------------|
| 1 (original) | 0 s | 30 s command timeout |
| 2 | 1 s | 30 s |
| 3 | 2 s | 30 s |
| 4 | 4 s | 30 s |
| 5 | 8 s | 30 s |
| 6 | 16 s | 30 s |
| 7+ | 30 s | 30 s |

Total worst-case time before giving up: **~97 s**.

### Step 4: Actionable error

If all retries exhaust, raise `SSHError` with:

- `message`: the original exception text from asyncssh (e.g., `Connection lost`).
- `hint`: a cluster-aware string, e.g., `SSH connection to jrlogin failed. Check network and try again. If on JURECA, verify the login node is reachable via jureca.fz-juelich.de.`
- `retryable`: `True` — the user can retry the same command and it may succeed.

---

## 5. Connection Pooling for Multi-Job Watch

`slurp watch` may monitor 20+ jobs simultaneously. Each job requires:
- a `tail -c +<offset>` poll every 2–5 s,
- a `sacct` query every 5 s,
- optionally a `progress.jsonl` read.

All of these execute as **separate asyncssh sessions multiplexed over a single control master socket**. On the remote login node, each session becomes an independent shell process. On the client side, there is only one TCP connection and one SSH transport.

**Concurrency limits:**

| Resource | Limit | Rationale |
|----------|-------|-----------|
| Max concurrent tail streams | 50 | Login node `MaxSessions` default is usually 10; slurp caps at 50 to be polite |
| Max concurrent sacct queries | 1 | `sacct` is expensive; batch all job IDs into one query |
| Control master socket file | 1 per profile | Shared across all slurp processes |

If the user runs `slurp watch` in two terminal windows, both windows share the same control master and socket. There is no need for a process-level connection pool.

---

## 6. Jump Host Routing

slurp delegates jump host routing entirely to OpenSSH via `~/.ssh/config`. The profile may specify a `proxy_jump` field, but if `~/.ssh/config` contains a matching `Host` entry, that entry takes precedence.

**Example `~/.ssh/config`:**

```ssh-config
Host jureca
    HostName jrlogin.fz-juelich.de
    User training2615
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump jgateway.fz-juelich.de
```

When slurp spawns the control master, it passes the profile's `hostname` as the SSH target. OpenSSH resolves `Host jureca` automatically, including `ProxyJump`, `IdentityFile`, and any `HostKeyAlgorithms` or `KexAlgorithms` overrides.

**If no `~/.ssh/config` entry exists**, slurp uses the profile's `hostname`, `username`, and `key_file` fields and passes them as explicit `ssh` flags:

```bash
ssh -MNf -l training2615 -i ~/.ssh/id_ed25519 jrlogin.fz-juelich.de
```

No jump host is configured in this case unless the profile explicitly sets `proxy_jump`, in which case slurp adds `-J <proxy_jump>`.

---

## 7. Key Management

slurp **never** manages SSH keys itself. It reads key paths from three sources, in order of precedence:

1. `~/.ssh/config` `IdentityFile` for the matching `Host` entry.
2. Profile `key_file` field.
3. OpenSSH default (`~/.ssh/id_rsa`, `~/.ssh/id_ed25519`, etc.).

**Key file handling:**
- Paths starting with `~/` are expanded via `os.path.expanduser()` before being passed to `ssh`.
- If the key file is missing, OpenSSH will prompt for a password or fail with `Permission denied`. slurp propagates this error verbatim.
- slurp does not support password authentication or key unlocking via Python. The user must add the key to `ssh-agent` beforehand.

---

## 8. File Locking for Control Master Respawn

A race condition exists when two concurrent slurp processes detect a dead control master simultaneously:

1. Process A runs `ssh -O check` → fails.
2. Process B runs `ssh -O check` → fails.
3. Both spawn `ssh -MNf` → one wins, the other gets `Address already in use` on the Unix socket.

**Resolution:** slurp uses `fcntl.flock` on a lock file `~/.slurp/sockets/<slug>.lock` during the respawn sequence.

```python
with open(lock_path, "w") as lf:
    fcntl.flock(lf, fcntl.LOCK_EX)        # block until we hold the lock
    if not socket_alive():                # double-check inside the lock
        spawn_control_master()
```

The lock is held only for the respawn sequence (typically < 2 s), never across long-running commands.

---

## 9. Jupyter Event Loop Handling

slurp's public Python API is synchronous (`job = slurp.submit(...)`), but the underlying transport is `asyncio`-based. The `SyncClient` in `client.py` manages the event loop internally.

### Normal Python script

```python
import slurp
job = slurp.submit("python train.py", gpus=4)   # blocking call
```

`SyncClient` creates a new `asyncio` event loop, runs the coroutine to completion, and closes the loop. No user interaction required.

### Jupyter / IPython

Jupyter already runs an `asyncio` event loop in the kernel thread. `SyncClient` detects this via `asyncio.get_running_loop()` and uses `asyncio.run_coroutine_threadsafe()` to submit work to the existing loop, then blocks the cell with a `concurrent.futures` waiter.

**No `nest_asyncio` required.** slurp does not attempt to patch the event loop. It cooperates with the loop Jupyter already owns.

**If the user calls `await` directly:**

```python
client = slurp.AsyncClient()   # v0.2
job = await client.submit("python train.py", gpus=4)
```

`AsyncClient` is deferred to v0.2. In v0.1, all async usage goes through `SyncClient`.

---

## Summary

| Concern | Implementation | Failure handling |
|---------|----------------|----------------|
| Control master | `ssh -MNf -S <socket>` via subprocess | Respawn on dead socket; `SSHError` after 7 retries |
| Command execution | `asyncssh` over Unix socket | Exponential backoff (1→2→4→…→30 s) |
| Multiplexing | Single socket shared by all processes | `fcntl.flock` prevents concurrent respawn races |
| Jump host | Delegated to OpenSSH `~/.ssh/config` | `SSHError` with hint if `ProxyJump` is unreachable |
| Key management | Read from `~/.ssh/config` or profile | Propagate OpenSSH errors verbatim |
| Jupyter | `SyncClient` uses existing loop or creates one | No `nest_asyncio`; no user action required |

