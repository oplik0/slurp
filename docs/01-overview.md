# 01 — Overview and Philosophy

## 1. What slurp Is

`slurp` is a Python library and command-line tool for submitting, monitoring, and managing machine-learning jobs on SLURM clusters. It is explicitly designed for researchers who understand PyTorch but do not want to learn SLURM internals, SBATCH directives, or multi-node launcher incantations.

The tool's guiding slogan is: **"Simpler than `sbatch` for people who don't know `sbatch`."** It is not a general-purpose workflow orchestrator, nor is it a replacement for `submitit`, `NeMo Run`, or `WandB Launch`. It is a narrow, opinionated wrapper that removes the friction between a local Python script and a remote GPU node.

---

## 2. Target User

The primary user is a researcher or PhD student with the following profile:

- Writes PyTorch training scripts locally on a laptop or workstation.
- Needs to run those scripts on a cluster (initially Jülich JURECA) with 4–64 GPUs.
- Does not want to write `.sbatch` files, manage SSH keys manually, or parse `squeue` output.
- Iterates rapidly: fix a bug, re-submit, check logs, repeat.

This user is **not** a cluster administrator, a CI/CD engineer, or a multi-step pipeline author. If the user needs DAG-based dependency management, persistent distributed storage, or fine-grained fair-share tuning, `slurp` will deliberately feel too simple — that is by design.

---

## 3. Core Design Principles

### 3.1 Zero-Config First Run

The very first invocation of `slurp submit` must not fail with "config file missing." Instead, the tool falls back to interactive prompts, learns the necessary connection details, and persists them as a **profile** for subsequent runs.

```bash
# First run — interactive fallback
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

**Edge case:** If `~/.ssh/config` already contains a Host entry for the target login node, no prompts are shown. The profile is auto-populated from SSH config and only asks for SLURM-specific defaults (partition, account). This path is common for users who already use `ssh jrlogin` daily.

**Edge case:** If the user declines to save the profile (`[n]`), the run proceeds but the next invocation will prompt again. This supports ephemeral or shared workstations where persisting credentials is undesirable.

### 3.2 Two Layers, No Hierarchy

Configuration is deliberately flat. There are only two layers:

1. **Built-in defaults** — cluster-agnostic values such as `nodes=1`, `time="2:00:00"`, and `cpus_per_task=8`.
2. **CLI flags / Python kwargs** — explicit overrides provided at invocation time.

**Profiles are not a third config layer.** A profile stores SSH connection metadata (`hostname`, `username`, `key_file`) plus per-profile SLURM defaults (`partition`, `account`). The user never thinks about "which layer wins." If a flag is passed, it wins. If no flag is passed, the profile default is used. If the profile has no default, the built-in default is used.

This avoids the surprise of `.slurprc.toml` + env vars + CLI flags + profile defaults + system defaults competing with one another. For teams that need shared defaults, a `.env` file with environment variables is sufficient and well-understood.

### 3.3 Cluster-Agnostic Core, Profile-Specific Boilerplate

All cluster-specific setup — `module load`, `jutil env activate`, MPI flags — lives in the profile's `prologue` field, not in `core/slurm.py`. When a new cluster is added, the user creates a new profile with a different `prologue`. The core code remains unchanged.

```toml
[profiles.jureca]
prologue = """
jutil env activate -p {account}
module load Stages/2024
module load CUDA/12
module load PyTorch
source $PROJECT/.venv/bin/activate
"""
```

**Rationale:** In previous iterations of this design, JURECA-specific defaults leaked into the core codebase. That made porting to other clusters (e.g., Helmholtz AI, LRZ) a refactoring exercise rather than a configuration change. By inverting the dependency — core calls profile prologue as an opaque string — the tool remains portable.

**Edge case:** If a profile's `prologue` contains a malformed shell command, the job fails at runtime. `slurp` does not attempt to validate shell syntax. The error appears in the job's stderr, and `slurp logs` surfaces it immediately.

### 3.4 SLURM Is the Source of Truth

Local state is a cache, not a ledger. The job store (`~/.local/share/slurp/jobs.json`) records metadata for convenience, but authoritative status always comes from `sacct` or `squeue`. If the local file is corrupted, deleted, or stale, `slurp list` reconciles against SLURM on the next call. There is no "local state machine" with grace periods or distributed consensus.

**Failure mode:** If the login node is down and `sacct` is unreachable, `slurp` shows the last cached status with a visual stale indicator. It does not invent terminal states.

### 3.5 In-Place by Default, Reproducibility Is Opt-In

For the rapid-iteration loop (fix bug → re-sync → re-submit), the default is in-place rsync. The remote working directory mirrors the local one. This means the latest code is always what runs.

For experiments that must be reproducible, `--snapshot` copies the rsynced tree into `$PROJECT/.slurp/runs/<job_id>/` before execution. This provides an immutable snapshot without forcing every submit to pay the copy cost.

**Edge case:** A dirty git workspace triggers a warning but does not block submission. The researcher is trusted to decide whether the uncommitted change is intentional.

---

## 4. User-Facing Surface (v0.1)

The v0.1 surface is intentionally tiny:

| Verb | Purpose |
|------|---------|
| `slurp submit` | Fire-and-forget job submission |
| `slurp run` | Blocking submit with live log streaming |
| `slurp submit-array` | Parameter sweep via SLURM job arrays |
| `slurp watch` | Live table of all jobs with progress |
| `slurp logs` | Tail stdout/stderr of a job |
| `slurp status` / `slurp list` | Query job state and filter by experiment |
| `slurp cancel` | Stop one or more jobs |
| `slurp sync` | Sync code without submitting |
| `slurp pull` | Download results from a completed job |
| `slurp config` | Add, list, or edit profiles |

Everything else — debug tunnels, web UI, pre-flight health checks, plugin system — is deferred to v0.2 or later. The goal is to ship a tool that solves 80% of daily friction with 10 commands, rather than solving 100% with 50.

---

## 5. Failure Philosophy

`slurp` errors are designed to be actionable. Every exception carries three fields:

- **message** — what failed, in plain English
- **hint** — the most likely fix, based on the error context
- **retryable** — whether retrying the same command might succeed

Examples:

- `SSHError("Connection to jrlogin timed out after 30s", hint="Check VPN and try again.", retryable=True)`
- `SlurmError("sbatch: error: Batch job submission failed: Invalid account or account/partition combination", hint="Verify account name in ~/.config/slurp/profiles.toml", retryable=False)`

The CLI prints these in a structured block (Rich panel) rather than a raw traceback. The Python API raises the same exception classes, so notebook users get identical diagnostics.

---

## 6. Scope Boundaries

What `slurp` does **not** do in v0.1:

- **No workflow DAGs.** Job dependencies exist (`depends_on=[job1]`), but there is no visual pipeline editor or implicit dependency inference.
- **No hyperparameter search.** `submit-array` maps parameter lists to SLURM arrays; the user still writes the training script.
- **No persistent experiment tracking.** The `experiment` tag is a convenience filter for `watch` and `cancel-all`; it is not a replacement for MLflow or WandB.
- **No checkpoint-aware resumption.** If a job times out, the user resubmits manually. Auto-resubmission with checkpoint detection is v0.2.

These boundaries keep the codebase small, the mental model flat, and the maintenance surface minimal.
