# slurp examples

Self-contained usage examples for the `slurp` Python API. Each file is
documented and runnable, though most require a configured cluster profile
and a live SLURM cluster to do anything useful.

## Two API styles

slurp supports two levels of API:

1. **String-based** (examples 01-06): `slurp.submit("python train.py")`.
   You write a script file and submit a command string. The script runs
   on the cluster as-is. No serialization, no cloudpickle dependency on
   the remote.

2. **Decorator-based** (examples 07-10): `@slurp.task` decorator.
   You decorate a function and call `.remote()` / `.distribute()`.
   The function is serialized via cloudpickle and shipped to the cluster.
   Return values are retrieved via `slurp.get()`. This is closer to how
   Ray and slurminade work.

The decorator API is built on top of the string-based API. Both can be
used together in the same project.

## Prerequisites

1. **Install slurp** (editable, from the repo root):

   ```bash
   uv sync --extra dev
   ```

2. **Configure a profile** — at minimum a `default` profile in
   `~/.config/slurp/profiles.toml`:

   ```bash
   slurp config add-profile default --hostname cluster.example.edu --user $USER
   ```

   See `templates/jureca.toml` for a full-featured profile example.

3. **For `@task` examples**: slurp must also be installed on the remote
   cluster (the worker runs `python -m slurp.worker`). Add it to your
   project venv or the profile prologue.

## Files

### String-based API (slurp.submit / SyncClient)

| File | What it shows |
|------|---------------|
| `01_quickstart.py` | Minimal submit + wait. The "hello world" of slurp. |
| `02_sync_client_lifecycle.py` | `SyncClient` as a context manager: submit, poll, refresh, pull. Connection reuse across operations. |
| `03_hyperparameter_sweep.py` | `submit_array()` for a parameter sweep. One `sbatch` call, N tasks, SLURM-native throttling. |
| `04_dependency_pipeline.py` | DAG with `depends_on`: preprocess -> train -> evaluate. Fire-and-forget chain. |
| `05_multi_node_distributed.py` | `nodes=2` triggers auto-generated `torchrun` wrapper. Shows what slurp generates internally. |
| `06_error_handling.py` | Exception hierarchy: `JobFailedError` log tails, `SSHError` retryability, `ProfileError` vs `SyncError`. |
| `train.py` | Sample training script using `slurp.log_progress()`. This is what you *submit* to the cluster. |

### Decorator API (@slurp.task)

| File | What it shows |
|------|---------------|
| `07_task_quickstart.py` | `@slurp.task` + `.remote()` + `slurp.get()`. Ray-style submit with result retrieval. |
| `08_task_distribute.py` | `.distribute()` + `slurp.join()` + `JobBundling`. Slurminade-style fire-and-forget with batching. |
| `09_task_batch_prediction.py` | `slurp.put()` shared model + `remote_batch()` job array. Translates the Ray batch prediction pattern. |
| `10_task_pipeline.py` | `@node_setup` + multi-stage pipeline with `@task`. preprocess -> train -> evaluate with blocking results. |

## Running the examples

```bash
# String-based (submits train.py to the cluster)
python examples/01_quickstart.py

# Decorator-based (submits the function itself via cloudpickle)
python examples/07_task_quickstart.py

# Local mode (no SLURM — for testing)
python -c "
import slurp
slurp.set_dispatcher(slurp.LocalDispatcher())
exec(open('examples/07_task_quickstart.py').read())
"
```

## Quick reference: API comparison

| Pattern | String-based | Decorator (@task) |
|---------|-------------|-------------------|
| Submit | `slurp.submit("python train.py", gpus=4)` | `train.remote(lr=0.001, gpus=4)` |
| Fire-and-forget | `slurp.submit(...)` (discard Job) | `train.distribute(lr=0.001)` |
| Wait for result | `job.wait()` → `JobResult` | `slurp.get(ref)` → deserialized value |
| Wait for all | — | `slurp.join()` |
| Batch | `slurp.submit_array(template, configs)` | `train.remote_batch(configs)` |
| Batch + buffer | — | `with slurp.JobBundling(n): ...` |
| Shared objects | file paths | `slurp.put(obj)` → `ObjectRef` |
| Dependencies | `depends_on=[job]` | (same — pass Refs) |
| Local testing | just run the script | `slurp.set_dispatcher(LocalDispatcher())` |
