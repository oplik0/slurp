# Current workflow & NeMo Run limitations

The DUCI project currently uses NVIDIA’s NeMo Run to submit jobs to JURECA. For example, a typical invocation builds a `SlurmExecutor` with SSH tunneling and then runs an experiment via `run.Experiment`【31†L73-L75】【30†L22-L30】. This hides raw `sbatch` scripts but adds complexity: users must configure executors, packagers (like GitArchivePackager), and often debug NeMo’s behavior. In practice the author found log streaming to be awkward (e.g. `--tail-logs` simply runs a blocking `tail -f` with only raw stdout) and lacks a programmatic channel for progress. In DUCI’s CLI, they resorted to parsing a custom `progress.jsonl` file on the remote side via SSH and displaying a Rich table of metrics【43†L775-L783】【43†L810-L818】. These hacks show that a simpler, custom tool could better integrate PyTorch training progress, live metrics, and multi-job tracking. 

# JURECA Slurm environment

JURECA is a Slurm-based cluster accessed via a set of login (jump) nodes【39†L610-L618】. All compute jobs must be submitted from these login nodes (or via an SSH jump). Compute nodes have **no internet access** (by security policy)【41†L898-L901】, so code and data must be staged or bundled beforehand. The Slurm partitions (`dc-gpu`, `dc-cpu`, etc.) determine available GPUs and memory. In practice, jobs run exclusively per node (no sharing) and you request resources with `sbatch` (or `srun`) arguments. Since there is no container engine (Pyxis/Singularity), one typically loads environment modules (e.g. `jutil env activate`) on the login node or in the job script. The new tool should assume an SSH key login to the JURECA gateway and then execute all Slurm commands (like `sbatch`, `squeue`, `sacct`) on the cluster via that SSH session. 

# Existing tools and patterns

Several Python libraries address Slurm submission. **Submitit** (Meta’s open-source tool) is a lightweight Python API that lets you submit Python functions to Slurm with a familiar executor/job pattern【46†L274-L279】. It handles output folders, job IDs, and lets you retrieve stdout/stderr or call `job.result()`. However, Submitit assumes you are already on a login node (it doesn’t manage SSH or code syncing). 

Another project is **srunx**【34†L363-L364】【36†L414-L415】, which provides a unified CLI/Python interface for Slurm. Notably, srunx _rsyncs your code_, runs `sbatch` on the remote cluster, and **streams logs back** to your local shell【34†L363-L364】. It has commands like `srunx sbatch`, `srunx tail <job_id>` and even a web UI. It fully integrates SSH (`--profile`) so that normal Slurm commands work transparently over the gateway【35†L378-L385】【36†L414-L415】. These tools confirm that code-sync + remote sbatch + log streaming is feasible. 

Given the user’s goals, a custom library can borrow ideas: use an SSH + rsync workflow (like srunx) and a simple Python API (like Submitit’s executor or TorchX). We avoid heavy abstractions (no full workflow/DAG engine), focusing just on job submission and tracking for ML tasks.

# Designing the new Slurm-run API

The library (say, `mlslurm`) would provide a **Python API and optional CLI** that abstracts SLURM. Key design points:

- **SSH connectivity:** Use an SSH session (via subprocess `ssh` or libraries like [paramiko](https://github.com/paramiko/paramiko) or [asyncssh](https://github.com/ronf/asyncssh)) to the cluster’s login/jump host. Support `ProxyJump` in `ssh_config` to reach the Slurm gateway. All subsequent Slurm commands (`sbatch`, `squeue`, etc.) are run over this SSH.

- **Code packaging/sync:** On first use, rsync or SCP the local code directory to a remote project folder (e.g. `$HOME/projects/exp123/`). We could use `git archive` or simple `rsync --exclude` patterns. This ensures the remote job has the exact code, without needing internet on compute nodes. (Similar to NeMo’s GitArchivePackager but simpler.)

- **Job submission:** Provide a function like `job = slurm.submit(command, name, gpus=1, nodes=1, time="2:00:00", ...)`. Under the hood this creates an SBATCH script or uses `sbatch --wrap`, including resource directives. We can allow optional “setup commands” (e.g. module loads) via parameters. The function runs `ssh ... sbatch myscript.sh` and captures the job ID from stdout. (Alternatively, use `ssh slurm@srv "sbatch ..."` directly with a one-line script.)

- **Log files:** Ensure the job’s stdout/stderr are written to known files (by `#SBATCH --output` or `--error`). After submission, the client can offer to **stream logs**. For example, open an SSH channel to run `tail -f job-<id>.out` on the login node. In Python, spawn a thread or async task that reads that output and prints it (or passes it to a progress logger). This would avoid the blocking behavior seen in NeMo Run. 

- **Progress reporting:** For ML jobs, we can standardize logging to (say) a JSONL file (e.g. `/path/to/run/progress.jsonl`) with metrics (epoch, loss, acc). The library can periodically `ssh`-cat that file to fetch the latest line (as DUCI did【43†L823-L831】) and parse metrics. These can be reported back in the client (e.g. via Rich progress bars or live tables). If the user code can write a simple JSON line each epoch, the tool can show real-time status of all jobs.

- **Multi-task orchestration:** The API should allow submitting multiple jobs and tracking them together. For example, `jobs = [slurm.submit(...) for _ in range(N)]`. The client can then poll `squeue` or `sacct` via SSH for all these job IDs, and display a summary (total progress, pending/running/completed counts). Alternatively, implement job arrays if all tasks are similar (though that complicates log gathering). The interface might include a `watch(jobs)` or a CLI `mlslurm watch <job_group_name>` (similar to DUCI’s `watch`) that shows a combined live table of all tasks.

- **Results retrieval:** After jobs finish, the client should offer to **fetch results** (e.g. model files, output CSVs) via `rsync`【42†L722-L729】. For simplicity, the library could define a remote “output” directory (maybe the job’s working dir) and `rsync` it down. This could happen automatically on job completion if `--download` is set.

- **CLI/front-end:** In addition to Python API, a small CLI (via `argparse` or `typer`) can mirror common actions: `mlslurm submit ...`, `mlslurm status JOB`, `mlslurm watch [JOB]`, `mlslurm logs JOB`, `mlslurm cancel JOB`, `mlslurm download JOB`. These would just call the underlying API. For UX, using [Rich](https://github.com/Textualize/rich) can give nice tables and progress bars (like the DUCI CLI did【43†L867-L874】).

- **Error handling & detach:** Allow a `detach` mode (return after submission without blocking) and a “follow logs” mode. Handle re-submissions or retries if needed. Keep job metadata (maybe in a local SQLite or JSON file) to correlate submitted jobs and their remote names/IDs.

# Remote debugging

It is possible to enable remote debugging of a Slurm job. A common approach is to insert a debug server in the training script. For instance, one can use [debugpy](https://github.com/microsoft/debugpy) in the Python code: have the master process run `debugpy.listen(("0.0.0.0",12345))` and wait for a client【48†L311-L314】. Then from your local machine use VSCode’s Remote-SSH or port forwarding to attach to that port on the compute node. (An example helper for Slurm jobs is shown in 【48†L311-L314】: it reads `SLURM_NODELIST`, chooses a node IP, and does `debugpy.listen((master_addr,port))`.) Alternatively, one can use VSCode’s “Remote-SSH” extension to login to the cluster (and even spawn a debug session on an allocated node). In short, the library can simply document that: you can run a debugpy hook in your script and then use SSH/VSC to attach, without changing the job orchestration code.

# Summary of plan

In summary, the new Python library will:

- **Focus on jump-host workflow:** run all commands via SSH to JURECA’s login node, sync code with rsync or git, submit jobs remotely.
- **Simplify Slurm use:** allow users to specify resources via function args or CLI flags, without writing SBATCH scripts manually.
- **Stream logs and metrics:** launch background log-tailers and parse JSON progress so users see live output and progress bars.
- **Manage multiple jobs:** support submitting arrays or batches of jobs and track them together in a unified way.
- **Provide simple CLI & API:** similar in spirit to srunx or submitit, but tailored to JURECA’s environment and the user’s needs.
- **Optional remote debugging:** document or include utilities for debugpy/VSCode integration on the cluster.

By combining SSH/rsync for file transfer with simple sbatch wrappers and log polling, this tool can avoid the heavy abstractions of NeMo Run while giving a clean, Pythonic interface. All pieces (SSH commands, rsync) rely on standard tools, and progress/log streaming can use Python threads or asyncio plus Rich for nice display. Existing examples (srunx【34†L363-L364】 and DUCI’s CLI code【42†L722-L729】【43†L775-L783】) validate the approach. 

**Sources:** The above plan is based on the DUCI repository’s current use of NeMo Run【31†L73-L75】【30†L22-L30】, JURECA’s documentation on Slurm and access rules【39†L610-L618】【41†L898-L901】, and examples from libraries like srunx【34†L363-L364】【36†L414-L415】 and Submitit【46†L274-L279】. Remote debugging via SSH is illustrated in MS/PyCharm examples【48†L311-L314】.
