# AutoTuneX on BlueVela (LSF via SkyPilot)

Four recipes, in increasing order of cost and fidelity. All use
`space://steps/autotunex-tune` and the `skypilot/lsf/ibm-bluevela` environment.

| Recipe | GPUs | Needs | Purpose |
|--------|------|-------|---------|
| [`smoke/`](smoke) | 0 | nothing | Plumbing only: enroot launch, `additional_files` materialization, monitor events, `env://` artifact registration. Mock trials, no clone. |
| [`probe/`](probe) | 0 | nothing | Diagnostic. Reports what the runtime image contains from inside enroot — CUDA layout and `nvcc`, which `python`/`pip` win on `PATH`, identity and `$HOME`, whether a baked git credential is visible. Run this first when the image changes. |
| [`tune-fast/`](tune-fast) | 2 | `GITHUB_IBM_PAT` | A real sweep in ~15 min: fm-tune's bundled dataset, 350m model from `/proj`, 2 trials × 1 epoch, run one at a time. |
| [`tune-4gpu/`](tune-4gpu) | 4 | `GITHUB_IBM_PAT`, `HF_TOKEN` | The AutoTuneX-generated build reproduced exactly — same search space, flags, and `compute_config`. The fidelity reference, not for iteration. |

`tune-fast` and `tune-4gpu` differ by three `default:` values in the 787-line search
space (`num_samples`, `max_concurrent_trials`, `num_train_epochs`) plus the dataset,
model, and GPU count; `diff tune-4gpu/build.yaml tune-fast/build.yaml` is the
authoritative list.

> **`tune-4gpu` is not currently expected to pass.** It keeps `max_concurrent_trials: 4`
> verbatim, and concurrent trials on one node collide on the torch.distributed
> rendezvous port (`EADDRINUSE`). That is why `tune-fast` serialises. Fixing it
> properly means a per-trial port in fm-tune.

## Prerequisites

### 1. Repo and virtualenv

```bash
cd /path/to/granite.build
make g4os-skypilot-venv PYTHON=python3.13     # creates .venv with gb, gbserver, skypilot
```

Use `.venv/bin/gb` and `.venv/bin/gbserver` directly — not `uv run`.

> The target begins with `rm -rf .venv` and stops the local SkyPilot API server, so it
> rebuilds from scratch rather than updating in place. Skip it if you already have a
> working `.venv`, and stop any running gbserver first.

### 2. BlueVela access

An SSH key at **`~/.ssh/ibm-bluevela.key`** — the path is not configurable per user;
it is what the environment asset declares:

```yaml
# configurations/assets/environments/skypilot/lsf/ibm-bluevela/environment.yaml
cluster_ssh_configs:
  lsf:
    - Host: bluevela
      HostName: login4.bluevela.rmf.ibm.com
      User: granitebuild
      IdentityFile: ~/.ssh/ibm-bluevela.key
      IdentitiesOnly: "yes"
```

The account that key authenticates as must:

- log in to the BlueVela login node over SSH,
- be a member of the LSF group **`grp_granite_dot_build`** — the environment passes
  it as `bsub -G`, so jobs are rejected without it,
- have **write** access to two shared filesystem paths:
  - `/proj/data-eng/llmb-read-write/` — the env's `shared_workdir`, under which every
    per-run working directory and output lands,
  - `/proj/granite-build/g4os/` — the enroot cache, where imported container images
    are stored as `.sqsh` files.

Verify before running anything. `IdentitiesOnly=yes` matters: without it SSH offers
every key in your agent and BlueVela closes the connection with *"Too many
authentication failures"* before reaching the right one.

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/ibm-bluevela.key \
    granitebuild@login4.bluevela.rmf.ibm.com \
    'id; touch /proj/data-eng/llmb-read-write/.probe && rm /proj/data-eng/llmb-read-write/.probe && echo "shared_workdir writable"; ls -d /proj/granite-build/g4os/enroot >/dev/null && echo "enroot cache reachable"'
```

### 3. A github.ibm.com token, stored as a space secret

The step clones `fm-tune` over HTTPS using `$GITHUB_IBM_PAT`. Every secret in the
space's secret manager is injected into the job as an environment variable under its
own name, so the name must be exactly `GITHUB_IBM_PAT` and nothing needs declaring in
the `build.yaml`.

**Simplest route — reuse the token `gh` already holds.** No minting, and it is
known-good because `gh` keeps it refreshed:

```bash
gh auth status                                    # confirm you are logged in to github.ibm.com
umask 077
printf '%s' "$(gh auth token --hostname github.ibm.com)" > /tmp/pat.txt
gb secret create GITHUB_IBM_PAT --space public --from-file /tmp/pat.txt
shred -u /tmp/pat.txt
```

The `printf '%s' "$(...)"` is deliberate: it strips the newline `gh` prints.
`--from-file` stores the file's bytes verbatim, and a trailing newline inside the
secret breaks the clone with `fatal: credential url cannot be parsed`.

Caveat: `gh`'s `gho_` token is tied to your `gh` session and rotates when it
refreshes. For a durable setup, mint a classic PAT with `repo` scope at
<https://github.ibm.com/settings/tokens> and store it the same way.

Verify before spending a cluster launch — a rejected token costs you a full launch to
discover:

```bash
read -rs TOKEN                                    # paste, Enter; not echoed, not in history
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: token $TOKEN" \
  https://github.ibm.com/api/v3/user               # expect 200
unset TOKEN
```

**Note:** `gb auth login` is *not* this. It authenticates you to gbserver (and mints
lakehouse tokens from that identity); in standalone mode the CLI does not need it, and
it does not produce a token usable for cloning.

Full background, including how to check what actually got stored without revealing it:
[docs/secrets/workload-credentials.md](../../../docs/secrets/workload-credentials.md).

## Starting gbserver in standalone mode

```bash
cd /path/to/granite.build
GB_ENVIRONMENT=STANDALONE .venv/bin/gbserver standalone \
    --space-dir configurations/spaces/local \
    --port 8080 2>&1 | tee /tmp/gbserver.log
```

Notes:

- Leave it running. It serves the REST API *and* the dashboard at
  <http://127.0.0.1:8080>, and builds reuse the warm server.
- `tee /tmp/gbserver.log` is worth doing: the orchestration log is the only place the
  monitor's retrieved job output and event dispatch are visible.
- Any credential the server needs from its own environment must be exported in **this**
  shell before starting it. Space secrets created with `gb secret create` do not need a
  restart (they are re-read from disk per build), but a `GBSERVER_SECRET_*` environment
  variable is fixed at exec time and does.
- A restart is required after changing anything under `src/` — Python is imported once
  at startup. Changes to recipes, step assets, and `space.yaml` are picked up per build
  with no restart.

## Running a recipe

`GB_ENVIRONMENT=STANDALONE` is needed by the **client** too, not just the server:
without it the CLI defaults to `PROD` and tries to resolve the space remotely. It
prints a `Warning: GB_ENVIRONMENT is set to STANDALONE`, which is just the CLI noting
it is not the default.

Validate first — this does the full parameter substitution and schema check without
launching anything, and catches most mistakes for free:

```bash
GB_ENVIRONMENT=STANDALONE .venv/bin/gb build validate \
    -f recipes/autotunex/lsf/tune-fast/build.yaml
```

Then start it:

```bash
GB_ENVIRONMENT=STANDALONE .venv/bin/gb build start \
    -f recipes/autotunex/lsf/tune-fast/build.yaml \
    -m "AutoTuneX tune-fast: CUDA_HOME=/opt/share/cuda-12.9, 1 trial at a time"
```

`parameters.yaml` is picked up automatically as a sibling of `build.yaml`. Override
individual values with `--param KEY=VALUE`:

```bash
--param NUM_SAMPLES=1                              # fewer trials
--param IMAGE=quay.io/fedora/fedora-minimal:42     # different image
--param AUTOTUNEX_ARGS="--autotunex_server_url https://<subdomain>.ngrok-free.dev"
```

Parameters use `$${VAR}` — double dollar — and are rendered **strictly** by the CLI
before submission, so an undefined name aborts the submit. A single `${VAR}` is passed
through untouched for the shell on the compute node. Keep shell variables single-`$`.

## Watching a run, and where the logs are

```bash
gb build status <build-id>            # overall status, targets, output artifacts
gb build log --all <build-id>         # gbserver's event log for the build
```

| What | Where |
|------|-------|
| The **workload's own stdout** — the useful one: clone, install, Ray, trial results, tracebacks | `/tmp/sky-logs/gb-<launch-id>/job-1/run.log` |
| gbserver orchestration, monitor activity, event dispatch | `/tmp/gbserver.log` (only because of the `tee` above) |
| Build/event store the dashboard reads | `~/.granite.build/llmb-server.db` — query via `gb build log`, not directly |
| The output artifact itself | `/proj/data-eng/llmb-read-write/builds/builds/<build-id>/runs/<targetrun-id>/output` on BlueVela |

The `sky-logs` directory is named after the SkyPilot **launch**, not the build, so pick
the newest: `ls -dt /tmp/sky-logs/*/ | head -1`. Nothing is written under
`~/.granite.build/workdir/.../outputs/job.log` for these runs — that is the local/bash
execution layout; a SkyPilot job's output arrives through the monitor's periodic
retrieval instead.

To see the cluster while it exists:

```bash
.venv/bin/sky status                  # INIT -> UP; torn down when the build finishes
```

## Notes

- **Image import.** The AutoTuneX runtime is a multi-GB CUDA image. The first cold
  import takes ~30 minutes (measured: 31); the LSF provisioner allows an hour for
  container readiness, a window that exists to cover exactly this. Once imported it is
  cached as a flattened `.sqsh` under `/proj/granite-build/g4os/enroot/` and later runs
  skip the import entirely, reaching a running job in ~2 minutes. Do **not** hand-place
  a `.sqsh` there to "pre-seed" it: the flatten step only runs after a real import, so
  a manually copied file stays layered and has no working `/bin/sh`.
- **GPUs come from `launcher_config.resources.accelerators`**, not from
  `compute_config` — the SkyPilot launcher does not derive them from it. Omitting
  `accelerators` is what makes `smoke` and `probe` 0-GPU builds.
- **`venv_path` is not optional for this image.** The runtime image installs
  torch/flash_attn/ray into `/step_venv` and only symlinks it into
  `/root/.local/bin`, which enroot does not put on `PATH` (the container runs as your
  own uid with `HOME=/`, not root). Without `venv_path`, `pip` hits PEP 668 and
  `python` cannot import `ray`.
- **`CUDA_HOME` must point at a toolkit that exists.** DeepSpeed resolves
  `$CUDA_HOME/bin/nvcc` at import, and the AutoTuneX k8s build's
  `/usr/local/cuda-12.4` is absent from the image. The recipes use
  `/opt/share/cuda-12.9`, a BlueVela-provided toolkit whose major version matches
  torch's. Run `probe/` to re-check this if the image changes.
- **The AutoTuneX bridge is optional.** `AUTOTUNEX_ARGS` is empty by default, so a
  sweep runs without posting to the AutoTuneX server; gbserver still reports trial
  progress and the output artifact. Enabling it needs egress from the BlueVela compute
  node to the bridge URL.
- **A green build can still contain a failed trial.** fm-tune exits 0 even when its
  post-training cleanup errors, so `gb build status` reports success while the log says
  `Number of errored trials: 1`. Worth grepping the run log for `errored` after a run
  you care about.
- **Upstream templates.** fm-tune carries the maintained AutoTuneX granite.build
  templates at `granite.build/gb-single-node/build.yaml` and `gb-multi-node/build.yaml`.
  Both are k8s-only (`space://steps/custom_code`, `lh://` URIs); these recipes are the
  LSF counterparts.
