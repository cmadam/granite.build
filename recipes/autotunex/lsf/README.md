# AutoTuneX on BlueVela (LSF via SkyPilot)

Three recipes, in increasing order of cost and fidelity. All three use
`space://steps/autotunex-tune` and the `skypilot/lsf/ibm-bluevela` environment.

| Recipe | GPUs | Needs | Purpose |
|--------|------|-------|---------|
| [`smoke/`](smoke) | 0 | nothing | Plumbing only: enroot launch, `additional_files` materialization, monitor events, `env://` artifact registration. Mock trials, no clone. |
| [`tune-fast/`](tune-fast) | 2 | `GITHUB_IBM_PAT` | A real sweep in minutes: fm-tune's bundled dataset, 350m model from `/proj`, 2 trials × 1 epoch. |
| [`tune-4gpu/`](tune-4gpu) | 4 | `GITHUB_IBM_PAT`, `HF_TOKEN` | The AutoTuneX-generated build reproduced exactly — same search space, flags, and `compute_config`. Use as the reference, not for iteration. |

`tune-fast` and `tune-4gpu` differ by exactly three `default:` values in the 787-line search space
(`num_samples`, `max_concurrent_trials`, `num_train_epochs`) plus the dataset, model, and GPU count;
`diff tune-4gpu/build.yaml tune-fast/build.yaml` is the authoritative list.

## Running one

```bash
export GB_ENVIRONMENT=STANDALONE          # the CLI reads this too, not just the server
gb build validate -f recipes/autotunex/lsf/smoke/build.yaml
gb build start    -f recipes/autotunex/lsf/smoke/build.yaml
```

`parameters.yaml` is picked up automatically as a sibling of `build.yaml`; override any value with
`--param KEY=VALUE`. Parameters use `$${VAR}` (double dollar) and are rendered **strictly** by gbcli
before submission, so an undefined name aborts the submit — and a single `${VAR}` is left alone for the
shell on the compute node. Keep shell variables single-`$`.

## Supplying `GITHUB_IBM_PAT`

Create it as a space secret; every space secret is injected into the job as an environment variable
under its own name, so the step picks it up with nothing declared in the `build.yaml`:

```bash
gb secret create GITHUB_IBM_PAT --space public --from-file /path/to/token.txt
```

Full procedure — minting the token, keeping the value out of `ps` and the job log, when a restart is
needed, and why it must not be baked into the runtime image:
**[docs/secrets/workload-credentials.md](../../../docs/secrets/workload-credentials.md)**.

## Notes

- **Image import.** The AutoTuneX runtime is a multi-GB CUDA image. It is pulled cold; the LSF
  provisioner allows an hour for container readiness, a window that exists to cover `enroot import`
  (`sky/provision/lsf/utils.py`, `DEFAULT_READY_TIMEOUT`). If a cold import ever overruns, pre-seeding
  the `.sqsh` under the env's enroot `share_path` is the fallback.
- **GPUs come from `launcher_config.resources.accelerators`**, not from `compute_config` — the SkyPilot
  launcher does not derive them from it. Omitting `accelerators` is what makes `smoke/` a 0-GPU build.
- **The AutoTuneX bridge is optional.** `AUTOTUNEX_ARGS` is empty by default, so a sweep runs without
  posting to the AutoTuneX server. gbserver still reports trial progress and the output artifact.
  Enabling it requires egress from the BlueVela compute node to the bridge URL.
- **Upstream templates.** fm-tune carries the maintained AutoTuneX granite.build templates at
  `granite.build/gb-single-node/build.yaml` and `gb-multi-node/build.yaml`. Both are k8s-only
  (`space://steps/custom_code`, `lh://` URIs); these recipes are the LSF counterparts.
