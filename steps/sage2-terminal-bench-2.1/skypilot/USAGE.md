# sage2-terminal-bench-2.1 (SkyPilot)

Scores a checkpoint on **Terminal-Bench 2.1** (Sage2 metric: pass@1[avg-of-8] resolve
rate). The job serves the model with vLLM and runs each task as a
[harbor](https://github.com/harbor-framework/harbor) trial: the reference Terminus 2
agent drives the model inside an enroot sandbox started from the task's prebuilt image,
then harbor's verifier runs the task's tests. It writes one `results.json`.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image. Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-terminal-bench-2.1
```

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals tbench image, pinned by tag. |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all runnable tasks) | **Smoke knob:** first N tasks by name. |
| `repeats` | `""` (8) | Independent agent runs per task (avg-of-k). |
| `workers` | `8` | Concurrent tasks. Tasks that serve on fixed ports run one at a time. |
| `dataset` / `dataset_revision` | `""` | Override the tasks pinned in sage2-evals (upstream `harborframework/terminal-bench-2.1` at a fixed commit, checked against the published task digests). |
| `options` | `""` | Space-separated `key=value` benchmark options: `temperature`, `top_p`, `max_tokens`, `max_turns`, `timeout_multiplier`, `max_retries`, `tasks` (regex), `exclude` (comma list, or `none`), `agent=oracle`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `/proj/granite-build/g4os/sage2/enroot-cache` | Shared squashfs cache for task images. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The dataset and the task images (Docker Hub) are public; no token is
needed. Tasks need outbound internet (package installs in the tests).

`agent=oracle` runs each task's reference solution instead of a model (set
`model_path: none`). Use it to validate the images, the sandbox and the tests on a new
cluster.

Tasks that can't run on the enroot sandbox are excluded from the score, so `n` is
87 of the 89 tasks. `configure-git-webserver` and `git-multibranch` run sshd and
clone over ssh to localhost:22. On the shared host network that reaches the node's
own sshd instead. The score is not directly comparable with an 89-task leaderboard
number. `results.json` records `n`, `n_total` (89) and `excluded` (task: reason).

## Output

`sage2_results` (dataset): the `results.json` file. Declare it on the target:

```yaml
outputs:
  sage2_results:
    uri: "env://{{ binding.path }}"
```

Harbor's trial directories (agent trajectory, terminal recording, verifier output,
`result.json`) are kept next to it under `output/repeat-<k>/<task>/`.
