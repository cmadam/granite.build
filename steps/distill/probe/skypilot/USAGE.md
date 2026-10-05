# probe (SkyPilot / LSF)

Runs one of three cheap checks on a checkpoint, used to settle questions before a
distillation run is spent on them. It trains nothing and produces no artifact: the
answer is a `PROBE VERDICT` line in the target's log.

| probe | question | needs |
|---|---|---|
| `load-student` | does the model load in this image, with FlashAttention-2 and with eager attention? | 1 GPU |
| `tokenizer` | does the checkpoint's declared `tokenizer_class` change how text is split into tokens? | CPU |
| `tokenizer-fit` | which way of splitting do the checkpoint's weights prefer, measured by NLL per byte? | 1 GPU |

## Minimal use

One target per probe. All three can run against the same checkpoint in one build:

```yaml
targets:
  load-student:
    environment_uri: space://environments/skypilot/lsf/ibm-bluevela
    steps:
      - step_uri: space://steps/distill/probe
        config:
          compute_config: {num_nodes: 1, num_gpus_per_node: 1}
          launcher_config:
            resources: {accelerators: "H100:1", cluster: "bluevela", zone: "normal", memory: 256}
          probe_config:
            probe: load-student
            checkpoint: /proj/run/checkpoints/epoch_hf_2
  tokenizer-fit:
    environment_uri: space://environments/skypilot/lsf/ibm-bluevela
    steps:
      - step_uri: space://steps/distill/probe
        config:
          compute_config: {num_nodes: 1, num_gpus_per_node: 1}
          launcher_config:
            resources: {accelerators: "H100:1", cluster: "bluevela", zone: "normal", memory: 256}
          probe_config:
            probe: tokenizer-fit
            checkpoint: /proj/run/checkpoints/epoch_hf_2
            corpus: /proj/run/corpus/eval.jsonl
            rows: 200
```

## Config (`probe_config`)

| key | default | notes |
|---|---|---|
| `probe` | — | **required.** `load-student`, `tokenizer` or `tokenizer-fit` |
| `checkpoint` | — | **required.** An HF checkpoint directory. It is only read: the tokenizer probes edit a scratch copy |
| `corpus` | `""` | **required for `tokenizer-fit`.** A JSONL of `{"messages": [...]}` rows |
| `rows` | `200` | `tokenizer-fit` only: how many rows to score; at least 1 |
| `python` | `/stage/.venv/bin/python` | interpreter in the image |

## Reading the verdict

| probe | verdict line | meaning |
|---|---|---|
| `load-student` | `PROBE VERDICT loaded=flash_attention_2,eager` | the attention implementations that loaded; `loaded=NONE` means neither did |
| `tokenizer` | `PROBE VERDICT differing=N/M` | `N > 0`: the declared class splits text differently, so any eval recorded with it needs a tokenizer-control comparison |
| `tokenizer-fit` | `PROBE VERDICT nll_per_byte asis=… pinned=… prefers=… ratio=…` | `prefers=pinned`: the declared class is the mismatch and pinning fixes it; `prefers=asis`: pinning would be a regression |

The target succeeds whatever the verdict is. A wrong answer is a finding to read, not
a failure. It fails only on bad arguments, for example `tokenizer-fit` without a
`corpus`, or when the probe itself crashes.

## Things to know

- The step deliberately does not set `HF_HUB_OFFLINE`. `load-student` asks whether
  transformers can fetch its kernels lazily in this image, and an offline hub would
  answer a different question.
- What each probe does, and why, is in `src/distill_probe.py`.

Needs torch and transformers from the image, one GPU for `load-student` and
`tokenizer-fit`, and no trainer source.
