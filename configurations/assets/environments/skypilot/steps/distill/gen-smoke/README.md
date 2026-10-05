# gen-smoke (SkyPilot / LSF)

Asks whether a trained model still writes usable text. It greedy-decodes eight raw
prompts (code in five languages and three short prose prompts) from each checkpoint in a
ladder. For each checkpoint it reports the fraction of generated lines that sit inside a
run of three or more identical lines.

That is the failure a distilled model showed in build df8512e0: one line repeated about
fifty times running. No training-time metric caught it. A few minutes on one GPU here is
much cheaper than finding it in a full eval.

## Minimal use

```yaml
targets:
  gen-smoke:
    environment_uri: space://environments/skypilot/lsf/ibm-bluevela
    inputs:
      export_500:  {binding: export-500.hf_model}
      export_1000: {binding: export-1000.hf_model}
    outputs:
      repetition_report:
        uri: "env://{{ binding.path }}"
        type: fileset
    steps:
      - step_uri: space://steps/distill/gen-smoke
        config:
          compute_config: {num_nodes: 1, num_gpus_per_node: 1}
          launcher_config:
            resources: {accelerators: "H100:1", cluster: "bluevela", zone: "normal", memory: 256}
          gen_smoke_config:
            rungs:
              - "500:/proj/run/export-500"
              - "1000:/proj/run/export-1000"
            output_dir: /proj/run/gen-smoke
```

Binding the export targets as inputs makes gen-smoke wait for every checkpoint it reads.

## Config (`gen_smoke_config`)

| key | default | notes |
|---|---|---|
| `rungs` | `[]` | **required.** One `"<step>:<hf_model_dir>"` per checkpoint, in ladder order; `<step>` is an integer |
| `max_repetition` | `0.15` | a checkpoint is degenerate when its mean looped fraction is above this |
| `new_tokens` | `256` | greedy tokens per prompt |
| `gate_final_rung` | `true` | `true` fails the target when the **last** rung is degenerate; `false` only reports |
| `output_dir` | `gen-smoke` | where `repetition.json` is written; a relative path lands under `GB_BUILD_WORKDIR` |
| `python` | `/stage/.venv/bin/python` | interpreter in the image |

Only the last rung can fail the target, because it is the checkpoint a later target would
pick up. A degenerate earlier rung is a finding to read in the report, not a reason to
fail a build whose later checkpoints may be fine. Set `gate_final_rung: false` for a
recipe whose job is to measure a whole ladder.

## Output

- **`repetition_report`**: `repetition.json`, with the threshold and, per rung, the step,
  path, mean and worst looped fraction, mean adjacent-repeat rate, the `degenerate`
  verdict, and each prompt's completion (first 400 characters).

The log has one `GEN-SMOKE step=… looped=… DEGENERATE|ok` line per rung, then a table of
all of them.

## Things to know

- The verdict is on runs of three or more identical lines, not on adjacent repeats. The
  adjacent-repeat rate is reported but never gates: a correct short function can score
  high on it. Why, and the measurements behind the threshold, are in `src/gen_smoke.py`.
- The step runs with `HF_HUB_OFFLINE=1`. Each `<hf_model_dir>` must be a complete local
  export.

Needs one GPU, which the build supplies, and no trainer source.
