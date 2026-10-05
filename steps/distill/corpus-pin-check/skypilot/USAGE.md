# corpus-pin-check (SkyPilot / LSF)

Checks that a pinned corpus, one built by an earlier run and reused by this one, was
built the way this build would have built it. It runs before any GPU allocation is
held, so a stale or mismatched corpus fails in seconds rather than after a training run.

## When to use it

When a recipe skips corpus prep and reads an existing corpus directory instead. Bind
its `pin_check` output into every target that reads the pinned corpus: that makes the
check an ordering edge, so nothing can train on the corpus before it is accepted.

## Minimal use

```yaml
targets:
  corpus-pin-check:
    environment_uri: space://environments/skypilot/lsf/ibm-bluevela
    inputs:
      pinned_corpus:
        uri: "env:///proj/run/corpus/train.jsonl"
        type: dataset
      tokenizer:
        binding: align.retagged_student
    outputs:
      pin_check:
        uri: "env://{{ binding.path }}"
        type: fileset
    steps:
      - step_uri: space://steps/distill/corpus-pin-check
        config:
          compute_config: {num_nodes: 1, num_cpus_per_node: 4}
          launcher_config:
            resources: {cluster: "bluevela", zone: "normal", memory: 32}
          pin_check_config:
            corpus_dir: /proj/run/corpus
            teacher_model: /proj/models/granite-4.1-3b-pinned
            max_length: 4096
            think_policy: keep
            documents_policy: keep
            eval_fraction: 0.005
            tokenizer_dir: "{{ bindings.tokenizer.binding.path }}"
            output_dir: /proj/run/corpus-pin
  train:
    inputs:
      pin_check: {binding: corpus-pin-check.pin_check}
    # ...
```

## Config (`pin_check_config`)

| key | default | notes |
|---|---|---|
| `corpus_dir` | — | **required.** Holds `corpus_manifest.json`, `train.jsonl`, `eval.jsonl` |
| `teacher_model` | — | **required.** Its basename must equal the manifest's `tokenizer_identity` |
| `max_length` | — | **required.** Must equal the manifest's |
| `think_policy` | — | **required.** Must equal the manifest's |
| `documents_policy` | — | **required.** Must equal the manifest's |
| `eval_fraction` | — | **required.** Must equal the manifest's |
| `tokenizer_dir` | — | **required.** This build's retagged tokenizer |
| `output_dir` | `corpus-pin` | where `corpus_pin.json` is written; a relative path lands under `GB_BUILD_WORKDIR` |
| `python` | `/stage/.venv/bin/python` | interpreter in the image |

Use the same values the recipe would pass to corpus prep. A pin check fed different
values from the prep it stands in for checks the wrong thing.

## What it checks

- `train.jsonl` and `eval.jsonl` are both present.
- The manifest's `tokenizer_identity` is the basename of `teacher_model`.
- `max_length`, `think_policy`, `documents_policy` and `eval_fraction` match, and the
  manifest's `completion_boundary` is `last_message`.
- When the manifest's `tokenizer_path` still exists, its `tokenizer.json` and
  `chat_template.jinja` are byte-identical to the ones in `tokenizer_dir`. When it no
  longer exists, that comparison is skipped and `corpus_pin.json` says so.

The reasoning behind each comparison is in `src/check_corpus_pin.py`.

## Output

- **`pin_check`**: `corpus_pin.json`, written only on acceptance. It records the
  corpus directory, which tokenizer check ran, and the manifest's tokenizer identity,
  policies, eval fraction, seed, counts and splits.

## Failures

Every mismatch is reported in one run, under a `CORPUS-PIN REJECTED` line, and the
target fails with `FATAL [corpus-pin-check]: N mismatch(es)`. A directory with no
`corpus_manifest.json` fails immediately.

CPU-only, standard library only, and no trainer source.
