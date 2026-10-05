# corpus-sources (SkyPilot / LSF)

Combines several raw SFT splits into one `train.jsonl` that corpus prep can read.
Corpus prep (`space://steps/distill/corpus-prep`) takes a single file, reads the
`messages` key, and stops after its first N kept rows. This step:

1. renames `conversations` to `messages` in every row that uses the old key;
2. samples each split in proportion to its share of the whole, so a subset still
   contains every split rather than only the largest one;
3. shuffles the result, so a downstream row limit still sees every split.

## Minimal use

```yaml
targets:
  sources:
    environment_uri: space://environments/skypilot/lsf/ibm-bluevela
    outputs:
      corpus_source:
        uri: "env://{{ binding.path }}"
        type: dataset
    steps:
      - step_uri: space://steps/distill/corpus-sources
        config:
          compute_config: {num_nodes: 1, num_cpus_per_node: 4}
          launcher_config:
            resources: {cluster: "bluevela", zone: "normal", memory: 64}
          sources_config:
            sources: [/data/general.jsonl, /data/tools.jsonl, /data/rag.jsonl]
            target_rows: 100000
            shuffle_seed: 42
            output_dir: /proj/run/sources
  prep:
    inputs:
      source_dataset: {binding: sources.corpus_source}
    steps:
      - step_uri: space://steps/distill/corpus-prep
        # ...
```

## Config (`sources_config`)

| key | default | notes |
|---|---|---|
| `sources` | `[]` | **required.** `.jsonl` paths visible from the compute node |
| `target_rows` | `0` | rows to keep across all splits; `0`, or anything at least the corpus size, keeps every row |
| `shuffle_seed` | `0` | seeds the per-split sampling and the final shuffle |
| `output_dir` | `sources` | where `train.jsonl` is written; a relative path lands under `GB_BUILD_WORKDIR` |
| `python` | `/stage/.venv/bin/python` | interpreter in the image |

The order of `sources` matters: each entry's position names its random stream, so
reordering the list changes which rows are picked even with the same seed. The same
list, `target_rows` and `shuffle_seed` always give the same file.

## Output

- **`corpus_source`**: the written `train.jsonl`, one `{"messages": [...]}` row per line.

The log shows the row count and quota per split (`SOURCES count`, `SOURCES quota`), how
many rows were sampled from each, and how many rows were renamed, already used
`messages`, or were unusable and skipped.

## Failures

- No sources given, a source that is not a file, or every source empty.
- No row anywhere carries `conversations` or `messages`. That means the source schema
  has changed, and prep would otherwise drop every row.

Unusable individual rows are skipped and counted in a `SOURCES WARNING` line, not fatal.
The sampling method, and the measurements behind it, are in `src/build_sources.py`.

CPU-only, standard library only, and no trainer source.
