# Vendored source: granite-evals

The benchmark runtime is vendored from
[laminair/granite-evals](https://github.com/laminair/granite-evals) at commit
`dce96b3` (Apache-2.0), so the eval steps do not depend on an external repository.

Layout (the images assemble one `granite_evals` package from the shared core
plus one family's modules; paths relative to this directory). Each family's
vendored unit tests sit beside its modules at `<family>/tests/`:

| Directory                    | Contents                                                                      |
|------------------------------|-------------------------------------------------------------------------------|
| `src/core/`                  | CLI, registry, data, meter, serving, sandbox, `benchmarks/nemo_skills.py`, `pyproject.toml`, `uv.lock` |
| `bfcl/granite_evals/`        | `bfcl`                                                                        |
| `bird/granite_evals/`        | `birdbench`                                                                   |
| `ifbench/granite_evals/`     | `ifbench` (split out of upstream `chat_if.py`)                                |
| `judged/granite_evals/`      | `judge_general` (profbench, gdpval), `mcpatlas`, `multi_challenge`, `safety`  |
| `lmeval/granite_evals/`      | `chat_if` (mmlu-prox-lite)                                                    |
| `nemoskills/granite_evals/`  | `nemo_skills_code`, `nemo_skills_extended`, `nemo_skills_ruler`, `scicode_prefill`, `tools_agentic` |
| `swebench/granite_evals/`    | `swebench`                                                                    |
| `tau/granite_evals/`         | `tau`                                                                         |
| `tbench/granite_evals/`      | `tbench`                                                                      |

Changes from upstream:

- Internal service endpoints (judge gateway, MCP search) are no longer defaulted;
  pass `--option judge_base_url=...` / `--option search_mcp_url=...`.
- `languages=ibm` is now `languages=extended`.
- Site-specific scripts, docs, registry defaults and author metadata were dropped.
- `chat_if.py` is split so IFBench ships in its own family image.

Build the images with `make -C steps/eval images REGISTRY=<registry>`.
