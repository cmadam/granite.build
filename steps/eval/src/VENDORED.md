# Vendored source: granite-evals

The benchmark runtime under this directory is vendored from
[laminair/granite-evals](https://github.com/laminair/granite-evals) at commit
`dce96b3` (Apache-2.0), so the eval steps do not depend on an external repository.

Layout (the images assemble one `granite_evals` package from `core` plus one family):

| Directory    | Contents                                                                      |
|--------------|-------------------------------------------------------------------------------|
| `core/`      | CLI, registry, data, meter, serving, sandbox, `benchmarks/nemo_skills.py`, `pyproject.toml`, `uv.lock` |
| `bfcl/`      | `bfcl`                                                                        |
| `bird/`      | `birdbench`                                                                   |
| `ifbench/`   | `ifbench` (split out of upstream `chat_if.py`)                                |
| `judged/`    | `judge_general` (profbench, gdpval), `mcpatlas`, `multi_challenge`, `safety`  |
| `lmeval/`    | `chat_if` (mmlu-prox-lite)                                                    |
| `nemoskills/`| `nemo_skills_code`, `nemo_skills_extended`, `nemo_skills_ruler`, `scicode_prefill`, `tools_agentic` |
| `swebench/`, `tau/`, `tbench/` | one module each                                             |

Changes from upstream:

- Internal service endpoints (judge gateway, MCP search) are no longer defaulted;
  pass `--option judge_base_url=...` / `--option search_mcp_url=...`.
- `languages=ibm` is now `languages=extended`.
- Site-specific scripts, docs, registry defaults and author metadata were dropped.
- `chat_if.py` is split so IFBench ships in its own family image.

Build the images with `make -C steps/eval images REGISTRY=<registry>`.
