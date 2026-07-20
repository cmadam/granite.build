#!/usr/bin/env python3
"""Generate a per-epoch checkpoint-eval k8s SFT build.

Starting from a parameters file and an eval catalog, emit a build.yaml that:
  1. tokenizes the tuning data once (CPU byoi),
  2. fine-tunes granite4-350m for NUM_EPOCHS, emitting one HF-checkpoint output
     per *evaluated* epoch (binding_id: epoch_<M>, 1-indexed — see the forked
     open-instruct step's per-epoch log_monitor),
  3. for each selected epoch, fans a selected eval suite out against that epoch's
     checkpoint (each eval target binds sft-training.epoch_<M>),
  4. rolls each epoch's eval logs into a per-epoch CSV, then rolls all per-epoch
     CSVs into one combined benchmark x epoch table.

This mirrors recipes/granite4-350m/lsf/rl-checkpoint-eval/generate_build.py; the
target block shapes are transcribed from the proven k8s builds
(sft-eval-smoke/build.yaml + sft-eval-full-dataset/build.yaml). The large,
invariant command literals (the tokenize base64 chat-template script and the
bfcl script) live verbatim under templates/; the small, parameterized commands
(sage eval, per-epoch/combined export) are built in Python.

Usage:
    python generate_build.py \\
        --parameters-path ../sft-eval-smoke/parameters.yaml \\
        --catalog-path eval-catalog.yaml \\
        --num-epochs 2 --eval-epochs all \\
        --param 'EVAL_SETS=[multilingual-eval, bfcl]' \\
        --output ../sft-eval-smoke/build.yaml

Any parameter in parameters.yaml can be overridden with --param KEY=VALUE (dot
notation supported, mirroring src/gbcli/utils/buildutil.py). The $${...}
placeholders in the emitted build are resolved at `gb build start` time against
the resolved parameters file this script writes alongside build.yaml.
"""

import argparse
import os
import sys

import yaml

# Every target runs on the space's default k8s environment (matches the proven
# sft-eval-smoke / sft-eval-full-dataset builds).
ENVIRONMENT_URI = "space://environments/{{ space.variables.DEFAULT_ENVIRONMENT }}"
# The forked open-instruct step emits one artifact per epoch's HF checkpoint,
# keyed epoch_<M> (1-indexed). The training outputs and eval bindings use this.
EPOCH_OUTPUT_PREFIX = "epoch_"
# Where sage writes eval outputs inside the pod (PVC).
OUTPUT_BASE_PATH = "/gb-read-write/sage"

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES = os.path.join(_HERE, "templates")


def _load_template(name):
    with open(os.path.join(_TEMPLATES, name), "r", encoding="utf-8") as f:
        return f.read()


# ─── YAML string quoting ──────────────────────────────────────────────────────
# The build engine substitutes $${...} placeholders into the emitted YAML and
# re-parses it (src/gbcli/utils/buildutil.py:apply_parameters). Match the
# hand-written recipes: double-quote scalar string values, single-quote values
# that embed a double quote (e.g. the byoi command), and keep multiline command
# scripts as literal block scalars. Mapping keys stay unquoted.
_SINGLE_QUOTE_TOKENS = ()


class _DQ(str):
    """A string emitted with double-quote style."""


class _SQ(str):
    """A string emitted with single-quote style."""


class _LIT(str):
    """A (multiline) string emitted as a literal block scalar."""


yaml.SafeDumper.add_representer(
    _DQ,
    lambda dumper, data: dumper.represent_scalar(
        "tag:yaml.org,2002:str", str(data), style='"'
    ),
)
yaml.SafeDumper.add_representer(
    _SQ,
    lambda dumper, data: dumper.represent_scalar(
        "tag:yaml.org,2002:str", str(data), style="'"
    ),
)
yaml.SafeDumper.add_representer(
    _LIT,
    lambda dumper, data: dumper.represent_scalar(
        "tag:yaml.org,2002:str", str(data), style="|"
    ),
)


def _quote_values(obj):
    """Recursively wrap string *values* (not keys) in a quoting style.

    Multiline strings become literal block scalars (matching the proven builds'
    command blocks); strings embedding a double quote (and no single quote) are
    single-quoted; everything else is double-quoted. Non-string scalars pass
    through unquoted.
    """
    if isinstance(obj, dict):
        return {k: _quote_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_quote_values(v) for v in obj]
    if isinstance(obj, str):
        if "\n" in obj:
            return _LIT(obj)
        if any(tok in obj for tok in _SINGLE_QUOTE_TOKENS) or (
            '"' in obj and "'" not in obj
        ):
            return _SQ(obj)
        return _DQ(obj)
    return obj


# ─── Parameter loading + KEY=VALUE overrides ──────────────────────────────────
# Mirrors add_parameter / add_key_value in src/gbcli/utils/buildutil.py so
# --param behaves identically to `gb build start --param`.
def add_key_value(data, key, value):
    """Add a key/value pair, supporting dot notation ('a.b.c=value')."""

    def add_branch(data_rec, prefix, key_vector, value):
        key = key_vector[0]
        if not isinstance(data_rec, dict):
            raise ValueError(
                f"param {prefix}.{key} cannot be used: prefix {prefix} is already in use."
            )
        if len(key_vector) == 1:
            data_rec[key] = value
        else:
            data_rec[key] = add_branch(
                data_rec.get(key, {}), f"{prefix}.{key}", key_vector[1:], value
            )
        return data_rec

    return add_branch(data, "", key.split("."), value)


def apply_override(data, param):
    """Apply one 'key=value' override; value is parsed as YAML for typing/lists."""
    key, sep, raw = param.partition("=")
    if not sep:
        raise ValueError(f"Invalid parameter {param!r}. Use the format 'key=value'.")
    try:
        value = yaml.safe_load(raw.strip())
    except yaml.YAMLError:
        value = raw.strip()
    return add_key_value(data, key.strip(), value)


def load_params(parameters_path, overrides):
    with open(parameters_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}
    for param in overrides:
        params = apply_override(params, param)
    return params


# ─── Epoch schedule ───────────────────────────────────────────────────────────
def compute_eval_epochs(params):
    """1-indexed epochs to evaluate. EVAL_EPOCHS = 'all' | [ints] | 'a,b' | int."""
    n = int(params["NUM_EPOCHS"])
    if n < 1:
        raise ValueError("NUM_EPOCHS must be >= 1")
    raw = params.get("EVAL_EPOCHS", "all")
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.lower() == "all":
            return list(range(1, n + 1))
        raw = [x.strip() for x in raw.split(",") if x.strip()]
    if isinstance(raw, int):
        raw = [raw]
    epochs = sorted({int(x) for x in raw})
    bad = [e for e in epochs if e < 1 or e > n]
    if bad:
        raise ValueError(f"EVAL_EPOCHS {bad} out of range 1..{n} (NUM_EPOCHS)")
    if not epochs:
        raise ValueError("EVAL_EPOCHS resolved to an empty selection")
    return epochs


# ─── Eval-set resolution ──────────────────────────────────────────────────────
def resolve_eval_names(params, catalog):
    """Expand EVAL_SETS (set names and/or individual eval names) to eval names.

    Preserves first-seen order and de-dups. Raises on unknown names.
    """
    raw = params.get("EVAL_SETS", [])
    if isinstance(raw, str):
        parsed = yaml.safe_load(raw)
        raw = parsed if isinstance(parsed, list) else [raw]
    if not isinstance(raw, list):
        raise ValueError(f"EVAL_SETS must be a list, got {type(raw).__name__}")

    evals = catalog["evals"]
    sets = catalog.get("sets", {})
    resolved = []
    seen = set()
    for name in raw:
        if name in sets:
            candidates = sets[name]
        elif name in evals:
            candidates = [name]
        else:
            known = sorted(set(sets) | set(evals))
            raise ValueError(
                f"Unknown EVAL_SETS entry {name!r}. Known sets/evals: {', '.join(known)}"
            )
        for eval_name in candidates:
            if eval_name not in evals:
                raise ValueError(f"set {name!r} references unknown eval {eval_name!r}")
            if eval_name not in seen:
                seen.add(eval_name)
                resolved.append(eval_name)
    if not resolved:
        raise ValueError("EVAL_SETS resolved to an empty selection")
    return resolved


# ─── Per-epoch experiment namespace ───────────────────────────────────────────
def experiment_for_epoch(m):
    """Per-epoch experiment namespace so results land in sibling PVC dirs.

    A flat NAME suffix ("$${EXPERIMENT}-ep_<M>"), NOT a "/"-separated subdir, so
    each epoch's eval/export outputs land at
    /gb-read-write/sage/$${EXPERIMENT}-ep_<M>/... as siblings (mirrors RL's
    $${EXPERIMENT}-ckpt_<step>).
    """
    return "$${EXPERIMENT}-ep_" + str(m)


# ─── Shared k8s fragments ─────────────────────────────────────────────────────
def _hf_token_env():
    return {
        "HF_TOKEN": {
            "valueFrom": {
                "secretKeyRef": {
                    "name": "{{ setup_config.space.secret }}",
                    "key": "HF_TOKEN",
                }
            }
        }
    }


def _byoi_command():
    return '/bin/bash -c "chmod +x /tmp/command.sh && /tmp/command.sh"'


# ─── Target builders ──────────────────────────────────────────────────────────
def build_tokenize_target():
    """The `tokenize` block, verbatim from the proven k8s build (CPU byoi)."""
    return {
        "environment_uri": ENVIRONMENT_URI,
        "inputs": {
            "tuning_data": {"uri": "$${TUNING_DATA_URI}"},
            "model_to_tune": {"uri": "$${MODEL_URI}"},
        },
        "outputs": {
            "tokenized_dataset": {
                "type": "dataset",
                "uri": "hf://huggingface.co/datasets/ibm-research/"
                "g4-350m-tokenized_{{ run_metadata.targetsteprun_id | short_hash }}",
            }
        },
        "steps": [
            {
                "step_uri": "space://steps/byoi",
                "config": {
                    "compute_config": {
                        "num_gpus_per_node": 0,
                        "num_nodes": 1,
                        "total_memory_per_node": "32Gi",
                        "num_cpus_per_node": 8,
                    },
                    "byoi_config": {"command": _byoi_command()},
                    "k8s": {
                        "image": "$${SFT_IMAGE}",
                        "additional_files": {
                            "/tmp/command.sh": _load_template("tokenize_command.sh")
                        },
                    },
                },
            }
        ],
    }


def build_training_target(eval_epochs):
    """The `sft-training` block, with one HF-checkpoint output per eval epoch."""
    outputs = {
        f"{EPOCH_OUTPUT_PREFIX}{m}": {
            "type": "model",
            "uri": "hf://huggingface.co/datasets/ibm-research/"
            "g4-350m-sft_{{ run_metadata.targetsteprun_id | short_hash }}_epoch"
            + str(m),
        }
        for m in eval_epochs
    }
    return {
        "environment_uri": ENVIRONMENT_URI,
        "inputs": {
            "model_to_tune": {"uri": "$${MODEL_URI}"},
            "tokenized_dataset": {
                "binding": "tokenize.tokenized_dataset",
                "wait_for_push": True,
            },
        },
        "outputs": outputs,
        "steps": [
            {
                "step_uri": "git+ssh://github.ibm.com/cmadam/assets.git@gbspace-config-dev#subdirectory=steps/open-instruct",
                "config": {
                    "k8s": {
                        "image": "$${SFT_IMAGE}",
                        "env": {
                            **_hf_token_env(),
                            "NCCL_P2P_DISABLE": {"value": "1"},
                            "NCCL_DEBUG": {"value": "WARN"},
                        },
                    },
                    "tuning_config": {
                        "tokenized_data": "{{ bindings.tokenized_dataset.binding.path }}",
                        "chat_template_name": "$${CHAT_TEMPLATE_NAME}",
                        "hf_dir": "/gb-read-write/outputs/open-instruct/hf/"
                        "g4-350m-sft-{{ run_metadata.targetsteprun_id | short_hash }}",
                        "use_flash_attn": True,
                        "gradient_checkpointing": True,
                        "num_train_epochs": "$${NUM_EPOCHS}",
                        "per_device_train_batch_size": "$${PER_DEVICE_BATCH_SIZE}",
                        "gradient_accumulation_steps": "$${GRADIENT_ACCUMULATION_STEPS}",
                        "learning_rate": "$${LEARNING_RATE}",
                        "lr_scheduler_type": "$${LR_SCHEDULER_TYPE}",
                        "warmup_ratio": "$${WARMUP_RATIO}",
                        "weight_decay": "$${WEIGHT_DECAY}",
                        "clip_grad_norm": "$${CLIP_GRAD_NORM}",
                        "max_seq_length": "$${MAX_SEQ_LEN}",
                        "seed": "$${SEED}",
                        "reduce_loss": "$${REDUCE_LOSS}",
                        "checkpointing_steps": "$${CHECKPOINTING_STEPS}",
                        "keep_last_n_checkpoints": "$${KEEP_LAST_N_CHECKPOINTS}",
                        "logging_steps": "$${LOGGING_STEPS}",
                    },
                    "compute_config": {
                        "num_gpus_per_node": "$${SFT_NUM_GPUS}",
                        "num_nodes": "$${SFT_NUM_NODES}",
                    },
                },
            }
        ],
    }


def _sage_command(entry, experiment):
    """Build the sage eval /tmp/command.sh for one eval + epoch.

    Mirrors the proven sage byoi command: fixed EXPERIMENT/MODEL_PATH/NUM_GPUS
    exports, then MAX_LENGTH/BATCH_SIZE (overridable via extra_env), then any
    remaining extra_env keys as extra exports (e.g. MULTIPLE_LANG,
    OE_EVAL_BCB_API_URL), then run the gb script and tee its log.
    """
    extra = dict(entry.get("extra_env") or {})
    max_length = extra.pop("MAX_LENGTH", "$${MAX_LENGTH}")
    batch_size = extra.pop("BATCH_SIZE", "$${BATCH_SIZE}")
    lines = [
        "set -e",
        f"export EXPERIMENT={experiment}",
        'export MODEL_PATH="{{ bindings.model.binding.path }}"',
        "export NUM_GPUS=$${EVAL_NUM_GPUS}",
        f"export MAX_LENGTH={max_length}",
        f"export BATCH_SIZE={batch_size}",
    ]
    for key, value in extra.items():
        lines.append(f"export {key}={value}")
    lines += [
        f"mkdir -p {OUTPUT_BASE_PATH}/{experiment}",
        f"LOG={OUTPUT_BASE_PATH}/{experiment}-{entry['log_suffix']}.log",
        f"bash /workspace/sage/sage/cluster/gb/scripts/{entry['script']} 2>&1 | tee \"$LOG\"",
        'echo "LLMB_ARTIFACT_ID:eval_log LLMB_ARTIFACT_PATH:$LOG"',
    ]
    return "\n".join(lines) + "\n"


def build_sage_eval_target(entry, m):
    """A sage byoi eval target bound to sft-training.epoch_<m>."""
    experiment = experiment_for_epoch(m)
    return {
        "environment_uri": ENVIRONMENT_URI,
        "inputs": {"model": {"binding": f"sft-training.{EPOCH_OUTPUT_PREFIX}{m}"}},
        "outputs": {
            "eval_log": {
                "type": "fileset",
                "uri": "hf://huggingface.co/datasets/ibm-research/"
                f"g4-350m-{entry['log_suffix']}-ep{m}-"
                "{{ run_metadata.targetsteprun_id | short_hash }}",
            }
        },
        "steps": [
            {
                "step_uri": "space://steps/byoi",
                "config": {
                    "compute_config": {
                        "num_gpus_per_node": "$${EVAL_NUM_GPUS}",
                        "num_nodes": 1,
                        "total_memory_per_node": "96Gi",
                        "num_cpus_per_node": 8,
                    },
                    "byoi_config": {"command": _byoi_command()},
                    "k8s": {
                        "image": "$${" + entry["image"] + "}",
                        "env": {
                            **_hf_token_env(),
                            "OUTPUT_BASE_PATH": {"value": OUTPUT_BASE_PATH},
                            "UV_CACHE_DIR": {"value": "/tmp/uv-cache"},
                        },
                        "additional_files": {
                            "/tmp/command.sh": _sage_command(entry, experiment)
                        },
                    },
                },
            }
        ],
    }


def build_bfcl_eval_target(entry, m):
    """The bfcl byoi eval target bound to sft-training.epoch_<m>."""
    experiment = experiment_for_epoch(m)
    # The verbatim bfcl command uses $${EXPERIMENT} for EXP_NAME + OUTPUT_DIR;
    # rewrite both to the per-epoch experiment.
    command = _load_template("bfcl_command.sh").replace("$${EXPERIMENT}", experiment)
    return {
        "environment_uri": ENVIRONMENT_URI,
        "inputs": {"model": {"binding": f"sft-training.{EPOCH_OUTPUT_PREFIX}{m}"}},
        "outputs": {
            "bfcl_results": {
                "type": "fileset",
                "uri": "hf://huggingface.co/datasets/ibm-research/"
                f"g4-350m-bfcl-ep{m}-"
                "{{ run_metadata.targetsteprun_id | short_hash }}",
            }
        },
        "steps": [
            {
                "step_uri": "space://steps/byoi",
                "config": {
                    "compute_config": {
                        "num_gpus_per_node": "$${EVAL_NUM_GPUS}",
                        "num_nodes": 1,
                        "total_memory_per_node": "96Gi",
                        "num_cpus_per_node": 8,
                    },
                    "byoi_config": {"command": _byoi_command()},
                    "k8s": {
                        "image": "$${BFCL_IMAGE}",
                        "env": {
                            **_hf_token_env(),
                            "OUTPUT_BASE_PATH": {"value": OUTPUT_BASE_PATH},
                        },
                        "additional_files": {"/tmp/command.sh": command},
                    },
                },
            }
        ],
    }


# The HF-token redaction copied from the proven export-results block; raw string
# keeps the sed backslashes intact.
_REDACT_LINE = (
    r"""find %s -name "*.sh" -exec sed -i 's/hf_[A-Za-z0-9]\{10,\}/REDACTED/g' {} +"""
)


def _export_env():
    return {"OUTPUT_BASE_PATH": {"value": OUTPUT_BASE_PATH}}


def _export_compute_config():
    return {
        "num_gpus_per_node": 0,
        "num_nodes": 1,
        "total_memory_per_node": "10Gi",
        "num_cpus_per_node": 1,
    }


def build_epoch_export_target(m, eval_target_names, eval_output_ids):
    """Per-epoch exporter (CPU byoi), gated on that epoch's eval targets.

    Runs exporter.py over the single per-epoch folder into an epoch CSV, redacts
    HF tokens from any generated scripts, and emits epoch_csv.
    """
    experiment = experiment_for_epoch(m)
    folder = f"{OUTPUT_BASE_PATH}/{experiment}"
    inputs = {
        f"wait_{name.replace('-', '_')}": {
            "binding": f"{name}.{output_id}",
            "wait_for_push": True,
        }
        for name, output_id in zip(eval_target_names, eval_output_ids)
    }
    command = "\n".join(
        [
            "set -e",
            "cd /workspace/sage/sage/exporters/",
            f"python exporter.py -in-folder {folder} -stack $${{EXPORT_STACK}} "
            f"-o {folder}/{experiment}.csv",
            _REDACT_LINE % folder,
            f'echo "LLMB_ARTIFACT_ID:epoch_csv LLMB_ARTIFACT_PATH:{folder}/{experiment}.csv"',
        ]
    ) + "\n"
    return {
        "environment_uri": ENVIRONMENT_URI,
        "inputs": inputs,
        "outputs": {
            "epoch_csv": {
                "type": "fileset",
                "uri": "hf://huggingface.co/datasets/ibm-research/"
                f"g4-350m-eval-results-ep{m}-"
                "{{ run_metadata.targetsteprun_id | short_hash }}",
            }
        },
        "steps": [
            {
                "step_uri": "space://steps/byoi",
                "config": {
                    "compute_config": _export_compute_config(),
                    "byoi_config": {"command": _byoi_command()},
                    "k8s": {
                        "image": "$${SAGE_OLMES_IMAGE}",
                        "env": _export_env(),
                        "additional_files": {"/tmp/command.sh": command},
                    },
                },
            }
        ],
    }


def build_combined_export_target(epoch_export_names, eval_epochs):
    """Combined roll-up (CPU byoi), gated on every per-epoch epoch_csv output.

    One exporter.py call over all epoch folders (ascending) joins them into a
    benchmark x epoch table.
    """
    inputs = {
        f"wait_{name.replace('-', '_')}": {
            "binding": f"{name}.epoch_csv",
            "wait_for_push": True,
        }
        for name in epoch_export_names
    }
    folders = " ".join(
        f"{OUTPUT_BASE_PATH}/{experiment_for_epoch(m)}" for m in eval_epochs
    )
    out_dir = f"{OUTPUT_BASE_PATH}/$${{EXPERIMENT}}"
    command = "\n".join(
        [
            "set -e",
            "cd /workspace/sage/sage/exporters/",
            f"mkdir -p {out_dir}",
            f"python exporter.py -in-folder {folders} -stack $${{EXPORT_STACK}} "
            f"-o {out_dir}/combined.csv",
            _REDACT_LINE % out_dir,
            f'echo "LLMB_ARTIFACT_ID:combined LLMB_ARTIFACT_PATH:{out_dir}/combined.csv"',
        ]
    ) + "\n"
    return {
        "environment_uri": ENVIRONMENT_URI,
        "inputs": inputs,
        "outputs": {
            "results": {
                "type": "fileset",
                "uri": "hf://huggingface.co/datasets/ibm-research/"
                "g4-350m-eval-results-combined-"
                "{{ run_metadata.targetsteprun_id | short_hash }}",
            }
        },
        "steps": [
            {
                "step_uri": "space://steps/byoi",
                "config": {
                    "compute_config": _export_compute_config(),
                    "byoi_config": {"command": _byoi_command()},
                    "k8s": {
                        "image": "$${SAGE_OLMES_IMAGE}",
                        "env": _export_env(),
                        "additional_files": {"/tmp/command.sh": command},
                    },
                },
            }
        ],
    }


# ─── Assembly ─────────────────────────────────────────────────────────────────
def generate(params, catalog):
    eval_epochs = compute_eval_epochs(params)
    eval_names = resolve_eval_names(params, catalog)

    targets = {
        "tokenize": build_tokenize_target(),
        "sft-training": build_training_target(eval_epochs),
    }
    epoch_export_names = []
    for m in eval_epochs:
        names = []
        output_ids = []
        for name in eval_names:
            entry = catalog["evals"][name]
            tname = f"{name}-ep{m}"
            if entry["category"] == "bfcl":
                targets[tname] = build_bfcl_eval_target(entry, m)
                output_ids.append("bfcl_results")
            else:
                targets[tname] = build_sage_eval_target(entry, m)
                output_ids.append("eval_log")
            names.append(tname)
        ename = f"export-ep{m}"
        targets[ename] = build_epoch_export_target(m, names, output_ids)
        epoch_export_names.append(ename)

    targets["export-combined"] = build_combined_export_target(
        epoch_export_names, eval_epochs
    )

    build = {
        "granite.build": {
            "name": params.get("BUILD_NAME", "g4-350m-sft-ckpt-eval-k8s"),
            "retries": {"max_retries": 2},
            "targets": _quote_values(targets),
        }
    }
    return build, eval_epochs, eval_names


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Generate a per-epoch checkpoint-eval k8s SFT build.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    here = _HERE
    p.add_argument(
        "--parameters-path",
        # Task 8 creates this file; default the path so the generator is runnable
        # once it exists.
        default=os.path.join(here, "parameters.yaml"),
        help="Base parameters file.",
    )
    p.add_argument(
        "--catalog-path",
        default=os.path.join(here, "eval-catalog.yaml"),
        help="Eval catalog file.",
    )
    common = p.add_argument_group(
        "common parameters",
        "Frequently-changed knobs. Equivalent to --param <NAME>=<value>; "
        "anything omitted falls back to the parameters file.",
    )
    common.add_argument(
        "--model", metavar="URI", help="Base model to fine-tune (MODEL_URI)."
    )
    common.add_argument(
        "--tuning-data", metavar="URI", help="Tuning dataset (TUNING_DATA_URI)."
    )
    common.add_argument(
        "--num-epochs",
        metavar="N",
        help="Number of training epochs (NUM_EPOCHS); drives the eval schedule.",
    )
    common.add_argument(
        "--eval-epochs",
        metavar="SPEC",
        help="Which 1-indexed epochs to evaluate (EVAL_EPOCHS): 'all', a CSV "
        "'2,4', or a YAML list '[2, 4]'.",
    )
    common.add_argument(
        "--eval-sets",
        metavar="LIST",
        help="Which evaluations to run (EVAL_SETS), e.g. 'multilingual-eval,bfcl' "
        "or 'full-eval'. Comma-separated or a YAML list.",
    )
    common.add_argument(
        "--experiment",
        metavar="NAME",
        help="Experiment namespace for eval/export outputs (EXPERIMENT). "
        "Per-epoch results land under <OUTPUT_BASE_PATH>/<EXPERIMENT>-ep_<M>/.",
    )
    common.add_argument(
        "--data-fraction",
        metavar="FRAC",
        help="Fraction (or count) of the tuning data to use (DATA_FRACTION).",
    )
    p.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a parameter (dot notation supported). Repeatable. "
        "Takes precedence over the common flags above.",
    )
    p.add_argument(
        "--output",
        default=os.path.join(here, "build.yaml"),
        help="Where to write the generated build.yaml ('-' for stdout).",
    )
    p.add_argument(
        "--params-out",
        default=None,
        help="Where to write the merged parameters (base file + flags + "
        "--param). Defaults to a 'parameters-resolved.yaml' sibling of "
        "--output. Pass this to `gb build start --parameters-path` so the "
        "overrides set here are honored when the build's $${...} placeholders "
        "are resolved. Ignored when --output is '-'.",
    )
    return p.parse_args(argv)


# Maps each common flag (argparse dest) to the parameter name it overrides.
COMMON_FLAG_PARAMS = {
    "model": "MODEL_URI",
    "tuning_data": "TUNING_DATA_URI",
    "num_epochs": "NUM_EPOCHS",
    "eval_epochs": "EVAL_EPOCHS",
    "eval_sets": "EVAL_SETS",
    "experiment": "EXPERIMENT",
    "data_fraction": "DATA_FRACTION",
}


def _flag_overrides(args):
    """Turn the provided common flags into KEY=VALUE override strings.

    EVAL_SETS accepts a comma-separated shorthand ('a,b') as well as a YAML list
    ('[a, b]'); normalize the shorthand to a YAML list so apply_override types it
    as a list rather than a bare string.
    """
    overrides = []
    for dest, name in COMMON_FLAG_PARAMS.items():
        value = getattr(args, dest, None)
        if value is None:
            continue
        if name == "EVAL_SETS" and "[" not in value:
            items = [v.strip() for v in value.split(",") if v.strip()]
            value = "[" + ", ".join(items) + "]"
        overrides.append(f"{name}={value}")
    return overrides


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    # Common flags first, then --param, so an explicit --param wins on conflict.
    params = load_params(args.parameters_path, _flag_overrides(args) + args.param)

    # keep_last_n_checkpoints must be >= NUM_EPOCHS so no evaluated epoch dir is
    # pruned before its eval reads it.
    num_epochs = int(params["NUM_EPOCHS"])
    keep = int(params.get("KEEP_LAST_N_CHECKPOINTS", 1) or 1)
    params["KEEP_LAST_N_CHECKPOINTS"] = str(max(keep, num_epochs))

    with open(args.catalog_path, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    build, eval_epochs, eval_names = generate(params, catalog)

    dumped = yaml.safe_dump(build, sort_keys=False, default_flow_style=False)
    params_out = None
    if args.output == "-":
        sys.stdout.write(dumped)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(dumped)
        params_out = args.params_out or os.path.join(
            os.path.dirname(os.path.abspath(args.output)),
            "parameters-resolved.yaml",
        )
        with open(params_out, "w", encoding="utf-8") as f:
            f.write(yaml.safe_dump(params, sort_keys=False))

    n_eval_targets = len(eval_epochs) * len(eval_names)
    total_targets = len(build["granite.build"]["targets"])
    msg = (
        "[generate_build] k8s sft-checkpoint-eval\n"
        f"  epochs evaluated ({len(eval_epochs)}): {eval_epochs}\n"
        f"  evals ({len(eval_names)}): {eval_names}\n"
        f"  eval targets: {len(eval_epochs)} epochs x {len(eval_names)} evals "
        f"= {n_eval_targets}\n"
        f"  total targets (incl. tokenize/training/exports): {total_targets}\n"
    )
    if args.output != "-":
        msg += f"  wrote build:  {args.output}\n"
        msg += f"  wrote params: {params_out}\n"
        msg += (
            "  start with:   gb build start -f "
            f"{args.output} --parameters-path {params_out} --space <space>\n"
        )
    sys.stderr.write(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
