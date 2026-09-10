#!/usr/bin/env python3
"""Render the flat YAML config the kd-sandbox GOLD trainer consumes.

Why this is a Python script rather than a heredoc in the step's ``run:`` block.
Jinja does work there, so this is not about capability — it is about which
mistakes stay silent. Three of the trainer's requirements corrupt a run without
producing an error, and each is handled badly by a shell heredoc:

* ``learning_rate`` and the scheduler's ``min_lr`` must be YAML floats. PyYAML
  parses a bare ``1e-05`` as a *string*, which crashes the trainer's min_lr
  handling with a str/float TypeError partway into a run. Emitting through
  ``yaml.safe_dump`` of an actual float makes that structural instead of a
  formatting convention one edit away from breaking.
* ``min_lr`` must be nested under ``lr_scheduler_kwargs``, never top level. The
  trainer parses with TRL's ``parse_args_and_config``, which rejects unknown
  top-level keys, so a flat ``min_lr`` fails the run outright — after the model
  and teacher have already loaded.
* Booleans must be lower-case YAML. ``safe_dump`` does that by construction;
  templating emits ``True`` unless every site remembers to lower-case it.
* The six on-policy keys must be emitted **only** when ``vllm_num_servers > 0``.
  That is one testable branch here, versus ``{% if %}`` nested inside a quoted
  heredoc inside a YAML literal block — and it is exactly the seam the on-policy
  phase will reopen.

Dumping with the same library the trainer parses with also means a config that
renders is a config the trainer can read.
"""

import argparse
import sys
from typing import Any, Dict

import yaml

# Fields emitted only on the on-policy (online) path. Keeping an off-policy
# config free of them is not cosmetic: their presence is what the launcher and
# trainer read to decide whether to expect a vLLM server at all.
_ONLINE_ONLY = (
    "vllm_num_servers",
    "top_p",
    "use_sampled_opd_loss",
    "last_message_only",
    "clip_alpha",
    "opd_importance_sampling",
)


def _lr(value: float) -> float:
    """Return a learning rate as a float with two-decimal exponent precision.

    Round-tripping through ``%.2e`` keeps the rendered value identical to the
    validated reference configs (``1.00e-05``) while remaining a float, so PyYAML
    reads a number rather than a string.
    """
    return float(f"{float(value):.2e}")


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Assemble the gold config mapping from parsed arguments."""
    online = args.vllm_num_servers > 0

    if online and args.total_nodes < 2:
        raise ValueError(
            "on-policy needs at least 2 nodes (one vLLM server plus one "
            f"trainer); got total_nodes={args.total_nodes}"
        )
    if online and args.vllm_num_servers >= args.total_nodes:
        raise ValueError(
            f"vllm_num_servers ({args.vllm_num_servers}) must be < total nodes "
            f"({args.total_nodes}); every node would serve and none would train"
        )

    config: Dict[str, Any] = {
        "model_name_or_path": args.model_name_or_path,
        "teacher_model_name_or_path": args.teacher_model_name_or_path,
        "dataset_name": args.dataset_name,
        "num_train_epochs": float(args.num_train_epochs),
        "learning_rate": _lr(args.learning_rate),
        "warmup_ratio": float(args.warmup_ratio),
        "lr_scheduler_type": args.lr_scheduler_type,
        # min_lr belongs to the SCHEDULER, not the top level. The trainer parses
        # its config with TRL's parse_args_and_config, which rejects unknown
        # top-level keys outright:
        #   ValueError: Unknown arguments from config file: ['--min_lr', ...]
        # Every validated config in kd-sandbox/configs/gold/ nests it this way.
        "lr_scheduler_kwargs": {
            "min_lr": _lr(args.min_lr),
        },
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_completion_length": args.max_completion_length,
        "max_length": args.max_length,
        "gradient_checkpointing": args.gradient_checkpointing,
        # Set explicitly by every validated reference config.
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "logging_steps": args.logging_steps,
        "dataset_num_proc": args.dataset_num_proc,
        "temperature": float(args.temperature),
        "lmbda": float(args.lmbda),
        "beta": float(args.beta),
        # granite's logits_scaling=10 overflows the fused bf16 JSD kernel and
        # yields NaN loss; the non-fused generalized_jsd_loss is stable.
        "use_liger_fused_jsd": args.use_liger_fused_jsd,
        # Required: locates the completion span for loss masking. The nothink
        # chat template carries no {% generation %} tag, so without this the
        # trainer cannot tell prompt from completion.
        "response_template": args.response_template,
    }

    if online:
        config.update(
            {
                "vllm_num_servers": args.vllm_num_servers,
                "top_p": float(args.top_p),
                "use_sampled_opd_loss": args.use_sampled_opd_loss,
                "last_message_only": args.last_message_only,
                "clip_alpha": float(args.clip_alpha),
                "opd_importance_sampling": args.opd_importance_sampling,
            }
        )
    return config


def _bool(value: str) -> bool:
    """Parse a YAML-ish boolean from the step's shell-rendered arguments."""
    lowered = str(value).strip().lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off", ""):
        return False
    raise argparse.ArgumentTypeError(f"not a boolean: {value!r}")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", required=True, help="Path to write the YAML to.")
    p.add_argument(
        "--total-nodes",
        type=int,
        required=True,
        help="Nodes in the allocation, used to validate the split.",
    )

    p.add_argument("--model-name-or-path", required=True, help="Student init.")
    p.add_argument("--teacher-model-name-or-path", required=True)
    p.add_argument("--dataset-name", required=True)

    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=1.0e-05)
    p.add_argument("--min-lr", type=float, default=1.0e-06)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--lr-scheduler-type", default="cosine_with_min_lr")
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=6)
    p.add_argument("--max-completion-length", type=int, default=4096)
    p.add_argument("--max-length", type=int, default=16384)
    p.add_argument("--gradient-checkpointing", type=_bool, default=True)
    p.add_argument(
        "--save-strategy",
        default="steps",
        help="Set explicitly by every validated reference config.",
    )
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=20)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--dataset-num-proc", type=int, default=64)

    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--lmbda",
        type=float,
        default=0.0,
        help="0 => off-policy (no vLLM); >0 => online.",
    )
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--use-liger-fused-jsd", type=_bool, default=False)
    p.add_argument("--response-template", default="<|im_start|>assistant")

    p.add_argument(
        "--vllm-num-servers",
        type=int,
        default=0,
        help=">0 emits the on-policy block and splits the nodes.",
    )
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--use-sampled-opd-loss", type=_bool, default=False)
    p.add_argument("--last-message-only", type=_bool, default=False)
    p.add_argument("--clip-alpha", type=float, default=0.1)
    p.add_argument("--opd-importance-sampling", type=_bool, default=False)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        config = build_config(args)
    except ValueError as e:
        print(f"render_gold_config: {e}", file=sys.stderr)
        return 2
    # sort_keys=False keeps the emitted order stable and readable against the
    # reference configs; default_flow_style=False forces block style.
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    print(
        f"render_gold_config: wrote {args.output} "
        f'({"on-policy" if args.vllm_num_servers > 0 else "off-policy"}, '
        f"{args.total_nodes} node(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
