#!/usr/bin/env python3
"""Emit the `gb build start` commands that re-evaluate the RoPE-repaired models.

WHY THERE IS ANYTHING TO RE-RUN. Every capability number in the GOLD distillation epic that
was taken from an export directory was served through vLLM, which reads the transformers-4
config schema and therefore resolved a RoPE base of 10000.0 instead of the trained
10000000. Those numbers are void -- not wrong about the models, wrong about which model was
measured. Everything measured through transformers (`eval-transfer`, `gen-smoke`) loaded the
nested config correctly and stands. See scripts/repair_export_rope_theta.py for the bug and
the repair; this script is only about re-measuring afterwards.

WHAT TO RE-RUN, AND WHAT NOT TO:

    eval-bfcl        RE-RUN, every rung and every baseline. Served through vLLM.
    full-eval        RE-RUN on selected rungs. Every sage-eval benchmark serves through
                     vLLM, so all of them are void too; the ones this emits are those the
                     voided post-mortem headline rested on, plus full-eval's own `bfcl`
                     target at the WIDE category set -- the standalone sweep below covers
                     every model at `simple`, so `all` here is the widening the report left
                     open rather than a repeat.
    eval-transfer    DO NOT. Loads via transformers; its numbers were always correct.
    gen-smoke        DO NOT. Same reason. (Note it read healthy because its prompts are
                     ~10 tokens, not because it validated the export -- a RoPE base 1000x
                     too small barely shows at short range. It vouches for nothing here.)
    training         DO NOT. The checkpoints were never affected.

THE TRAP THIS SCRIPT EXISTS TO AVOID. `EXPERIMENT` must be unique per model. Two eval
targets sharing one register the same artifact URI; the second registration is refused, the
target still reports SUCCESS with an empty output list, and its consumers wait forever. That
is build b5f030cd, and doing this by hand across 52 models is how it happens again. Every
name here carries the arm, the build id and the rung.

USAGE. Feed it the model directories -- generate the list on a host where /proj is mounted
(BlueVela: `ssh -F $HOME/.lsf/config bluevela`) and run the generator wherever `gb` is:

    ssh -F $HOME/.lsf/config bluevela \\
      'find /proj/granite-build/g4os/distill -mindepth 3 -maxdepth 4 \\
         \\( -name "export" -o -name "export-*" -o -name "retagged_student" \\) -type d' \\
      > /tmp/reeval-models.txt

    python3 scripts/gen_rope_reeval_commands.py --dirs-file /tmp/reeval-models.txt

It prints commands and runs nothing: launching cluster work is the operator's call, and a
review pass over 50-odd queued H100 jobs before they are queued is cheap. Pipe to `sh` when
they look right, or take the subset you want.

COST. ~10 min on one H100 per BFCL run, all independent. Against the ~56 GPU-h per training
arm already spent, re-evaluating is close to free.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BFCL_RECIPE = "recipes/granite4-350m/lsf/bfcl-eval"
FULL_EVAL_RECIPE = "recipes/granite4-350m/lsf/full-eval"

# `simple` for comparability with the numbers already on record -- the whole point is a
# before/after on one metric, and BFCL's `simple` category is what every voided figure in
# this epic was measured on. Widen afterwards, not in the same comparison.
DEFAULT_CATEGORIES = "simple"

# The sage-eval benchmarks the voided post-mortem headline rested on:
# HumanEval 40.9% -> 0.6% and MultiPL-E Java 21.1% -> 0.0% are what "the objective is wrong,
# not merely run too long" was concluded from, and the whole v2 programme was designed off
# that conclusion. ifeval is here because instruction-following is the other capability a
# scrambled RoPE base would wreck at length while sparing short-prompt fluency.
#
# `bfcl` is full-eval's own 26th target, and it is the WIDE run: the standalone sweep covers
# every model at `simple` for comparability with the numbers on record, so repeating
# `simple` here would buy nothing. See WIDE_EVAL_NAME for the collision this would otherwise
# cause.
FULL_EVAL_TARGETS = ("evalplus-humaneval", "multiple-java", "olmes-ifeval", "bfcl")

# THE ARTIFACT-URI COLLISION between the two recipes, and why the eval name is what moves.
#
# full-eval's `bfcl` target registers
#     env://$BFCL_OUTPUT_DIR/$EXPERIMENT/$BFCL_EVAL_NAME
# and the standalone bfcl-eval recipe registers
#     env://$OUTPUT_DIR/$EXPERIMENT/$EVAL_NAME
# with BFCL_OUTPUT_DIR == OUTPUT_DIR == /proj/granite-build/g4os/bfcl and both eval names
# defaulting to `bfclv3`. So for the prioritised models -- which get BOTH -- one EXPERIMENT
# means one URI registered twice: the second registration is refused, the target still
# reports SUCCESS with an empty output list, and consumers wait forever.
#
# EXPERIMENT is left alone as the model's identity, because that is what makes a sage result
# and a bfcl result for the same rung joinable. The eval NAME carries the difference
# instead, which is also the honest label: it is a different category set, not a different
# launch of the same measurement.
STANDALONE_EVAL_NAME = "bfclv3"
WIDE_EVAL_NAME = "bfclv3-all"
WIDE_CATEGORIES = "all"

# A prefix on every EXPERIMENT so the re-run's artifacts are separable at a glance from the
# ones taken through the bug. The old numbers are kept, not overwritten: they are the
# evidence that the bug was real.
PREFIX = "ropefix"

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def experiment_name(model_dir: Path) -> str:
    """A unique, URI-safe EXPERIMENT for one model directory.

    Carries arm + build id + rung. The build id is not decoration: one arm can hold several
    builds (lr0 holds two, each with its own align/retagged_student), so arm+rung alone
    collides -- and a collision here does not fail loudly, it silently drops a result.
    """
    parts = model_dir.parts
    if model_dir.name == "retagged_student":
        # .../<arm>/<build-id>/align/retagged_student
        arm, build_id, rung = parts[-4], parts[-3], "baseline"
    else:
        # .../<arm>/<build-id>/export  or  .../<arm>/<build-id>/export-N
        arm, build_id = parts[-3], parts[-2]
        rung = model_dir.name[len("export-") :] if "-" in model_dir.name else "final"

    arm = arm.removeprefix("distill-350m-")
    name = f"{PREFIX}-{arm}-{build_id.split('-')[0]}-{rung}"
    return _UNSAFE.sub("-", name)[:64].strip("-")


def bfcl_command(model_dir: Path, *, categories: str = DEFAULT_CATEGORIES) -> str:
    return (
        f"gb build start {BFCL_RECIPE}/build.yaml"
        f" --parameters-path {BFCL_RECIPE}/parameters.yaml"
        f" --tag {PREFIX}"
        f" --param MODEL_PATH={model_dir}"
        f" --param EXPERIMENT={experiment_name(model_dir)}"
        f" --param TEST_CATEGORIES={categories}"
        f" --param EVAL_NAME={STANDALONE_EVAL_NAME}"
    )


def full_eval_command(model_dir: Path) -> str:
    return (
        f"gb build start {FULL_EVAL_RECIPE}/build.yaml {' '.join(FULL_EVAL_TARGETS)}"
        f" --parameters-path {FULL_EVAL_RECIPE}/parameters.yaml"
        f" --tag {PREFIX}"
        f" --param MODEL_PATH={model_dir}"
        f" --param EXPERIMENT={experiment_name(model_dir)}"
        f" --param BFCL_TEST_CATEGORIES={WIDE_CATEGORIES}"
        f" --param BFCL_EVAL_NAME={WIDE_EVAL_NAME}"
    )


def is_prioritised_for_full_eval(model_dir: Path) -> bool:
    """The two ends of the anchor range at the two ends of the descent, plus the run the
    voided conclusion came from.

    Deliberately small. Every full-eval target is an H100 job and the BFCL sweep already
    answers the ranking question; these exist to re-establish or withdraw a specific claim.
    """
    arm = (
        model_dir.parts[-3]
        if model_dir.name != "retagged_student"
        else model_dir.parts[-4]
    )
    rung = model_dir.name
    if arm == "distill-350m-stage1" and rung == "export":
        return (
            True  # the 8,150-step stage-1 run: the v2 programme's founding conclusion
        )
    if arm in ("distill-350m-s1v2-ce0", "distill-350m-s1v2-ce040"):
        return rung in ("export-25", "export-300", "retagged_student")
    return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dirs-file",
        required=True,
        help="File of model directories, one per line ('-' for stdin). See USAGE.",
    )
    p.add_argument(
        "--categories",
        default=DEFAULT_CATEGORIES,
        help=f"BFCL test categories (default: {DEFAULT_CATEGORIES}).",
    )
    p.add_argument(
        "--no-full-eval",
        action="store_true",
        help="Emit only the BFCL sweep.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text = (
        sys.stdin.read() if args.dirs_file == "-" else Path(args.dirs_file).read_text()
    )
    dirs = [Path(ln.strip()) for ln in text.splitlines() if ln.strip()]
    if not dirs:
        print("FATAL: no model directories given", file=sys.stderr)
        return 2

    names = [experiment_name(d) for d in dirs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        # Refuse rather than emit. A duplicate EXPERIMENT does not fail the build, it
        # silently registers nothing -- so this is the last place it can be caught.
        print(f"FATAL: EXPERIMENT collision for {dupes}", file=sys.stderr)
        return 2

    print(f"# {len(dirs)} models. BFCL categories={args.categories}.")
    print("# Review, then pipe to sh, or run the subset you want.")
    print(f"\n# ---- eval-bfcl: every rung and every baseline ({len(dirs)} jobs)")
    for d in sorted(dirs):
        print(bfcl_command(d, categories=args.categories))

    if not args.no_full_eval:
        chosen = [d for d in sorted(dirs) if is_prioritised_for_full_eval(d)]
        print(
            f"\n# ---- full-eval on {len(chosen)} prioritised models"
            f" x {len(FULL_EVAL_TARGETS)} targets"
        )
        print(
            "# Do NOT re-run eval-transfer or gen-smoke: they load via transformers and"
        )
        print("# their numbers were never affected.")
        for d in chosen:
            print(full_eval_command(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
