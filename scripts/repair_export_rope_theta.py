#!/usr/bin/env python3
"""Repair published model configs whose RoPE base was relocated out of reach.

WHAT WENT WRONG. `distill-hf-export` ran under transformers 5.8.0, which writes RoPE
settings into a nested `rope_parameters` dict and no top-level `rope_theta`. Every
transformers-4-schema consumer -- vLLM 0.11.0, which the whole eval stack serves through,
and bfcl-eval -- reads the top-level field, does not find it, and silently falls back to
the class default 10000.0 instead of the trained 10000000. Measured under 4.55.4:

    align/retagged_student (4.57.6-authored)   rope_theta -> 10000000    healthy
    export-N               (5.8.0-authored)    rope_theta ->     10000.0  1000x too small
    export-N + the hoist                       rope_theta -> 10000000    repaired

A RoPE base 1000x too small destroys long-range position information and leaves local
coherence intact, so the affected models wrote correct code and correct arithmetic and
never emitted a `<tool_call>`. That read as "the distillation objective destroyed
structured behaviour", and it was an export bug: restoring the one key took `export-1`
from BFCL 0.0000 to 0.8025 against a 0.8000-0.8125 baseline. No weights were ever harmed
and no retraining is needed.

WHAT THIS SCRIPT IS FOR. The export step now normalises this at publish time, but 38
directories were published before it did. This repairs those in place.

It does NOT define what a correct config is. It imports `normalise_model_config` and
`assert_config_portable` from the `distill-hf-export` step, so a repaired directory is
byte-for-byte what the fixed export step would write today and the two cannot drift. That
is the point of the import; do not inline the logic here.

DRY RUN BY DEFAULT. These are registered artifacts of completed builds. Nothing is written
without `--apply`, and `--apply` keeps `config.json.bak` plus a `config_repair.json`
sidecar saying what changed and why, because silently mutating a finished build's output
destroys the audit trail that makes the re-evaluation meaningful.

USAGE, on a host where the artifacts are mounted (BlueVela: `ssh -F $HOME/.lsf/config
bluevela`). The step's src and the upstream distillation package must both be importable:

    export PYTHONPATH=/path/to/distill-hf-export/src:/proj/granite-build/g4os/gb-steps-collection-post-training/src
    python3 repair_export_rope_theta.py --root /proj/granite-build/g4os/distill          # report
    python3 repair_export_rope_theta.py --root /proj/granite-build/g4os/distill --apply  # repair

Then confirm a repaired directory resolves the trained value, which is the only check that
matters before spending GPU time -- if transformers overrode the top-level key from the
nested dict on load, the repair would have to rewrite the nested dict instead:

    python3 -c "
    import sys, warnings; warnings.filterwarnings('ignore')
    from transformers import AutoConfig
    print(AutoConfig.from_pretrained(sys.argv[1]).rope_theta)" <a-repaired-dir>
    # must print 10000000
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

# A published model directory is one with a config AND weights. Keyed on the weights rather
# than on the directory NAME on purpose: the first scan of this bug globbed
# `*/*/export-*/config.json` and therefore missed a directory named plain `export` -- which
# is where the 8,150-step stage-1 run lives, the single run the voided "the objective is
# wrong" conclusion was drawn from. It was reported as "not found" and it was there all
# along. Name patterns are how that happened; presence of weights is the real signal.
_WEIGHT_NAMES = ("model.safetensors",)
_WEIGHT_PREFIX = "model-"
_WEIGHT_SUFFIX = ".safetensors"

CONFIG = "config.json"
BACKUP = "config.json.bak"
SIDECAR = "config_repair.json"

_REASON = (
    "transformers 5.8.0 nests rope_theta under rope_parameters and writes no top-level "
    "key; vLLM 0.11.0 and every other transformers-4-schema consumer read the top-level "
    "field and silently fell back to the class default 10000.0 instead of the trained "
    "10000000, serving a positionally scrambled model. Repaired by hoisting the nested "
    "value to a top-level key (the nested dict is kept, so transformers-5 consumers are "
    "unaffected). Weights were never affected."
)


def load_normaliser() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Import the export step's own normalisation, or explain how to make it importable.

    Deliberately not vendored. If this file carried its own copy of the rule, a later fix
    to the export step would leave the repaired directories describing a contract that no
    longer exists -- which is the same class of drift as the bug being repaired.
    """
    try:
        from export_hf_model import (  # noqa: PLC0415
            assert_config_portable,
            normalise_model_config,
        )
    except ImportError as exc:
        raise SystemExit(
            f"cannot import the export step's normalisation ({exc}).\n"
            "Put both of these on PYTHONPATH:\n"
            "  <granite.build>/steps/distill-hf-export/skypilot/src\n"
            "  /proj/granite-build/g4os/gb-steps-collection-post-training/src\n"
            "This script imports the rule rather than restating it, so that a repaired "
            "config is exactly what the fixed export step would write."
        ) from exc
    return normalise_model_config, assert_config_portable


def _has_weights(d: Path) -> bool:
    for entry in d.iterdir():
        if not entry.is_file():
            continue
        if entry.name in _WEIGHT_NAMES:
            return True
        if entry.name.startswith(_WEIGHT_PREFIX) and entry.name.endswith(
            _WEIGHT_SUFFIX
        ):
            return True
    return False


def find_model_dirs(root: Path) -> list[Path]:
    """Every published model directory under `root`, at any depth."""
    out: list[Path] = []
    for cfg in sorted(root.rglob(CONFIG)):
        d = cfg.parent
        try:
            if _has_weights(d):
                out.append(d)
        except OSError:
            continue
    return out


def repair_dir(
    d: Path,
    normalise: Callable[[dict], tuple[dict, list[str]]],
    assert_portable: Callable[[dict], Any],
    *,
    apply: bool,
) -> dict[str, Any]:
    """Repair one directory. Returns a record; never raises for a per-directory problem.

    Actions: clean | would-repair | repaired | refused | failed.
    """
    rec: dict[str, Any] = {
        "dir": str(d),
        "action": "clean",
        "detail": "",
        "changed": [],
    }
    cfg_path = d / CONFIG
    try:
        raw = json.loads(cfg_path.read_text())
    except (OSError, ValueError) as exc:
        rec.update(action="failed", detail=f"unreadable {CONFIG}: {exc}")
        return rec
    if not isinstance(raw, dict):
        rec.update(action="failed", detail=f"{CONFIG} is not a JSON object")
        return rec

    try:
        norm, changes = normalise(raw)
    except (
        Exception
    ) as exc:  # the step raises ExportError; do not import it just for this
        rec.update(action="failed", detail=f"{type(exc).__name__}: {exc}")
        return rec

    rec["changed"] = changes
    if not changes:
        return rec  # clean -- already correct, or already repaired

    if not apply:
        rec["action"] = "would-repair"
        return rec

    backup = d / BACKUP
    if backup.exists():
        # A backup already here while the config still needs repair means something other
        # than this script has been in this directory. Overwriting would replace the only
        # copy of the original with a half-repaired one.
        rec.update(
            action="refused",
            detail=f"{BACKUP} already exists but {CONFIG} still needs repair; "
            "not overwriting the only copy of the original",
        )
        return rec

    try:
        assert_portable(norm)
    except Exception as exc:
        rec.update(
            action="failed",
            detail=f"repaired config still fails the export step's own guard -- "
            f"{type(exc).__name__}: {exc}",
        )
        return rec

    try:
        shutil.copy2(cfg_path, backup)
        cfg_path.write_text(json.dumps(norm, indent=2, ensure_ascii=False) + "\n")
        (d / SIDECAR).write_text(
            json.dumps(
                {
                    "repaired_at": datetime.datetime.now().astimezone().isoformat(),
                    "repaired_by": Path(__file__).name,
                    "reason": _REASON,
                    "changed": changes,
                    "original": BACKUP,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
    except OSError as exc:
        rec.update(action="failed", detail=f"write failed: {exc}")
        return rec

    rec["action"] = "repaired"
    return rec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--root",
        required=True,
        help="Directory to walk, e.g. /proj/granite-build/g4os/distill",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually write. Without it, report only (the default).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"FATAL: --root is not a directory: {root}", file=sys.stderr)
        return 2

    normalise, assert_portable = load_normaliser()
    dirs = find_model_dirs(root)
    print(
        f"{len(dirs)} published model director{'y' if len(dirs)==1 else 'ies'} under {root}"
    )
    if not args.apply:
        print("DRY RUN -- nothing will be written. Re-run with --apply to repair.\n")

    records = [
        repair_dir(d, normalise, assert_portable, apply=args.apply) for d in dirs
    ]
    width = max((len(r["action"]) for r in records), default=6)
    for r in sorted(records, key=lambda r: (r["action"], r["dir"])):
        rel = Path(r["dir"]).relative_to(root)
        print(f"  {r['action']:<{width}}  {rel}")
        if r["detail"]:
            print(f"  {'':<{width}}    {r['detail']}")

    tally: dict[str, int] = {}
    for r in records:
        tally[r["action"]] = tally.get(r["action"], 0) + 1
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))

    if tally.get("would-repair"):
        print(
            f"\n{tally['would-repair']} director{'y' if tally['would-repair']==1 else 'ies'} "
            "need repair. Re-run with --apply."
        )
    # Non-zero only for outcomes a human has to look at. A dry run that found work to do is
    # a successful report, not a failure.
    return 1 if (tally.get("failed") or tally.get("refused")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
