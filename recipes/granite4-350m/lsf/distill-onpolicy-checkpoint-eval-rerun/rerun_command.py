#!/usr/bin/env python3
"""Print the `gb build start` command that finishes a distill build's measurement.

Given the id of a finished distill build (distill-onpolicy-switch-study,
distill-onpolicy-checkpoint-eval, ...), read its targets, find every per-rung eval and
aggregation that never succeeded, and print the distill-onpolicy-checkpoint-eval-rerun
command that re-runs exactly those:

    lost export-<N>           NEW_RUNGS (+ NEW_RUNG_EVALS when the rung had a subset)
    failed eval-<name>-ck<N>  RERUN_EVALS, then merge-sage + export-sage for the rung
    failed eval-bfcl-ck<N>    RERUN_BFCL, then export-bfcl
    failed eval-transfer-<N>  RERUN_TRANSFER
    failed export-sage-ck<N>  RERUN_SAGE_EXPORT, when every sage eval of the rung passed
    failed export-bfcl-ck<N>  RERUN_BFCL_EXPORT, when the rung's BFCL eval passed
    failed export-combined    RUN_COMBINED, which also runs whenever anything above does

"Failed" means no attempt of the target succeeded, which covers targets that never
dispatched. It is the gbserver status only: a sage eval that reported SUCCESS with
another eval's file is not caught here (check the rung's CSV for MISSING cells).

The build's own parameters come from its archive (parameters-applied.yaml). The command
reads the parent recipe's parameters.yaml, as the rerun recipe's header prescribes, and
adds a --param for every value the rerun recipe reads that the build ran with a
different value of (CORPUS_DIR, EVAL_QUEUE, ...). The command is rendered once before
it is printed, so a command this prints at least renders.

Run from the repo root, with gbserver up (the gb CLI must reach it):

    python recipes/granite4-350m/lsf/distill-onpolicy-checkpoint-eval-rerun/rerun_command.py \\
        2738af3c-cb43-4e66-a1bf-4c86e87dea79

Exit status: 0 with a command on stdout, 0 with nothing on stdout when there is nothing
to re-run, 1 on an error, 2 when the build has not finished (--allow-running overrides).
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile

import yaml

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
_RECIPE = _HERE / "build.yaml"
_LSF = _HERE.parent
_SAGE_RERUN_ROOT = "/proj/granite-build/g4os/sage-rerun"

# Set from the build itself, never copied over from parameters-applied.yaml.
_COMPUTED = {
    "EXPERIMENT",
    "SRC_BUILD_DIR",
    "CKPT_SRC_DIR",
    "NEW_RUNGS",
    "NEW_RUNG_EVALS",
    "RERUN_EVALS",
    "RERUN_BFCL",
    "RERUN_TRANSFER",
    "RERUN_SAGE_EXPORT",
    "RERUN_BFCL_EXPORT",
    "RERUN_SAGE_DIR",
    "RUN_COMBINED",
    "OE_EVAL_BCB_API_URL",
    "EVAL_CATALOG",
}

# What plan() decides, in the order the command lists it.
_PLAN_KEYS = (
    "NEW_RUNGS",
    "NEW_RUNG_EVALS",
    "RERUN_EVALS",
    "RERUN_BFCL",
    "RERUN_TRANSFER",
    "RERUN_SAGE_EXPORT",
    "RERUN_BFCL_EXPORT",
    "RUN_COMBINED",
)

_EVAL_TARGET = re.compile(r"^eval-(.+)-ck(\d+)$")


class RerunError(Exception):
    pass


def _gb(*args):
    env = dict(os.environ)
    # Unset means PROD, where every server round-trip hangs.
    env.setdefault("GB_ENVIRONMENT", "STANDALONE")
    proc = subprocess.run(
        ["gb", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        raise RerunError(f"gb {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def fetch_build(build_id):
    """The build's status, its target definitions, and its applied parameters."""
    status = json.loads(
        _gb(
            "build",
            "status",
            build_id,
            "--format",
            "json",
            "-q",
            "--skip-version-check",
        )
    )
    definition = json.loads(
        _gb(
            "build",
            "describe",
            build_id,
            "--format",
            "json",
            "-q",
            "--skip-version-check",
        )
    )
    with tempfile.TemporaryDirectory(prefix="rerun-command-") as tmp:
        _gb(
            "build",
            "init",
            f"{tmp}/src",
            "--from-build",
            build_id,
            "-q",
            "--skip-version-check",
        )
        applied_path = pathlib.Path(tmp, "src", "parameters-applied.yaml")
        if not applied_path.is_file():
            raise RerunError(
                f"build {build_id}'s archive has no parameters-applied.yaml"
            )
        applied = yaml.safe_load(applied_path.read_text(encoding="utf-8"))
    return status, definition, applied


def parent_recipe(build_name):
    """The recipe directory whose build.yaml is named build_name, with a parameters.yaml."""
    for build_yaml in sorted(_LSF.glob("*/build.yaml")):
        match = re.search(
            r"^  name:\s*(\S+)", build_yaml.read_text(encoding="utf-8"), re.M
        )
        if match and match.group(1) == build_name:
            params = build_yaml.parent / "parameters.yaml"
            if params.is_file():
                return params
    raise RerunError(
        f"no recipe under {_LSF.relative_to(_REPO)} is named {build_name!r} and has a "
        "parameters.yaml; pass --parameters-path"
    )


def _succeeded(status):
    """Target names with at least one successful attempt."""
    return {t["target_name"] for t in status["targets"] if t["status"] == "success"}


def _outputs_of_failed(status, name):
    """Output URIs a never-successful target registered anyway."""
    return [
        o["uri"]
        for t in status["targets"]
        if t["target_name"] == name
        for o in t.get("output_artifacts", [])
    ]


def plan(status, definition, applied, catalog):
    """The rerun recipe's build-specific parameters, and the notes that go with them.

    `definition` is the build's target list (`gb build describe --format json`), so a
    target that never dispatched counts as failed; `catalog` is the parent recipe's
    EVAL_CATALOG, the names RERUN_EVALS may use.
    """
    ok = _succeeded(status)
    defined = [t["target_name"] for t in definition]
    by_name = {t["target_name"]: t for t in definition}
    sage_names = {e["name"] for e in catalog}
    notes = []

    rungs = [n.strip() for n in str(applied["CKPT_LADDER"]).split(",") if n.strip()]
    cats = str(applied.get("EVAL_CATEGORIES", "")).replace(" ", "").lower().split(",")
    full_set = {e["name"] for e in catalog if e["category"] in cats}
    if "bfcl" in cats:
        full_set.add("bfcl")

    evals_of = {n: [] for n in rungs}
    for name in defined:
        match = _EVAL_TARGET.match(name)
        if match and match.group(2) in evals_of:
            evals_of[match.group(2)].append(match.group(1))
    unknown = sorted(
        {e for es in evals_of.values() for e in es} - sage_names - {"bfcl"}
    )
    if unknown:
        raise RerunError(
            f"evals {unknown} are not in the parent recipe's EVAL_CATALOG; pass the "
            "--parameters-path the build was started with"
        )

    checkpoints = {}
    for t in status["targets"]:
        if t["target_name"].startswith("train") and t["status"] == "success":
            for o in t["output_artifacts"]:
                m = re.search(r"/checkpoint-(\d+)$", o["uri"])
                if m:
                    checkpoints[m.group(1)] = o["uri"].removeprefix("env://")

    new, new_only, rerun, rerun_b, rerun_t, rerun_sx, rerun_bx = (
        [],
        {},
        {},
        [],
        [],
        [],
        [],
    )
    lost = []
    for n in rungs:
        if f"export-{n}" in defined and f"export-{n}" not in ok:
            if n not in checkpoints:
                lost.append(n)
                continue
            new.append(n)
            if set(evals_of[n]) != full_set:
                new_only[n] = sorted(evals_of[n])
            continue
        if f"eval-transfer-{n}" in defined and f"eval-transfer-{n}" not in ok:
            rerun_t.append(n)
        failed = [e for e in evals_of[n] if e != "bfcl" and f"eval-{e}-ck{n}" not in ok]
        if failed:
            rerun[n] = failed
        elif f"export-sage-ck{n}" in defined and f"export-sage-ck{n}" not in ok:
            rerun_sx.append(n)
        if "bfcl" in evals_of[n] and f"eval-bfcl-ck{n}" not in ok:
            rerun_b.append(n)
        elif f"export-bfcl-ck{n}" in defined and f"export-bfcl-ck{n}" not in ok:
            rerun_bx.append(n)

    if lost:
        notes.append(
            f"rungs {','.join(lost)}: export failed and training left no checkpoint for "
            "them; nothing here can measure them"
        )
    ckpt_dirs = {str(pathlib.PurePosixPath(checkpoints[n]).parent) for n in new}
    if len(ckpt_dirs) > 1:
        raise RerunError(
            f"the new rungs' checkpoints are in several directories: {ckpt_dirs}"
        )

    # A CSV the failed target registered anyway is refused when this build emits it,
    # which strands export-combined (see the rerun recipe's header).
    for name in [f"export-sage-ck{n}" for n in rerun_sx] + [
        f"export-bfcl-ck{n}" for n in rerun_bx
    ]:
        if _outputs_of_failed(status, name):
            notes.append(
                f"{name} failed but registered {_outputs_of_failed(status, name)}: this "
                "build's copy is dropped as a duplicate, and export-combined never runs"
            )
    for n in rerun_b:
        if _outputs_of_failed(status, f"eval-bfcl-ck{n}"):
            notes.append(
                f"eval-bfcl-ck{n} failed but registered an output URI; the re-run's is "
                f"dropped as a duplicate and export-bfcl-ck{n} never runs"
            )

    anything = new or rerun or rerun_b or rerun_t or rerun_sx or rerun_bx
    combined = "export-combined" in by_name and (
        anything or "export-combined" not in ok
    )
    if not anything and not combined:
        return None, notes

    params = {
        "NEW_RUNGS": ",".join(new),
        "NEW_RUNG_EVALS": ";".join(f"{n}:{','.join(e)}" for n, e in new_only.items()),
        "RERUN_EVALS": ";".join(f"{n}:{','.join(e)}" for n, e in rerun.items()),
        "RERUN_BFCL": ",".join(rerun_b),
        "RERUN_TRANSFER": ",".join(rerun_t),
        "RERUN_SAGE_EXPORT": ",".join(rerun_sx),
        "RERUN_BFCL_EXPORT": ",".join(rerun_bx),
        "RUN_COMBINED": "true" if combined else "false",
    }
    if new:
        params["CKPT_SRC_DIR"] = ckpt_dirs.pop()
    return params, notes


def src_build_dir(status, applied, build_id):
    for t in status["targets"]:
        if re.fullmatch(r"export-\d+", t["target_name"]) and t["status"] == "success":
            for o in t["output_artifacts"]:
                return str(
                    pathlib.PurePosixPath(o["uri"].removeprefix("env://")).parent
                )
    return (
        f"{str(applied['WORKDIR_ROOT']).rstrip('/')}/{applied['RUN_NAME']}/{build_id}"
    )


def overrides(applied, parent):
    """--params for values the rerun recipe reads that the build ran with differently."""
    # Names inside template markers only: the recipe's comments name parameters too.
    text = _RECIPE.read_text(encoding="utf-8")
    markers = re.findall(r"\$\$\{(.*?)\}|<%(.*?)%>", text, re.S)
    code = " ".join(part for pair in markers for part in pair)
    referenced = set(re.findall(r"\b[A-Z][A-Z0-9_]+\b", code))
    out = {}
    for key in sorted(referenced & set(applied) - _COMPUTED):
        value = applied[key]
        if isinstance(value, (dict, list)):
            continue
        if key not in parent or str(parent[key]) != str(value):
            if value is None:
                value = ""
            elif isinstance(value, bool):
                value = str(value).lower()
            out[key] = str(value)
    return out


def _display(path):
    path = pathlib.Path(path).resolve()
    return str(path.relative_to(_REPO)) if path.is_relative_to(_REPO) else str(path)


def render_check(parameters_path, params):
    """Render the rerun recipe as `gb build start` would; the target names it yields."""
    sys.path.insert(0, str(_REPO / "src"))
    from gbcli.services.service_build import get_params_from_file
    from gbcli.utils.buildutil import apply_parameters

    merged = get_params_from_file(str(parameters_path))
    merged.update(params)
    with tempfile.TemporaryDirectory() as tmp:
        text = apply_parameters(_RECIPE.read_text(encoding="utf-8"), [], merged, tmp)
    return list(yaml.safe_load(text)["granite.build"]["targets"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("build_id", help="the full UUID of the build to finish")
    parser.add_argument(
        "--parameters-path",
        help="the parent recipe's parameters.yaml (default: the recipe the build is named after)",
    )
    parser.add_argument("--space", default="public")
    parser.add_argument(
        "--rerun-sage-dir",
        help=f"where re-run sage evals write (default: {_SAGE_RERUN_ROOT}/<EXPERIMENT>-<now>). "
        "Pass an earlier rerun's directory to reuse the evals it finished",
    )
    parser.add_argument(
        "--bcb-url", help="a live bcb-server URL, if BigCodeBench re-runs"
    )
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help="plan against a build that has not finished (its retries may still succeed)",
    )
    args = parser.parse_args(argv)

    try:
        status, definition, applied = fetch_build(args.build_id)
        build_status = status["details"]["status"]
        finished = build_status in ("success", "failed", "cancelled", "invalid")
        if not finished and not args.allow_running:
            print(
                f"build {args.build_id} is {build_status}; its own retries may still "
                "succeed. Re-run this once it finishes, or pass --allow-running.",
                file=sys.stderr,
            )
            return 2
        parameters_path = pathlib.Path(
            args.parameters_path or parent_recipe(status["details"]["name"])
        )
        parent = yaml.safe_load(parameters_path.read_text(encoding="utf-8"))
        params, notes = plan(status, definition, applied, parent["EVAL_CATALOG"])
    except RerunError as e:
        print(f"rerun_command: {e}", file=sys.stderr)
        return 1

    for note in notes:
        print(f"NOTE: {note}", file=sys.stderr)
    if params is None:
        print(
            f"build {args.build_id} ({build_status}): every eval and aggregation succeeded; "
            "nothing to re-run.",
            file=sys.stderr,
        )
        return 0

    experiment = applied["EXPERIMENT"]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    bcb_url = args.bcb_url or applied.get("OE_EVAL_BCB_API_URL", "")
    params.update(
        {
            "EXPERIMENT": experiment,
            "SRC_BUILD_DIR": src_build_dir(status, applied, args.build_id),
            "RERUN_SAGE_DIR": args.rerun_sage_dir
            or f"{_SAGE_RERUN_ROOT}/{experiment}-{stamp}",
            "OE_EVAL_BCB_API_URL": bcb_url,
        }
    )
    extra = overrides(applied, parent)
    try:
        targets = render_check(parameters_path, {**extra, **params})
    except Exception as e:  # pylint: disable=broad-except
        print(f"rerun_command: the command does not render: {e}", file=sys.stderr)
        return 1
    bcb_evals = {e["name"] for e in parent["EVAL_CATALOG"] if e.get("needs_bcb_url")}
    if not args.bcb_url and any(
        _EVAL_TARGET.match(t) and _EVAL_TARGET.match(t).group(1) in bcb_evals
        for t in targets
    ):
        print(
            f"NOTE: BigCodeBench re-runs against {bcb_url}, the build's bcb-server; "
            "check that it still answers, or pass --bcb-url",
            file=sys.stderr,
        )

    print(
        f"build {args.build_id} ({build_status}), experiment {experiment}:",
        file=sys.stderr,
    )
    for key in _PLAN_KEYS:
        if params[key]:
            print(f"  {key:<18} {params[key]}", file=sys.stderr)
    if extra:
        print(f"  ran with non-default {', '.join(sorted(extra))}", file=sys.stderr)
    print(
        f"  renders to {len(targets)} targets: {' '.join(sorted(targets))}",
        file=sys.stderr,
    )

    lines = [
        f"gb build start -f {_RECIPE.relative_to(_REPO)}",
        f"--parameters-path {shlex.quote(_display(parameters_path))}",
        f"--space {shlex.quote(args.space)}",
    ]
    # NEW_RUNGS and RERUN_EVALS have no default in the recipe; the other lists do.
    order = ["EXPERIMENT", "SRC_BUILD_DIR", "CKPT_SRC_DIR", *_PLAN_KEYS]
    order += ["RERUN_SAGE_DIR", "OE_EVAL_BCB_API_URL"]
    lines += [
        f"--param {shlex.quote(f'{k}={params[k]}')}"
        for k in order
        if k in params and (params[k] or k in ("NEW_RUNGS", "RERUN_EVALS"))
    ]
    lines += [f"--param {shlex.quote(f'{k}={v}')}" for k, v in extra.items()]
    print(" \\\n  ".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
