"""Tests for distill-onpolicy-checkpoint-eval-rerun/rerun_command.py.

The helper turns a finished distill build's target statuses into the rerun recipe's
parameters. The shapes here are the switch study's (build 2738af3c, arm A1u0): a
screening subset on every rung, the full battery on FULL_EVAL_RUNGS.
"""

import importlib.util
import pathlib

import pytest
import yaml

_REPO = pathlib.Path(__file__).resolve().parents[3]
_LSF = _REPO / "recipes" / "granite4-350m" / "lsf"
_SCRIPT = _LSF / "distill-onpolicy-checkpoint-eval-rerun" / "rerun_command.py"
_PARAMS = _LSF / "distill-onpolicy-switch-study" / "parameters.yaml"

_spec = importlib.util.spec_from_file_location("rerun_command", _SCRIPT)
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

_CATALOG = yaml.safe_load(_PARAMS.read_text(encoding="utf-8"))["EVAL_CATALOG"]
_SCREEN = ["salad-bench", "attaq", "global-mmlu"]
_CKPTS = "/proj/x/builds/b/runs/r/checkpoints/run_node2"
_APPLIED = {
    "CKPT_LADDER": "2250,4500",
    "EVAL_CATEGORIES": "code,general,math,safety,multilingual,bfcl",
    "EXPERIMENT": "exp",
    "WORKDIR_ROOT": "/proj/x/distill",
    "RUN_NAME": "run",
}


def _rung_targets(n, evals):
    names = [f"export-{n}", f"eval-transfer-{n}", f"export-sage-ck{n}"]
    names += [f"eval-{e}-ck{n}" for e in evals]
    if "bfcl" in evals:
        names.append(f"export-bfcl-ck{n}")
    return names


def _build(overrides=None, missing=()):
    """A finished build: every target succeeded except the ones named."""
    overrides = overrides or {}
    full = [e["name"] for e in _CATALOG] + ["bfcl"]
    names = ["train-gold"] + _rung_targets("2250", _SCREEN + ["bfcl"])
    names += _rung_targets("4500", full) + ["export-combined"]
    definition = [{"target_name": n} for n in names]
    targets = []
    for n in names:
        if n in missing:
            continue
        outputs = []
        if n == "train-gold":
            outputs = [
                {"uri": f"env://{_CKPTS}/checkpoint-{c}"} for c in ("2250", "4500")
            ]
        if n.startswith("export-") and n[7:].isdigit():
            outputs = [{"uri": f"env:///proj/x/distill/run/b/{n}"}]
        for status in overrides.get(n, ["success"]):
            targets.append(
                {"target_name": n, "status": status, "output_artifacts": outputs}
            )
    status = {"details": {"status": "failed", "name": "x"}, "targets": targets}
    return status, definition


def _plan(**kw):
    status, definition = _build(**kw)
    return rc.plan(status, definition, _APPLIED, _CATALOG)


def test_a_fully_measured_build_needs_nothing():
    params, notes = _plan()
    assert params is None and notes == []


def test_a_failure_its_own_retry_fixed_needs_nothing():
    """A1u0's case: both failed evals succeeded on the build's retry."""
    params, _ = _plan(
        overrides={"eval-olmes-gsm-symbolic-ck4500": ["failed", "success"]}
    )
    assert params is None


def test_failed_evals_rerun_with_their_rungs_csv_and_the_table():
    params, _ = _plan(
        overrides={
            "eval-olmes-gsm-symbolic-ck4500": ["failed", "failed"],
            "eval-olmes-minerva-math-ck4500": ["failed"],
        },
        missing={"export-sage-ck4500", "export-combined"},
    )
    assert params["RERUN_EVALS"] == "4500:olmes-gsm-symbolic,olmes-minerva-math"
    # The rerun recipe writes the rung's CSV behind its merge; no separate re-export.
    assert params["RERUN_SAGE_EXPORT"] == ""
    assert params["RUN_COMBINED"] == "true"


def test_an_eval_that_never_dispatched_counts_as_failed():
    params, _ = _plan(missing={"eval-attaq-ck2250", "export-sage-ck2250"})
    assert params["RERUN_EVALS"] == "2250:attaq"


def test_a_lost_sage_csv_is_re_exported_alone():
    params, _ = _plan(overrides={"export-sage-ck2250": ["failed"]})
    assert params["RERUN_SAGE_EXPORT"] == "2250"
    assert params["RERUN_EVALS"] == ""
    assert params["RUN_COMBINED"] == "true"


def test_bfcl_reruns_the_eval_or_only_the_csv():
    params, _ = _plan(
        overrides={"eval-bfcl-ck2250": ["failed"], "export-bfcl-ck4500": ["failed"]},
        missing={"export-bfcl-ck2250"},
    )
    assert params["RERUN_BFCL"] == "2250"
    assert params["RERUN_BFCL_EXPORT"] == "4500"


def test_a_failed_transfer_eval_reruns():
    params, _ = _plan(overrides={"eval-transfer-4500": ["failed"]})
    assert params["RERUN_TRANSFER"] == "4500"


def test_only_the_table_lost():
    params, _ = _plan(overrides={"export-combined": ["failed"]})
    assert params["RUN_COMBINED"] == "true"
    assert not any(params[k] for k in rc._PLAN_KEYS if k != "RUN_COMBINED")


def test_a_lost_export_becomes_a_new_rung_with_its_screening_subset():
    rung = [n for n in _rung_targets("2250", _SCREEN + ["bfcl"]) if n != "export-2250"]
    params, _ = _plan(overrides={"export-2250": ["failed"]}, missing=set(rung))
    assert params["NEW_RUNGS"] == "2250"
    assert params["NEW_RUNG_EVALS"] == "2250:" + ",".join(sorted(_SCREEN + ["bfcl"]))
    assert params["CKPT_SRC_DIR"] == _CKPTS


def test_a_lost_export_on_a_full_rung_needs_no_eval_filter():
    params, _ = _plan(overrides={"export-4500": ["failed"]})
    assert params["NEW_RUNGS"] == "4500"
    assert params["NEW_RUNG_EVALS"] == ""


def test_a_rung_without_a_checkpoint_is_reported_not_planned():
    status, definition = _build(overrides={"export-2250": ["failed"]})
    for t in status["targets"]:
        if t["target_name"] == "train-gold":
            t["output_artifacts"] = t["output_artifacts"][1:]
    params, notes = rc.plan(status, definition, _APPLIED, _CATALOG)
    assert params is None
    assert "2250" in notes[0]


def _died_after_last_save(on_disk):
    """Build d7966914's shape: train-gold FAILED after writing 4500 but before
    publishing it, and export-2250 lost to an SSH submit error."""
    status, definition = _build(
        overrides={"train-gold": ["failed"], "export-2250": ["failed"]},
        missing={"export-4500", "export-combined"},
    )
    for t in status["targets"]:
        if t["target_name"] == "train-gold":
            t["output_artifacts"] = t["output_artifacts"][:1]
    seen = []

    def ckpt_on_disk(ckpt_dir, n):
        seen.append((ckpt_dir, n))
        return on_disk

    params, notes = rc.plan(
        status, definition, _APPLIED, _CATALOG, ckpt_on_disk=ckpt_on_disk
    )
    return params, notes, seen


def test_a_failed_train_targets_published_checkpoint_is_used():
    params, _, seen = _died_after_last_save(on_disk=True)
    assert "2250" in params["NEW_RUNGS"].split(",")
    assert params["CKPT_SRC_DIR"] == _CKPTS
    # Published checkpoints are not looked for on disk; only the unpublished one is.
    assert seen == [(_CKPTS, "4500")]


def test_an_unpublished_checkpoint_found_on_disk_is_planned():
    params, notes, _ = _died_after_last_save(on_disk=True)
    assert params["NEW_RUNGS"] == "2250,4500"
    assert params["NEW_RUNG_EVALS"] == "2250:" + ",".join(sorted(_SCREEN + ["bfcl"]))
    assert any("checkpoint-4500" in n and "found complete" in n for n in notes)


def test_an_unpublished_checkpoint_missing_on_disk_is_lost():
    params, notes, _ = _died_after_last_save(on_disk=False)
    assert params["NEW_RUNGS"] == "2250"
    assert any("rungs 4500" in n for n in notes)


def test_an_unpublished_checkpoint_that_cannot_be_checked_is_assumed_and_noted():
    params, notes, _ = _died_after_last_save(on_disk=None)
    assert params["NEW_RUNGS"] == "2250,4500"
    assert any("checkpoint-4500" in n and "assumed" in n for n in notes)


def test_a_successful_train_target_is_never_second_guessed_on_disk():
    status, definition = _build(overrides={"export-2250": ["failed"]})
    for t in status["targets"]:
        if t["target_name"] == "train-gold":
            t["output_artifacts"] = t["output_artifacts"][1:]
    params, _ = rc.plan(
        status, definition, _APPLIED, _CATALOG, ckpt_on_disk=lambda d, n: True
    )
    assert params is None


def test_the_ssh_check_reads_the_remote_answer(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        out = {"yes": "yes\n", "no": "no\n"}[answer]
        return rc.subprocess.CompletedProcess(
            cmd, 255 if answer == "down" else 0, out, ""
        )

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    check = rc.ssh_ckpt_check("bluevela", "")
    answer = "yes"
    assert check("/d", "4500") is True
    assert calls[0][-2] == "bluevela"
    assert all(f"/d/checkpoint-4500/{s}" in calls[0][-1] for s in rc.CKPT_SENTINELS)
    answer = "no"
    assert check("/d", "4500") is False


def test_the_ssh_check_is_unknown_when_ssh_fails(monkeypatch):
    monkeypatch.setattr(
        rc.subprocess,
        "run",
        lambda cmd, **kw: rc.subprocess.CompletedProcess(cmd, 255, "", "refused"),
    )
    assert rc.ssh_ckpt_check("bluevela", "")("/d", "4500") is None


def test_a_failed_csv_that_registered_its_uri_is_flagged():
    status, definition = _build(overrides={"export-sage-ck2250": ["failed"]})
    for t in status["targets"]:
        if t["target_name"] == "export-sage-ck2250":
            t["output_artifacts"] = [{"uri": "env:///csv"}]
    _, notes = rc.plan(status, definition, _APPLIED, _CATALOG)
    assert any("export-combined never runs" in n for n in notes)


def test_an_eval_missing_from_the_catalog_is_an_error():
    status, definition = _build()
    definition.append({"target_name": "eval-no-such-eval-ck2250"})
    with pytest.raises(rc.RerunError):
        rc.plan(status, definition, _APPLIED, _CATALOG)


def test_the_planned_parameters_render(tmp_path):
    params, _ = _plan(
        overrides={
            "eval-olmes-gsm-symbolic-ck4500": ["failed"],
            "export-sage-ck2250": ["failed"],
            "eval-transfer-2250": ["failed"],
        },
    )
    params.update(
        EXPERIMENT="exp",
        SRC_BUILD_DIR="/proj/x/distill/run/b",
        RERUN_SAGE_DIR="/proj/x/sage-rerun/t",
        OE_EVAL_BCB_API_URL="http://bcb/",
    )
    targets = rc.render_check(_PARAMS, params)
    assert sorted(targets) == [
        "eval-olmes-gsm-symbolic-ck4500",
        "eval-transfer-2250",
        "export-combined",
        "export-sage-ck2250",
        "export-sage-ck4500",
        "merge-sage-ck4500",
    ]


def test_overrides_carry_only_what_the_rerun_recipe_reads():
    parent = yaml.safe_load(_PARAMS.read_text(encoding="utf-8"))
    applied = dict(parent, CORPUS_DIR="/other", RUN_NAME="run", LMBDA="0.5")
    # RUN_NAME is named in the recipe's comments only; LMBDA not at all.
    assert rc.overrides(applied, parent) == {"CORPUS_DIR": "/other"}
