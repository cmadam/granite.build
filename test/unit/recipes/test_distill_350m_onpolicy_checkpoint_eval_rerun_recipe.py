"""Tests for recipes/granite4-350m/lsf/distill-onpolicy-checkpoint-eval-rerun.

The rerun recipe takes its parameters from a parent recipe's parameters.yaml, plus a
handful of --params naming the build it finishes. These pin RERUN_BFCL, which BFCL
needs because BFCL is not in EVAL_CATALOG and so cannot ride on RERUN_EVALS. The
first build to need it was ck2500's lmbda-0 control (dfe3642d), whose eval-bfcl-ck2000
failed at provisioning.
"""

import pathlib

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_REPO = pathlib.Path(__file__).resolve().parents[3]
_LSF = _REPO / "recipes" / "granite4-350m" / "lsf"
_RECIPE = _LSF / "distill-onpolicy-checkpoint-eval-rerun" / "build.yaml"
_PARENT_PARAMS = _LSF / "distill-onpolicy-ck2500-continue" / "parameters.yaml"

_SRC = "/proj/x/distill/run/build"
_BUILD_PARAMS = {
    "EXPERIMENT": "exp",
    "SRC_BUILD_DIR": _SRC,
    "CKPT_SRC_DIR": "/proj/x/ckpts/run_node2",
    "NEW_RUNGS": "",
    "RERUN_EVALS": "500:multiple-js",
    "RERUN_SAGE_DIR": "/proj/x/sage-rerun/tag",
}


def _render(tmp_path, **overrides):
    params = get_params_from_file(str(_PARENT_PARAMS))
    params.update(_BUILD_PARAMS)
    params.update(overrides)
    contents = _RECIPE.read_text(encoding="utf-8")
    rendered = yaml.safe_load(apply_parameters(contents, [], params, str(tmp_path)))
    return rendered["granite.build"]["targets"]


def test_rerun_bfcl_is_optional(tmp_path):
    """Every launch command written before RERUN_BFCL existed must still render."""
    targets = _render(tmp_path)
    assert not [name for name in targets if "bfcl" in name]


@pytest.mark.parametrize("value", ["", " "])
def test_an_empty_rerun_bfcl_reruns_nothing(tmp_path, value):
    targets = _render(tmp_path, RERUN_BFCL=value)
    assert not [name for name in targets if "bfcl" in name]


def test_a_bfcl_rerun_reads_the_existing_export(tmp_path):
    """No export target here: the rung is already exported, and re-exporting it would
    register a second artifact at the original build's export URI."""
    targets = _render(tmp_path, RERUN_BFCL="2000")
    assert "export-2000" not in targets
    model = targets["eval-bfcl-ck2000"]["inputs"]["model"]
    assert model["uri"] == f"env://{_SRC}/export-2000"
    config = targets["eval-bfcl-ck2000"]["steps"][0]["config"]["bfcl_config"]
    assert config["model_path"] == f"{_SRC}/export-2000"
    assert config["experiment"] == "exp-ckpt_2000"


def test_a_bfcl_rerun_writes_where_the_original_would_have(tmp_path):
    """bfcl-export reads <BFCL_OUTPUT_DIR>/<experiment>-ckpt_<N>; a re-run anywhere else
    would leave the rung's CSV empty."""
    params = get_params_from_file(str(_PARENT_PARAMS))
    targets = _render(tmp_path, RERUN_BFCL="2000")
    out = targets["eval-bfcl-ck2000"]["outputs"]["bfcl_results"]["uri"]
    assert out == (
        f"env://{params['BFCL_OUTPUT_DIR']}/exp-ckpt_2000/{params['BFCL_EVAL_NAME']}"
    )


def test_the_rungs_csv_and_the_table_wait_for_the_rerun(tmp_path):
    targets = _render(tmp_path, RERUN_BFCL="1500,2000")
    for n in ("1500", "2000"):
        gate = targets[f"export-bfcl-ck{n}"]["inputs"]["gate_bfcl"]
        assert gate["binding"] == f"eval-bfcl-ck{n}.bfcl_results"
        combined = targets["export-combined"]["inputs"][f"gate_bfcl_{n}"]
        assert combined["binding"] == f"export-bfcl-ck{n}.bfcl_export_csv"
