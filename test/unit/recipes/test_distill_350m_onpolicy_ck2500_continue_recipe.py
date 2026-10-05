#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for recipes/granite4-350m/lsf/distill-onpolicy-ck2500-continue.

The recipe is distill-onpolicy-checkpoint-eval continued from its step-2,500 export, run
as three arms that differ only in LMBDA: 0 (an off-policy control), and linear ramps from
0.25 to 0.5 and to 0.75. The parent's suite guards everything the two share, so this file
guards that the copy IS a copy, and the three things the copy adds:

* the parameters differ from the parent in exactly the keys the continuation changes, and
  build.yaml only in its name, its mem:// URIs and the lmbda-0 conditionals;
* every arm's trainer config passes the renderer's own checks -- the ones that would
  otherwise fail after the allocation is held -- and the control's is the off-policy
  recipe's, so the arms differ in lmbda and in nothing else;
* the ramp arms read the patched trainer checkout the linear schedule needs, and the
  patch file states the commit the recipe pins.
"""

import argparse
import difflib
import importlib.util
import pathlib
import re

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_REPO = pathlib.Path(__file__).resolve().parents[3]
_LSF = _REPO / "recipes" / "granite4-350m" / "lsf"
_RECIPE = _LSF / "distill-onpolicy-ck2500-continue"
_PARENT = _LSF / "distill-onpolicy-checkpoint-eval"
_OFFPOLICY = _LSF / "distill-checkpoint-eval"
_GOLD_SRC = _REPO / "steps" / "distill" / "distill-gold" / "skypilot"
_PATCH = _GOLD_SRC / "patches" / "lmbda_linear_schedule.diff"

# The CLI delivers --param values as STRINGS, so the arms are spelled the way a command
# line spells them, "0.0" included.
_ARMS = ["0", "0.0", "0.5", "0.75"]
_ON_POLICY_ARMS = ["0.5", "0.75"]

_CHANGED_PARAMS = {
    "RUN_NAME": "distill-350m-onpolicy-ck2500",
    "EXPERIMENT": "distill-350m-onpolicy-ck2500",
    "STUDENT_MODEL": (
        "/proj/granite-build/g4os/distill/distill-350m-onpolicy-c2000_20261003_1119/"
        "68b7c4b2-7d39-449d-9f0b-fb448ac2f5ad/export-2500"
    ),
    "CODE_DIR": "/proj/granite-build/g4os/gb-steps-distillation-lmbda-ramp",
    "CODE_EXPECT_REF": "288bfc6bdab589eca394ca9f7229f1138a6da1c4",
    "GOLD_DATA_SEED": 44,
    "GOLD_LEARNING_RATE": "1.0e-06",
    "LMBDA": 0.5,
    "LMBDA_SCHEDULE": "linear",
    "LMBDA_INIT": 0.25,
    "VLLM_MAX_LIFETIME_SECONDS": 129600,
}

_GATE = "<% if LMBDA | float > 0 %>"


def _params(path=_RECIPE, **overrides):
    params = get_params_from_file(str(path / "parameters.yaml"))
    params.update(overrides)
    return params


def _render(tmp_path, path=_RECIPE, **overrides):
    contents = (path / "build.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(
        apply_parameters(contents, [], _params(path, **overrides), str(tmp_path))
    )


def _targets(rendered):
    return rendered["granite.build"]["targets"]


def _config(rendered, target):
    return _targets(rendered)[target]["steps"][0]["config"]


def _code_lines(path):
    text = (path / "build.yaml").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


def _renderer():
    spec = importlib.util.spec_from_file_location(
        "render_gold_config_under_test", _GOLD_SRC / "src" / "render_gold_config.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _renderer_args(module, gold_config, total_nodes) -> argparse.Namespace:
    """The step hands render_gold_config.py each gold_config key as its kebab-case flag.
    Keys the renderer has no flag for are the step's own, and are dropped here as there.
    """
    flags = set(
        re.findall(
            r'"--([a-z0-9-]+)"',
            (_GOLD_SRC / "src" / "render_gold_config.py").read_text(),
        )
    )
    argv = ["--output", "/dev/null", "--total-nodes", str(total_nodes)]
    for key, value in gold_config.items():
        flag = key.replace("_", "-")
        if flag in flags:
            argv += [f"--{flag}", str(value)]
    return module._parse_args(argv)  # pylint: disable=protected-access


@pytest.fixture(name="arm", params=_ARMS)
def fixture_arm(request):
    return request.param


# ─── A copy of the parent ──────────────────────────────────────────────────────


def test_the_parameters_differ_from_the_parent_only_where_intended():
    mine, theirs = _params(), _params(_PARENT)
    assert set(mine) == set(theirs)
    assert {k: mine[k] for k in mine if mine[k] != theirs[k]} == _CHANGED_PARAMS


def test_the_build_differs_from_the_parent_only_where_intended():
    """Comments aside: the build name, the two mem:// URIs -- which must differ, or two
    arms' registrations collide -- and the lmbda-0 conditionals around the server pair,
    the trainer's server input and its on-policy keys."""
    diff = list(
        difflib.unified_diff(
            _code_lines(_PARENT), _code_lines(_RECIPE), n=0, lineterm=""
        )
    )
    removed = [d[1:] for d in diff if d.startswith("-") and not d.startswith("---")]
    added = [d[1:] for d in diff if d.startswith("+") and not d.startswith("+++")]
    assert removed == [
        "  name: distill-350m-onpolicy-ckpt-eval",
        '          uri: "mem://distill-onpolicy-ckpt-eval-vllm"',
        '          uri: "mem://distill-onpolicy-ckpt-eval-vllm-cluster"',
    ]
    assert added == [
        "  name: distill-350m-onpolicy-ck2500-continue",
        _GATE,
        '          uri: "mem://distill-onpolicy-ck2500-vllm-$${RUN_NAME}"',
        '          uri: "mem://distill-onpolicy-ck2500-vllm-cluster-$${RUN_NAME}"',
        "<% endif %>",
        "",
        _GATE,
        "<% endif %>",
        _GATE,
        "<% else %>",
        "              vllm_num_servers: 0",
        "<% endif %>",
        _GATE,
        "<% endif %>",
    ]


# ─── The three arms ────────────────────────────────────────────────────────────


def test_the_server_pair_exists_exactly_when_the_student_generates(arm, tmp_path):
    targets = _targets(_render(tmp_path, LMBDA=arm))
    on_policy = float(arm) > 0
    assert ("vllm-server" in targets) is on_policy
    assert ("teardown" in targets) is on_policy
    assert ("vllm" in targets["train-gold"]["inputs"]) is on_policy


def test_every_arm_passes_the_renderers_checks(arm, tmp_path):
    """The refusals that matter here -- a server URL at lmbda 0, a linear schedule
    off-policy, a floor off-policy -- all fire in render_gold_config.build_config, which
    runs on the allocation. Running it here makes each one a unit-test failure instead.
    """
    rendered = _render(tmp_path, LMBDA=arm)
    module = _renderer()
    gold = _config(rendered, "train-gold")["gold_config"]
    nodes = int(_config(rendered, "train-gold")["compute_config"]["num_nodes"])
    config = module.build_config(_renderer_args(module, gold, nodes))
    assert float(config["lmbda"]) == pytest.approx(float(arm))
    if float(arm) > 0:
        assert config["lmbda_schedule"] == "linear"
        assert config["lmbda_init"] == pytest.approx(0.25)
    else:
        assert "lmbda_schedule" not in config


def test_the_control_trains_like_the_off_policy_recipe(tmp_path):
    """Every gold_config key the off-policy recipe sets, set to the same value, except
    the ones the continuation exists to change. No on-policy key at all."""
    mine = _config(_render(tmp_path, LMBDA="0"), "train-gold")["gold_config"]
    theirs = _config(_render(tmp_path, _OFFPOLICY), "train-gold")["gold_config"]
    assert set(mine) - set(theirs) == {"data_seed"}
    assert set(theirs) - set(mine) == set()
    differing = {k for k in theirs if mine[k] != theirs[k]}
    assert differing == {
        "run_name",
        "max_steps",
        "learning_rate",
        "save_steps",
        "save_total_limit",
    }
    assert mine["vllm_num_servers"] == 0


def test_the_arms_differ_from_each_other_only_in_lmbda(tmp_path):
    ramps = [
        _config(_render(tmp_path, LMBDA=a), "train-gold")["gold_config"]
        for a in _ON_POLICY_ARMS
    ]
    control = _config(_render(tmp_path, LMBDA="0"), "train-gold")["gold_config"]
    assert {k for k in ramps[0] if ramps[0][k] != ramps[1][k]} == {"lmbda"}
    shared = set(control) - {"lmbda", "vllm_num_servers"}
    assert {k for k in shared if control[k] != ramps[0][k]} == set()


@pytest.mark.parametrize("target", ["train-gold", "vllm-server"])
def test_every_arm_reads_the_patched_trainer(target, tmp_path):
    for arm in ["0", *_ON_POLICY_ARMS]:
        targets = _targets(_render(tmp_path, LMBDA=arm))
        if target not in targets:
            continue
        code = targets[target]["steps"][0]["config"]["code_config"]
        assert code["code_dir"] == _CHANGED_PARAMS["CODE_DIR"], (arm, target)
        assert code["expect_ref"] == _CHANGED_PARAMS["CODE_EXPECT_REF"], (arm, target)


def test_the_patch_states_the_commit_the_recipe_pins():
    """CODE_EXPECT_REF can be stated before the checkout exists only because the patch
    header fixes the commit identity and date; the two must name the same commit."""
    header = _PATCH.read_text(encoding="utf-8").split("diff --git", 1)[0]
    assert _CHANGED_PARAMS["CODE_EXPECT_REF"] in header
    assert "a5d59bc45524a8d75706e20d44ae1a254f273f23" in header


def test_the_patch_only_moves_lines():
    """The fix relocates the stranded linear branch and changes none of it, so what the
    patch removes and what it adds are the same lines -- the claim the constant path's
    "byte-identical training" rests on."""
    body = "diff --git" + _PATCH.read_text(encoding="utf-8").split("diff --git", 1)[1]
    assert body.count("diff --git") == 1
    assert body.splitlines()[0].endswith("custom_gold_trainer.py")
    lines = body.splitlines()
    removed = sorted(l[1:] for l in lines if l[:1] == "-" and not l.startswith("---"))
    added = sorted(l[1:] for l in lines if l[:1] == "+" and not l.startswith("+++"))
    assert removed and removed == added


def test_two_arms_never_register_the_same_server_uri(tmp_path):
    uris = {
        _targets(_render(tmp_path, LMBDA=a, RUN_NAME=f"run-{a}"))["vllm-server"][
            "outputs"
        ][o]["uri"]
        for a in _ON_POLICY_ARMS
        for o in ("vllm_url", "cluster_name")
    }
    assert len(uris) == 4


def test_the_start_is_ck2500_and_not_the_parents_start():
    assert _params()["STUDENT_MODEL"].endswith("/export-2500")
    assert _params()["STUDENT_MODEL"] != _params(_PARENT)["STUDENT_MODEL"]


def test_the_sampler_seed_is_neither_earlier_runs():
    assert _params()["GOLD_DATA_SEED"] not in (
        42,
        43,
        _params(_PARENT)["GOLD_DATA_SEED"],
    )


def test_the_lr_peak_is_below_the_first_runs():
    assert float(_params()["GOLD_LEARNING_RATE"]) < float(
        _params(_PARENT)["GOLD_LEARNING_RATE"]
    )
