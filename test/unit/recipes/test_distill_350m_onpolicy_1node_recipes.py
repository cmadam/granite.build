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

"""Unit tests for the single-node on-policy recipes.

distill-onpolicy-v2-1node and distill-onpolicy-checkpoint-eval-1node are copies of their
two-node parents with the trainer on one 8-GPU node. Their parents' own suites guard
everything else, so this file guards only that the copies ARE copies: the parameters
differ in exactly the keys the variant exists to change, build.yaml differs only in
comments, the build name and the mem:// URIs, and the effective batch is unchanged.
"""

import difflib
import pathlib

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_REPO = pathlib.Path(__file__).resolve().parents[3]
_LSF = _REPO / "recipes" / "granite4-350m" / "lsf"

# Off-policy on this pair: 8,150 steps in 13h20 on 2 x 8 H100 (build df8512e0).
_OFFPOLICY_SECONDS_PER_STEP = (13 * 3600 + 20 * 60) / 8150

_GEOMETRY = {"GOLD_NUM_NODES": 1, "GOLD_GRADIENT_ACCUMULATION_STEPS": 6}

_VARIANTS = {
    "distill-onpolicy-v2-1node": {
        "parent": "distill-onpolicy-v2",
        "params": {
            **_GEOMETRY,
            "RUN_NAME": "distill-350m-onpolicy-v2-1node",
        },
        "lines": {
            "  name: distill-350m-onpolicy-v2": "  name: distill-350m-onpolicy-v2-1node",
            '          uri: "mem://distill-onpolicy-v2-vllm"': (
                '          uri: "mem://distill-onpolicy-v2-1node-vllm"'
            ),
            '          uri: "mem://distill-onpolicy-v2-vllm-cluster"': (
                '          uri: "mem://distill-onpolicy-v2-1node-vllm-cluster"'
            ),
        },
    },
    "distill-onpolicy-checkpoint-eval-1node": {
        "parent": "distill-onpolicy-checkpoint-eval",
        "params": {
            **_GEOMETRY,
            "RUN_NAME": "distill-350m-onpolicy-c2000-1node",
            "EXPERIMENT": "distill-350m-onpolicy-c2000-1node",
            "VLLM_MAX_LIFETIME_SECONDS": 129600,
        },
        "lines": {
            "  name: distill-350m-onpolicy-ckpt-eval": (
                "  name: distill-350m-onpolicy-ckpt-eval-1node"
            ),
            '          uri: "mem://distill-onpolicy-ckpt-eval-vllm"': (
                '          uri: "mem://distill-onpolicy-ckpt-eval-1node-vllm"'
            ),
            '          uri: "mem://distill-onpolicy-ckpt-eval-vllm-cluster"': (
                '          uri: "mem://distill-onpolicy-ckpt-eval-1node-vllm-cluster"'
            ),
        },
    },
}


def _params(name):
    return get_params_from_file(str(_LSF / name / "parameters.yaml"))


def _render(tmp_path, name):
    contents = (_LSF / name / "build.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(apply_parameters(contents, [], _params(name), str(tmp_path)))


def _code_lines(name):
    text = (_LSF / name / "build.yaml").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


@pytest.fixture(name="variant", params=sorted(_VARIANTS))
def fixture_variant(request):
    return request.param


def test_the_parameters_differ_from_the_parent_only_where_intended(variant):
    spec = _VARIANTS[variant]
    mine, theirs = _params(variant), _params(spec["parent"])
    assert set(mine) == set(theirs)
    changed = {k: mine[k] for k in mine if mine[k] != theirs[k]}
    assert changed == spec["params"]


def test_the_build_differs_from_the_parent_only_where_intended(variant):
    """Comments aside, the only changed lines are the build name and the mem:// URIs --
    which must differ, or the two recipes' artifacts collide in the space."""
    spec = _VARIANTS[variant]
    diff = list(
        difflib.unified_diff(
            _code_lines(spec["parent"]), _code_lines(variant), n=0, lineterm=""
        )
    )
    removed = [d[1:] for d in diff if d.startswith("-") and not d.startswith("---")]
    added = [d[1:] for d in diff if d.startswith("+") and not d.startswith("+++")]
    assert removed == list(spec["lines"])
    assert added == list(spec["lines"].values())


def test_the_effective_batch_is_the_parents(variant):
    def product(p):
        return (
            p["GOLD_PER_DEVICE_TRAIN_BATCH_SIZE"]
            * p["GOLD_GRADIENT_ACCUMULATION_STEPS"]
            * p["GOLD_NUM_NODES"]
            * p["GOLD_NUM_GPUS"]
        )

    assert (
        product(_params(variant))
        == product(_params(_VARIANTS[variant]["parent"]))
        == 96
    )


def test_the_trainer_renders_onto_one_node(variant, tmp_path):
    target = _render(tmp_path, variant)["granite.build"]["targets"]["train-gold"]
    config = target["steps"][0]["config"]
    assert int(config["compute_config"]["num_nodes"]) == 1
    assert int(config["compute_config"]["num_gpus_per_node"]) == 8
    assert config["launcher_config"]["resources"]["accelerators"] == "H100:8"


def test_the_server_outlives_a_run_on_half_the_gpus():
    """The ceiling covers 3x the off-policy step time, doubled for one node instead of
    two -- the same margin the parent's test asserts for two nodes."""
    p = _params("distill-onpolicy-checkpoint-eval-1node")
    steps = int(p["GOLD_MAX_STEPS"])
    assert (
        int(p["VLLM_MAX_LIFETIME_SECONDS"])
        >= steps * 3 * 2 * _OFFPOLICY_SECONDS_PER_STEP
    )
