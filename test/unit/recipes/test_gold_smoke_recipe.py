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

"""Unit tests for the gold-smoke distillation recipe (issue #60).

The recipe is a build.yaml + parameters.yaml pair, so the thing that can break it
is a parameter that exists in one file and not the other — which on a live run
surfaces only after LSF has queued a 2-node H100 allocation. These tests render
the pair through the same substitution gbcli uses (StrictUndefined, so a missing
parameter raises) and assert the rendered result: the three lineage inputs, their
agreement with what gold_config passes the trainer, and the value TYPES, since a
float written unquoted in parameters.yaml round-trips into build.yaml as a string.
"""

import pathlib

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_RECIPE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "recipes"
    / "granite4-gold"
    / "lsf"
    / "gold-smoke"
)


@pytest.fixture
def rendered(tmp_path):
    """The recipe rendered against its own parameters.yaml.

    apply_parameters writes a parameters-applied side-effect file into the folder
    it is handed, so it gets tmp_path rather than the recipe dir.
    """
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    return yaml.safe_load(apply_parameters(contents, [], params, str(tmp_path)))


def _target(rendered):
    targets = rendered["granite.build"]["targets"]
    assert list(targets) == ["gold-smoke"], "recipe is a single-target build"
    return targets["gold-smoke"]


def test_every_parameter_resolves(rendered):
    """No $${...} survives rendering.

    apply_parameters uses StrictUndefined, so a parameter missing from
    parameters.yaml raises during the fixture. This asserts the other direction:
    that nothing was left literal by a malformed marker (e.g. ${VAR} or $${VAR}}).
    """
    assert "$${" not in yaml.safe_dump(rendered)


def test_three_lineage_inputs_declared(rendered):
    """The teacher, student and dataset are lineage inputs.

    Lineage inputs[] is built from the target's input_artifacts, i.e. this block —
    see WandBLineageStore.create_jobstats_for_target. A lineage-supported `type` is
    required for the artifact to appear.
    """
    inputs = _target(rendered)["inputs"]
    assert set(inputs) == {"teacher_model", "student_model", "training_dataset"}
    assert inputs["teacher_model"]["type"] == "model"
    assert inputs["student_model"]["type"] == "model"
    assert inputs["training_dataset"]["type"] == "dataset"
    for name, spec in inputs.items():
        # env:// needs an absolute path (pullasset_envstore rejects a relative
        # one), so the scheme is followed by three slashes once /proj/... lands.
        assert spec["uri"].startswith("env:///"), name


def test_inputs_agree_with_what_the_trainer_is_given(rendered):
    """The lineage record and the trainer's arguments cannot disagree.

    Both sides resolve from the same parameter, so this pins the wiring: an input
    renamed on one side only would make lineage describe a run that did not happen.
    """
    target = _target(rendered)
    inputs = target["inputs"]
    gold = target["steps"][0]["config"]["gold_config"]

    assert inputs["student_model"]["uri"] == "env://" + gold["model_name_or_path"]
    assert (
        inputs["teacher_model"]["uri"] == "env://" + gold["teacher_model_name_or_path"]
    )
    assert inputs["training_dataset"]["uri"] == "env://" + gold["dataset_name"]


def test_dataset_is_think_filtered(rendered):
    """An inline <think>...</think> in an assistant turn breaks gold's completion
    extraction, silently. The reference and smoke datasets are *_nothink.jsonl."""
    gold = _target(rendered)["steps"][0]["config"]["gold_config"]
    assert "nothink" in gold["dataset_name"]


def test_numeric_and_boolean_parameters_keep_their_types(rendered):
    """Guards the parameters.yaml float-quoting trap.

    A float written unquoted in parameters.yaml is rendered by Jinja as its Python
    repr (1e-05), which PyYAML then re-reads as a string. Quoted ("1.0e-05") it is
    emitted verbatim and parses as a float.
    """
    target = _target(rendered)
    gold = target["steps"][0]["config"]["gold_config"]

    assert isinstance(gold["learning_rate"], float)
    assert isinstance(gold["min_lr"], float)
    assert isinstance(gold["lmbda"], float)
    assert isinstance(gold["max_length"], int)
    assert isinstance(gold["save_steps"], int)
    assert isinstance(gold["gradient_checkpointing"], bool)
    assert isinstance(gold["use_liger_fused_jsd"], bool)
    assert isinstance(target["steps"][0]["config"]["compute_config"]["num_nodes"], int)


def test_smoke_run_is_off_policy_and_not_liger(rendered):
    """The two settings whose wrong value corrupts a run rather than failing it:
    liger's fused bf16 JSD overflows on granite's logits_scaling=10 (NaN loss), and
    vllm_num_servers > 0 would dedicate nodes to serving instead of training."""
    gold = _target(rendered)["steps"][0]["config"]["gold_config"]
    assert gold["use_liger_fused_jsd"] is False
    assert gold["lmbda"] == 0.0
    assert gold["vllm_num_servers"] == 0


def test_effective_batch_invariant_is_recorded(rendered):
    """Effective batch = per_device x grad_accum x nodes x gpus_per_node, and the LR
    schedule was tuned against the reference 192. The smoke run deliberately does
    NOT hold 192 (it keeps the step count low), so this asserts the arithmetic is
    what the recipe claims rather than asserting the reference value."""
    target = _target(rendered)
    gold = target["steps"][0]["config"]["gold_config"]
    compute = target["steps"][0]["config"]["compute_config"]

    effective = (
        gold["per_device_train_batch_size"]
        * gold["gradient_accumulation_steps"]
        * compute["num_nodes"]
        * compute["num_gpus_per_node"]
    )
    assert effective == 16, "2 nodes x 8 GPUs x 1 x 1 for the smoke run"
