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

"""Unit tests for the distill-onpolicy-v2 recipe.

This recipe has not run. What it guards against, therefore, is not regression but the
set of on-policy mistakes that are already documented as having happened once, every one
of which produces a run that COMPLETES and reports a plausible loss:

* serving the teacher instead of the student -- a different algorithm, not a crash;
* allocating a server and then generating locally, because --use_vllm defaulted False
  (build d77546a9, which logged not one request against an idle 8-GPU server);
* a server whose address never reaches the trainer, which is what the in-allocation
  split does: it passes no --vllm_server_host;
* an immediate-EOS collapse, where the teacher scores empty completions as rollouts;
* a stranded SERVICE allocation, because LSF clusters never autostop and teardown is
  gated on an artifact a crashed trainer never emits.

It also asserts the recipe is stage 1 v2 plus on-policy and nothing else, because the
question it exists to answer is where the sequences come from -- not what the objective
is, which stage 1 v2 already changed.
"""

import pathlib
import re

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_LSF = pathlib.Path(__file__).resolve().parents[3] / "recipes" / "granite4-350m" / "lsf"
_RECIPE = _LSF / "distill-onpolicy-v2"
_OFFPOLICY = _LSF / "distill-stage1-v2"

_PLAIN_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_]+)\}")
_LOOP_VARS = {"N"}


def _params(path=_RECIPE, **overrides):
    params = get_params_from_file(str(path / "parameters.yaml"))
    params.update(overrides)
    return params


def _render(tmp_path, **overrides):
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(
        apply_parameters(contents, [], _params(**overrides), str(tmp_path))
    )


def _targets(rendered):
    return rendered["granite.build"]["targets"]


def _config(rendered, target):
    return _targets(rendered)[target]["steps"][0]["config"]


@pytest.fixture(name="on")
def fixture_on(tmp_path):
    return _render(tmp_path, INCLUDE_SFT=False)


def test_every_marker_has_a_parameter():
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    assert (set(_PLAIN_MARKER.findall(contents)) - _LOOP_VARS) - set(_params()) == set()


def test_every_parameter_is_referenced():
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    referenced = set(_PLAIN_MARKER.findall(contents)) | {"CKPT_LADDER"}
    assert set(_params()) - referenced - {"INCLUDE_SFT"} == set()


def test_the_graph_has_a_server_a_trainer_and_a_teardown(on):
    names = list(_targets(on))
    assert names.index("vllm-server") < names.index("train-gold")
    assert "teardown" in names


class TestTheServerServesTheStudent:
    def test_the_served_model_is_a_binding_not_a_parameter(self, on):
        """The copy being served and the copy being trained have to be the same
        weights. A binding makes that structural; two parameters agreeing is a
        convention that survives until someone edits one of them."""
        cfg = _config(on, "vllm-server")["vllm_config"]
        assert cfg["model_path"] == "{{ bindings.student.binding.path }}"
        assert _targets(on)["vllm-server"]["inputs"]["student"]["binding"] == (
            "align.retagged_student"
        )

    def test_it_is_not_the_teacher(self, on):
        """The silent trap: on-policy has the STUDENT generate and the teacher score
        those generations. Serving the teacher is a different algorithm that runs to
        completion and reports a plausible loss."""
        cfg = _config(on, "vllm-server")["vllm_config"]
        assert _params()["TEACHER_MODEL"] not in str(cfg["model_path"])

    def test_the_server_window_covers_the_trainers_prompts(self, on):
        """A max_model_len below the trainer's max_length means a prompt the trainer
        accepts is one the server refuses, mid-run."""
        server = _config(on, "vllm-server")["vllm_config"]
        trainer = _config(on, "train-gold")["gold_config"]
        assert int(server["max_model_len"]) >= int(trainer["max_length"])

    def test_the_url_travels_over_mem_not_env(self, on):
        """env:// runs the payload through filesystem-path normalisation and mangles
        http://host:8001 into /http:/host:8001."""
        outputs = _targets(on)["vllm-server"]["outputs"]
        assert outputs["vllm_url"]["uri"].startswith("mem://")
        assert "type" not in outputs["vllm_url"]

    def test_the_trainer_gates_on_the_servers_health(self, on):
        """The binding resolves only once the server answers /health, so it is the
        ordering primitive as well as the address."""
        assert _targets(on)["train-gold"]["inputs"]["vllm"]["binding"] == (
            "vllm-server.vllm_url"
        )


class TestTheTrainerActuallyUsesTheServer:
    def test_the_external_url_is_wired(self, on):
        """Its presence is what selects the external-server path over the
        in-allocation split, which passes no --vllm_server_host at all."""
        gold = _config(on, "train-gold")["gold_config"]
        assert gold["vllm_server_url"] == "{{ bindings.vllm.binding.state }}"
        assert gold["vllm_mode"] == "server"

    def test_one_external_server_is_declared(self, on):
        """At 0 the renderer refuses the URL outright: the on-policy keys would not be
        emitted and the run would train off-policy while a server sat idle."""
        assert int(_config(on, "train-gold")["gold_config"]["vllm_num_servers"]) == 1

    def test_the_student_actually_generates(self, on):
        """lmbda 0 with a server allocated is the d77546a9 shape: a held allocation
        that is never asked for anything. The renderer refuses it."""
        assert float(_config(on, "train-gold")["gold_config"]["lmbda"]) > 0

    def test_weights_are_synced_every_step(self, on):
        """A stale served copy generates from a model that no longer exists, which is
        off-policy training paying on-policy's costs."""
        assert int(_config(on, "train-gold")["gold_config"]["vllm_sync_frequency"]) >= 1

    def test_the_lmbda_schedule_is_valid_and_paired(self, on):
        gold = _config(on, "train-gold")["gold_config"]
        assert gold["lmbda_schedule"] in ("constant", "linear")
        if gold["lmbda_schedule"] == "linear":
            assert 0.0 <= float(gold["lmbda_init"]) <= 1.0
            assert float(gold["lmbda_init"]) < float(gold["lmbda"]), "a ramp must rise"


class TestTheGenerationFloor:
    def test_it_is_set_on_both_sides(self, on):
        """Not redundancy. The trainer's copy bounds what it ASKS for; the server's is
        where the sampling decision is taken, and trl's vllm_serve has no min_tokens
        field, so run_vllm_serve.py patches SamplingParams from GOLD_MIN_TOKENS, which
        the step exports from its own key."""
        assert int(_config(on, "train-gold")["gold_config"]["min_completion_length"]) > 0
        assert int(_config(on, "vllm-server")["vllm_config"]["min_completion_length"]) > 0

    def test_the_two_sides_agree(self, on):
        assert int(_config(on, "train-gold")["gold_config"]["min_completion_length"]) == (
            int(_config(on, "vllm-server")["vllm_config"]["min_completion_length"])
        )

    def test_the_step_exports_it_to_the_server_process(self):
        """The recipe key is inert unless the step turns it into GOLD_MIN_TOKENS."""
        step = (
            _RECIPE.parents[3]
            / "configurations"
            / "assets"
            / "environments"
            / "skypilot"
            / "steps"
            / "vllm-server"
            / "step.yaml"
        ).read_text(encoding="utf-8")
        assert "GOLD_MIN_TOKENS" in step
        assert "min_completion_length" in step


class TestTheAllocationIsReleased:
    def test_teardown_exists_and_is_gated_on_training_finishing(self, on):
        """LSF SERVICE clusters never auto-stop and never get a terminal-status
        cleanup, so without this the server's GPUs are held until someone runs
        `sky down` by hand."""
        inputs = _targets(on)["teardown"]["inputs"]
        assert inputs["trained"]["binding"] == "train-gold.checkpoint"
        assert inputs["vllm_cluster"]["binding"] == "vllm-server.cluster_name"

    def test_teardown_passes_a_list(self, on):
        names = _config(on, "teardown")["teardown_config"]["cluster_names"]
        assert isinstance(names, list) and len(names) == 1

    def test_the_server_carries_its_own_ceiling(self, on):
        """Teardown is gated on an artifact a CRASHED trainer never emits, and
        gbserver has no on-failure semantics. Build d77546a9 stranded 8 H100s that
        way. This is the backstop, and it must outlast the run it guards."""
        cfg = _config(on, "vllm-server")["vllm_config"]
        lifetime = int(cfg["max_lifetime_seconds"])
        assert lifetime > int(cfg["health_timeout_seconds"])
        # ~4 s/it measured on this pair, so a 2,000-step run needs ~2.2 h plus startup.
        steps = int(_config(on, "train-gold")["gold_config"]["max_steps"])
        assert lifetime >= steps * 4, "the ceiling would kill the server mid-run"


class TestItIsStage1V2PlusOnPolicy:
    """One variable. Stage 1 v2 already changed the objective; this changes where the
    sequences come from, and nothing else — otherwise an on-policy result is not
    comparable to the off-policy arm it is supposed to be judged against."""

    @pytest.mark.parametrize(
        "key",
        [
            "CE_COEF",
            "LOG_STUDENT_ENTROPY",
            "ENTROPY_GUARD_DROP_FRAC",
            "ENTROPY_GUARD_BASELINE_STEPS",
            "ENTROPY_GUARD_PATIENCE",
            "BETA",
            "TEMPERATURE",
            "GOLD_MAX_STEPS",
            "CKPT_LADDER",
            "GOLD_SAVE_STEPS",
            "GOLD_SAVE_TOTAL_LIMIT",
            "MAX_LENGTH",
            "TARGET_ROWS",
            "GOLD_LEARNING_RATE",
            "KD_CODE_DIR",
            "KD_EXPECT_REF",
            "NCCL_DEBUG",
        ],
    )
    def test_it_matches_the_off_policy_arm(self, key):
        assert _params()[key] == _params(_OFFPOLICY)[key], key

    def test_the_anchor_is_still_on(self, on):
        """It matters more here, not less: on-policy training on a student's own
        output is the regime where a collapse feeds itself."""
        assert float(_config(on, "train-gold")["gold_config"]["ce_coef"]) > 0

    def test_the_effective_batch_is_unchanged_by_the_server_node(self):
        """The server has its own allocation, so GOLD_NUM_NODES still counts trainers
        only and the product must still be 96 — otherwise the on-policy arm is also a
        different optimization."""
        p = _params()
        product = (
            p["GOLD_PER_DEVICE_TRAIN_BATCH_SIZE"]
            * p["GOLD_GRADIENT_ACCUMULATION_STEPS"]
            * p["GOLD_NUM_NODES"]
            * p["GOLD_NUM_GPUS"]
        )
        assert product == 96


def test_the_student_default_cannot_silently_train_the_wrong_model():
    """On-policy continues from a good off-policy checkpoint. A plausible-looking
    default — the base retagged student, say — would run, report a loss, and answer a
    question nobody asked. An invalid one fails at align in minutes."""
    student = _params()["STUDENT_MODEL"]
    assert not student.startswith("/"), student
    assert "SET-ME" in student.upper()
    assert student != _params(_OFFPOLICY)["STUDENT_MODEL"]
