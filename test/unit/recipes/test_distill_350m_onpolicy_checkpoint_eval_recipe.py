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

"""Unit tests for the distill-onpolicy-checkpoint-eval recipe.

On-policy GOLD from c2000 -- the off-policy step-2,000 checkpoint of
distill-checkpoint-eval -- with that recipe's mid-run ladder and full-eval fanout, and
distill-onpolicy-v2's vLLM server and teardown. Its two parents' tests each guard half of
it; this file guards that the halves were joined without losing either, and the three
things that are new:

* the student is c2000 and the corpus is the one c2000 trained on, so a rung here is
  comparable to c2000's own row;
* the trainer's data sampler is reseeded, because at c2000's corpus, geometry and seed a
  fresh run draws exactly the batches c2000 already trained on;
* the server's lifetime covers an on-policy run of this length plus the queue wait, which
  distill-onpolicy-v2's 5 h ceiling does not.

And it asserts the one-variable claim: everything that defined c2000's run is unchanged
except where a quarter of the sequences come from.
"""

import json
import pathlib
import re
import subprocess
import textwrap

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_REPO = pathlib.Path(__file__).resolve().parents[3]
_LSF = _REPO / "recipes" / "granite4-350m" / "lsf"
_RECIPE = _LSF / "distill-onpolicy-checkpoint-eval"
_OFFPOLICY = _LSF / "distill-checkpoint-eval"
_ONPOLICY_V2 = _LSF / "distill-onpolicy-v2"
_GOLD_STEP = (
    _REPO
    / "configurations"
    / "assets"
    / "environments"
    / "skypilot"
    / "steps"
    / "distill"
    / "distill-gold"
    / "step.yaml"
)

_PLAIN_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_]+)\}")
_HEREDOC = re.compile(r"<<'PYSRC'\n(.*?)\n\s*PYSRC(?:\n|$)", re.S)
_LOOP_VARS = {"N", "e", "k", "v", "name", "CATS", "SAGE_NAMES"}

_LADDER = ["500", "1000", "1500", "2000", "2500"]
_C2000_BUILD = "1e2e969e-0a26-46f2-be2b-74e351e6ca24"
# What build 1e2e969e's corpus-pin-check accepted (corpus-pin/corpus_pin.json).
_C2000_CORPUS = (
    "/proj/granite-build/g4os/distill/distill-350m-stage1/"
    "df8512e0-be93-4762-85ad-813d9080d72f/corpus"
)
# Off-policy on this pair: 8,150 steps in 13h20 on 2 x 8 H100 (build df8512e0).
_OFFPOLICY_SECONDS_PER_STEP = (13 * 3600 + 20 * 60) / 8150


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


def _sage_names():
    cats = str(_params()["EVAL_CATEGORIES"]).replace(" ", "").lower().split(",")
    return [e["name"] for e in _params()["EVAL_CATALOG"] if e["category"] in cats]


@pytest.fixture(name="built")
def fixture_built(tmp_path):
    return _render(tmp_path)


# ─── The parameter surface ─────────────────────────────────────────────────────


def test_every_marker_has_a_parameter():
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    names = set(_PLAIN_MARKER.findall(contents)) - _LOOP_VARS
    assert names - set(_params()) == set()


def test_every_parameter_is_referenced():
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    # Reached through an expression rather than a plain marker -- see the off-policy
    # recipe's copy of this test for each one.
    referenced = set(_PLAIN_MARKER.findall(contents)) | {
        "CKPT_LADDER",
        "CORPUS_DIR",
        "EVAL_CATEGORIES",
        "EVAL_CATALOG",
        "SAGE_BATCH_SIZE",
        "SAGE_MAX_LENGTH",
    }
    assert set(_params()) - referenced - {"INCLUDE_SFT"} == set()


def test_no_comment_uses_a_template_variable_outside_its_scope():
    """The CLI templates comments too, so a marker in a comment is evaluated."""
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    for line in contents.splitlines():
        if line.lstrip().startswith("#"):
            assert "$${" not in line, line


def test_it_renders_in_every_switch_combination(tmp_path):
    for overrides in (
        {},
        {"CORPUS_DIR": ""},
        {"INCLUDE_SFT": True},
        {"EVAL_CATEGORIES": "bfcl"},
        {"EVAL_CATEGORIES": "math,general"},
        {"CKPT_LADDER": "2,4", "GOLD_MAX_STEPS": 4, "GOLD_SAVE_STEPS": 2},
        {"GOLD_DATA_SEED": -1},
    ):
        assert _targets(_render(tmp_path, **overrides)), overrides


def test_teardown_is_declared_exactly_once():
    """distill-onpolicy-v2 carries a second teardown INSIDE its CKPT_LADDER loop, which
    yaml.safe_load silently collapses (the last duplicate key wins). The rendered graph
    looks right and the source is wrong; this asserts the source."""
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    assert len(re.findall(r"^    teardown:$", contents, re.M)) == 1


# ─── Where it starts: c2000, on c2000's corpus ─────────────────────────────────


class TestItStartsFromC2000:
    def test_the_student_is_c2000s_export(self):
        """The export, not the raw checkpoint-2000: it is what full-eval measured, so
        the init is the model whose row this run is read against."""
        student = _params()["STUDENT_MODEL"]
        assert student.startswith("/proj/"), student
        assert _C2000_BUILD in student
        assert student.endswith("/export-2000")

    def test_it_is_not_the_off_policy_runs_own_init(self):
        assert _params()["STUDENT_MODEL"] != _params(_OFFPOLICY)["STUDENT_MODEL"]

    def test_align_server_and_trainer_all_see_that_student(self, built):
        """align is what the student path reaches; the server and the trainer both bind
        align's output, so one parameter decides what is served AND what is trained."""
        align = _config(built, "align")["align_config"]
        assert align["student_model"] == _params()["STUDENT_MODEL"]
        for target in ("vllm-server", "train-gold"):
            assert _targets(built)[target]["inputs"]["student"]["binding"] == (
                "align.retagged_student"
            ), target

    def test_the_corpus_is_pinned_to_the_one_c2000_trained_on(self, built):
        """Pinned by DEFAULT here. The held-out split has to be c2000's for
        eval-transfer to be comparable with c2000's eval-transfer-2000, and the train
        split has to be c2000's for GOLD_DATA_SEED to be reseeding the same corpus."""
        assert _params()["CORPUS_DIR"] == _C2000_CORPUS
        assert "sources" not in _targets(built)
        assert "corpus" not in _targets(built)
        inputs = _targets(built)["train-gold"]["inputs"]
        assert inputs["corpus"]["uri"] == f"env://{_C2000_CORPUS}/train.jsonl"
        assert inputs["corpus_pin"]["binding"] == "corpus-pin-check.pin_check"
        for rung in _LADDER:
            assert _config(built, f"eval-transfer-{rung}")["eval_config"]["corpus"] == (
                f"{_C2000_CORPUS}/eval.jsonl"
            )

    def test_the_server_is_not_gated_on_the_corpus(self, built):
        """vllm-server never reads the corpus; a corpus input would also have given it
        the pin check's ordering edge."""
        assert "corpus" not in (_targets(built)["vllm-server"].get("inputs") or {})


# ─── The data sampler is reseeded ──────────────────────────────────────────────


class TestTheDataIsNotReplayed:
    """c2000 trained on this pinned corpus at this geometry with the sampler seeded from
    HF Trainer's default `seed` 42. A fresh run at the same three draws the same
    permutation, so 2,000 of its 2,500 steps would be c2000's own batches again."""

    def test_the_sampler_seed_is_set_and_is_not_c2000s(self):
        seed = int(_params()["GOLD_DATA_SEED"])
        assert seed >= 0, "-1 omits data_seed and restores the replay"
        assert seed != 42
        assert seed != int(_params()["SEED"])

    def test_it_reaches_the_trainer_config(self, built):
        gold = _config(built, "train-gold")["gold_config"]
        assert int(gold["data_seed"]) == int(_params()["GOLD_DATA_SEED"])

    def test_the_published_step_passes_it_to_the_renderer(self):
        """The recipe key is inert unless the PUBLISHED step forwards it; a source-only
        change would render data_seed in the recipe and drop it on the cluster."""
        step = _GOLD_STEP.read_text(encoding="utf-8")
        assert re.search(
            r"--data-seed \{\{ config\.gold_config\.data_seed \| default\(-1\) \}\}",
            step,
        )

    def test_the_geometry_that_makes_the_replay_exact_is_c2000s(self):
        """The replay argument holds only because these match. If any one of them moves,
        the permutation moves with it -- which is also a different experiment."""
        for key in (
            "GOLD_PER_DEVICE_TRAIN_BATCH_SIZE",
            "GOLD_GRADIENT_ACCUMULATION_STEPS",
            "GOLD_NUM_NODES",
            "GOLD_NUM_GPUS",
            "SEED",
        ):
            assert _params()[key] == _params(_OFFPOLICY)[key], key


# ─── The ladder: derived, and every rung exists ────────────────────────────────


def test_the_run_is_bounded_by_steps_and_the_last_rung_is_that_bound():
    """With max_steps set the trainer's final step is exactly GOLD_MAX_STEPS, so unlike
    distill-checkpoint-eval the last rung is derived rather than measured."""
    params = _params()
    rungs = [int(n) for n in str(params["CKPT_LADDER"]).split(",")]
    assert int(params["GOLD_MAX_STEPS"]) == 2500
    assert rungs[-1] == int(params["GOLD_MAX_STEPS"])


def test_every_rung_is_a_multiple_of_the_save_grid():
    params = _params()
    save_steps = int(params["GOLD_SAVE_STEPS"])
    for rung in str(params["CKPT_LADDER"]).split(","):
        assert int(rung) % save_steps == 0, rung


def test_the_ladder_is_the_500_step_grid():
    assert str(_params()["CKPT_LADDER"]).split(",") == _LADDER


def test_no_checkpoint_is_evicted_before_its_export_reads_it():
    params = _params()
    assert int(params["GOLD_SAVE_TOTAL_LIMIT"]) >= len(_LADDER)


# ─── The mid-run fanout ────────────────────────────────────────────────────────


def test_every_export_is_gated_on_its_own_mid_run_checkpoint(built):
    """The load-bearing edge of distill-checkpoint-eval, and the one distill-onpolicy-v2
    does NOT have (its exports bind the final checkpoint). On-policy makes it matter
    more: rung 500's evals are the earliest read on a collapse, and they arrive while
    there is still a run to cancel."""
    for rung in _LADDER:
        binding = _targets(built)[f"export-{rung}"]["inputs"]["checkpoint_dir"][
            "binding"
        ]
        assert binding == f"train-gold.checkpoint_{rung}"
        cfg = _config(built, f"export-{rung}")["export_config"]
        assert cfg["checkpoint"] == "{{ bindings.checkpoint_dir.binding.path }}"


def test_train_gold_turns_the_watcher_on_and_declares_every_rung(built):
    cfg = _config(built, "train-gold")
    assert cfg["emit_checkpoint_artifacts"] is True
    assert int(cfg["checkpoint_watch_interval_seconds"]) > 0
    outputs = _targets(built)["train-gold"]["outputs"]
    assert "checkpoint" in outputs
    for rung in _LADDER:
        assert f"checkpoint_{rung}" in outputs


def test_the_watcher_detection_latency_is_bounded():
    params = _params()
    assert int(params["LOG_RETRIEVAL_INTERVAL_SECONDS"]) <= 300
    assert int(params["CKPT_WATCH_INTERVAL_SECONDS"]) <= 300


# ─── On-policy: the server serves the student, the trainer uses it ─────────────


class TestTheOnPolicyTopology:
    """distill-onpolicy-v2's guards, re-asserted on this graph, because every one of them
    is a run that COMPLETES and reports a plausible loss when it goes wrong."""

    def test_the_server_comes_before_the_trainer(self, built):
        names = list(_targets(built))
        assert names.index("vllm-server") < names.index("train-gold")

    def test_the_served_model_is_a_binding_and_not_the_teacher(self, built):
        cfg = _config(built, "vllm-server")["vllm_config"]
        assert cfg["model_path"] == "{{ bindings.student.binding.path }}"
        assert _params()["TEACHER_MODEL"] not in str(cfg["model_path"])

    def test_the_server_window_covers_the_trainers_prompts(self, built):
        server = _config(built, "vllm-server")["vllm_config"]
        trainer = _config(built, "train-gold")["gold_config"]
        assert int(server["max_model_len"]) >= int(trainer["max_length"])

    def test_the_url_travels_over_mem_not_env(self, built):
        outputs = _targets(built)["vllm-server"]["outputs"]
        assert outputs["vllm_url"]["uri"].startswith("mem://")
        assert outputs["cluster_name"]["uri"].startswith("mem://")
        assert "type" not in outputs["vllm_url"]

    def test_its_mem_names_are_its_own(self, built, tmp_path):
        """Two on-policy recipes in one space must not share a mem:// slot."""
        theirs = _targets(_render(tmp_path, _ONPOLICY_V2, INCLUDE_SFT=False))[
            "vllm-server"
        ]["outputs"]
        mine = _targets(built)["vllm-server"]["outputs"]
        for key in ("vllm_url", "cluster_name"):
            assert mine[key]["uri"] != theirs[key]["uri"], key

    def test_the_trainer_gates_on_the_servers_health_and_uses_it(self, built):
        assert _targets(built)["train-gold"]["inputs"]["vllm"]["binding"] == (
            "vllm-server.vllm_url"
        )
        gold = _config(built, "train-gold")["gold_config"]
        assert gold["vllm_server_url"] == "{{ bindings.vllm.binding.state }}"
        assert gold["vllm_mode"] == "server"
        assert int(gold["vllm_num_servers"]) == 1
        assert float(gold["lmbda"]) > 0
        assert int(gold["vllm_sync_frequency"]) >= 1

    def test_neither_side_asks_for_a_floor_the_pin_cannot_deliver(self, built):
        """0 is the state of the pinned code, not a preference: see distill-onpolicy-v2's
        TestTheGenerationFloor. The immediate-EOS collapse is therefore unmitigated and
        has to be read off rung 500."""
        assert int(_params()["MIN_COMPLETION_LENGTH"]) == 0
        assert (
            int(_config(built, "train-gold")["gold_config"]["min_completion_length"])
            == 0
        )
        assert (
            int(_config(built, "vllm-server")["vllm_config"]["min_completion_length"])
            == 0
        )

    def test_the_trainer_and_the_server_read_the_same_revision(self, built):
        overrides = {}
        for name, target in _targets(built).items():
            for step in target["steps"]:
                if "code_config" in step.get("config", {}):
                    overrides[name] = step["config"]["code_config"]
        assert set(overrides) == {"train-gold", "vllm-server"}, overrides
        assert len({tuple(sorted(c.items())) for c in overrides.values()}) == 1
        assert _params()["CODE_DIR"] == ""


class TestTheAllocationIsReleased:
    def test_teardown_is_gated_on_training_finishing(self, built):
        inputs = _targets(built)["teardown"]["inputs"]
        assert inputs["trained"]["binding"] == "train-gold.checkpoint"
        assert inputs["vllm_cluster"]["binding"] == "vllm-server.cluster_name"
        names = _config(built, "teardown")["teardown_config"]["cluster_names"]
        assert isinstance(names, list) and len(names) == 1

    def test_the_server_ceiling_outlasts_this_run(self, built):
        """distill-onpolicy-v2's 5 h assumed 2,000 steps at ~4 s/it. This run is 2,500
        steps, on-policy is budgeted at 2-3x the measured off-policy 5.9 s/step, and the
        clock starts before the trainer's 16-GPU allocation clears the queue. 3x is the
        floor asserted here; the parameter's comment says what the rest of the margin is
        for."""
        cfg = _config(built, "vllm-server")["vllm_config"]
        lifetime = int(cfg["max_lifetime_seconds"])
        steps = int(_config(built, "train-gold")["gold_config"]["max_steps"])
        assert lifetime > int(cfg["health_timeout_seconds"])
        assert lifetime >= steps * 3 * _OFFPOLICY_SECONDS_PER_STEP
        assert lifetime > int(_params(_ONPOLICY_V2)["VLLM_MAX_LIFETIME_SECONDS"])


# ─── One variable: everything else is c2000's run ──────────────────────────────


class TestOneVariable:
    """A rung here is read against c2000's row and against the off-policy rungs after
    it. That comparison means something only if the objective, the data, the geometry,
    the schedule and the measurement are all the off-policy run's."""

    @pytest.mark.parametrize(
        "key",
        [
            "TEACHER_MODEL",
            "CHAT_TEMPLATE",
            "REQUIRE_CHATML",
            "RESPONSE_TEMPLATE",
            "THINK_POLICY",
            "DOCUMENTS_POLICY",
            "EVAL_FRACTION",
            "MAX_LENGTH",
            "TARGET_ROWS",
            "SHUFFLE_SEED",
            "MAX_COMPLETION_LENGTH",
            "TEMPERATURE",
            "BETA",
            "CE_COEF",
            "LOG_STUDENT_ENTROPY",
            "ENTROPY_GUARD_DROP_FRAC",
            "ENTROPY_GUARD_ACTION",
            "USE_LIGER_FUSED_JSD",
            "GOLD_LEARNING_RATE",
            "GOLD_MIN_LR",
            "WARMUP_RATIO",
            "LR_SCHEDULER_TYPE",
            "CODE_DIR",
            "CODE_EXPECT_REF",
            "DS_CONFIG",
            "IMAGE_ID",
            "EVAL_METRICS",
            "EVAL_MAX_SAMPLES",
            "EVAL_CATEGORIES",
            "EVAL_CATALOG",
            "SAGE_BATCH_SIZE",
            "SAGE_MAX_LENGTH",
            "EXPORT_STACK",
            "BFCL_TEST_CATEGORIES",
            "BFCL_MODEL_ID",
            "CHAT_TEMPLATE_THINKING",
        ],
    )
    def test_it_matches_c2000s_run(self, key):
        assert _params()[key] == _params(_OFFPOLICY)[key], key

    @pytest.mark.parametrize(
        "key",
        [
            "LMBDA",
            "LMBDA_SCHEDULE",
            "LMBDA_INIT",
            "MIN_COMPLETION_LENGTH",
            "VLLM_MODE",
            "VLLM_SYNC_FREQUENCY",
            "VLLM_PORT",
            "VLLM_NUM_GPUS",
            "VLLM_HEALTH_TIMEOUT_SECONDS",
            "TOP_P",
            "USE_SAMPLED_OPD_LOSS",
            "LAST_MESSAGE_ONLY",
            "CLIP_ALPHA",
            "OPD_IMPORTANCE_SAMPLING",
        ],
    )
    def test_the_on_policy_settings_are_distill_onpolicy_v2s(self, key):
        assert _params()[key] == _params(_ONPOLICY_V2)[key], key

    def test_the_lmbda_is_a_constant_quarter(self, built):
        """The constant arm first: a ramp confounds "on-policy helped" with "on-policy
        helped once the student was further along"."""
        gold = _config(built, "train-gold")["gold_config"]
        assert float(gold["lmbda"]) == pytest.approx(0.25)
        assert gold["lmbda_schedule"] == "constant"

    def test_the_effective_batch_is_96(self):
        p = _params()
        assert (
            p["GOLD_PER_DEVICE_TRAIN_BATCH_SIZE"]
            * p["GOLD_GRADIENT_ACCUMULATION_STEPS"]
            * p["GOLD_NUM_NODES"]
            * p["GOLD_NUM_GPUS"]
        ) == 96

    def test_the_front_end_is_the_off_policy_recipes(self, built, tmp_path):
        """align and corpus-pin-check are distill-checkpoint-eval's, verbatim, once both
        are pointed at the same corpus and the per-run names are normalised away."""
        theirs = _render(tmp_path, _OFFPOLICY, CORPUS_DIR=_C2000_CORPUS)
        mine_p, theirs_p = _params(), _params(_OFFPOLICY)

        def normalise(config, params):
            text = yaml.safe_dump(config)
            for key in ("RUN_NAME", "STUDENT_MODEL"):
                text = text.replace(str(params[key]), key)
            return text

        for name in ("align", "corpus-pin-check"):
            assert normalise(_config(built, name), mine_p) == normalise(
                _config(theirs, name), theirs_p
            ), name

    def test_the_trainer_differs_from_c2000s_only_where_it_should(
        self, built, tmp_path
    ):
        """The whole one-variable claim, on the rendered trainer config: every gold_config
        key the off-policy run set is set identically here, except the horizon, the save
        grid and the on-policy keys."""
        theirs = _config(
            _render(tmp_path, _OFFPOLICY, CORPUS_DIR=_C2000_CORPUS), "train-gold"
        )["gold_config"]
        mine = _config(built, "train-gold")["gold_config"]
        expected_to_differ = {
            "run_name",
            "max_steps",
            "save_steps",
            "save_total_limit",
            "lmbda",
            "vllm_num_servers",
        }
        for key, value in theirs.items():
            if key in expected_to_differ:
                continue
            assert mine[key] == value, key
        added = set(mine) - set(theirs)
        assert added == {
            "data_seed",
            "vllm_server_url",
            "vllm_mode",
            "vllm_sync_frequency",
            "top_p",
            "use_sampled_opd_loss",
            "last_message_only",
            "clip_alpha",
            "opd_importance_sampling",
            "lmbda_schedule",
            "lmbda_init",
            "min_completion_length",
        }, added


# ─── Full-eval and the roll-up ─────────────────────────────────────────────────


def test_every_rung_is_crossed_with_every_selected_eval(built):
    names = _sage_names()
    assert len(names) == 26
    for rung in _LADDER:
        for name in names + ["bfcl"]:
            target = _targets(built)[f"eval-{name}-ck{rung}"]
            assert target["inputs"]["model"]["binding"] == f"export-{rung}.hf_model"


def test_its_results_cannot_land_on_the_off_policy_runs(built):
    """Both recipes write into the shared sage and bfcl trees, keyed by EXPERIMENT and
    rung. ckpt_2000 here is c2000 + 2,000 on-policy steps; written under the off-policy
    run's experiment name it would overwrite c2000's own results."""
    mine = _params()["EXPERIMENT"]
    assert mine != _params(_OFFPOLICY)["EXPERIMENT"]
    assert not mine.startswith("distill-350m-ckpt-eval")
    assert _params()["RUN_NAME"] != _params(_OFFPOLICY)["RUN_NAME"]
    for rung in _LADDER:
        cfg = _config(built, f"eval-olmes-gsm8k-ck{rung}")["sage_eval_config"]
        assert cfg["experiment"] == f"{mine}-ckpt_{rung}"
        assert "/" not in cfg["experiment"]


def test_the_bcb_url_reaches_the_only_eval_that_needs_it(built):
    url = _params()["OE_EVAL_BCB_API_URL"]
    for rung in _LADDER:
        env = _config(built, f"eval-olmes-bigcodebench-ck{rung}")["sage_eval_config"][
            "extra_env"
        ]
        assert env["OE_EVAL_BCB_API_URL"] == url


def test_every_per_rung_csv_name_carries_a_numeric_step(built):
    for rung in _LADDER:
        for target, output in (
            (f"export-sage-ck{rung}", "sage_export_csv"),
            (f"export-bfcl-ck{rung}", "bfcl_export_csv"),
        ):
            uri = _targets(built)[target]["outputs"][output]["uri"]
            found = re.search(r"ckpt_(\d+)-", uri)
            assert found and found.group(1) == rung, uri


def test_the_roll_up_waits_for_every_rung_of_both_eval_kinds(built):
    gates = _targets(built)["export-combined"]["inputs"]
    bindings = {g["binding"] for g in gates.values()}
    assert len(gates) == 2 * len(_LADDER)
    for rung in _LADDER:
        assert f"export-sage-ck{rung}.sage_export_csv" in bindings
        assert f"export-bfcl-ck{rung}.bfcl_export_csv" in bindings


def test_evals_ride_the_preemptable_queue_and_training_does_not(built):
    params = _params()
    assert params["EVAL_QUEUE"] == "preemptable"
    assert params["QUEUE"] == "normal"
    for target in ("train-gold", "vllm-server"):
        assert _config(built, target)["launcher_config"]["resources"]["zone"] == (
            "normal"
        ), target


# ─── The instruments do not act ────────────────────────────────────────────────


def test_gen_smoke_reports_without_gating(built):
    command = _config(built, "gen-smoke")["command_config"]["command"]
    body = textwrap.dedent(_HEREDOC.findall(command)[0])
    assert "sys.exit" not in body
    assert "degenerate" in body


def test_gen_smoke_hands_the_interpreter_every_rung(built, tmp_path):
    """Executes the assembly rather than parsing it: a blank line after a `\\` ends the
    command and bash runs the next rung as a program name (build bb779f1f)."""
    rendered = _render(tmp_path, WORKDIR_ROOT=str(tmp_path))
    cmd = _config(rendered, "gen-smoke")["command_config"]["command"]
    argv_log = tmp_path / "argv.json"
    stub = tmp_path / "python-stub"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "sys.stdin.read()\n"
        f"json.dump(sys.argv[1:], open({str(argv_log)!r}, 'w'))\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    script = cmd.replace("/stage/.venv/bin/python", str(stub))
    result = subprocess.run(
        ["bash"], input=script, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(argv_log.read_text(encoding="utf-8"))[1:]
    assert [a.partition(":")[0] for a in argv[3:]] == _LADDER
    for rung, got in zip(_LADDER, argv[3:]):
        assert got.endswith(f"/export-{rung}"), got


def test_nothing_downstream_is_gated_on_an_instrument(built):
    instruments = {f"eval-transfer-{rung}" for rung in _LADDER} | {"gen-smoke"}
    for name, target in _targets(built).items():
        for key, spec in (target.get("inputs") or {}).items():
            producer = spec.get("binding", "").split(".")[0]
            assert producer not in instruments, (name, key)


def test_there_is_no_baseline_eval_target(built):
    """c2000's row is the baseline and it is already measured, in build 1e2e969e."""
    assert not [t for t in _targets(built) if t.endswith("-baseline")]


def test_the_anchor_and_the_guard_are_off(built):
    gold = _config(built, "train-gold")["gold_config"]
    assert float(gold["ce_coef"]) == 0.0
    assert gold["log_student_entropy"] is False
    assert float(gold["entropy_guard_drop_frac"]) == 0.0


# ─── The graph as a whole ──────────────────────────────────────────────────────


def test_the_target_count_is_what_the_readme_says(built):
    # align, corpus-pin-check, vllm-server, train-gold, teardown, gen-smoke and
    # export-combined, plus per rung: export, eval-transfer, 26 sage, bfcl and two CSVs.
    assert len(_targets(built)) == 7 + 31 * len(_LADDER) == 162


def test_retries_reuse_what_already_succeeded(built):
    assert built["granite.build"]["retries"]["target_reuse_enabled"] is True
