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

"""Unit tests for recipes/granite4-350m/lsf/distill-onpolicy-switch-study.

One recipe runs every training arm of the G4-350m on-policy study, each arm a set of
--param overrides. The parent (distill-onpolicy-ck2500-continue) has its own suite; this
file guards what the study adds:

* every arm in _ARMS -- the same overrides the README's launch commands use -- renders, and
  its trainer config passes the renderer's own checks, which otherwise run only once the
  allocation is held;
* each arm sets the lever it exists to vary, and only the rungs its ladder names exist;
* the eval protocol: screening evals on every rung, the full battery only on
  FULL_EVAL_RUNGS, and the per-rung CSVs gated on exactly the evals that rung ran;
* the generation floor travels with the trainer, not the server, and the trainer checkout
  is the one the patch header names.
"""

import argparse
import importlib.util
import pathlib
import re

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_REPO = pathlib.Path(__file__).resolve().parents[3]
_LSF = _REPO / "recipes" / "granite4-350m" / "lsf"
_RECIPE = _LSF / "distill-onpolicy-switch-study"
_GOLD_SRC = _REPO / "steps" / "distill" / "gold" / "skypilot"
_PATCH = _GOLD_SRC / "patches" / "onpolicy_study.diff"

_PLAIN_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_]+)\}")
_LOOP_VARS = {"N", "e", "k", "v", "name", "CATS", "SAGE_NAMES", "SCREEN", "FULL"}

_CK2000 = (
    "/proj/granite-build/g4os/distill/distill-350m-ckpt-eval-20260929/"
    "1e2e969e-0a26-46f2-be2b-74e351e6ca24/export-2000"
)
_SFT = (
    "/proj/granite-build/g4os/skypilot-test/sft/checkpoints/"
    "v0-20260529-20260529_212150-hf/epoch_hf_2"
)
_CK2500 = (
    "/proj/granite-build/g4os/distill/distill-350m-onpolicy-c2000_20261003_1119/"
    "68b7c4b2-7d39-449d-9f0b-fb448ac2f5ad/export-2500"
)
_C2000_SEED = "/proj/data-eng/llmb-read-write/resume-src/c2000-b6b92d72"
_FRONTLOAD = "/proj/granite-build/g4os/distill/corpora/safety-frontload-x5-w1000"

_SHORT_LADDER = {"CKPT_LADDER": "250,500,1000,1500,2000,2500", "GOLD_SAVE_STEPS": "250"}
_A5 = {"ADAM_BETA2": "0.95", "WARMUP_RATIO": "0.02", **_SHORT_LADDER}
_FRONT = {"CORPUS_DIR": _FRONTLOAD, "TRAIN_SAMPLING_STRATEGY": "sequential"}
_B = {"STUDENT_MODEL": _SFT, "GOLD_DATA_SEED": "42", "WARMUP_RATIO": "0.4", **_FRONT}
_D = {
    "STUDENT_MODEL": _CK2500,
    "GOLD_DATA_SEED": "44",
    "WARMUP_RATIO": "0.05",
    "LMBDA_INIT": "0.25",
    "MIN_COMPLETION_LENGTH": "0",
    "CKPT_LADDER": "500,1000,1500,2000,2500",
}

# Spelled as a command line spells them: the CLI delivers every --param as a string.
_ARMS = {
    "A0": {"LMBDA": "0"},
    "A1": {
        "RESUME_FROM_CHECKPOINT_DIR": _C2000_SEED,
        "RESUME_EMIT_SEEDED": "false",
        "GOLD_MAX_STEPS": "0",
        "STOP_AT_STEP": "4500",
        "GOLD_DATA_SEED": "-1",
        "GOLD_LEARNING_RATE": "2.0e-06",
        "WARMUP_RATIO": "0.05",
        "LMBDA_RAMP_START_STEP": "2000",
        "LMBDA_RAMP_STEPS": "2500",
        "CKPT_LADDER": "2250,2500,3000,3500,4000,4500",
        "GOLD_SAVE_STEPS": "250",
        "FULL_EVAL_RUNGS": "4500",
    },
    "A2": {},
    "A3": {"WARMUP_RATIO": "0.05", **_SHORT_LADDER},
    "A4": {"LMBDA": "0.25", "LMBDA_SCHEDULE": "constant"},
    "A5": dict(_A5),
    "A6": {**_A5, **_FRONT},
    "B0": {**_B, "LMBDA": "0", "GOLD_LEARNING_RATE": "2.0e-06"},
    "B1": {**_B, "LMBDA_RAMP_STEPS": "1250", "GOLD_LEARNING_RATE": "1.0e-06"},
    "B2": {**_B, "LMBDA_SCHEDULE": "constant", "GOLD_LEARNING_RATE": "1.0e-06"},
    "C1": {"LOSS_ARM": "distillm2_like"},
    "C2": {"BETA": "0.9"},
    "D1": {**_D, "LMBDA": "0.5", "LOSS_ARM": "distillm2_like"},
    "D2": {**_D, "LMBDA": "0.5", "BETA": "0.9"},
    "D3": {**_D, "LMBDA": "1.0"},
}


def _params(**overrides):
    params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    params.update(overrides)
    return params


def _render(tmp_path, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(
        apply_parameters(contents, [], _params(**overrides), str(tmp_path))
    )


def _targets(rendered):
    return rendered["granite.build"]["targets"]


def _gold(rendered):
    return _targets(rendered)["train-gold"]["steps"][0]["config"]["gold_config"]


def _renderer():
    spec = importlib.util.spec_from_file_location(
        "render_gold_config_under_test", _GOLD_SRC / "src" / "render_gold_config.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trainer_config(module, target) -> dict:
    """The step hands render_gold_config.py each gold_config key as its kebab-case flag;
    keys it has no flag for are the step's own and are dropped here as there."""
    config = target["steps"][0]["config"]
    flags = set(
        re.findall(
            r'"--([a-z0-9-]+)"',
            (_GOLD_SRC / "src" / "render_gold_config.py").read_text(),
        )
    )
    argv = [
        "--output",
        "/dev/null",
        "--total-nodes",
        str(config["compute_config"]["num_nodes"]),
    ]
    argv += ["--model-name-or-path", "/student", "--teacher-model-name-or-path"]
    argv += ["/teacher", "--dataset-name", "/corpus"]
    if "vllm" in target["inputs"]:
        argv += ["--vllm-server-url", "http://vllm-host:8001"]
    for key, value in config["gold_config"].items():
        if key.replace("_", "-") in flags:
            argv += [f"--{key.replace('_', '-')}", str(value)]
    return module.build_config(
        module._parse_args(argv)
    )  # pylint: disable=protected-access


def _catalog_names():
    return [e["name"] for e in _params()["EVAL_CATALOG"]]


@pytest.fixture(name="arm", params=sorted(_ARMS))
def fixture_arm(request):
    return request.param


# ─── The parameter surface ─────────────────────────────────────────────────────


def test_every_marker_has_a_parameter():
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    assert set(_PLAIN_MARKER.findall(contents)) - _LOOP_VARS - set(_params()) == set()


def test_every_parameter_is_referenced():
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    referenced = set(_PLAIN_MARKER.findall(contents)) | {
        "CKPT_LADDER",
        "CORPUS_DIR",
        "EVAL_CATEGORIES",
        "EVAL_CATALOG",
        "SAGE_BATCH_SIZE",
        "SAGE_MAX_LENGTH",
        "SCREEN_EVALS",
        "FULL_EVAL_RUNGS",
    }
    assert set(_params()) - referenced - {"INCLUDE_SFT"} == set()


def test_no_comment_uses_a_template_variable():
    """The CLI templates comments too, so a marker in a comment is evaluated."""
    for line in (_RECIPE / "build.yaml").read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            assert "$${" not in line, line


# ─── Every arm ─────────────────────────────────────────────────────────────────


def test_every_arm_passes_the_renderers_checks(arm, tmp_path):
    rendered = _render(tmp_path, **_ARMS[arm])
    config = _trainer_config(_renderer(), _targets(rendered)["train-gold"])
    assert float(config["lmbda"]) == pytest.approx(
        float(_params(**_ARMS[arm])["LMBDA"])
    )


def test_every_ladder_rung_is_a_checkpoint_the_trainer_writes(arm):
    p = _params(**_ARMS[arm])
    rungs = [int(n) for n in str(p["CKPT_LADDER"]).split(",")]
    assert all(n % int(p["GOLD_SAVE_STEPS"]) == 0 for n in rungs)
    assert rungs == sorted(rungs)
    last = int(p["STOP_AT_STEP"]) or int(p["GOLD_MAX_STEPS"])
    assert rungs[-1] == last
    # Every checkpoint the run writes is kept, so no bound export waits on an evicted rung.
    first = int(p["STOP_AT_STEP"]) and int(p["LMBDA_RAMP_START_STEP"])
    assert (last - first) // int(p["GOLD_SAVE_STEPS"]) + 1 <= int(
        p["GOLD_SAVE_TOTAL_LIMIT"]
    )
    for n in str(p["FULL_EVAL_RUNGS"]).split(","):
        assert int(n) in rungs


def test_the_server_pair_exists_exactly_when_the_student_generates(arm, tmp_path):
    targets = _targets(_render(tmp_path, **_ARMS[arm]))
    on_policy = float(_params(**_ARMS[arm])["LMBDA"]) > 0
    assert ("vllm-server" in targets) is on_policy
    assert ("vllm" in targets["train-gold"]["inputs"]) is on_policy


def test_the_floor_is_the_trainers_and_the_server_stays_at_zero(tmp_path):
    rendered = _render(tmp_path)
    assert _gold(rendered)["min_completion_length"] == 16
    server = _targets(rendered)["vllm-server"]["steps"][0]["config"]["vllm_config"]
    assert server["min_completion_length"] == 0
    config = _trainer_config(_renderer(), _targets(rendered)["train-gold"])
    assert config["min_completion_length"] == 16


# ─── The lever each arm varies ─────────────────────────────────────────────────


def _cfg(tmp_path, arm):
    rendered = _render(tmp_path, **_ARMS[arm])
    return _trainer_config(_renderer(), _targets(rendered)["train-gold"])


def test_the_default_is_a2(tmp_path):
    config = _cfg(tmp_path, "A2")
    assert config["model_name_or_path"] == "/student"
    assert _params()["STUDENT_MODEL"] == _CK2000
    assert config["lmbda_schedule"] == "linear"
    assert config["lmbda_init"] == 0.0 and config["lmbda"] == 1.0
    assert config["warmup_ratio"] == pytest.approx(0.4)
    assert config["learning_rate"] == pytest.approx(1.0e-06)
    assert config["data_seed"] == 43
    assert "adam_beta2" not in config and "train_sampling_strategy" not in config
    assert "lmbda_ramp_steps" not in config and "stop_at_step" not in config


def test_a0_is_off_policy_with_no_on_policy_keys(tmp_path):
    config = _cfg(tmp_path, "A0")
    assert config["lmbda"] == 0.0
    assert not {"lmbda_schedule", "min_completion_length", "vllm_num_servers"} & {
        k for k, v in config.items() if k != "vllm_num_servers" or v
    }


def test_a1_resumes_c2000_on_its_own_schedule(tmp_path):
    rendered = _render(tmp_path, **_ARMS["A1"])
    step = _targets(rendered)["train-gold"]["steps"][0]["config"]
    assert step["resume_from_checkpoint_dir"] == _C2000_SEED
    assert step["resume_emit_seeded"] is False
    config = _trainer_config(_renderer(), _targets(rendered)["train-gold"])
    # No max_steps: the epoch defines the 8,150-step schedule c2000 was built with.
    assert "max_steps" not in config and "data_seed" not in config
    assert config["stop_at_step"] == 4500
    assert config["learning_rate"] == pytest.approx(2.0e-06)
    assert config["warmup_ratio"] == pytest.approx(0.05)
    assert (config["lmbda_ramp_start_step"], config["lmbda_ramp_steps"]) == (2000, 2500)


def test_a4_holds_the_first_runs_lambda(tmp_path):
    config = _cfg(tmp_path, "A4")
    assert config["lmbda"] == pytest.approx(0.25)
    assert "lmbda_schedule" not in config


def test_a5_is_the_short_memory_optimizer(tmp_path):
    config = _cfg(tmp_path, "A5")
    assert config["adam_beta2"] == pytest.approx(0.95)
    assert config["warmup_ratio"] == pytest.approx(0.02)


def test_a6_is_a5_on_the_front_loaded_corpus_in_file_order(tmp_path):
    a5, a6 = _cfg(tmp_path / "a", "A5"), _cfg(tmp_path / "b", "A6")
    assert a6["train_sampling_strategy"] == "sequential"
    assert {k for k in a6 if a5.get(k) != a6[k]} == {"train_sampling_strategy"}
    rendered = _render(tmp_path, **_ARMS["A6"])
    corpus = _targets(rendered)["train-gold"]["inputs"]["corpus"]["uri"]
    assert corpus == f"env://{_FRONTLOAD}/train.jsonl"


def test_track_b_starts_from_sft_on_the_front_loaded_corpus(tmp_path):
    for arm in ("B0", "B1", "B2"):
        assert _params(**_ARMS[arm])["STUDENT_MODEL"] == _SFT
        assert _cfg(tmp_path / arm, arm).get("train_sampling_strategy", "") == (
            "sequential"
        )
    b1 = _cfg(tmp_path / "b1", "B1")
    assert b1["lmbda_schedule"] == "linear" and b1["lmbda_ramp_steps"] == 1250
    b2 = _cfg(tmp_path / "b2", "B2")
    assert b2["lmbda"] == 1.0 and "lmbda_schedule" not in b2


def test_track_c_changes_only_the_divergence(tmp_path):
    a2 = _cfg(tmp_path / "a2", "A2")
    c1, c2 = _cfg(tmp_path / "c1", "C1"), _cfg(tmp_path / "c2", "C2")
    assert {k for k in c1 if a2.get(k) != c1[k]} == {"use_distillm2_like"}
    assert {k for k in c2 if a2.get(k) != c2[k]} == {"beta"}


def test_track_d_continues_ck2500_with_the_existing_arms_settings(tmp_path):
    for arm in ("D1", "D2", "D3"):
        config = _cfg(tmp_path / arm, arm)
        assert config["data_seed"] == 44
        assert config["lmbda_init"] == pytest.approx(0.25)
        assert "min_completion_length" not in config
    assert _cfg(tmp_path / "d1", "D1")["use_distillm2_like"] is True


# ─── The evaluation protocol ───────────────────────────────────────────────────


def test_every_screening_eval_exists():
    names = set(_catalog_names()) | {"bfcl"}
    assert set(str(_params()["SCREEN_EVALS"]).split(",")) <= names


def _evals_by_rung(targets):
    by_rung = {}
    for name in targets:
        m = re.fullmatch(r"eval-(.+)-ck(\d+)", name)
        if m and m.group(1) != "transfer":
            by_rung.setdefault(m.group(2), set()).add(m.group(1))
    return by_rung


@pytest.mark.parametrize("arm", ["A2", "A3", "A1"])
def test_screening_on_every_rung_and_the_battery_on_the_full_rungs(arm, tmp_path):
    p = _params(**_ARMS[arm])
    targets = _targets(_render(tmp_path, **_ARMS[arm]))
    by_rung = _evals_by_rung(targets)
    screen = set(str(p["SCREEN_EVALS"]).split(","))
    full = set(str(p["FULL_EVAL_RUNGS"]).split(","))
    everything = set(_catalog_names()) | {"bfcl"}
    assert set(by_rung) == set(str(p["CKPT_LADDER"]).split(","))
    for rung, evals in by_rung.items():
        assert evals == (everything if rung in full else screen), rung


def test_each_rungs_csv_waits_for_exactly_the_evals_that_rung_ran(tmp_path):
    targets = _targets(_render(tmp_path, **_ARMS["A3"]))
    for rung, evals in _evals_by_rung(targets).items():
        gates = set(targets[f"export-sage-ck{rung}"]["inputs"])
        assert gates == {f"gate_{n.replace('-', '_')}" for n in evals - {"bfcl"}}, rung
    combined = set(targets["export-combined"]["inputs"])
    rungs = _evals_by_rung(targets)
    assert {g for g in combined if g.startswith("gate_bfcl_")} == {
        f"gate_bfcl_{r}" for r, e in rungs.items() if "bfcl" in e
    }


def test_screening_only_still_writes_the_roll_up(tmp_path):
    targets = _targets(_render(tmp_path, FULL_EVAL_RUNGS=""))
    assert "export-combined" in targets
    assert all(
        e == set(str(_params()["SCREEN_EVALS"]).split(","))
        for e in _evals_by_rung(targets).values()
    )


# ─── The trainer checkout ──────────────────────────────────────────────────────


@pytest.mark.parametrize("target", ["train-gold", "vllm-server"])
def test_every_target_reads_the_patched_trainer(target, tmp_path):
    code = _targets(_render(tmp_path))[target]["steps"][0]["config"]["code_config"]
    assert code["code_dir"] == _params()["CODE_DIR"]
    assert code["expect_ref"] == _params()["CODE_EXPECT_REF"]


def test_the_patch_states_the_commit_the_recipe_pins():
    header = _PATCH.read_text(encoding="utf-8").split("diff --git", 1)[0]
    assert _params()["CODE_EXPECT_REF"] in header
    assert "288bfc6bdab589eca394ca9f7229f1138a6da1c4" in header


def test_the_patch_adds_exactly_the_fields_the_renderer_gates():
    body = _PATCH.read_text(encoding="utf-8").split("diff --git", 1)[1]
    added = re.findall(r"^\+    ([a-z_]+): (?:int|float|bool|str) = field", body, re.M)
    assert set(added) == {
        "lmbda_ramp_start_step",
        "lmbda_ramp_steps",
        "min_completion_length",
        "stop_at_step",
    }
    assert (
        set(added) <= _renderer()._TRAINER_EXTENSION_KEYS
    )  # pylint: disable=protected-access


def test_two_arms_never_register_the_same_server_uri(tmp_path):
    uris = {
        _targets(_render(tmp_path / a, RUN_NAME=f"run-{a}", **_ARMS[a]))["vllm-server"][
            "outputs"
        ][o]["uri"]
        for a in ("A2", "A3")
        for o in ("vllm_url", "cluster_name")
    }
    assert len(uris) == 4
