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

"""Unit tests for recipes/granite4-350m/lsf/distill-model-screening-eval.

Rendered against distill-onpolicy-switch-study's parameters.yaml, as the README launches
it. Guards the target graph (one interp target per ratio, the selected evals per model,
CSVs and the roll-up gated on exactly those) and, where torch is importable (set
GB_TEST_TORCH_PYTHON to an interpreter that has it), runs the interpolation script itself.
"""

import json
import os
import pathlib
import re
import subprocess

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_REPO = pathlib.Path(__file__).resolve().parents[3]
_LSF = _REPO / "recipes" / "granite4-350m" / "lsf"
_RECIPE = _LSF / "distill-model-screening-eval"
_PARAMS = _LSF / "distill-onpolicy-switch-study" / "parameters.yaml"


def _params(**overrides):
    params = get_params_from_file(str(_PARAMS))
    params.update(overrides)
    return params


def _render(tmp_path, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(
        apply_parameters(contents, [], _params(**overrides), str(tmp_path))
    )["granite.build"]["targets"]


def _screen():
    return set(str(_params()["SCREEN_EVALS"]).split(","))


def _evals(targets, model_id):
    return {
        m.group(1)
        for name in targets
        if (m := re.fullmatch(rf"eval-(.+)-m{model_id}", name))
    }


def test_the_default_interpolates_three_ratios_and_screens_each(tmp_path):
    targets = _render(tmp_path, EXPERIMENT="x")
    assert {t for t in targets if t.startswith("interp-")} == {
        "interp-30",
        "interp-50",
        "interp-70",
    }
    for model_id in ("30", "50", "70"):
        assert _evals(targets, model_id) == _screen()
        assert targets[f"eval-salad-bench-m{model_id}"]["inputs"]["model"] == {
            "binding": f"interp-{model_id}.hf_model"
        }
        gates = set(targets[f"export-sage-m{model_id}"]["inputs"])
        assert gates == {f"gate_{n.replace('-', '_')}" for n in _screen() - {"bfcl"}}
    assert set(targets["export-combined"]["inputs"]) == {
        f"gate_{k}_{i}" for k in ("sage", "bfcl") for i in ("30", "50", "70")
    }


def test_a_plain_model_is_evaluated_from_its_path(tmp_path):
    targets = _render(
        tmp_path,
        EXPERIMENT="x",
        INTERP_RATIOS="",
        EVAL_MODELS="1=/m/ck2000;2=/m/teacher",
    )
    assert not [t for t in targets if t.startswith("interp-")]
    assert targets["eval-attaq-m2"]["inputs"]["model"] == {
        "uri": "env:///m/teacher",
        "type": "model",
    }
    assert targets["eval-bfcl-m1"]["inputs"]["model"]["uri"] == "env:///m/ck2000"
    sage = targets["eval-olmes-ifeval-m1"]["steps"][0]["config"]["sage_eval_config"]
    assert sage["experiment"] == "x-ckpt_1"


def test_evals_selects_the_battery(tmp_path):
    targets = _render(
        tmp_path, EXPERIMENT="x", INTERP_RATIOS="0.5", EVALS="salad-bench,attaq"
    )
    assert _evals(targets, "50") == {"salad-bench", "attaq"}
    assert "export-bfcl-m50" not in targets
    assert set(targets["export-combined"]["inputs"]) == {"gate_sage_50"}


def test_no_comment_uses_a_template_variable():
    for line in (_RECIPE / "build.yaml").read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            assert "$${" not in line, line


_TORCH_PY = os.environ.get("GB_TEST_TORCH_PYTHON", "")


@pytest.mark.skipif(
    not _TORCH_PY, reason="set GB_TEST_TORCH_PYTHON to a python with torch"
)
def test_the_interpolation_script_mixes_tensor_by_tensor(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    setup = f"""
import torch, pathlib
from safetensors.torch import save_file
for d, v in (("{a}", 1.0), ("{b}", 3.0)):
    p = pathlib.Path(d); p.mkdir()
    save_file({{"w": torch.full((2, 2), v, dtype=torch.bfloat16)}}, str(p / "model.safetensors"))
    (p / "tokenizer.json").write_text("{{}}"); (p / "config.json").write_text('{{"a": "%s"}}' % d)
"""
    subprocess.run([_TORCH_PY, "-c", setup], check=True)
    targets = _render(
        tmp_path / "r",
        EXPERIMENT="x",
        INTERP_RATIOS="0.25",
        INTERP_MODEL_A=str(a),
        INTERP_MODEL_B=str(b),
        INTERP_OUTPUT_DIR=str(tmp_path / "out"),
    )
    command = targets["interp-25"]["steps"][0]["config"]["command_config"]["command"]
    result = subprocess.run(
        ["bash", "-c", command.replace("/stage/.venv/bin/python", _TORCH_PY)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    dest = tmp_path / "out" / "x" / "interp-25"
    assert f"GB_ARTIFACT_PATH:{dest}" in result.stdout
    check = f"""
from safetensors.torch import load_file
t = load_file("{dest}/model.safetensors")["w"]
print(t.dtype, float(t[0, 0]))
"""
    out = subprocess.run(
        [_TORCH_PY, "-c", check], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["torch.bfloat16", "2.5"]  # 0.25 * 1 + 0.75 * 3
    assert json.loads((dest / "config.json").read_text())["a"] == str(a)


@pytest.mark.skipif(
    not _TORCH_PY, reason="set GB_TEST_TORCH_PYTHON to a python with torch"
)
def test_the_interpolation_script_refuses_different_tokenizers(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d, tok in ((a, "{}"), (b, '{"x": 1}')):
        d.mkdir()
        (d / "tokenizer.json").write_text(tok)
    targets = _render(
        tmp_path / "r",
        EXPERIMENT="x",
        INTERP_RATIOS="0.5",
        INTERP_MODEL_A=str(a),
        INTERP_MODEL_B=str(b),
        INTERP_OUTPUT_DIR=str(tmp_path / "out"),
    )
    command = targets["interp-50"]["steps"][0]["config"]["command_config"]["command"]
    result = subprocess.run(
        ["bash", "-c", command.replace("/stage/.venv/bin/python", _TORCH_PY)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "tokenizer.json differs" in result.stderr
