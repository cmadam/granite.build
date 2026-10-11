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

"""Unit tests for recipes/granite4-350m/lsf/distill-safety-frontload-corpus.

The two command targets are run here, for real, on a small synthetic corpus: the build
renders with test paths, each command's script is executed with this interpreter, and the
corpus it writes is checked against what distill-onpolicy-switch-study and the trainer
need from it -- the window's safety share, the base rows all present once, and a manifest
whose row count is the file's (the trainer raises ManifestDrift otherwise).
"""

import json
import pathlib
import subprocess
import sys

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_REPO = pathlib.Path(__file__).resolve().parents[3]
_RECIPE = (
    _REPO / "recipes" / "granite4-350m" / "lsf" / "distill-safety-frontload-corpus"
)

_BASE_ROWS = 2000
_SAFETY_ROWS = 40


def _render(tmp_path, **overrides):
    params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    params.update(overrides)
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(apply_parameters(contents, [], params, str(tmp_path)))


def _command(rendered, target, bindings=None):
    text = rendered["granite.build"]["targets"][target]["steps"][0]["config"][
        "command_config"
    ]["command"]
    for name, path in (bindings or {}).items():
        text = text.replace(f"{{{{ bindings.{name}.binding.path }}}}", str(path))
    return text.replace("/stage/.venv/bin/python", sys.executable)


def _run(command, cwd):
    return subprocess.run(
        ["bash", "-c", command], cwd=cwd, capture_output=True, text=True, check=False
    )


@pytest.fixture(name="world")
def fixture_world(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    with open(base / "train.jsonl", "w", encoding="utf-8") as f:
        for i in range(_BASE_ROWS):
            f.write(
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": f"b{i}"}],
                        "row_id": f"b{i}",
                    }
                )
                + "\n"
            )
    (base / "eval.jsonl").write_text('{"messages": [], "row_id": "e0"}\n')
    (base / "corpus_manifest.json").write_text(
        json.dumps(
            {
                "tokenizer_identity": "granite-4.1-3b-pinned",
                "policies": {"max_length": 8192},
                "eval_fraction": 0.005,
                "splits": {
                    "train": {
                        "path": str(base / "train.jsonl"),
                        "examples": _BASE_ROWS,
                    },
                    "eval": {"path": str(base / "eval.jsonl"), "examples": 1},
                },
                "rows": {
                    "training_rows": _BASE_ROWS,
                    "row_id_scheme": "blake2b-128/canonical-json",
                },
            }
        )
    )
    source = tmp_path / "general.jsonl"
    with open(source, "w", encoding="utf-8") as f:
        for i in range(300):
            domain = "safety" if i % 7 == 0 else "code"
            f.write(
                json.dumps(
                    {
                        "uuid": f"u{i}",
                        "domain": domain,
                        "source": "s",
                        "conversations": [{"role": "user", "content": str(i)}],
                    }
                )
                + "\n"
            )
    # What corpus-prep would have written from the filtered rows.
    prep = tmp_path / "out" / "safety-prep"
    prep.mkdir(parents=True)
    with open(prep / "train.jsonl", "w", encoding="utf-8") as f:
        for i in range(_SAFETY_ROWS):
            f.write(
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": f"s{i}"}],
                        "row_id": f"s{i}",
                    }
                )
                + "\n"
            )
    (tmp_path / "render").mkdir()
    rendered = _render(
        tmp_path / "render",
        BASE_CORPUS_DIR=str(base),
        SAFETY_SOURCE=str(source),
        OUTPUT_DIR=str(tmp_path / "out"),
        FRONT_STEPS="10",
        EFFECTIVE_BATCH="50",
        SAFETY_WINDOW_FRACTION="0.1",
    )
    return {
        "tmp": tmp_path,
        "base": base,
        "out": tmp_path / "out",
        "prep": prep,
        "rendered": rendered,
    }


def test_safety_source_keeps_only_the_domain_and_spells_messages(world):
    result = _run(_command(world["rendered"], "safety-source"), world["tmp"])
    assert result.returncode == 0, result.stderr
    rows = [
        json.loads(l) for l in open(world["out"] / "safety-source" / "safety.jsonl")
    ]
    assert len(rows) == len(range(0, 300, 7))
    assert all("messages" in r and "conversations" not in r for r in rows)
    assert "GB_ARTIFACT_ID:safety_source" in result.stdout


def test_frontload_puts_the_share_in_the_window_and_keeps_every_base_row(world):
    command = _command(
        world["rendered"], "frontload", {"safety_corpus": world["prep"] / "train.jsonl"}
    )
    result = _run(command, world["tmp"])
    assert result.returncode == 0, result.stderr
    rows = [json.loads(l) for l in open(world["out"] / "train.jsonl")]
    window = 10 * 50
    head = [r["row_id"] for r in rows[:window]]
    tail = [r["row_id"] for r in rows[window:]]
    assert sum(1 for r in head if r.startswith("s")) == 50
    assert not [r for r in tail if r.startswith("s")]
    base_ids = [r["row_id"] for r in rows if r["row_id"].startswith("b")]
    assert sorted(base_ids) == sorted(f"b{i}" for i in range(_BASE_ROWS))
    # Shuffled: the window is a sample of the corpus, not its first rows in prep order.
    assert [r for r in head if r.startswith("b")][:20] != [f"b{i}" for i in range(20)]

    manifest = json.loads((world["out"] / "corpus_manifest.json").read_text())
    assert manifest["rows"]["training_rows"] == len(rows) == _BASE_ROWS + 50
    assert manifest["splits"]["train"]["examples"] == len(rows)
    assert manifest["eval_fraction"] == 0.005
    assert manifest["tokenizer_identity"] == "granite-4.1-3b-pinned"
    assert manifest["frontload"]["safety_rows_in_window"] == 50
    assert manifest["frontload"]["requires"] == "train_sampling_strategy sequential"
    assert (world["out"] / "eval.jsonl").read_text() == (
        world["base"] / "eval.jsonl"
    ).read_text()
    origins = [
        json.loads(l)["origin"] for l in open(world["out"] / "corpus_rows.jsonl")
    ]
    assert origins.count("safety") == 50


def test_frontload_is_reproducible_from_its_seed(world):
    command = _command(
        world["rendered"], "frontload", {"safety_corpus": world["prep"] / "train.jsonl"}
    )
    assert _run(command, world["tmp"]).returncode == 0
    first = (world["out"] / "train.jsonl").read_text()
    assert _run(command, world["tmp"]).returncode == 0
    assert (world["out"] / "train.jsonl").read_text() == first


def test_the_default_output_names_its_settings():
    params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    assert "w1000" in params["OUTPUT_DIR"] and int(params["FRONT_STEPS"]) == 1000
    # The training recipe's effective batch: 2 x 3 x 2 nodes x 8 GPUs.
    assert int(params["EFFECTIVE_BATCH"]) == 96
