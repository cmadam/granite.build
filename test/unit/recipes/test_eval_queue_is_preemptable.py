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

"""Every recipe's evals default to BlueVela's preemptable queue.

Internal policy keeps evaluations off the `normal` queue by default, so any
parameters.yaml that declares EVAL_QUEUE must default it to preemptable. A launch can
still pass --param EVAL_QUEUE=normal; this guards the default only.
"""

import pathlib

import pytest
import yaml

_RECIPES = pathlib.Path(__file__).resolve().parents[3] / "recipes"
_PARAMS = sorted(
    p
    for p in _RECIPES.rglob("parameters.yaml")
    if "EVAL_QUEUE" in (yaml.safe_load(p.read_text(encoding="utf-8")) or {})
)


def test_some_recipe_declares_eval_queue():
    assert _PARAMS


@pytest.mark.parametrize("path", _PARAMS, ids=lambda p: str(p.relative_to(_RECIPES)))
def test_eval_queue_is_preemptable(path):
    params = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert params["EVAL_QUEUE"] == "preemptable"
