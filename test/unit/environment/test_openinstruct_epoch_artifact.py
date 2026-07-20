"""Unit tests for the open-instruct step's per-epoch checkpoint artifact.

The open-instruct log_monitor emits a NEWARTIFACT event for each HF-converted
checkpoint (…/epoch_hf_<k>, 0-indexed). This test drives the REAL log-line
parser (Environment.get_events_from_log_line) against the REAL step.yaml config
and asserts that each epoch gets a distinct, 1-indexed binding_id
(epoch_hf_0 -> epoch_1) while the path field still strips the trailing filename.
"""

import asyncio
import os
from pathlib import Path

import pytest
import yaml

from gbserver.environment.environment import (
    Environment,
    EventLogLineParserConfig,
)

# Resolve the assets checkout portably: env override wins, else a path relative
# to this repo's parent dir (granite.build/../assets), which is the known layout.
# test file: granite.build/test/unit/environment/test_*.py
#   parents[0]=environment [1]=unit [2]=test [3]=granite.build ; .parent = repos root
_DEFAULT_STEP = (
    Path(__file__).resolve().parents[3].parent
    / "assets"
    / "steps"
    / "open-instruct"
    / "step.yaml"
)
STEP = Path(os.environ.get("OPEN_INSTRUCT_STEP_YAML", _DEFAULT_STEP))

pytestmark = pytest.mark.skipif(
    not STEP.exists(),
    reason=f"open-instruct step.yaml not found at {STEP}; set OPEN_INSTRUCT_STEP_YAML",
)


class _CapturingMessenger:
    """Minimal MessagingBase stand-in.

    When a messenger is provided, get_events_from_log_line returns plain dicts
    shaped {"type", "event_id", "data": event_data} and publishes each one.
    """

    def __init__(self):
        self.published = []

    async def publish(self, payload, suffix):
        self.published.append(payload)


def _artifact_cfg() -> EventLogLineParserConfig:
    doc = yaml.safe_load(STEP.read_text())
    cfgs = doc["environment_configs"]["K8s"]["monitors"]["log_monitor"]["config"][
        "event_configs"
    ]
    art = [
        c
        for c in cfgs
        if str(c["event_type"]).lower() == "newartifact_in_environment_event"
    ]
    assert len(art) == 1, f"expected exactly one artifact event config, got {len(art)}"
    return EventLogLineParserConfig.model_validate(art[0])


def _parse(line: str) -> dict:
    cfg = _artifact_cfg()
    messenger = _CapturingMessenger()
    events = asyncio.run(
        Environment.get_events_from_log_line(
            line, [cfg], messenger=messenger
        )
    )
    assert len(events) == 1, f"expected exactly one event, got {len(events)}"
    return events[0]["data"]


def test_epoch0_maps_to_epoch_1_binding():
    line = (
        "tokenizer config file saved in "
        "/gb-read-write/outputs/open-instruct/hf/g4-350m-sft-abc/epoch_hf_0/"
        "tokenizer_config.json"
    )
    data = _parse(line)
    assert data["binding_id"] == "epoch_1"
    assert data["data"]["path"].endswith("/epoch_hf_0")


def test_epoch3_maps_to_epoch_4_binding():
    line = "tokenizer config file saved in /x/epoch_hf_3/tokenizer_config.json"
    data = _parse(line)
    assert data["binding_id"] == "epoch_4"
    assert data["data"]["path"].endswith("/epoch_hf_3")
