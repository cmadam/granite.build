"""Tests for the gold config renderer (../src/render_gold_config.py).

Run from the step directory with `make test`.

These pin the trainer requirements that fail *silently* — a run that starts, then
either crashes hours in or trains on the wrong objective. Each of the three rules
below cost real debugging time on the ansible path before it was understood, which
is why the renderer exists as a testable script rather than a shell heredoc.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_RENDER = Path(__file__).resolve().parent.parent / "src" / "render_gold_config.py"

_REQUIRED = {
    "--model-name-or-path": "/proj/kd/student_overlays/granite-4.1-3b-base-hub",
    "--teacher-model-name-or-path": "/proj/kd/teacher_overlays/granite-4.2-30b",
    "--dataset-name": "/proj/kd/data/subsampled_0.4_shuffled_nothink.jsonl",
}


def _render(tmp_path, total_nodes=2, extra=None, expect_rc=0):
    """Invoke the renderer the way the step's run block does; return the config."""
    out = tmp_path / "gold_config.yaml"
    cmd = [
        sys.executable,
        str(_RENDER),
        "--output",
        str(out),
        "--total-nodes",
        str(total_nodes),
    ]
    for flag, value in _REQUIRED.items():
        cmd += [flag, value]
    cmd += list(extra or [])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert (
        result.returncode == expect_rc
    ), f"rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    if expect_rc != 0:
        return result
    return yaml.safe_load(out.read_text())


class TestLearningRateIsAFloat:
    """The most expensive failure the renderer prevents.

    PyYAML parses a bare ``1e-05`` as a *string*. The trainer's min_lr handling
    then raises a str/float TypeError — not at startup, but once the scheduler is
    first consulted, so a multi-node job burns its queue time before failing.
    """

    def test_learning_rate_round_trips_as_a_float(self, tmp_path):
        config = _render(tmp_path, extra=["--learning-rate", "1e-05"])
        assert isinstance(config["learning_rate"], float)

    def test_min_lr_round_trips_as_a_float(self, tmp_path):
        config = _render(tmp_path, extra=["--min-lr", "1e-06"])
        assert isinstance(config["min_lr"], float)

    def test_rendered_text_carries_a_decimal_exponent(self, tmp_path):
        """Matches the validated reference configs (1.00e-05), not 1e-05."""
        out = tmp_path / "gold_config.yaml"
        cmd = [
            sys.executable,
            str(_RENDER),
            "--output",
            str(out),
            "--total-nodes",
            "2",
            "--learning-rate",
            "1e-05",
        ]
        for flag, value in _REQUIRED.items():
            cmd += [flag, value]
        subprocess.run(cmd, check=True, capture_output=True)
        assert "1.0e-05" in out.read_text()

    @pytest.mark.parametrize("value", ["1e-05", "1.0e-05", "0.00001"])
    def test_every_input_form_yields_the_same_float(self, tmp_path, value):
        config = _render(tmp_path, extra=["--learning-rate", value])
        assert config["learning_rate"] == pytest.approx(1e-05)


class TestBooleansAreLowerCaseYaml:
    """A Python ``True`` in YAML is a string, not a boolean."""

    def test_gradient_checkpointing_emits_yaml_true(self, tmp_path):
        out = tmp_path / "gold_config.yaml"
        cmd = [
            sys.executable,
            str(_RENDER),
            "--output",
            str(out),
            "--total-nodes",
            "2",
            "--gradient-checkpointing",
            "true",
        ]
        for flag, value in _REQUIRED.items():
            cmd += [flag, value]
        subprocess.run(cmd, check=True, capture_output=True)
        text = out.read_text()
        assert "gradient_checkpointing: true" in text
        assert "True" not in text

    def test_parsed_back_as_a_real_boolean(self, tmp_path):
        config = _render(tmp_path, extra=["--gradient-checkpointing", "true"])
        assert config["gradient_checkpointing"] is True

    @pytest.mark.parametrize(
        "given,expected",
        [
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
        ],
    )
    def test_accepts_the_forms_a_shell_renders(self, tmp_path, given, expected):
        config = _render(tmp_path, extra=["--use-liger-fused-jsd", given])
        assert config["use_liger_fused_jsd"] is expected


class TestOnPolicyBlockGating:
    """The six online keys must appear only on the online path.

    Their presence is what the trainer reads to decide whether to expect a vLLM
    server, so leaking them into an off-policy config makes it wait for a server
    nobody started.
    """

    ONLINE_KEYS = (
        "vllm_num_servers",
        "top_p",
        "use_sampled_opd_loss",
        "last_message_only",
        "clip_alpha",
        "opd_importance_sampling",
    )

    def test_off_policy_omits_all_of_them(self, tmp_path):
        config = _render(tmp_path)
        assert not [k for k in self.ONLINE_KEYS if k in config]

    def test_on_policy_emits_all_of_them(self, tmp_path):
        config = _render(
            tmp_path, total_nodes=2, extra=["--vllm-num-servers", "1", "--lmbda", "1.0"]
        )
        assert all(k in config for k in self.ONLINE_KEYS)

    def test_off_policy_still_carries_lmbda_and_beta(self, tmp_path):
        """These are loss knobs, not online-only ones; lmbda=0 IS off-policy."""
        config = _render(tmp_path)
        assert config["lmbda"] == 0.0
        assert config["beta"] == 0.0


class TestNodeSplitValidation:
    """Reject splits that cannot train, at render time rather than on the cluster."""

    def test_rejects_servers_equal_to_node_count(self, tmp_path):
        result = _render(
            tmp_path, total_nodes=2, extra=["--vllm-num-servers", "2"], expect_rc=2
        )
        assert "must be < total nodes" in result.stderr

    def test_rejects_servers_exceeding_node_count(self, tmp_path):
        _render(tmp_path, total_nodes=2, extra=["--vllm-num-servers", "3"], expect_rc=2)

    def test_rejects_on_policy_on_a_single_node(self, tmp_path):
        """The floor is 2: one server plus one trainer."""
        result = _render(
            tmp_path, total_nodes=1, extra=["--vllm-num-servers", "1"], expect_rc=2
        )
        assert "at least 2 nodes" in result.stderr

    def test_allows_off_policy_on_a_single_node(self, tmp_path):
        config = _render(tmp_path, total_nodes=1)
        assert "vllm_num_servers" not in config

    def test_allows_one_server_and_three_trainers(self, tmp_path):
        config = _render(tmp_path, total_nodes=4, extra=["--vllm-num-servers", "1"])
        assert config["vllm_num_servers"] == 1


class TestTrainerContract:
    """Values whose default matters, and which must survive rendering verbatim."""

    def test_response_template_is_preserved_exactly(self, tmp_path):
        """Required for completion masking; the nothink chat template has no
        {% generation %} tag, so this string is what locates the span."""
        config = _render(tmp_path)
        assert config["response_template"] == "<|im_start|>assistant"

    def test_fused_jsd_defaults_off(self, tmp_path):
        """granite's logits_scaling=10 overflows the fused bf16 kernel -> NaN."""
        config = _render(tmp_path)
        assert config["use_liger_fused_jsd"] is False

    def test_lmbda_defaults_to_off_policy(self, tmp_path):
        config = _render(tmp_path)
        assert config["lmbda"] == 0.0

    def test_reference_hyperparameters_are_the_defaults(self, tmp_path):
        config = _render(tmp_path)
        assert config["max_length"] == 16384
        assert config["max_completion_length"] == 4096
        assert config["per_device_train_batch_size"] == 1
        assert config["save_total_limit"] == 20

    def test_model_and_data_paths_pass_through(self, tmp_path):
        config = _render(tmp_path)
        assert config["model_name_or_path"] == _REQUIRED["--model-name-or-path"]
        assert config["dataset_name"] == _REQUIRED["--dataset-name"]

    def test_output_is_parseable_by_the_trainers_own_loader(self, tmp_path):
        """safe_load is what the trainer uses, so rendering implies readability."""
        config = _render(tmp_path)
        assert isinstance(config, dict) and config
