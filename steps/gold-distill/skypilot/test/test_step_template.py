"""Contract tests for step-template.yaml.

The launcher's ``run:`` block is a ~100-line shell script carrying runtime Jinja.
Nothing else validates it before a cluster does, and its failures are expensive:
a shell syntax error costs a queue slot, and a missing rank guard costs N
duplicate artifact registrations on a multi-node run.

These tests read the template directly (not the rendered Space), so they hold
whether or not ``make space`` has been run.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

_STEP = Path(__file__).resolve().parent.parent / "step-template.yaml"


@pytest.fixture(scope="module")
def step():
    return yaml.safe_load(_STEP.read_text())


@pytest.fixture(scope="module")
def launcher(step):
    return step["environment_configs"]["Skypilot"]["launchers"]["gold"]["config"]


@pytest.fixture(scope="module")
def run_script(launcher):
    return launcher["run"]


def _as_shell(script):
    """Approximate what fill_objtemplate leaves behind, for a syntax check."""
    script = re.sub(r"\{%.*?%\}", "", script, flags=re.S)
    return re.sub(r"\{\{.*?\}\}", "X", script, flags=re.S)


class TestRunScriptIsValidShell:
    def test_bash_accepts_the_rendered_script(self, run_script):
        result = subprocess.run(
            ["bash", "-n"],
            input=_as_shell(run_script),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_container_venv_is_put_on_path(self, run_script):
        """The image's venv is not on PATH by default."""
        assert "export PATH=/stage/.venv/bin:$PATH" in run_script

    def test_no_login_shell_anywhere(self, run_script):
        """A login shell re-runs /etc/profile and drops the venv from PATH."""
        assert "bash -lc" not in run_script
        assert "#!/bin/bash -l" not in run_script


class TestRankHandling:
    """accelerate owns per-process rank; the provisioner's vars are node-level."""

    def test_inherited_rank_vars_are_unset_before_launch(self, run_script):
        assert "unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT" in run_script
        assert run_script.index("unset RANK") < run_script.index("accelerate launch")

    def test_topology_is_snapshotted_before_being_unset(self, run_script):
        for var in ("RANK", "TOTAL_NODES", "NUM_GPUS_PER_NODE", "MASTER_ADDR"):
            assert f"${{{var}" in run_script

    def test_master_addr_is_required_not_defaulted(self, run_script):
        """A silently-wrong master address hangs NCCL init until the timeout, so
        fail loudly instead of substituting a default."""
        assert "${MASTER_ADDR:?" in run_script

    def test_accelerate_receives_the_snapshotted_topology(self, run_script):
        for flag in (
            "--machine_rank",
            "--main_process_ip",
            "--main_process_port",
            "--num_machines",
            "--num_processes",
        ):
            assert flag in run_script


class TestRankZeroGuards:
    """The executor streams every node into one driver log.

    So anything that must happen once per RUN, rather than once per NODE, has to
    be guarded — an unguarded artifact marker registers N checkpoints.
    """

    def test_artifact_marker_is_guarded(self, run_script):
        marker = 'echo "GB_ARTIFACT_ID:checkpoint'
        assert marker in run_script
        guard = run_script.rindex('if [ "$NODE_RANK" = "0" ]; then')
        assert guard < run_script.index(marker)

    def test_commit_metadata_is_guarded(self, run_script):
        assert "GB_STEP_METADATA_KEY:kd_sandbox_commit" in run_script
        before = run_script[: run_script.index("GB_STEP_METADATA_KEY")]
        assert '[ "$NODE_RANK" = "0" ]' in before

    def test_config_echo_is_guarded(self, run_script):
        """Printing the config N times would bury the run's real output."""
        before = run_script[: run_script.index('cat "$CFG"')]
        assert '[ "$NODE_RANK" = "0" ]' in before


class TestIdentityComesFromTheAllocation:
    def test_config_name_uses_the_runtime_node_count(self, run_script):
        """Not a build parameter: the checkpoint path must not be able to claim a
        topology the run did not have."""
        assert (
            'CONFIG_NAME="{{ config.gold_config.run_name }}_node${NODES}"' in run_script
        )

    def test_checkpoint_dir_is_under_the_build_workdir(self, run_script):
        """Per-build and publishable, unlike the launcher's hardcoded path."""
        assert (
            'CKPT_DIR="${GB_BUILD_WORKDIR:-$PWD}/checkpoints/${CONFIG_NAME}"'
            in run_script
        )


class TestOnPolicyForwardCompatibility:
    """The vLLM/trainer split is a step-level branch, needing no fork change."""

    def test_last_nodes_serve(self, run_script):
        assert "TRAINER_NODES=$(( NODES - VLLM_SERVERS ))" in run_script
        assert '[ "$NODE_RANK" -ge "$TRAINER_NODES" ]' in run_script

    def test_trainer_process_count_excludes_server_nodes(self, run_script):
        assert "--num_processes $(( TRAINER_NODES * GPUS ))" in run_script
        assert '--num_machines "$TRAINER_NODES"' in run_script

    def test_off_policy_disables_vllm_explicitly(self, run_script):
        assert "--use_vllm=False" in run_script


class TestStepDeclaration:
    def test_is_a_training_step_publishing_a_model(self, step):
        assert step["type"] == "training"
        assert step["outputs"]["optional"]["checkpoint"]["type"] == "model"

    def test_restricted_to_the_lsf_subtype(self, step):
        """enroot and the LSF topology contract are LSF-specific, and the image
        is an SM90 build that cannot run on A100."""
        assert step["environment_configs"]["Skypilot"]["subtypes"] == ["lsf"]

    def test_no_setup_phase(self, launcher):
        """The trainer comes from /proj, so there is nothing to clone or install —
        and a setup phase would be one more thing to fail per node."""
        assert "setup" not in launcher

    def test_resources_are_left_to_the_build(self, launcher):
        """One step serves the smoke and reference runs."""
        assert launcher["resources"] == {}

    def test_renderer_is_shipped_as_a_file_mount(self, launcher):
        assert launcher["file_mounts"] == {"src": "src"}

    def test_nccl_timeout_env_is_present(self, launcher):
        """The 30B teacher forward plus ZeRO-3 collectives outlast the default
        watchdog on a healthy run, so the timeout must be set explicitly. Its
        value is templated — see TestDistributedDiagnostics."""
        for key in (
            "TORCH_NCCL_TIMEOUT_MS",
            "NCCL_TIMEOUT",
            "TORCH_NCCL_ENABLE_MONITORING",
        ):
            assert key in launcher["envs"]

    def test_monitor_uses_periodic_retrieval(self, step):
        """The default on_completion surfaces nothing until a multi-hour run ends."""
        monitor = step["environment_configs"]["Skypilot"]["monitors"][
            "skypilot_monitor"
        ]
        assert monitor["ref"] == "space://monitors/skypilot"
        assert "periodic" in monitor["config"]["log_retrieval"]["mode"]

    def test_defaults_that_break_granite_are_correct(self, step):
        gold = step["config"]["gold_config"]
        assert gold["use_liger_fused_jsd"] is False
        assert gold["response_template"] == "<|im_start|>assistant"
        assert gold["lmbda"] == 0.0
        assert gold["vllm_num_servers"] == 0

    def test_model_and_data_have_no_defaults(self, step):
        """Silently distilling the wrong model is worse than failing to start."""
        gold = step["config"]["gold_config"]
        for key in ("model_name_or_path", "teacher_model_name_or_path", "dataset_name"):
            assert gold[key] == ""


class TestRendererInvocation:
    """Every gold_config field must actually reach the renderer."""

    def test_all_renderer_flags_are_passed(self, run_script, step):
        gold = step["config"]["gold_config"]
        # Fields consumed by the run block or the launcher env rather than
        # forwarded to the renderer: the nccl_* knobs configure NCCL through the
        # environment, and have no place in the trainer's config file.
        step_only = {
            "kd_code_dir",
            "ds_config",
            "run_name",
            "nccl_debug",
            "nccl_debug_subsys",
            "nccl_timeout_ms",
            "nccl_enable_monitoring",
        }
        for key in gold:
            if key in step_only:
                continue
            flag = "--" + key.replace("_", "-")
            assert flag in run_script, f"{key} never reaches the renderer"

    def test_renderer_is_run_with_the_container_interpreter(self, run_script):
        """So the config is dumped by the same PyYAML the trainer parses with."""
        assert "/stage/.venv/bin/python ./src/render_gold_config.py" in run_script

    def test_total_nodes_is_passed_from_the_allocation(self, run_script):
        assert '--total-nodes "$NODES"' in run_script


class TestDistributedDiagnostics:
    """A hang must produce an error, not silence.

    The first 2-node run reached the training loop and then stalled on step 0 for
    an hour with no output, because the step copied the reference launcher's
    TORCH_NCCL_ENABLE_MONITORING=0 — which disables the thread that aborts a
    stalled collective. The allocation was held the whole time and nothing was
    learned from it.
    """

    def test_monitoring_defaults_on(self, step):
        """Deliberately diverging from the reference launcher: an abort with a
        named collective beats an indefinite hang."""
        assert step["config"]["gold_config"]["nccl_enable_monitoring"] is True

    def test_monitoring_is_templated_not_hardcoded(self, launcher):
        env = launcher["envs"]["TORCH_NCCL_ENABLE_MONITORING"]
        assert "{{" in env and "nccl_enable_monitoring" in env
        assert '"1"' in env and '"0"' in env, "must render 1/0, not True/False"

    def test_timeout_is_templated(self, launcher):
        for key in ("TORCH_NCCL_TIMEOUT_MS", "NCCL_TIMEOUT"):
            assert "nccl_timeout_ms" in launcher["envs"][key]

    def test_nccl_debug_is_available_and_off_by_default(self, step, launcher):
        """Off by default (very verbose), but reachable without editing the step —
        it is the only way to distinguish an IB path from a silent TCP fallback."""
        assert step["config"]["gold_config"]["nccl_debug"] == ""
        assert "nccl_debug" in launcher["envs"]["NCCL_DEBUG"]

    def test_reference_timeout_default_is_preserved(self, step):
        """A healthy 30B teacher forward is slow; the production default must stay
        generous even though debug builds lower it."""
        assert step["config"]["gold_config"]["nccl_timeout_ms"] == 3600000


class TestMasterAddressIsAnIp:
    """accelerate is given an IP, matching the reference launcher.

    The provisioner exports MASTER_ADDR as a short hostname. Rendezvous works with
    either form, but NCCL's bootstrap selects its interface from this value, so
    the validated path's choice is not something to assume equivalent.
    """

    def test_master_is_resolved_before_use(self, run_script):
        assert "/etc/hosts" in run_script
        assert run_script.index("MIP=") < run_script.index("--main_process_ip")

    def test_accelerate_receives_the_resolved_address(self, run_script):
        assert '--main_process_ip "$MIP"' in run_script
        assert '--main_process_ip "$MADDR"' not in run_script

    def test_resolution_falls_back_rather_than_failing(self, run_script):
        """A missing /etc/hosts entry must not abort the run: fall through to
        getent, then to the hostname, which is what worked before."""
        assert "getent ahostsv4" in run_script
        assert '[ -z "$MIP" ] && MIP="$MADDR"' in run_script
