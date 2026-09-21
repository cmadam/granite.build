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
        # Anchored on the FULL key rather than the bare GB_STEP_METADATA_KEY prefix.
        # The shared source-delivery region echoes three metadata keys of its own
        # (distill_code_dirty, _commit, _source) and runs above the rank split, so the
        # prefix stopped identifying THIS step's echo the moment that region was
        # spliced in: it matched the region's line and reported a missing guard for a
        # guard that is still there. A test that names the thing it is about survives
        # the file growing around it.
        marker = "GB_STEP_METADATA_KEY:kd_sandbox_commit"
        assert marker in run_script
        before = run_script[: run_script.index(marker)]
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

    def test_use_vllm_is_stated_in_both_directions(self, run_script):
        """Driven by the same vllm_num_servers as every other on-policy key.

        gold.py defaults --use_vllm to False and custom_gold_trainer.py branches on
        it, so a template emitting the flag only negatively leaves an on-policy run
        generating locally while its allocated server sits idle — build d77546a9 did
        exactly that. Asserted here as the unrendered expression, because this suite
        reads the template rather than a render;
        test/unit/builtins/steps/test_gold_distill.py renders both branches.
        """
        assert (
            "--use_vllm={{ 'True' if config.gold_config.vllm_num_servers "
            "| int > 0 else 'False' }}" in run_script
        )


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

    # Every gold_config key goes to exactly one of three places, and a key that
    # reaches NONE of them is configuration that does nothing — which reads as a
    # working knob to the next person who tunes it. The partition is asserted
    # rather than described so that adding a key forces a decision about which
    # kind it is.
    #
    # 1. STEP_ONLY — consumed by the run block or the launcher env. The nccl_*
    #    knobs configure NCCL through the environment and have no place in the
    #    trainer's config file; the rest name paths the step itself resolves.
    STEP_ONLY = {
        "kd_code_dir",
        # Whether the shared gb_steps_post_training checkout is delivered into the
        # container is a property of the ENVIRONMENT, not of the trainer's config, so it
        # is consumed by the run block's Jinja guard and deliberately never reaches the
        # renderer. Sending it there would put a key the trainer's dataclass does not
        # accept into the rendered config, which TrlParser rejects outright.
        "deliver_distill_source",
        "ds_config",
        "run_name",
        "nccl_debug",
        "nccl_debug_subsys",
        "nccl_timeout_ms",
        "nccl_enable_monitoring",
    }
    # 2. TRAINER_CLI_ONLY — handed to gold.py on its command line, NOT written
    #    into the rendered config. That mirrors the one launcher that has actually
    #    run on-policy: TRL's parse_args_and_config rejects unknown top-level keys,
    #    so putting these in the config file would bet on them being accepted
    #    config-file keys rather than merely accepted CLI flags.
    TRAINER_CLI_ONLY = {
        "vllm_mode": "--vllm_mode",
        "vllm_sync_frequency": "--vllm_sync_frequency",
    }

    # 3. Everything else reaches render_gold_config.py as --kebab-case.

    def test_all_renderer_flags_are_passed(self, run_script, step):
        for key in step["config"]["gold_config"]:
            if key in self.STEP_ONLY or key in self.TRAINER_CLI_ONLY:
                continue
            flag = "--" + key.replace("_", "-")
            assert flag in run_script, f"{key} never reaches the renderer"

    def test_trainer_cli_only_keys_reach_gold_py(self, run_script, step):
        """They bypass the renderer, so nothing else would catch them going
        nowhere — and a silently dropped vllm_mode is an on-policy run that
        cannot find its server."""
        for key, flag in self.TRAINER_CLI_ONLY.items():
            assert key in step["config"]["gold_config"], f"{key} is not a config key"
            assert flag in run_script, f"{key} never reaches gold.py"
            # And they must NOT be sent to the renderer, which would emit them
            # into the config file and risk the rejection described above.
            assert (
                "--" + key.replace("_", "-") not in run_script
            ), f"{key} is also passed to the renderer"

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


class TestExternalVllmServer:
    """The on-policy path where the server is a separate target, reached by URL.

    NOT YET RUN ON A CLUSTER. Everything here is a contract test; the open
    question is whether the trainer's NCCL weight-sync group can span two LSF
    allocations, which no test can answer.
    """

    def test_the_url_is_a_config_key(self, step):
        gold = step["config"]["gold_config"]
        assert gold["vllm_server_url"] == ""
        assert gold["vllm_mode"] == "server"
        assert gold["vllm_sync_frequency"] == 1

    def test_all_nodes_train_when_the_server_is_external(self, run_script):
        """No node is taken away from the trainer, because the server is not in
        this allocation. Getting this wrong wastes a node silently."""
        assert 'TRAINER_NODES="$NODES"' in run_script

    def test_the_in_allocation_split_is_still_present(self, run_script):
        """The external path is additive; the reference launcher's role split must
        remain reachable when no URL is given."""
        assert "TRAINER_NODES=$(( NODES - VLLM_SERVERS ))" in run_script
        assert "run_vllm_serve.py" in run_script

    def test_the_address_reaches_gold_py(self, run_script):
        for flag in ("--vllm_server_host", "--vllm_server_port", "--vllm_mode"):
            assert flag in run_script
        assert "$VLLM_ARGS" in run_script

    def test_the_url_is_parsed_at_run_time_not_templated(self, run_script):
        """The address is only known at run time — it arrives through a mem://
        binding — so it must be split in shell, not by Jinja."""
        assert 'VLLM_URL="{{ config.gold_config.vllm_server_url }}"' in run_script
        assert "_hostport" in run_script

    @pytest.mark.parametrize(
        "url,host,port",
        [
            ("http://host-42:8001", "host-42", "8001"),
            ("http://10.0.0.7:8001", "10.0.0.7", "8001"),
            # A trailing path is what an OpenAI-compatible base URL looks like.
            ("http://host-42:8001/v1", "host-42", "8001"),
            # No port: fall back to the reference launcher's 8001 rather than
            # handing the trainer an empty port that fails at connect time.
            ("http://host-42", "host-42", "8001"),
            ("https://host-42:9000", "host-42", "9000"),
        ],
    )
    def test_url_parsing_actually_works(self, run_script, url, host, port):
        """Runs the real parsing lines rather than pattern-matching them, because
        a wrong host here is a connection error minutes into an allocated run.
        """
        lines = _as_shell(run_script).splitlines()
        start = next(i for i, l in enumerate(lines) if "_hostport=" in l)
        # The first `esac` AFTER the function, not the first in the file. The shared
        # source-delivery region carries a case/esac inside its GIT_ASKPASS heredoc and
        # sits above this function, so scanning from zero selected that one -- making
        # lines[start:end + 1] EMPTY and every parametrization below compare an empty
        # stdout against the expected host. It failed loudly here, but the same slice
        # bug in a test that asserted something weaker would have gone on passing while
        # testing nothing at all.
        end = next(i for i, l in enumerate(lines) if i > start and "esac" in l)
        snippet = "\n".join(lines[start : end + 1]).replace(
            "${VLLM_URL#*://}", "${URL#*://}"
        )

        result = subprocess.run(
            ["bash", "-c", f'URL="{url}"\n{snippet}\necho "$VLLM_HOST $VLLM_PORT"'],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == f"{host} {port}"

    def test_an_unparseable_url_fails_loudly(self, run_script):
        """Rather than launching a trainer that cannot reach anything."""
        assert "could not parse a host out of vllm_server_url" in run_script


class TestDistillSourceDelivery:
    """The opt-in shared-checkout block, and the fact that OFF is the old behaviour.

    gold-distill is the one ported step that does not need gb_steps_post_training to run:
    its trainer comes from kd_code_dir. The block is here so the renderer's extra
    validators can be reached, and it is guarded so that adding it changed nothing for the
    five recipes that already exist. Both halves of that claim are asserted.
    """

    def test_the_contract_block_is_present(self, step):
        """test_source_contract.py asserts it is byte-identical to the reference; this only
        asserts gold-distill has one at all, so a failure here reads as "missing" rather
        than as an obscure ValueError from that file's .index()."""
        assert "code_config" in step["config"]
        cc = step["config"]["code_config"]
        assert (
            cc["code_dir"]
            == "/proj/granite-build/g4os/gb-steps-collection-post-training"
        )
        assert cc["python"] == "/stage/.venv/bin/python"
        # Empty on purpose: the default path needs no credential in the container.
        assert cc["repo"] == "" and cc["token_secret"] == ""

    def test_delivery_is_off_by_default(self, step):
        """The whole no-behaviour-change claim rests on this one value."""
        assert step["config"]["gold_config"]["deliver_distill_source"] is False

    def test_the_region_is_guarded_by_that_key(self, run_script):
        """Off must mean NOT RENDERED, not rendered-and-harmless: the block exits 1 when
        the checkout is absent, so an unguarded copy would turn every host without
        /proj/granite-build into a failing gold-distill run."""
        guard = "{% if config.gold_config.deliver_distill_source %}"
        assert guard in run_script
        begin = run_script.index("# --- distill source delivery: BEGIN")
        end = run_script.index("# --- distill source delivery: END")
        assert (
            run_script.index(guard) < begin
        ), "the guard opens after the region begins"
        assert end < run_script.index(
            "{% endif %}", end
        ), "the region is not closed inside the guard"

    def test_the_guard_wraps_the_region_without_entering_it(self, run_script):
        """The byte-identity assertion in test_source_contract.py extracts BEGIN..END
        inclusive, so a guard placed INSIDE those markers would silently break the
        contract for every other ported step at once."""
        begin = run_script.index("# --- distill source delivery: BEGIN")
        end = run_script.index("# --- distill source delivery: END ---")
        region = run_script[begin : end + len("# --- distill source delivery: END ---")]
        assert "deliver_distill_source" not in region

    def test_the_region_runs_above_the_rank_split(self, run_script):
        """Asserted because it is the reason two tests in this file had to name their
        full anchor, and because it is a property worth being explicit about: the region
        runs on EVERY node, so its three metadata echoes are emitted N times on an
        N-node run. That is the shared region's behaviour, byte-identical in all seven
        ported steps and untested in the reference step (which asserts only that the
        keys exist, not that they are guarded) -- so it is recorded here rather than
        diverged from, since guarding it in one step would break the byte-identity
        contract for the other six.
        """
        assert run_script.index(
            "# --- distill source delivery: BEGIN"
        ) < run_script.index('if [ "$NODE_RANK" = "0" ]; then')
        assert "GB_STEP_METADATA_KEY:distill_code_dirty" in run_script

    def test_pythonpath_reaches_the_renderer(self, run_script):
        """The point of delivering the source at all: the renderer is what imports it, so
        the export must precede the invocation rather than merely existing."""
        assert run_script.index('export PYTHONPATH="$CODE_DIR/src') < run_script.index(
            "/stage/.venv/bin/python ./src/render_gold_config.py"
        )
