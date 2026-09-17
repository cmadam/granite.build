"""Contract tests for step-template.yaml.

The launcher's ``run:`` block is a ~140-line shell script carrying runtime Jinja,
and it hands off to ``src/run-align.sh``. Nothing else validates either before a
cluster does, and the failures are expensive: a shell syntax error costs a queue
slot, and an artifact id printed by the script but not declared in the template
makes the buildrun resolver drop the event, so the target completes with no output
AND no error.

These tests read the template and the script directly (not the rendered Space), so
they hold whether or not ``make space`` has been run, and they need no checkout of
the upstream distillation package.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve().parent.parent
_STEP = _HERE / "step-template.yaml"
_RUN_ALIGN = _HERE / "src" / "run-align.sh"


@pytest.fixture(scope="module")
def step():
    return yaml.safe_load(_STEP.read_text())


@pytest.fixture(scope="module")
def launcher(step):
    return step["environment_configs"]["Skypilot"]["launchers"]["align"]["config"]


@pytest.fixture(scope="module")
def run_script(launcher):
    return launcher["run"]


@pytest.fixture(scope="module")
def align_sh():
    return _RUN_ALIGN.read_text()


def _as_shell(script):
    """Approximate what fill_objtemplate leaves behind, for a syntax check.

    Block tags become a SPACE rather than nothing: an ``{% if %}--flag{% else %}
    --no-flag{% endif %}`` pair would otherwise collapse into the single token
    ``--flag--no-flag``, which is neither valid shell nor a real flag name.
    """
    script = re.sub(r"\{%.*?%\}", " ", script, flags=re.S)
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

    def test_no_login_shell_anywhere(self, run_script):
        """A login shell re-runs /etc/profile and drops the venv from PATH."""
        assert "bash -lc" not in run_script

    def test_no_jinja_comment_sequence(self, run_script):
        """``${#VAR}`` opens a Jinja comment.

        The renderer does not override ``comment_start_string``, so it stays the
        default ``{#`` — and a bash string-length expansion contains exactly that.
        Measured: a build.yaml carrying ``${#TOKEN}`` fails validation with
        "Missing end of comment tag" rather than anything naming the real cause.
        """
        assert "${#" not in run_script

    def test_metadata_emission_cannot_abort_the_run(self, run_script):
        """``[ -n "$X" ] && echo`` as the last statement of an if-body returns 1
        when X is empty, and the launcher prefixes ``set -eu``, so that aborts the
        run. gold-distill's template still carries this latent bug; this one must
        not, so every metadata echo is written as a full if/then/fi.
        """
        assert not re.search(
            r'\[\s+-n\s+"\$\w+"\s+\]\s+&&\s+echo\s+"GB_STEP_METADATA', run_script
        )


class TestSourceDeliveryContract:
    """The block asserted byte-identical across the ported steps.

    Its shape is load-bearing, not stylistic: the default must need no credential,
    the clone branch must refuse rather than hang when it has none, and the
    resolved commit must be recorded whichever branch ran.
    """

    EXPECTED_KEYS = {
        "code_dir",
        "expect_ref",
        "repo",
        "ref",
        "workdir",
        "token_secret",
        "python",
        "setup_command",
    }

    def test_code_config_has_exactly_the_contract_keys(self, step):
        assert set(step["config"]["code_config"]) == self.EXPECTED_KEYS

    def test_filesystem_path_is_the_default(self, step):
        """No credential reaches the container by default. Every other step in this
        repo either clones nothing or clones a public repo unauthenticated."""
        code = step["config"]["code_config"]
        assert code["code_dir"].startswith("/"), "the default must be a real path"
        assert code["repo"] == "", "the clone branch must be opt-in"
        assert code["ref"] == ""
        assert code["token_secret"] == "", "the default path must need no secret"

    def test_the_pin_is_a_full_sha(self, step):
        """A branch name here makes two runs a week apart different runs while
        reporting the same provenance."""
        assert re.fullmatch(
            r"[0-9a-f]{40}", step["config"]["code_config"]["expect_ref"]
        )

    def test_both_branches_are_present(self, run_script):
        assert 'CODE_SOURCE="filesystem"' in run_script
        assert 'CODE_SOURCE="git"' in run_script

    def test_package_root_goes_on_pythonpath(self, run_script):
        """The package ROOT, not the distillation/ directory: every module is
        imported by its full dotted path. Omitting this killed two upstream jobs
        at import time."""
        assert 'export PYTHONPATH="$CODE_DIR/src' in run_script

    def test_the_resolved_commit_is_recorded(self, run_script):
        assert "GB_STEP_METADATA_KEY:distill_code_commit" in run_script
        assert "GB_STEP_METADATA_KEY:distill_code_source" in run_script

    def test_a_dirty_shared_checkout_is_reported(self, run_script):
        """A shared checkout is mutable state, so a run whose code came from a
        dirty tree must not report a bare commit as its provenance."""
        assert "status --porcelain" in run_script
        assert "GB_STEP_METADATA_KEY:distill_code_dirty" in run_script

    def test_an_unexpected_checkout_commit_is_fatal(self, run_script):
        assert "expect_ref" in run_script
        assert re.search(r"AT.*!=.*EXPECT_REF|EXPECT_REF.*!=.*AT", run_script)

    def test_the_clone_branch_refuses_without_a_credential(self, run_script):
        """Unauthenticated against a private repo either hangs on a prompt or 404s,
        so it must say which secret is missing instead."""
        assert "GIT_TERMINAL_PROMPT=0" in run_script
        assert "has no secret" in run_script

    def test_the_token_never_reaches_argv_or_git_config(self, run_script):
        """A token in the URL is readable via /proc on a shared node and persists
        into .git/config on the SHARED filesystem. Verified on BlueVela: with
        GIT_ASKPASS, .git/config had 0 matches for the token."""
        assert "GIT_ASKPASS" in run_script
        assert "x-access-token:" not in run_script
        assert "@github" not in run_script

    def test_source_delivery_region_is_delimited(self, run_script):
        """test_source_contract.py extracts this region from every ported step and
        asserts they are byte-identical, so the markers must exist."""
        assert run_script.count("--- distill source delivery: BEGIN") == 1
        assert run_script.count("--- distill source delivery: END") == 1

    def test_no_setup_phase(self, launcher):
        """``task.setup`` is unexercised on the LSF cloud in this repo: the only
        shipped step with one is byoc, whose build tests are slurm/aws. Source
        delivery happens in ``run``, which gold-distill has proven on LSF."""
        assert "setup" not in launcher


class TestLauncher:
    def test_lsf_only(self, step):
        """SM90 image, and the contract assumes the LSF provisioner's
        identity-mounted /proj workdir."""
        assert step["environment_configs"]["Skypilot"]["subtypes"] == ["lsf"]

    def test_runs_in_a_prebuilt_registry_image(self, launcher):
        """A non-image step: no Dockerfile, so no ${IMAGE_REF} to substitute."""
        assert launcher["image_id"].startswith("docker:")
        assert "IMAGE_REF" not in launcher["image_id"]

    def test_no_dockerfile_so_common_mk_treats_it_as_non_image(self):
        assert not (_HERE / "Dockerfile").exists()

    def test_resources_come_from_the_build(self, launcher):
        """No GPU is needed; node count and accelerators come from build.yaml so
        one step serves a smoke run and the reference run."""
        assert launcher["resources"] == {}

    def test_the_launcher_script_is_shipped(self, launcher):
        assert launcher["file_mounts"] == {"src": "src"}

    def test_uses_the_shipped_monitor(self, step):
        monitors = step["environment_configs"]["Skypilot"]["monitors"]
        assert monitors["skypilot_monitor"]["ref"] == "space://monitors/skypilot"


class TestArtifactContract:
    """Declared outputs and printed markers must be the same set.

    An undeclared output makes the resolver drop the NEWARTIFACT event and the
    target completes with no output and no error. An id declared but never printed
    is the mirror failure: a consumer waits on a binding that never arrives.
    """

    def test_declared_and_printed_ids_match(self, step, align_sh):
        declared = set(step["outputs"]["required"])
        printed = set(re.findall(r"GB_ARTIFACT_ID:(\w+)", align_sh))
        assert declared == printed, f"declared={declared} printed={printed}"

    def test_all_three_are_declared_as_models(self, step):
        for name, spec in step["outputs"]["required"].items():
            assert spec["type"] == "model", name

    def test_artifact_paths_are_absolutised(self, run_script):
        """The monitor hands these to the env:// store, possibly from another host,
        and a relative env: URI is rejected at config load."""
        assert 'OUT_DIR="$WORK/$OUT_DIR"' in run_script

    def test_no_legacy_marker_prefix(self, align_sh):
        """The monitor accepts LLMB_ too, but GB_ is what every other step prints."""
        assert "LLMB_ARTIFACT_ID:" not in align_sh


class TestFlagSurface:
    """template -> run-align.sh drift.

    This seam is where executing the rendered command found two real defects
    upstream, so it is asserted in both directions: every config key must reach
    the script, and every flag the template passes must be one the script parses.
    """

    def test_every_align_config_key_reaches_the_script(self, step, run_script):
        for key in step["config"]["align_config"]:
            flag = "--" + key.replace("_", "-")
            if key in ("verify", "dry_run"):
                # booleans are a --flag/--no-flag pair, asserted below
                continue
            assert flag in run_script, f"{key} never reaches run-align.sh"

    def test_booleans_are_flag_pairs_not_values(self, step, run_script):
        """Jinja renders a YAML boolean with PYTHON casing, so ``--flag {{ x }}``
        arrives as the literal string ``False``: nothing errors and the setting is
        silently inverted. Measured upstream on LSF job 1138651."""
        for key in ("verify", "dry_run"):
            assert key in step["config"]["align_config"]
            flag = "--" + key.replace("_", "-")
            assert f"{{% if config.align_config.{key} %}}{flag}" in run_script
            assert f"--no-{key.replace('_', '-')}" in run_script
            assert f"{flag} {{{{ config.align_config.{key} }}}}" not in run_script

    def test_script_parses_every_flag_the_template_passes(self, run_script, align_sh):
        """The reverse direction: a flag the script's case statement does not know
        makes it exit 2 with "unknown argument"."""
        handled = set(re.findall(r"^\s*(--[a-z-]+)\)", align_sh, re.M))
        # Comments first: a "# --- banner ---" line is full of dashes and would
        # otherwise be read as a flag named "---".
        body = "\n".join(
            line
            for line in _as_shell(run_script).splitlines()
            if not line.lstrip().startswith("#")
        )
        # A real flag starts with a letter and is not preceded by another dash.
        for flag in set(re.findall(r"(?<![-\w])(--[a-z][a-z0-9-]*)", body)):
            if flag in ("--quiet", "--porcelain", "--all"):  # git's own flags
                continue
            assert flag in handled, f"run-align.sh does not parse {flag}"

    def test_a_relative_chat_template_resolves_against_the_checkout(
        self, step, run_script
    ):
        """The marked-up template lives in the delivered checkout. Upstream defaults
        this to an in-image /opt path that does not exist in this image."""
        assert not step["config"]["align_config"]["chat_template"].startswith("/")
        assert 'CHAT_TEMPLATE="$CODE_DIR/$CHAT_TEMPLATE"' in run_script


class TestConfigDefaults:
    def test_model_paths_are_not_defaulted(self, step):
        """A wrong default here trains a plausible, meaningless run."""
        align = step["config"]["align_config"]
        assert align["teacher_model"] == ""
        assert align["student_model"] == ""

    def test_verify_defaults_on(self, step):
        """verify() is what catches a resolved backend disagreeing with
        tokenizer.json — the failure mode this step exists to prevent."""
        assert step["config"]["align_config"]["verify"] is True
        assert step["config"]["align_config"]["dry_run"] is False

    def test_step_type_is_a_real_enum_member(self, step):
        from gbcommon.types.stepconfig import StepType

        assert step["type"] in {m.value for m in StepType}
