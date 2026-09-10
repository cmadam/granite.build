"""Types related to the SkyPilot environment."""

from typing import Optional

from pydantic import Field

from gbserver.types.environment.environment import StepEnvConfig


class StepSkypilotConfig(StepEnvConfig):
    """Config specific to SkyPilot environments, extracted from step.yaml."""

    resources: dict = Field(default_factory=dict)
    # Descriptive only. The launch path does NOT read this model: the live object
    # is the untyped `config` dict on StepLauncherConfig, Jinja-filled by
    # fill_objtemplate and read via launcher_config.get(...). Declared here so
    # the field is discoverable, and because num_nodes belongs to sky.Task rather
    # than sky.Resources — it must not be nested under `resources`.
    num_nodes: int = 1
    setup: str = ""
    run: str = ""
    envs: dict = Field(default_factory=dict)
    file_mounts: dict = Field(default_factory=dict)
    idle_minutes_to_autostop: int = 10
    image_id: Optional[str] = None
