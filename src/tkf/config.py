from dataclasses import dataclass


@dataclass
class TkfConfig:
    """Global and pipeline default configurations."""
    default_ttl_seconds_after_finished: int = 300  # Auto-delete Jobs 5 mins after completion
    default_storage_class: str = "local-path"
    default_workspace_mount: str = "/workspace"
