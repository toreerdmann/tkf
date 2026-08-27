from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OutputRef:
    """Reference to a task output parameter or artifact."""

    task_name: str
    name: str
    kind: str = "param"  # "param", "artifact", "dataset", "model"

    def __str__(self) -> str:
        if self.kind == "param":
            return f"{{{{ tasks.{self.task_name}.outputs.{self.name} }}}}"
        # For artifacts, datasets, and models:
        return f"{{{{ tasks.{self.task_name}.artifacts.{self.name} }}}}"

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# In-Container Task Helpers (Functions invoked inside the task's script)
# ---------------------------------------------------------------------------


def set_output(name: str, value: Any) -> None:
    """Set an output parameter for the current task.

    Writes to /dev/termination-log (or fallback file) so the controller
    can record it in PipelineRun status and pass to downstream tasks.
    """
    val_str = str(value) if not isinstance(value, (dict, list)) else json.dumps(value)

    # 1. Update termination log for K8s controller
    term_log_path = Path("/dev/termination-log")
    current_outputs = {}

    try:
        if term_log_path.exists() and term_log_path.stat().st_size > 0:
            try:
                current_outputs = json.loads(term_log_path.read_text())
            except Exception:
                current_outputs = {}
        current_outputs[name] = val_str
        term_log_path.write_text(json.dumps(current_outputs))
    except Exception:
        pass  # In local simulation or non-root container

    # 2. Also persist to shared volume directory
    vol = os.environ.get("VOLUME", "/workspace")
    run_name = os.environ.get("TKF_RUN_NAME")
    task_name = os.environ.get("TKF_TASK_NAME", "default")
    if run_name:
        out_dir = Path(vol) / "runs" / run_name / "outputs" / task_name
    else:
        out_dir = Path(vol) / ".tkf" / "outputs" / task_name
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / name).write_text(val_str)
    except Exception:
        pass


def artifact_path(name: str, create_parents: bool = True) -> Path:
    """Get the absolute path for an output artifact in the shared volume."""
    vol = os.environ.get("VOLUME", "/workspace")
    run_name = os.environ.get("TKF_RUN_NAME")
    task_name = os.environ.get("TKF_TASK_NAME", "default")
    if run_name:
        path = Path(vol) / "runs" / run_name / "artifacts" / task_name / name
    else:
        path = Path(vol) / "artifacts" / task_name / name
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def dataset_path(name: str, create_parents: bool = True) -> Path:
    """Alias for artifact_path (semantic clarity for datasets)."""
    return artifact_path(name, create_parents=create_parents)


def model_path(name: str, create_parents: bool = True) -> Path:
    """Alias for artifact_path (semantic clarity for model files)."""
    return artifact_path(name, create_parents=create_parents)
