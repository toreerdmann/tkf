from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class Phase(str, Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    SKIPPED = "Skipped"


class VolumeSpec(BaseModel):
    enabled: bool = True
    name: str | None = None
    size: str = "1Gi"
    mount_path: str = Field(default="/workspace", alias="mountPath")
    temp: bool = False
    storage_class: str = Field(default="local-path", alias="storageClass")

    model_config = {"populate_by_name": True}


class TaskSpec(BaseModel):
    name: str
    image: str
    command: list[str] = Field(default_factory=list)
    args: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list, alias="dependsOn")
    env: dict[str, str] = Field(default_factory=dict)
    cpu: str | None = "500m"
    memory: str | None = "512Mi"
    gpu: int | None = None

    model_config = {"populate_by_name": True}


class PipelineRunSpec(BaseModel):
    volume: VolumeSpec = Field(default_factory=VolumeSpec)
    tasks: list[TaskSpec] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class TaskStatus(BaseModel):
    phase: Phase = Phase.PENDING
    job_name: str | None = Field(default=None, alias="jobName")
    pod_name: str | None = Field(default=None, alias="podName")
    start_time: str | None = Field(default=None, alias="startTime")
    completion_time: str | None = Field(default=None, alias="completionTime")
    exit_code: int | None = Field(default=None, alias="exitCode")
    outputs: dict[str, str] = Field(default_factory=dict)
    message: str | None = None

    model_config = {"populate_by_name": True}


class PipelineRunStatus(BaseModel):
    phase: Phase = Phase.PENDING
    message: str | None = None
    start_time: str | None = Field(default=None, alias="startTime")
    completion_time: str | None = Field(default=None, alias="completionTime")
    pvc_name: str | None = Field(default=None, alias="pvcName")
    tasks: dict[str, TaskStatus] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}
