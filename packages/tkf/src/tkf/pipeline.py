from __future__ import annotations

import collections
import collections.abc
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from kubernetes import client, config as k8s_config

from tkf.io import OutputRef
from tkf.models import Phase, PipelineRunSpec, PipelineRunStatus, TaskSpec, VolumeSpec

REF_REGEX = re.compile(r"\{\{\s*tasks\.([a-zA-Z0-9_-]+)\.(outputs|artifacts)\.([a-zA-Z0-9_.-]+)\s*\}\}")


def to_k8s_name(raw_name: str) -> str:
    """Convert an arbitrary string to a DNS-1123 compliant name."""
    s = raw_name.lower().replace("_", "-").replace(" ", "-").strip()
    cleaned = "".join(c for c in s if c.isalnum() or c == "-")
    return cleaned.strip("-") or "task"


@dataclass
class VolumeConfig:
    """Configuration for the shared PVC used by pipeline tasks."""

    name: str | None = None
    size: str = "1Gi"
    mount_path: str = "/workspace"
    local_path: str = "local_pipeline_volume"
    temp: bool = False
    storage_class: str = "local-path"
    create_if_missing: bool = True
    enabled: bool = True

    def to_spec(self, fallback_name: str) -> VolumeSpec:
        if self.name is None and self.temp:
            self.name = f"temp-{uuid.uuid4().hex[:8]}"
        vol_name = self.name or f"{fallback_name}-pvc"
        return VolumeSpec(
            enabled=self.enabled,
            name=vol_name,
            size=self.size,
            mountPath=self.mount_path,
            temp=self.temp,
            storageClass=self.storage_class,
        )


@dataclass(frozen=True)
class ComputeResources:
    """Resource descriptor for a Task."""

    cpu: str = "500m"
    memory: str = "512Mi"
    gpu: int | None = None


@dataclass(frozen=True)
class Task:
    """Represents a single step in the pipeline DAG."""

    name: str
    command: list[Any] = field(default_factory=list)
    args: list[Any] = field(default_factory=list)
    docker_image: str = "python:3.12-slim"
    packages: list[str] = field(default_factory=list)  # e.g. ["pandas", "scikit-learn"] (uses uv run --with)
    image_pull_secrets: list[str] = field(default_factory=list)  # e.g. ["ecr-image-pull-secret"]
    env: dict[str, Any] = field(default_factory=dict)
    resources: ComputeResources = field(default_factory=ComputeResources)
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    disable_istio: bool = True

    def __post_init__(self):
        object.__setattr__(self, "name", to_k8s_name(self.name))
        if any(arg is None for arg in self.command):
            raise TypeError(f"Task '{self.name}' command list cannot contain None values.")
        if any(arg is None for arg in self.args):
            raise TypeError(f"Task '{self.name}' args list cannot contain None values.")

    def output(self, name: str) -> OutputRef:
        """Reference an output parameter produced by this task."""
        return OutputRef(task_name=self.name, name=name, kind="param")

    def artifact(self, name: str) -> OutputRef:
        """Reference an output artifact file produced by this task."""
        return OutputRef(task_name=self.name, name=name, kind="artifact")

    def dataset(self, name: str) -> OutputRef:
        """Reference an output dataset produced by this task."""
        return OutputRef(task_name=self.name, name=name, kind="dataset")

    def model(self, name: str) -> OutputRef:
        """Reference an output model file produced by this task."""
        return OutputRef(task_name=self.name, name=name, kind="model")

    def find_referenced_parent_tasks(self) -> set[str]:
        """Inspect command, args, and env for references to other tasks."""
        parents = set()
        items_to_scan = list(self.command) + list(self.args) + list(self.env.values())
        for item in items_to_scan:
            if isinstance(item, OutputRef):
                parents.add(item.task_name)
            elif isinstance(item, str):
                for match in REF_REGEX.finditer(item):
                    parents.add(match.group(1))
        return parents


def make_task_config(task: str, /, **kwargs) -> dict:
    """Helper for config-driven pipelines."""
    return {"task": task, **kwargs}


def make_parallel_config(branches: collections.abc.Iterable[list]) -> list[list]:
    """Helper to structure parallel branches."""
    result = []
    for branch in branches:
        if isinstance(branch, (str, dict)) or not isinstance(branch, collections.abc.Iterable):
            raise TypeError("Each branch must be an iterable of task items.")
        result.append(list(branch))
    return result


def get_current_user() -> str:
    """Detect current user from TKF_USER, GITHUB_USER, GITHUB_ACTOR, USER, or LOGNAME."""
    raw_user = (
        os.environ.get("TKF_USER")
        or os.environ.get("GITHUB_USER")
        or os.environ.get("GITHUB_ACTOR")
        or os.environ.get("USER")
        or os.environ.get("LOGNAME")
        or "default"
    )
    # Sanitize to valid k8s label value (alphanumeric, -, _, ., max 63 chars)
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "-", str(raw_user).strip()).strip("-.")[:63].lower()
    return sanitized or "default"


class Pipeline:
    """A directed acyclic graph (DAG) of tasks to execute on tkf or locally."""

    def __init__(
        self,
        name: str = "default-run",
        volume: VolumeConfig | None = None,
        namespace: str = "default",
        user: str | None = None,
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
        ttl_seconds_after_finished: int | None = 300,
    ):
        self.name = to_k8s_name(name)
        self.volume = volume or VolumeConfig()
        self.namespace = namespace
        self.user = user or get_current_user()
        self.labels = labels.copy() if labels else {}
        self.labels.setdefault("tkf/user", self.user)
        self.annotations = annotations or {}
        self.ttl_seconds_after_finished = ttl_seconds_after_finished

        # DAG representation: node_id -> Task, parent_id -> list[child_ids]
        self._nodes: dict[int, Task] = {}
        self._adj: dict[int, list[int]] = collections.defaultdict(list)

    def add_task(self, task: Task) -> "Pipeline":
        """Add a task node to the graph and auto-wire any referenced parent tasks."""
        if id(task) not in self._nodes:
            self._nodes[id(task)] = task
            referenced_parents = task.find_referenced_parent_tasks()
            for parent_name in referenced_parents:
                parent_task = next((t for t in self._nodes.values() if t.name == parent_name), None)
                if parent_task and id(parent_task) != id(task):
                    self.add_dependency(parent_task, task)
        return self

    def add_dependency(self, parent_task: Task, child_task: Task) -> "Pipeline":
        """Add a directed edge making child_task depend on parent_task."""
        self.add_task(parent_task)
        self.add_task(child_task)

        parent_id, child_id = id(parent_task), id(child_task)
        if child_id not in self._adj[parent_id]:
            self._adj[parent_id].append(child_id)
        return self

    def append_task(self, task: Task) -> "Pipeline":
        """Add a task that depends on all current leaf tasks."""
        self.add_task(task)
        new_id = id(task)
        leaves = [
            node for nid, node in self._nodes.items()
            if nid != new_id and (nid not in self._adj or not self._adj[nid])
        ]
        for leaf in leaves:
            self.add_dependency(leaf, task)
        return self

    def prepend_task(self, task: Task) -> "Pipeline":
        """Add a task that all current root tasks will depend on."""
        self.add_task(task)
        new_id = id(task)
        all_children = {cid for cids in self._adj.values() for cid in cids}
        roots = [node for nid, node in self._nodes.items() if nid != new_id and nid not in all_children]
        for root in roots:
            self.add_dependency(task, root)
        return self

    def build_from_list(self, definition: list) -> "Pipeline":
        """Populate the DAG from a nested list (sequential and parallel branches)."""
        self._build_recursive(definition, parents=[])
        return self

    def _build_recursive(self, definition: list, parents: list[Task]) -> list[Task]:
        last_nodes = parents
        for item in definition:
            is_parallel = isinstance(item, list) and item and isinstance(item[0], list)
            if is_parallel:
                branch_ends = []
                for branch in item:
                    branch_ends.extend(self._build_recursive(branch, parents=last_nodes))
                last_nodes = branch_ends
            elif isinstance(item, list):
                last_nodes = self._build_recursive(item, parents=last_nodes)
            else:
                self.add_task(item)
                for p in last_nodes:
                    self.add_dependency(p, item)
                last_nodes = [item]
        return last_nodes

    def build_from_config(
        self,
        config: list,
        tasks: dict[str, Task | Callable[..., Task]],
    ) -> "Pipeline":
        """Build DAG from a config list mapping step keys to factories or Task objects."""
        def _resolve(item: Any) -> Any:
            if isinstance(item, str):
                if item not in tasks:
                    raise ValueError(f"Unknown task '{item}'. Available: {sorted(tasks)}")
                entry = tasks[item]
                if not isinstance(entry, Task):
                    raise TypeError(f"Task '{item}' is a factory and requires kwargs.")
                return entry
            if isinstance(item, dict):
                item = dict(item)
                key = item.pop("task")
                if key not in tasks:
                    raise ValueError(f"Unknown task '{key}'. Available: {sorted(tasks)}")
                entry = tasks[key]
                return entry if isinstance(entry, Task) else entry(**item)
            if isinstance(item, list):
                return [_resolve(sub) for sub in item]
            raise TypeError(f"Invalid config item: {item!r}")

        resolved = [_resolve(x) for x in config]
        return self.build_from_list(resolved)

    def topological_sort(self) -> list[Task]:
        """Perform topological sort (Kahn's algorithm)."""
        in_degree = dict.fromkeys(self._nodes, 0)
        for children in self._adj.values():
            for cid in children:
                in_degree[cid] += 1

        queue = collections.deque([nid for nid, deg in in_degree.items() if deg == 0])
        ordered: list[Task] = []

        while queue:
            nid = queue.popleft()
            ordered.append(self._nodes[nid])
            for cid in self._adj.get(nid, []):
                in_degree[cid] -= 1
                if in_degree[cid] == 0:
                    queue.append(cid)

        if len(ordered) != len(self._nodes):
            raise RuntimeError("Pipeline graph contains a cycle!")
        return ordered

    def get_task(self, name: str) -> Task:
        """Find a Task in the pipeline by name."""
        k8s_name = to_k8s_name(name)
        for task in self._nodes.values():
            if task.name == k8s_name:
                return task
        raise KeyError(f"Task '{name}' not found in pipeline '{self.name}'. Available: {[t.name for t in self._nodes.values()]}")

    def get_upstream_tasks(self, target: str | Task) -> set[str]:
        """Returns the set of all upstream prerequisite task names for target (including target)."""
        target_name = target.name if isinstance(target, Task) else to_k8s_name(target)
        target_task = self.get_task(target_name)

        rev_adj = collections.defaultdict(list)
        for pid, cids in self._adj.items():
            for cid in cids:
                rev_adj[cid].append(pid)

        visited = set()
        queue = collections.deque([id(target_task)])
        while queue:
            curr_id = queue.popleft()
            if curr_id not in visited:
                visited.add(curr_id)
                for pid in rev_adj.get(curr_id, []):
                    queue.append(pid)
        return {self._nodes[nid].name for nid in visited}

    def get_downstream_tasks(self, target: str | Task) -> set[str]:
        """Returns the set of all downstream dependent task names for target (including target)."""
        target_name = target.name if isinstance(target, Task) else to_k8s_name(target)
        target_task = self.get_task(target_name)

        visited = set()
        queue = collections.deque([id(target_task)])
        while queue:
            curr_id = queue.popleft()
            if curr_id not in visited:
                visited.add(curr_id)
                for cid in self._adj.get(curr_id, []):
                    queue.append(cid)
        return {self._nodes[nid].name for nid in visited}

    def stepper(self, local: bool = True, namespace: str | None = None) -> "PipelineStepper":
        """Create an interactive step-by-step runner for this pipeline."""
        return PipelineStepper(self, local=local, namespace=namespace)

    def print(self) -> None:
        """Pretty-print the DAG structure to console."""
        try:
            sorted_tasks = self.topological_sort()
        except RuntimeError:
            print("(!) Cycle detected.")
            return

        if not sorted_tasks:
            print(f"{self.name}/ (empty)")
            return

        print(f"\n{self.name}/")
        parents_map = collections.defaultdict(list)
        for pid, cids in self._adj.items():
            for cid in cids:
                parents_map[cid].append(pid)

        for task in sorted_tasks:
            tid = id(task)
            p_names = [self._nodes[pid].name for pid in parents_map.get(tid, [])]
            dep_str = f" <- [{', '.join(p_names)}]" if p_names else " (root)"
            pkgs = f" (pkgs: {', '.join(task.packages)})" if task.packages else ""
            print(f"  * {task.name}{dep_str}{pkgs}")
        print()

    def to_manifest(self) -> dict[str, Any]:
        """Generate the Kubernetes CustomResource manifest dictionary."""
        sorted_tasks = self.topological_sort()
        parents_map = collections.defaultdict(list)
        for pid, cids in self._adj.items():
            for cid in cids:
                parents_map[cid].append(pid)

        task_specs: list[dict[str, Any]] = []
        for task in sorted_tasks:
            p_names = [self._nodes[pid].name for pid in parents_map.get(id(task), [])]

            cmd_strs = [str(x) for x in task.command]
            args_strs = [str(x) for x in task.args]
            env_strs = {k: str(v) for k, v in task.env.items()}

            task_specs.append({
                "name": task.name,
                "image": task.docker_image,
                "command": cmd_strs,
                "args": args_strs,
                "dependsOn": p_names,
                "packages": task.packages,
                "imagePullSecrets": task.image_pull_secrets,
                "env": env_strs,
                "cpu": task.resources.cpu,
                "memory": task.resources.memory,
                "gpu": task.resources.gpu,
            })

        vol_spec = self.volume.to_spec(fallback_name=self.name).model_dump(by_alias=True, exclude_none=True)

        return {
            "apiVersion": "tkf.dev/v1alpha1",
            "kind": "PipelineRun",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
            },
            "spec": {
                "volume": vol_spec,
                "tasks": task_specs,
            },
        }

    def to_yaml(self) -> str:
        """Return manifest formatted as YAML."""
        return yaml.dump(self.to_manifest(), sort_keys=False)

    def run(
        self,
        local: bool = False,
        direct: bool = True,
        namespace: str | None = None,
        wait: bool = True,
        until: str | Task | None = None,
        from_task: str | Task | None = None,
    ) -> Any:
        """Execute the pipeline.

        - local=True: Runs locally on host with simulated /workspace.
        - direct=True (Default): Runs directly on Kubernetes Jobs without CRD/Operator.
        - direct=False: Submits a PipelineRun CustomResource to Kubernetes operator.
        - until: Executes tasks up to and including the target task/step, then stops.
        - from_task: Resumes execution starting from target task onward.
        """
        target_ns = namespace or self.namespace
        if local:
            return self._run_local(until=until, from_task=from_task)
        if direct:
            from tkf.runner import DirectRunner
            runner = DirectRunner(pipeline=self, namespace=target_ns)
            return runner.run(stream_logs=True)
        return self.submit(namespace=target_ns, wait=wait)

    def submit_job(
        self,
        namespace: str | None = None,
        service_account: str | None = None,
    ) -> str:
        """Submit the pipeline as a fire-and-forget in-cluster Launcher Job.

        You can safely close your terminal or disconnect from Codespaces.
        """
        from tkf.launcher import submit_launcher_job
        target_ns = namespace or self.namespace
        return submit_launcher_job(
            pipeline=self,
            namespace=target_ns,
            service_account=service_account,
        )

    def _run_local(
        self,
        until: str | Task | None = None,
        from_task: str | Task | None = None,
    ) -> bool:
        """Execute tasks locally sequentially in topological order with parameter substitution."""
        print(f"--- Starting Local Simulation: {self.name} ---")
        vol_path = Path(self.volume.local_path).resolve()
        vol_path.mkdir(parents=True, exist_ok=True)
        print(f"Shared Volume directory: {vol_path}")

        tasks = self.topological_sort()

        # Filter by until / from_task if requested
        if until is not None:
            ancestor_names = self.get_upstream_tasks(until)
            tasks = [t for t in tasks if t.name in ancestor_names]
            print(f"[Run Until Mode] Running {len(tasks)} prerequisite tasks up to '{until if isinstance(until, str) else until.name}'")

        if from_task is not None:
            descendant_names = self.get_downstream_tasks(from_task)
            tasks = [t for t in tasks if t.name in descendant_names]
            print(f"[Resume Mode] Resuming {len(tasks)} downstream tasks starting from '{from_task if isinstance(from_task, str) else from_task.name}'")

        task_outputs: dict[str, dict[str, str]] = collections.defaultdict(dict)

        for i, task in enumerate(tasks, start=1):
            print(f"\n[{i}/{len(tasks)}] Running task '{task.name}'...")

            def substitute(val: Any) -> str:
                s = str(val)
                def repl(match):
                    p_name, kind, o_name = match.group(1), match.group(2), match.group(3)
                    if kind == "artifacts":
                        p = vol_path / "runs" / self.name / "artifacts" / p_name / o_name
                        if not p.exists() and (vol_path / "artifacts" / p_name / o_name).exists():
                            return str(vol_path / "artifacts" / p_name / o_name)
                        return str(p)
                    if o_name in task_outputs[p_name]:
                        return task_outputs[p_name][o_name]
                    param_file = vol_path / "runs" / self.name / "outputs" / p_name / o_name
                    if not param_file.exists():
                        param_file = vol_path / ".tkf" / "outputs" / p_name / o_name
                    if param_file.exists():
                        return param_file.read_text().strip()
                    return match.group(0)
                return REF_REGEX.sub(repl, s)

            cmd = [substitute(x) for x in task.command]
            args = [substitute(x) for x in task.args]
            env = os.environ.copy()
            env["VOLUME"] = str(vol_path)
            env["TKF_TASK_NAME"] = task.name
            env["TKF_RUN_NAME"] = self.name
            for k, v in task.env.items():
                env[k] = substitute(v)

            # If task has packages, prefix with uv run --with ...
            if task.packages:
                with_args = [f"--with={pkg}" for pkg in task.packages]
                if cmd and cmd[0] in ("python", "python3"):
                    cmd = ["uv", "run"] + with_args + cmd

            full_cmd = cmd + args
            try:
                subprocess.run(full_cmd, check=True, env=env)
                print(f"-> Task '{task.name}' Succeeded.")

                task_out_dir = vol_path / ".tkf" / "outputs" / task.name
                if task_out_dir.exists():
                    for f in task_out_dir.iterdir():
                        if f.is_file():
                            task_outputs[task.name][f.name] = f.read_text().strip()
            except subprocess.CalledProcessError as e:
                print(f"-> Task '{task.name}' FAILED with code {e.returncode}")
                return False

        print(f"\n--- Local Simulation Finished Successfully ---")
        return True


class PipelineStepper:
    """Interactive execution stepper for debugging, pausing, and inspecting intermediate pipeline steps.

    Supports local host simulation as well as remote Kubernetes/cluster execution (sync or async).
    """

    def __init__(
        self,
        pipeline: Pipeline,
        local: bool = True,
        namespace: str | None = None,
    ):
        self.pipeline = pipeline
        self.local = local
        self.namespace = namespace or pipeline.namespace
        self.sorted_tasks = pipeline.topological_sort()
        self.executed_tasks: list[str] = []
        self.task_outputs: dict[str, dict[str, str]] = collections.defaultdict(dict)
        self.current_index: int = 0
        self.vol_path = Path(pipeline.volume.local_path).resolve()
        self.vol_path.mkdir(parents=True, exist_ok=True)

        if not self.local:
            from tkf.runner import DirectRunner
            self.runner = DirectRunner(pipeline=self.pipeline, namespace=self.namespace)
            self.pvc_name = self.runner._ensure_volume()
        else:
            self.runner = None
            self.pvc_name = None

    @property
    def current_task(self) -> Task | None:
        if self.current_index < len(self.sorted_tasks):
            return self.sorted_tasks[self.current_index]
        return None

    @property
    def is_finished(self) -> bool:
        return self.current_index >= len(self.sorted_tasks)

    def step(self, wait: bool = True, stream_logs: bool = True) -> Any:
        """Executes the next pending task in topological order.

        - local=True: runs task locally on host.
        - local=False and wait=True: dispatches to Kubernetes Job, streams pod logs, and blocks until finished.
        - local=False and wait=False: dispatches to Kubernetes Job asynchronously and returns RemoteTaskHandle immediately.
        """
        if self.is_finished:
            print("✔ All tasks in pipeline have already completed.")
            return None

        task = self.sorted_tasks[self.current_index]
        backend_name = "Local Simulation" if self.local else f"Remote K8s ({self.namespace})"
        print(f"\n[Step {self.current_index + 1}/{len(self.sorted_tasks)}] Executing task '{task.name}' via {backend_name}...")

        if self.local:
            success = self._execute_local_task(task)
            if not success:
                raise RuntimeError(f"Task '{task.name}' failed during interactive local step.")
            self.executed_tasks.append(task.name)
            self.current_index += 1
            return task
        else:
            return self._execute_remote_task(task, wait=wait, stream_logs=stream_logs)

    def step_async(self) -> Any:
        """Asynchronously dispatch the next task to the remote Kubernetes cluster without blocking."""
        return self.step(wait=False, stream_logs=False)

    def run_until(self, target: str | Task, stream_logs: bool = True) -> bool:
        """Executes tasks step-by-step until target task has completed, then pauses."""
        target_name = target.name if isinstance(target, Task) else to_k8s_name(target)
        # Ensure target exists
        self.pipeline.get_task(target_name)

        while not self.is_finished:
            task = self.step(wait=True, stream_logs=stream_logs)
            if task and getattr(task, "name", None) == target_name:
                print(f"\n⏸ Paused at target task '{target_name}'. You can inspect intermediate artifacts or call .step() / .run_remaining().")
                return True
        return True

    def run_remaining(self, stream_logs: bool = True) -> bool:
        """Runs all remaining tasks to completion."""
        while not self.is_finished:
            task = self.step(wait=True, stream_logs=stream_logs)
            if task is None:
                break
        print(f"\n✔ Pipeline '{self.pipeline.name}' finished completely.")
        return True

    def get_artifact_path(self, task_name: str | Task, filename: str) -> Path:
        """Get filesystem path of an artifact produced by a completed task."""
        tname = task_name.name if isinstance(task_name, Task) else to_k8s_name(task_name)
        p = self.vol_path / "runs" / self.pipeline.name / "artifacts" / tname / filename
        if not p.exists():
            p = self.vol_path / "artifacts" / tname / filename
        return p

    def get_output(self, task_name: str | Task, name: str) -> str | None:
        """Get output parameter produced by a completed task."""
        tname = task_name.name if isinstance(task_name, Task) else to_k8s_name(task_name)
        return self.task_outputs.get(tname, {}).get(name)

    def _execute_local_task(self, task: Task) -> bool:
        def substitute(val: Any) -> str:
            s = str(val)
            def repl(match):
                p_name, kind, o_name = match.group(1), match.group(2), match.group(3)
                if kind == "artifacts":
                    p = self.vol_path / "runs" / self.pipeline.name / "artifacts" / p_name / o_name
                    if not p.exists() and (self.vol_path / "artifacts" / p_name / o_name).exists():
                        return str(self.vol_path / "artifacts" / p_name / o_name)
                    return str(p)
                if o_name in self.task_outputs[p_name]:
                    return self.task_outputs[p_name][o_name]
                param_file = self.vol_path / "runs" / self.pipeline.name / "outputs" / p_name / o_name
                if not param_file.exists():
                    param_file = self.vol_path / ".tkf" / "outputs" / p_name / o_name
                if param_file.exists():
                    return param_file.read_text().strip()
                return match.group(0)
            return REF_REGEX.sub(repl, s)

        cmd = [substitute(x) for x in task.command]
        args = [substitute(x) for x in task.args]
        env = os.environ.copy()
        env["VOLUME"] = str(self.vol_path)
        env["TKF_TASK_NAME"] = task.name
        env["TKF_RUN_NAME"] = self.pipeline.name
        for k, v in task.env.items():
            env[k] = substitute(v)

        if task.packages:
            with_args = [f"--with={pkg}" for pkg in task.packages]
            if cmd and cmd[0] in ("python", "python3"):
                cmd = ["uv", "run"] + with_args + cmd

        full_cmd = cmd + args
        try:
            subprocess.run(full_cmd, check=True, env=env)
            print(f"-> Task '{task.name}' Succeeded.")
            task_out_dir = self.vol_path / ".tkf" / "outputs" / task.name
            if task_out_dir.exists():
                for f in task_out_dir.iterdir():
                    if f.is_file():
                        self.task_outputs[task.name][f.name] = f.read_text().strip()
            return True
        except subprocess.CalledProcessError as e:
            print(f"-> Task '{task.name}' FAILED with code {e.returncode}")
            return False

    def _execute_remote_task(self, task: Task, wait: bool = True, stream_logs: bool = True) -> Any:
        import asyncio
        from tkf.runner import RemoteTaskHandle

        if self.runner is None:
            raise RuntimeError("Runner is not initialized for remote execution.")

        self.runner.task_outputs.update(self.task_outputs)
        
        # 1. Dispatch the Kubernetes Job to the cluster
        job_name = self.runner.create_task_job(task, self.pvc_name)
        handle = RemoteTaskHandle(runner=self.runner, task=task, job_name=job_name)

        if wait:
            success = handle.wait(stream_logs=stream_logs)
            if not success:
                raise RuntimeError(f"Remote task '{task.name}' failed on cluster.")
            self.task_outputs.update(self.runner.task_outputs)
            self.executed_tasks.append(task.name)
            self.current_index += 1
            return task
        else:
            # Advance stepper index and return handle immediately
            self.executed_tasks.append(task.name)
            self.current_index += 1
            return handle

    def submit(self, namespace: str | None = None, wait: bool = True) -> dict[str, Any]:
        """Submit the PipelineRun CustomResource to Kubernetes."""
        try:
            k8s_config.load_kube_config()
        except Exception:
            k8s_config.load_incluster_config()

        custom_api = client.CustomObjectsApi()
        target_ns = namespace or self.namespace
        manifest = self.to_manifest()
        manifest["metadata"]["namespace"] = target_ns

        run_name = f"{self.name}-{uuid.uuid4().hex[:6]}"
        manifest["metadata"]["name"] = run_name

        print(f"Submitting PipelineRun '{run_name}' to namespace '{target_ns}'...")
        custom_api.create_namespaced_custom_object(
            group="tkf.dev",
            version="v1alpha1",
            namespace=target_ns,
            plural="pipelineruns",
            body=manifest,
        )
        print(f"PipelineRun '{run_name}' created successfully.")

        if wait:
            return self._watch_run(run_name, target_ns)
        return manifest

    def _watch_run(self, name: str, namespace: str) -> dict[str, Any]:
        """Watch a pipeline run until completion and print status."""
        custom_api = client.CustomObjectsApi()
        print(f"Waiting for PipelineRun '{name}' to complete...")

        last_phase = None
        while True:
            obj = custom_api.get_namespaced_custom_object(
                group="tkf.dev",
                version="v1alpha1",
                namespace=namespace,
                plural="pipelineruns",
                name=name,
            )
            status = obj.get("status", {})
            phase = status.get("phase", "Pending")
            tasks = status.get("tasks", {})

            if phase != last_phase:
                print(f"Status changed -> {phase}")
                last_phase = phase

            if phase in ("Succeeded", "Failed"):
                print(f"\nPipeline finished with phase: {phase}")
                for tname, tstatus in tasks.items():
                    t_phase = tstatus.get("phase", "Unknown")
                    outputs = tstatus.get("outputs", {})
                    out_str = f" | Outputs: {outputs}" if outputs else ""
                    print(f"  - Task {tname}: {t_phase}{out_str}")
                return obj

            time.sleep(2)
