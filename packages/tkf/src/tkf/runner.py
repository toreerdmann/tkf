from __future__ import annotations

import asyncio
import collections
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kubernetes import client, config as k8s_config, watch
from rich.console import Console
from rich.status import Status
from rich.table import Table

if TYPE_CHECKING:
    from tkf.pipeline import Pipeline, Task

REF_REGEX = re.compile(r"\{\{\s*tasks\.([a-zA-Z0-9_-]+)\.(outputs|artifacts)\.([a-zA-Z0-9_.-]+)\s*\}\}")
console = Console()


class DirectRunner:
    """Executes a Pipeline directly against Kubernetes Jobs without requiring a CRD or Operator."""

    def __init__(self, pipeline: Pipeline, namespace: str = "default"):
        self.pipeline = pipeline
        self.namespace = namespace
        self.run_id = f"{pipeline.name}-{uuid.uuid4().hex[:6]}"
        self._init_k8s()
        
        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()
        
        # State tracking
        self.task_statuses: dict[str, str] = {}  # task_name -> "Pending" | "Running" | "Succeeded" | "Failed" | "Skipped"
        self.task_outputs: dict[str, dict[str, str]] = collections.defaultdict(dict)
        self.task_jobs: dict[str, str] = {}
        self._pvc_name: str | None = None

    def _init_k8s(self):
        try:
            k8s_config.load_incluster_config()
        except Exception:
            try:
                k8s_config.load_kube_config()
            except Exception as e:
                console.print(f"[red]Error loading Kubernetes config: {e}[/red]")
                raise

    def run(self, stream_logs: bool = True) -> bool:
        """Run the entire DAG pipeline synchronously, dispatching ready tasks in parallel."""
        console.print(f"\n[bold cyan]=== Starting Direct Pipeline Run: {self.run_id} ===[/bold cyan]")
        console.print(f"Target Namespace: [bold green]{self.namespace}[/bold green]\n")

        # 1. Setup shared PVC if volume enabled
        pvc_name = self._ensure_volume()
        
        # 2. Get all tasks in DAG
        tasks_map = {t.name: t for t in self.pipeline._nodes.values()}
        for tname in tasks_map:
            self.task_statuses[tname] = "Pending"

        # 3. DAG loop: run until all tasks reach terminal state
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(self._execute_dag(tasks_map, pvc_name, stream_logs))
        finally:
            loop.close()

        # 4. Optional cleanup if temp volume
        if self.pipeline.volume.temp and self.pipeline.volume.create_if_missing and pvc_name:
            console.print(f"[yellow]Cleaning up temporary PVC '{pvc_name}'...[/yellow]")
            try:
                self.core_v1.delete_namespaced_persistent_volume_claim(name=pvc_name, namespace=self.namespace)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not delete temp PVC: {e}[/yellow]")

        if success:
            console.print(f"\n[bold green]✔ Pipeline Run '{self.run_id}' Finished Successfully![/bold green]\n")
        else:
            console.print(f"\n[bold red]✖ Pipeline Run '{self.run_id}' FAILED.[/bold red]\n")
        return success

    def _ensure_volume(self) -> str | None:
        if self._pvc_name:
            return self._pvc_name

        vol = self.pipeline.volume
        if not vol.enabled:
            return None

        # 1. If explicit PVC name is provided (and not temp), verify it exists
        if vol.name and not vol.temp:
            pvc_name = vol.name
            try:
                self.core_v1.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=self.namespace)
                console.print(f"Using existing PVC: [bold]{pvc_name}[/bold]")
                return pvc_name
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    try:
                        pvcs = self.core_v1.list_namespaced_persistent_volume_claim(namespace=self.namespace)
                        available = [p.metadata.name for p in pvcs.items]
                    except Exception:
                        available = []
                    avail_str = f" Available PVCs in '{self.namespace}': {available}" if available else ""
                    raise RuntimeError(
                        f"PersistentVolumeClaim '{pvc_name}' does not exist in namespace '{self.namespace}'.{avail_str}\n"
                        f"To auto-provision a clean temporary volume instead, use VolumeConfig(temp=True, storage_class='...')."
                    ) from None
                raise

        # 2. Dynamic temporary or auto-created volume
        pvc_name = vol.name or (f"temp-{uuid.uuid4().hex[:8]}" if vol.temp else f"{self.pipeline.name}-pvc")
        
        try:
            self.core_v1.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=self.namespace)
            console.print(f"Using existing PVC: [bold]{pvc_name}[/bold]")
            self._pvc_name = pvc_name
            return pvc_name
        except client.exceptions.ApiException as e:
            if e.status != 404:
                raise
            if not vol.create_if_missing and not vol.temp:
                raise RuntimeError(f"PVC '{pvc_name}' not found and create_if_missing is False.")

        # Pre-flight check: Validate StorageClass exists in cluster if possible
        if vol.storage_class:
            try:
                storage_v1 = client.StorageV1Api()
                sc_list = storage_v1.list_storage_class()
                valid_scs = [sc.metadata.name for sc in sc_list.items]
                if valid_scs and vol.storage_class not in valid_scs:
                    raise RuntimeError(
                        f"StorageClass '{vol.storage_class}' is not available in the cluster.\n"
                        f"Available StorageClasses: {valid_scs}\n"
                        f"Please set VolumeConfig(storage_class='...') to a supported StorageClass (e.g. 'azurefile-csi')."
                    )
            except client.exceptions.ApiException as sc_err:
                if sc_err.status not in (403, 404):
                    raise

        console.print(f"Creating shared PVC '[bold]{pvc_name}[/bold]' ({vol.size}) in namespace '{self.namespace}'...")
        pvc_labels = {"tkf/run": self.run_id}
        pvc_labels.update(self.pipeline.labels)
        pvc = client.V1PersistentVolumeClaim(
            metadata=client.V1ObjectMeta(
                name=pvc_name,
                namespace=self.namespace,
                labels=pvc_labels,
            ),
            spec=client.V1PersistentVolumeClaimSpec(
                access_modes=["ReadWriteMany" if vol.storage_class != "local-path" else "ReadWriteOnce"],
                storage_class_name=vol.storage_class,
                resources=client.V1VolumeResourceRequirements(
                    requests={"storage": vol.size}
                ),
            ),
        )
        self.core_v1.create_namespaced_persistent_volume_claim(namespace=self.namespace, body=pvc)
        console.print(f"[green]PVC '{pvc_name}' created successfully.[/green]")
        self._pvc_name = pvc_name
        return pvc_name

    async def _execute_dag(self, tasks_map: dict[str, Task], pvc_name: str | None, stream_logs: bool) -> bool:
        """Async scheduler loop that dispatches parallel tasks as their dependencies resolve."""
        running_tasks: dict[str, asyncio.Task] = {}
        parents_map = collections.defaultdict(list)
        for pid, cids in self.pipeline._adj.items():
            parent_name = self.pipeline._nodes[pid].name
            for cid in cids:
                child_name = self.pipeline._nodes[cid].name
                parents_map[child_name].append(parent_name)

        while True:
            # 1. Check for newly ready tasks (all parents Succeeded)
            for tname, task in tasks_map.items():
                if self.task_statuses[tname] == "Pending":
                    parents = parents_map.get(tname, [])
                    if any(self.task_statuses.get(p) == "Failed" for p in parents):
                        self.task_statuses[tname] = "Skipped"
                        console.print(f"[magenta]Task '{tname}' Skipped (parent failed)[/magenta]")
                    elif all(self.task_statuses.get(p) == "Succeeded" for p in parents):
                        # Launch task!
                        self.task_statuses[tname] = "Running"
                        running_tasks[tname] = asyncio.create_task(
                            self._run_task_job(task, pvc_name, stream_logs)
                        )

            if not running_tasks:
                # No tasks are running. Are all terminal?
                terminal = {"Succeeded", "Failed", "Skipped"}
                if all(status in terminal for status in self.task_statuses.values()):
                    break
                await asyncio.sleep(0.5)
                continue

            # Wait for at least one running task to finish
            done, _ = await asyncio.wait(running_tasks.values(), return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                # Find task name
                finished_name = next(k for k, v in running_tasks.items() if v == t)
                del running_tasks[finished_name]
                task_success = t.result()
                self.task_statuses[finished_name] = "Succeeded" if task_success else "Failed"

        # Overall pipeline outcome
        all_succeeded = all(status == "Succeeded" for status in self.task_statuses.values())
        return all_succeeded

    def create_task_job(self, task: Task, pvc_name: str | None = None) -> str:
        """Submit a single Kubernetes Job to the cluster and return its job_name."""
        job_name = f"{self.run_id}-{task.name}"
        self.task_jobs[task.name] = job_name
        mount_path = self.pipeline.volume.mount_path

        # Parameter and artifact substitution
        def substitute(val: Any) -> str:
            s = str(val)
            def repl(match):
                p_name, kind, o_name = match.group(1), match.group(2), match.group(3)
                if kind == "artifacts":
                    return f"{mount_path}/runs/{self.run_id}/artifacts/{p_name}/{o_name}"
                return str(self.task_outputs.get(p_name, {}).get(o_name, match.group(0)))
            res = REF_REGEX.sub(repl, s)
            # Automatic translation from local workspace directory to remote PVC mount path
            local_vol_name = self.pipeline.volume.local_path
            if res.startswith(f"{local_vol_name}/") or res.startswith(f"./{local_vol_name}/"):
                rel = res.removeprefix("./").removeprefix(f"{local_vol_name}/")
                return f"{mount_path}/{rel}"
            return res

        # Build command and arguments
        # If task specifies `packages` and default image, wrap with `uv run --with pkg1,pkg2`
        command = [substitute(x) for x in task.command] or None
        args = [substitute(x) for x in task.args] or None
        image = task.docker_image

        if task.packages:
            # Use astral uv image if standard python image
            if image == "python:3.12-slim" or "uv" not in image:
                image = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
            with_args = [f"--with={pkg}" for pkg in task.packages]
            if command and command[0] in ("python", "python3"):
                command = ["uv", "run"] + with_args + command
            elif not command and args:
                command = ["uv", "run"] + with_args + args
                args = None

        # Build env vars
        env_vars = [
            client.V1EnvVar(name="VOLUME", value=mount_path),
            client.V1EnvVar(name="TKF_TASK_NAME", value=task.name),
            client.V1EnvVar(name="TKF_RUN_NAME", value=self.run_id),
        ]
        for k, v in task.env.items():
            env_vars.append(client.V1EnvVar(name=k, value=substitute(v)))

        # Volumes and mounts
        volumes = []
        volume_mounts = []
        actual_pvc = pvc_name or self._ensure_volume()
        if actual_pvc:
            volumes.append(
                client.V1Volume(
                    name="tkf-workspace",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=actual_pvc),
                )
            )
            volume_mounts.append(
                client.V1VolumeMount(name="tkf-workspace", mount_path=mount_path)
            )

        # Image pull secrets
        image_pull_secrets = [
            client.V1LocalObjectReference(name=s) for s in task.image_pull_secrets
        ] if task.image_pull_secrets else None

        # Resource limits
        resources_req = {}
        if task.resources.cpu:
            resources_req["cpu"] = str(task.resources.cpu)
        if task.resources.memory:
            resources_req["memory"] = str(task.resources.memory)
        resources = client.V1ResourceRequirements(
            requests=resources_req, limits=resources_req
        ) if resources_req else None

        container = client.V1Container(
            name=task.name,
            image=image,
            command=command,
            args=args,
            env=env_vars,
            volume_mounts=volume_mounts,
            resources=resources,
            image_pull_policy="IfNotPresent",
            termination_message_path="/dev/termination-log",
            termination_message_policy="File",
        )

        job_labels = {
            "tkf/run": self.run_id,
            "tkf/task": task.name,
        }
        job_labels.update(self.pipeline.labels)
        job_labels.update(task.labels)

        pod_annotations = {}
        if task.disable_istio:
            pod_annotations["sidecar.istio.io/inject"] = "false"
        pod_annotations.update(self.pipeline.annotations)
        pod_annotations.update(task.annotations)

        job = client.V1Job(
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self.namespace,
                labels=job_labels,
            ),
            spec=client.V1JobSpec(
                backoff_limit=0,
                ttl_seconds_after_finished=getattr(self.pipeline, "ttl_seconds_after_finished", 300),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels=job_labels,
                        annotations=pod_annotations if pod_annotations else None,
                    ),
                    spec=client.V1PodSpec(
                        restart_policy="Never",
                        containers=[container],
                        volumes=volumes,
                        image_pull_secrets=image_pull_secrets,
                    ),
                ),
            ),
        )

        console.print(f"[bold blue]▶ Dispatching Task Job:[/bold blue] [cyan]{task.name}[/cyan] ([dim]{job_name}[/dim])")
        self.batch_v1.create_namespaced_job(namespace=self.namespace, body=job)
        return job_name

    async def _run_task_job(self, task: Task, pvc_name: str | None, stream_logs: bool) -> bool:
        """Submit a single Kubernetes Job and wait for its completion."""
        job_name = self.create_task_job(task, pvc_name)
        return await self._wait_and_stream_task(task.name, job_name, stream_logs)

    async def _wait_and_stream_task(self, task_name: str, job_name: str, stream_logs: bool) -> bool:
        """Poll job and stream logs from the active pod with full diagnostic reporting on failure."""
        pod_name = None
        
        # 1. Find pod name
        for _ in range(30):
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"batch.kubernetes.io/job-name={job_name}",
            )
            if pods.items:
                pod_name = pods.items[0].metadata.name
                break
            await asyncio.sleep(1.0)

        if not pod_name:
            console.print(f"[bold red]✖ Could not find pod for Job '{job_name}'[/bold red]")
            return False

        # 2. Wait until container starts or terminates
        while True:
            try:
                pod = self.core_v1.read_namespaced_pod_status(name=pod_name, namespace=self.namespace)
                phase = pod.status.phase
                if phase in ("Running", "Succeeded", "Failed"):
                    break
                
                # Check for container waiting errors (e.g. ImagePullBackOff, ErrImagePull)
                if pod.status.container_statuses:
                    for cs in pod.status.container_statuses:
                        if cs.state.waiting and cs.state.waiting.reason in ("ImagePullBackOff", "ErrImagePull", "CrashLoopBackOff"):
                            console.print(f"[bold red]✖ Container error in pod {pod_name}: {cs.state.waiting.reason} ({cs.state.waiting.message})[/bold red]")
                            return False
            except Exception:
                pass
            await asyncio.sleep(1.0)

        # 3. Wait for final Job completion
        while True:
            job = self.batch_v1.read_namespaced_job_status(name=job_name, namespace=self.namespace)
            if (job.status.succeeded and job.status.succeeded > 0) or (job.status.failed and job.status.failed > 0):
                break
            await asyncio.sleep(1.0)

        # 4. Fetch logs after execution
        logs = ""
        try:
            raw_logs = self.core_v1.read_namespaced_pod_log(name=pod_name, namespace=self.namespace, container=task_name)
            logs = raw_logs.decode("utf-8") if isinstance(raw_logs, bytes) else str(raw_logs)
        except Exception:
            pass

        if logs.strip() and stream_logs:
            console.print(f"[dim]── Logs for {task_name} ──────────────────────────────────────[/dim]")
            console.print(logs.strip())
            console.print(f"[dim]─────────────────────────────────────────────────────────────[/dim]")

        # 5. Check outcome
        if job.status.succeeded and job.status.succeeded > 0:
            pod = self.core_v1.read_namespaced_pod(name=pod_name, namespace=self.namespace)
            if pod.status.container_statuses:
                term = pod.status.container_statuses[0].state.terminated
                if term and term.message:
                    try:
                        self.task_outputs[task_name] = json.loads(term.message)
                    except Exception:
                        self.task_outputs[task_name] = {"output": term.message.strip()}
            
            console.print(f"[bold green]✔ Task '{task_name}' Succeeded.[/bold green]")
            return True
        else:
            pod = self.core_v1.read_namespaced_pod(name=pod_name, namespace=self.namespace)
            exit_code = None
            reason = "Failed"
            if pod.status.container_statuses and pod.status.container_statuses[0].state.terminated:
                term = pod.status.container_statuses[0].state.terminated
                exit_code = term.exit_code
                reason = term.reason or "Failed"
            console.print(f"[bold red]✖ Task '{task_name}' FAILED (Reason: {reason}, ExitCode: {exit_code}).[/bold red]")
            return False


class RemoteTaskHandle:
    """Handle to a running or completed remote Kubernetes task job."""

    def __init__(self, runner: DirectRunner, task: Task, job_name: str):
        self.runner = runner
        self.task = task
        self.job_name = job_name

    def status(self) -> str:
        """Query Kubernetes Job status ('Pending', 'Running', 'Succeeded', 'Failed')."""
        try:
            job = self.runner.batch_v1.read_namespaced_job_status(name=self.job_name, namespace=self.runner.namespace)
            if job.status.succeeded and job.status.succeeded > 0:
                return "Succeeded"
            if job.status.failed and job.status.failed > 0:
                return "Failed"
            if job.status.active and job.status.active > 0:
                return "Running"
        except Exception:
            pass
        return "Pending"

    def logs(self) -> str:
        """Fetch remote logs for this task container."""
        pods = self.runner.core_v1.list_namespaced_pod(
            namespace=self.runner.namespace,
            label_selector=f"batch.kubernetes.io/job-name={self.job_name}",
        )
        if pods.items:
            pod_name = pods.items[0].metadata.name
            try:
                return self.runner.core_v1.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=self.runner.namespace,
                    container=self.task.name,
                )
            except Exception:
                return ""
        return ""

    def wait(self, stream_logs: bool = True) -> bool:
        """Synchronously wait for the remote task Job to complete and stream logs."""
        loop = asyncio.new_event_loop()
        try:
            success = loop.run_until_complete(self.runner._wait_and_stream_task(self.task.name, self.job_name, stream_logs))
            return success
        finally:
            loop.close()
