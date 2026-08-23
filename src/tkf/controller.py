from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
import kopf
from kubernetes import client, config as k8s_config

# Load k8s configuration on import
try:
    k8s_config.load_incluster_config()
except Exception:
    k8s_config.load_kube_config()

REF_REGEX = re.compile(r"\{\{\s*tasks\.([a-zA-Z0-9_-]+)\.(outputs|artifacts)\.([a-zA-Z0-9_.-]+)\s*\}\}")


def get_iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def substitute_params(val: str, task_statuses: dict[str, Any], mount_path: str) -> str:
    """Replace {{ tasks.<parent>.outputs.<param> }} and {{ tasks.<parent>.artifacts.<file> }}."""
    def repl(match):
        p_name, kind, o_name = match.group(1), match.group(2), match.group(3)
        if kind == "artifacts":
            return f"{mount_path}/artifacts/{p_name}/{o_name}"
        p_outputs = task_statuses.get(p_name, {}).get("outputs", {})
        return str(p_outputs.get(o_name, match.group(0)))
    return REF_REGEX.sub(repl, str(val))


def ensure_pvc(spec: dict[str, Any], meta: dict[str, Any], logger: kopf.Logger) -> str | None:
    """Ensure shared PersistentVolumeClaim exists for the pipeline."""
    vol = spec.get("volume", {})
    if not vol.get("enabled", True):
        return None

    namespace = meta.get("namespace", "default")
    run_name = meta.get("name", "pipeline")
    pvc_name = vol.get("name") or f"{run_name}-pvc"
    size = vol.get("size", "1Gi")
    storage_class = vol.get("storageClass", "local-path")

    core_v1 = client.CoreV1Api()
    try:
        core_v1.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace)
        return pvc_name
    except client.exceptions.ApiException as e:
        if e.status != 404:
            logger.error(f"Error checking PVC {pvc_name}: {e}")
            raise

    logger.info(f"Creating shared PVC '{pvc_name}' ({size}) in namespace '{namespace}'...")
    pvc_body = client.V1PersistentVolumeClaim(
        metadata=client.V1ObjectMeta(
            name=pvc_name,
            namespace=namespace,
            labels={"tkf.dev/pipeline": run_name},
        ),
        spec=client.V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteOnce"],
            storage_class_name=storage_class,
            resources=client.V1VolumeResourceRequirements(
                requests={"storage": size}
            ),
        ),
    )
    kopf.adopt(pvc_body)
    core_v1.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc_body)
    logger.info(f"PVC '{pvc_name}' created successfully.")
    return pvc_name


def launch_task_job(
    task: dict[str, Any],
    pvc_name: str | None,
    spec: dict[str, Any],
    meta: dict[str, Any],
    task_statuses: dict[str, Any],
    logger: kopf.Logger,
) -> str:
    """Create a Kubernetes Job for an individual DAG task with resolved parameters."""
    batch_v1 = client.BatchV1Api()
    namespace = meta.get("namespace", "default")
    run_name = meta.get("name", "pipeline")
    task_name = task["name"]
    job_name = f"{run_name}-{task_name}"

    # Build container env and substitute parameters
    vol = spec.get("volume", {})
    mount_path = vol.get("mountPath", "/workspace")
    env_vars = [
        client.V1EnvVar(name="VOLUME", value=mount_path),
        client.V1EnvVar(name="TKF_TASK_NAME", value=task_name),
        client.V1EnvVar(name="TKF_RUN_NAME", value=run_name),
    ]
    for k, v in task.get("env", {}).items():
        env_vars.append(client.V1EnvVar(name=k, value=substitute_params(str(v), task_statuses, mount_path)))

    # Substitute parameters in command and args
    command = [substitute_params(x, task_statuses, mount_path) for x in task.get("command", [])] or None
    args = [substitute_params(x, task_statuses, mount_path) for x in task.get("args", [])] or None

    # Volumes and mounts
    volumes = []
    volume_mounts = []
    effective_pvc = pvc_name or vol.get("name") or (f"{run_name}-pvc" if vol.get("enabled", True) else None)
    if effective_pvc:
        volumes.append(
            client.V1Volume(
                name="tkf-shared-workspace",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=effective_pvc
                ),
            )
        )
        volume_mounts.append(
            client.V1VolumeMount(
                name="tkf-shared-workspace",
                mount_path=mount_path,
            )
        )

    # Resources
    resources_req = {}
    if task.get("cpu"):
        resources_req["cpu"] = str(task["cpu"])
    if task.get("memory"):
        resources_req["memory"] = str(task["memory"])

    resources = client.V1ResourceRequirements(
        requests=resources_req,
        limits=resources_req,
    ) if resources_req else None

    container = client.V1Container(
        name=task_name,
        image=task["image"],
        command=command,
        args=args,
        env=env_vars,
        volume_mounts=volume_mounts,
        resources=resources,
        image_pull_policy="IfNotPresent",
        termination_message_path="/dev/termination-log",
        termination_message_policy="File",
    )

    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=job_name,
            namespace=namespace,
            labels={
                "tkf.dev/pipeline": run_name,
                "tkf.dev/task": task_name,
            },
        ),
        spec=client.V1JobSpec(
            backoff_limit=0,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={
                        "tkf.dev/pipeline": run_name,
                        "tkf.dev/task": task_name,
                    }
                ),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[container],
                    volumes=volumes,
                ),
            ),
        ),
    )

    kopf.adopt(job)
    logger.info(f"Launching Job '{job_name}' for task '{task_name}'...")
    batch_v1.create_namespaced_job(namespace=namespace, body=job)
    return job_name


def extract_pod_outputs(job_name: str, namespace: str, logger: kopf.Logger) -> dict[str, str]:
    """Read output parameters from pod termination log."""
    core_v1 = client.CoreV1Api()
    try:
        pods = core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"batch.kubernetes.io/job-name={job_name}",
        )
        if not pods.items:
            return {}

        pod = pods.items[0]
        if pod.status and pod.status.container_statuses:
            terminated = pod.status.container_statuses[0].state.terminated
            if terminated and terminated.message:
                try:
                    return json.loads(terminated.message)
                except Exception:
                    return {"output": terminated.message.strip()}
    except Exception as e:
        logger.warning(f"Could not read outputs from pod for job {job_name}: {e}")
    return {}


@kopf.on.create("tkf.dev", "v1alpha1", "pipelineruns")
@kopf.on.resume("tkf.dev", "v1alpha1", "pipelineruns")
def on_pipelinerun_created(spec: dict[str, Any], meta: dict[str, Any], status: dict[str, Any], patch: kopf.Patch, logger: kopf.Logger, **_):
    """Initial handler when a PipelineRun is submitted."""
    pvc_name = ensure_pvc(spec, meta, logger)
    now = get_iso_timestamp()

    patch.status["phase"] = "Running"
    patch.status["startTime"] = status.get("startTime") or now
    patch.status["pvcName"] = pvc_name

    task_statuses = dict(status.get("tasks", {}))
    tasks = spec.get("tasks", [])

    for task in tasks:
        tname = task["name"]
        if tname not in task_statuses:
            task_statuses[tname] = {"phase": "Pending", "outputs": {}}

    # Find root tasks (no dependencies)
    for task in tasks:
        tname = task["name"]
        depends_on = task.get("dependsOn", [])
        if not depends_on and task_statuses[tname]["phase"] == "Pending":
            job_name = launch_task_job(task, pvc_name, spec, meta, task_statuses, logger)
            task_statuses[tname] = {
                "phase": "Running",
                "jobName": job_name,
                "startTime": now,
                "outputs": {},
            }

    patch.status["tasks"] = task_statuses


@kopf.on.timer("tkf.dev", "v1alpha1", "pipelineruns", interval=2.0)
def reconcile_pipelinerun(spec: dict[str, Any], meta: dict[str, Any], status: dict[str, Any], patch: kopf.Patch, logger: kopf.Logger, **_):
    """Periodic reconciler to sync Job statuses and advance the DAG."""
    phase = status.get("phase", "Pending")
    if phase in ("Succeeded", "Failed"):
        return

    namespace = meta.get("namespace", "default")
    run_name = meta.get("name", "pipeline")
    vol = spec.get("volume", {})
    pvc_name = status.get("pvcName") or vol.get("name") or (f"{run_name}-pvc" if vol.get("enabled", True) else None)
    tasks = spec.get("tasks", [])
    task_statuses = dict(status.get("tasks", {}))

    batch_v1 = client.BatchV1Api()
    core_v1 = client.CoreV1Api()
    now = get_iso_timestamp()
    changed = False

    # 1. Update status of Running tasks by querying their Job
    for task in tasks:
        tname = task["name"]
        t_stat = task_statuses.get(tname, {"phase": "Pending", "outputs": {}})

        if t_stat.get("phase") == "Running":
            job_name = t_stat.get("jobName") or f"{run_name}-{tname}"
            try:
                job = batch_v1.read_namespaced_job_status(name=job_name, namespace=namespace)
                if job.status.succeeded and job.status.succeeded > 0:
                    t_stat["phase"] = "Succeeded"
                    t_stat["completionTime"] = now
                    # Capture output parameters
                    outputs = extract_pod_outputs(job_name, namespace, logger)
                    if outputs:
                        t_stat["outputs"] = outputs
                        logger.info(f"Task '{tname}' recorded outputs: {outputs}")

                    changed = True
                    logger.info(f"Task '{tname}' (Job '{job_name}') Succeeded!")
                elif job.status.failed and job.status.failed > 0:
                    t_stat["phase"] = "Failed"
                    t_stat["completionTime"] = now
                    changed = True
                    logger.error(f"Task '{tname}' (Job '{job_name}') Failed!")
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    logger.warning(f"Job {job_name} not found yet.")

        task_statuses[tname] = t_stat

    # 2. Check for newly ready tasks (all parents Succeeded)
    for task in tasks:
        tname = task["name"]
        t_stat = task_statuses.get(tname, {"phase": "Pending", "outputs": {}})

        if t_stat.get("phase") == "Pending":
            depends_on = task.get("dependsOn", [])
            parents_succeeded = all(
                task_statuses.get(p, {}).get("phase") == "Succeeded" for p in depends_on
            )
            parents_failed = any(
                task_statuses.get(p, {}).get("phase") == "Failed" for p in depends_on
            )

            if parents_failed:
                t_stat["phase"] = "Skipped"
                t_stat["message"] = "Parent task failed"
                changed = True
            elif parents_succeeded:
                job_name = launch_task_job(task, pvc_name, spec, meta, task_statuses, logger)
                t_stat["phase"] = "Running"
                t_stat["jobName"] = job_name
                t_stat["startTime"] = now
                changed = True

        task_statuses[tname] = t_stat

    # 3. Check overall pipeline completion
    all_phases = [task_statuses.get(t["name"], {}).get("phase") for t in tasks]

    if any(p == "Failed" for p in all_phases):
        patch.status["phase"] = "Failed"
        patch.status["completionTime"] = now
        changed = True
        logger.error(f"PipelineRun '{run_name}' Failed.")
    elif all(p == "Succeeded" for p in all_phases):
        patch.status["phase"] = "Succeeded"
        patch.status["completionTime"] = now
        changed = True
        logger.info(f"PipelineRun '{run_name}' Succeeded completely!")

        # If temp volume, delete PVC
        if spec.get("volume", {}).get("temp", False) and pvc_name:
            logger.info(f"Cleaning up temporary PVC '{pvc_name}'...")
            try:
                core_v1.delete_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace)
            except Exception as e:
                logger.warning(f"Failed to delete temp PVC: {e}")

    if changed:
        patch.status["tasks"] = task_statuses


def start_controller():
    """Run the Kopf operator."""
    import kopf._core.engines.indexing
    kopf.run()


if __name__ == "__main__":
    start_controller()
