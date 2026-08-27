from __future__ import annotations

import base64
import json
import uuid
from typing import TYPE_CHECKING, Any

from kubernetes import client, config as k8s_config
from rich.console import Console

if TYPE_CHECKING:
    from tkf.pipeline import Pipeline

console = Console()


def submit_launcher_job(
    pipeline: Pipeline,
    namespace: str = "default",
    service_account: str | None = None,
) -> str:
    """Deploy a self-contained in-cluster Launcher Job that runs the DAG asynchronously."""
    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()

    batch_v1 = client.BatchV1Api()
    core_v1 = client.CoreV1Api()
    run_id = f"{pipeline.name}-{uuid.uuid4().hex[:6]}"
    launcher_job_name = f"launcher-{run_id}"

    # Volume Pre-Flight Validation & Provisioning
    vol = pipeline.volume
    pvc_name = None
    if vol.enabled:
        if vol.name and not vol.temp:
            # 1. Verify named PVC exists
            try:
                core_v1.read_namespaced_persistent_volume_claim(name=vol.name, namespace=namespace)
                pvc_name = vol.name
                console.print(f"Using existing PVC: [bold]{pvc_name}[/bold]")
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    try:
                        pvcs = [p.metadata.name for p in core_v1.list_namespaced_persistent_volume_claim(namespace=namespace).items]
                    except Exception:
                        pvcs = []
                    avail_str = f" Available in '{namespace}': {pvcs}" if pvcs else ""
                    raise RuntimeError(
                        f"PersistentVolumeClaim '{vol.name}' does not exist in namespace '{namespace}'.{avail_str}\n"
                        f"To auto-provision a clean temporary volume instead, use VolumeConfig(temp=True, storage_class='...')."
                    ) from None
                raise
        else:
            # 2. Dynamic temporary PVC creation
            pvc_name = vol.name or f"temp-{uuid.uuid4().hex[:8]}"
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

            pvc_labels = {"tkf/run": run_id, **pipeline.labels}
            pvc = client.V1PersistentVolumeClaim(
                metadata=client.V1ObjectMeta(
                    name=pvc_name,
                    namespace=namespace,
                    labels=pvc_labels,
                ),
                spec=client.V1PersistentVolumeClaimSpec(
                    access_modes=["ReadWriteMany" if vol.storage_class != "local-path" else "ReadWriteOnce"],
                    storage_class_name=vol.storage_class,
                    resources=client.V1VolumeResourceRequirements(requests={"storage": vol.size}),
                ),
            )
            core_v1.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc)
            console.print(f"Created temporary shared PVC: [bold green]{pvc_name}[/bold green]")

    # Serialize pipeline manifest with exact resolved PVC name
    manifest = pipeline.to_manifest()
    if pvc_name and "volume" in manifest.get("spec", {}):
        manifest["spec"]["volume"]["name"] = pvc_name
    spec_json = json.dumps(manifest)
    spec_b64 = base64.b64encode(spec_json.encode("utf-8")).decode("utf-8")

    # In-cluster standalone runner script
    launcher_script = f"""
import asyncio, base64, collections, json, os, re, sys, time
from kubernetes import client, config

config.load_incluster_config()
batch_v1 = client.BatchV1Api()
core_v1 = client.CoreV1Api()

spec_data = json.loads(base64.b64decode('{spec_b64}').decode('utf-8'))
vol_spec = spec_data.get('spec', {{}}).get('volume', {{}})
tasks_spec = spec_data.get('spec', {{}}).get('tasks', [])
namespace = '{namespace}'
run_id = '{run_id}'
mount_path = vol_spec.get('mountPath', '/workspace')
pvc_name = '{pvc_name}' if '{pvc_name}' != 'None' else None

print(f"=== Starting In-Cluster Pipeline: {{run_id}} ===")
print(f"Namespace: {{namespace}} | PVC: {{pvc_name}}")

task_statuses = {{t['name']: 'Pending' for t in tasks_spec}}
task_outputs = collections.defaultdict(dict)
running_tasks = {{}}
REF_REGEX = re.compile(r"\\{{\\{{\\s*tasks\\.([a-zA-Z0-9_-]+)\\.(outputs|artifacts)\\.([a-zA-Z0-9_.-]+)\\s*\\}}\\}}")

def substitute(val):
    s = str(val)
    def repl(m):
        p_name, kind, o_name = m.group(1), m.group(2), m.group(3)
        if kind == 'artifacts':
            return f"{{mount_path}}/runs/{{run_id}}/artifacts/{{p_name}}/{{o_name}}"
        return str(task_outputs.get(p_name, {{}}).get(o_name, m.group(0)))
    return REF_REGEX.sub(repl, s)

async def run_task(task):
    tname = task['name']
    job_name = f"{{run_id}}-{{tname}}"
    print(f"▶ [{{tname}}] Launching Kubernetes Job {{job_name}}...")

    command = [substitute(x) for x in task.get('command', [])] or None
    args = [substitute(x) for x in task.get('args', [])] or None
    image = task['image']
    pkgs = task.get('packages', [])

    if pkgs:
        if image == "python:3.12-slim" or "uv" not in image:
            image = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
        with_args = [f"--with={{p}}" for p in pkgs]
        if command and command[0] in ("python", "python3"):
            command = ["uv", "run"] + with_args + command
        elif not command and args:
            command = ["uv", "run"] + with_args + args
            args = None

    env_vars = [
        client.V1EnvVar(name="VOLUME", value=mount_path),
        client.V1EnvVar(name="TKF_TASK_NAME", value=tname),
        client.V1EnvVar(name="TKF_RUN_NAME", value=run_id),
    ]
    for k, v in task.get('env', {{}}).items():
        env_vars.append(client.V1EnvVar(name=k, value=substitute(v)))

    volumes = []
    volume_mounts = []
    if pvc_name:
        volumes.append(client.V1Volume(name="tkf-ws", persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=pvc_name)))
        volume_mounts.append(client.V1VolumeMount(name="tkf-ws", mount_path=mount_path))

    secrets = [client.V1LocalObjectReference(name=s) for s in task.get('imagePullSecrets', [])] if task.get('imagePullSecrets') else None

    resources_req = {{}}
    if task.get('cpu'): resources_req['cpu'] = str(task['cpu'])
    if task.get('memory'): resources_req['memory'] = str(task['memory'])
    resources = client.V1ResourceRequirements(requests=resources_req, limits=resources_req) if resources_req else None

    container = client.V1Container(
        name=tname, image=image, command=command, args=args, env=env_vars,
        volume_mounts=volume_mounts, resources=resources, image_pull_policy="IfNotPresent",
        termination_message_path="/dev/termination-log", termination_message_policy="File",
    )
    job = client.V1Job(
        metadata=client.V1ObjectMeta(name=job_name, namespace=namespace, labels={{"tkf/run": run_id, "tkf/task": tname}}),
        spec=client.V1JobSpec(
            backoff_limit=0,
            ttl_seconds_after_finished=300,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={{"tkf/run": run_id, "tkf/task": tname}},
                    annotations={{"sidecar.istio.io/inject": "false"}},
                ),
                spec=client.V1PodSpec(restart_policy="Never", containers=[container], volumes=volumes, image_pull_secrets=secrets)
            )
        )
    )
    batch_v1.create_namespaced_job(namespace=namespace, body=job)

    # Wait for completion
    while True:
        await asyncio.sleep(2.0)
        j = batch_v1.read_namespaced_job_status(name=job_name, namespace=namespace)
        if j.status.succeeded and j.status.succeeded > 0:
            print(f"✔ [{{tname}}] Succeeded!")
            try:
                pods = core_v1.list_namespaced_pod(namespace=namespace, label_selector=f"batch.kubernetes.io/job-name={{job_name}}")
                if pods.items and pods.items[0].status.container_statuses:
                    term = pods.items[0].status.container_statuses[0].state.terminated
                    if term and term.message:
                        task_outputs[tname] = json.loads(term.message)
            except Exception:
                pass
            return True
        elif j.status.failed and j.status.failed > 0:
            print(f"✖ [{{tname}}] FAILED!")
            return False

async def main():
    while True:
        # Find ready tasks
        for task in tasks_spec:
            tname = task['name']
            if task_statuses[tname] == 'Pending':
                parents = task.get('dependsOn', [])
                if any(task_statuses.get(p) == 'Failed' for p in parents):
                    task_statuses[tname] = 'Skipped'
                    print(f"⊘ [{{tname}}] Skipped (parent failed)")
                elif all(task_statuses.get(p) == 'Succeeded' for p in parents):
                    task_statuses[tname] = 'Running'
                    running_tasks[tname] = asyncio.create_task(run_task(task))

        if not running_tasks:
            if all(s in ('Succeeded', 'Failed', 'Skipped') for s in task_statuses.values()):
                break
            await asyncio.sleep(1.0)
            continue

        done, _ = await asyncio.wait(running_tasks.values(), return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            finished_name = next(k for k, v in running_tasks.items() if v == t)
            del running_tasks[finished_name]
            success = t.result()
            task_statuses[finished_name] = 'Succeeded' if success else 'Failed'

    all_success = all(s == 'Succeeded' for s in task_statuses.values())
    if vol_spec.get('temp') and pvc_name:
        print(f"Cleaning up temporary PVC '{{pvc_name}}'...")
        try:
            core_v1.delete_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace)
        except Exception as e:
            print(f"Warning: could not delete temp PVC: {{e}}")

    print(f"\\n=== Pipeline {{run_id}} Finished: {{'SUCCESS' if all_success else 'FAILED'}} ===")
    sys.exit(0 if all_success else 1)

asyncio.run(main())
""".strip()

    launcher_image = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
    container = client.V1Container(
        name="launcher",
        image=launcher_image,
        command=["uv", "run", "--with=kubernetes", "python3", "-c", launcher_script],
        env=[client.V1EnvVar(name="PYTHONUNBUFFERED", value="1")],
        image_pull_policy="IfNotPresent",
    )

    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=launcher_job_name,
            namespace=namespace,
            labels={"tkf/launcher": run_id},
        ),
        spec=client.V1JobSpec(
            backoff_limit=0,
            ttl_seconds_after_finished=300,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={"tkf/launcher": run_id},
                    annotations={"sidecar.istio.io/inject": "false"},
                ),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    service_account_name=service_account or "default",
                    containers=[container],
                ),
            ),
        ),
    )

    console.print(f"\n[bold green]Submitting In-Cluster Launcher Job:[/bold green] [cyan]{launcher_job_name}[/cyan]")
    console.print(f"Target Namespace: [bold]{namespace}[/bold]")
    batch_v1.create_namespaced_job(namespace=namespace, body=job)

    console.print(f"\n[green]✔ Pipeline Launcher Job launched in Kubernetes![/green]")
    console.print(f"You can now safely disconnect or close your laptop.")
    console.print(f"To check live logs anytime, run:\n  [bold cyan]kubectl logs -f job/{launcher_job_name} -n {namespace}[/bold cyan]\n")
    return launcher_job_name
