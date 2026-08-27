from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional
import kopf
import typer
from kubernetes import client, config as k8s_config
from rich.console import Console
from rich.table import Table

# Import controller to register kopf handlers
import tkf.controller  # noqa: F401

app = typer.Typer(help="tkf - Tiny Kubeflow CLI")
console = Console()


def init_k8s():
    try:
        k8s_config.load_incluster_config()
    except Exception:
        try:
            k8s_config.load_kube_config()
        except Exception as e:
            console.print(f"[red]Error loading kubeconfig: {e}[/red]")
            raise typer.Exit(code=1)


def get_default_namespace() -> str:
    """Detect current namespace from env or active kubeconfig context."""
    import os
    if os.environ.get("TKF_NAMESPACE"):
        return os.environ["TKF_NAMESPACE"]
    try:
        _, active_context = k8s_config.list_kube_config_contexts()
        if active_context and active_context.get("context", {}).get("namespace"):
            return active_context["context"]["namespace"]
    except Exception:
        pass
    return "default"


@app.command()
def controller(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Namespace to watch (all if not set)"),
):
    """Start the tkf operator / scheduler locally."""
    console.print(f"[bold green]Starting tkf controller (namespace={namespace or 'all'})...[/bold green]")
    namespaces = [namespace] if namespace else ()
    asyncio.run(
        kopf.operator(
            namespaces=namespaces,
            clusterwide=not bool(namespace),
        )
    )


@app.command("list")
def list_runs(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Kubernetes namespace"),
):
    """List PipelineRuns in the cluster."""
    init_k8s()
    ns = namespace or get_default_namespace()
    custom_api = client.CustomObjectsApi()
    
    try:
        res = custom_api.list_namespaced_custom_object(
            group="tkf.dev",
            version="v1alpha1",
            namespace=ns,
            plural="pipelineruns",
        )
    except client.exceptions.ApiException as e:
        console.print(f"[red]Failed to list pipelineruns: {e}[/red]")
        raise typer.Exit(code=1)

    items = res.get("items", [])
    if not items:
        console.print(f"[yellow]No PipelineRuns found in namespace '{ns}'.[/yellow]")
        return

    table = Table(title=f"PipelineRuns ({ns})")
    table.add_column("Name", style="bold cyan")
    table.add_column("Owner", style="magenta")
    table.add_column("Phase", style="bold")
    table.add_column("Tasks (Done/Total)")
    table.add_column("PVC")
    table.add_column("Start Time")

    for item in items:
        name = item["metadata"]["name"]
        labels = item["metadata"].get("labels", {})
        owner = labels.get("tkf/user", "-")
        status = item.get("status", {})
        phase = status.get("phase", "Pending")
        pvc = status.get("pvcName", "-")
        start_time = status.get("startTime", "-")
        
        spec_tasks = item.get("spec", {}).get("tasks", [])
        status_tasks = status.get("tasks", {})
        succeeded_count = sum(1 for t in status_tasks.values() if t.get("phase") == "Succeeded")
        task_str = f"{succeeded_count}/{len(spec_tasks)}"

        phase_color = {
            "Succeeded": "green",
            "Running": "blue",
            "Failed": "red",
            "Pending": "yellow",
        }.get(phase, "white")

        table.add_row(name, owner, f"[{phase_color}]{phase}[/{phase_color}]", task_str, pvc, str(start_time))

    console.print(table)


@app.command()
def status(
    name: str = typer.Argument(..., help="PipelineRun name"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Kubernetes namespace"),
):
    """Show detailed status of a PipelineRun and its DAG tasks."""
    init_k8s()
    ns = namespace or get_default_namespace()
    custom_api = client.CustomObjectsApi()
    
    try:
        obj = custom_api.get_namespaced_custom_object(
            group="tkf.dev",
            version="v1alpha1",
            namespace=ns,
            plural="pipelineruns",
            name=name,
        )
    except client.exceptions.ApiException as e:
        console.print(f"[red]PipelineRun '{name}' not found: {e}[/red]")
        raise typer.Exit(code=1)

    spec = obj.get("spec", {})
    status = obj.get("status", {})
    phase = status.get("phase", "Pending")
    
    console.print(f"\n[bold]PipelineRun:[/bold] [cyan]{name}[/cyan] (Namespace: {ns})")
    console.print(f"[bold]Phase:[/bold] {phase}")
    console.print(f"[bold]PVC:[/bold] {status.get('pvcName', 'None')}")
    console.print(f"[bold]Started:[/bold] {status.get('startTime', 'N/A')}")
    console.print(f"[bold]Completed:[/bold] {status.get('completionTime', 'N/A')}\n")

    table = Table(title="Tasks")
    table.add_column("Task Name", style="bold")
    table.add_column("Phase")
    table.add_column("Depends On")
    table.add_column("Job Name")
    table.add_column("Started")
    table.add_column("Completed")

    tasks_spec = {t["name"]: t for t in spec.get("tasks", [])}
    tasks_status = status.get("tasks", {})

    for tname, tspec in tasks_spec.items():
        tstat = tasks_status.get(tname, {})
        tphase = tstat.get("phase", "Pending")
        depends = ", ".join(tspec.get("dependsOn", [])) or "-"
        job = tstat.get("jobName", "-")
        tstart = tstat.get("startTime", "-")
        tcomp = tstat.get("completionTime", "-")

        color = {
            "Succeeded": "green",
            "Running": "blue",
            "Failed": "red",
            "Skipped": "magenta",
            "Pending": "yellow",
        }.get(tphase, "white")

        table.add_row(tname, f"[{color}]{tphase}[/{color}]", depends, job, str(tstart), str(tcomp))

    console.print(table)


def complete_tasks(incomplete: str = "") -> list[str]:
    """Auto-complete available task names and run IDs from the cluster."""
    try:
        init_k8s()
        ns = get_default_namespace()
        core_v1 = client.CoreV1Api()
        pods = core_v1.list_namespaced_pod(namespace=ns, label_selector="tkf/run")
        results = []
        seen = set()
        for p in pods.items:
            tname = p.metadata.labels.get("tkf/task") if p.metadata.labels else None
            rname = p.metadata.labels.get("tkf/run") if p.metadata.labels else None
            if tname and tname not in seen and tname.startswith(incomplete):
                seen.add(tname)
                results.append(tname)
            if rname and rname not in seen and rname.startswith(incomplete):
                seen.add(rname)
                results.append(rname)
        return results
    except Exception:
        return []


@app.command()
def logs(
    target: str = typer.Argument(..., help="Task name, pod name, or PipelineRun ID", autocompletion=complete_tasks),
    task_name: Optional[str] = typer.Argument(None, help="Optional specific task name (if target is PipelineRun ID)", autocompletion=complete_tasks),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Kubernetes namespace"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow / stream live log output"),
):
    """View container logs for a specific task job or pipeline run."""
    init_k8s()
    ns = namespace or get_default_namespace()
    core_v1 = client.CoreV1Api()
    
    # 1. Determine search criteria
    if task_name:
        pipeline_name, t_name = target, task_name
        label_selector = f"tkf/task={t_name}"
    else:
        pipeline_name, t_name = None, target
        label_selector = f"tkf/task={target}"

    pods = core_v1.list_namespaced_pod(namespace=ns, label_selector=label_selector)
    if not pods.items and not task_name:
        # Fallback: search by run ID
        pods = core_v1.list_namespaced_pod(namespace=ns, label_selector=f"tkf/run={target}")

    if pipeline_name and pods.items:
        pods.items = [
            p for p in pods.items
            if p.metadata.labels and (
                p.metadata.labels.get("tkf/run") == pipeline_name
                or p.metadata.labels.get("tkf/pipeline") == pipeline_name
                or p.metadata.labels.get("tkf.dev/pipeline") == pipeline_name
                or p.metadata.labels.get("tkf.dev/run") == pipeline_name
            )
        ]
    
    if not pods.items:
        import difflib
        console.print(f"[yellow]No active pods found matching '{target}' in namespace '{ns}'.[/yellow]\n")
        
        # Fetch all available tasks & runs to provide smart suggestions
        try:
            all_pods = core_v1.list_namespaced_pod(namespace=ns, label_selector="tkf/run")
            available_tasks = {
                p.metadata.labels.get("tkf/task"): p.metadata.labels.get("tkf/run")
                for p in all_pods.items if p.metadata.labels and p.metadata.labels.get("tkf/task")
            }
            available_runs = {
                p.metadata.labels.get("tkf/run")
                for p in all_pods.items if p.metadata.labels and p.metadata.labels.get("tkf/run")
            }
            
            candidates = list(available_tasks.keys()) + list(available_runs)
            matches = difflib.get_close_matches(target, [c for c in candidates if c], n=3, cutoff=0.3)
            if matches:
                console.print("[bold cyan]Did you mean:[/bold cyan]")
                for m in matches:
                    console.print(f"  * [green]{m}[/green]")
                console.print()

            if all_pods.items:
                table = Table(title=f"Available Pipeline Tasks in '{ns}'")
                table.add_column("Task Name", style="bold green")
                table.add_column("Pipeline Run", style="cyan")
                table.add_column("Status")
                for p in all_pods.items:
                    t = p.metadata.labels.get("tkf/task", "-") if p.metadata.labels else "-"
                    r = p.metadata.labels.get("tkf/run", "-") if p.metadata.labels else "-"
                    st = p.status.phase
                    table.add_row(t, r, st)
                console.print(table)
        except Exception:
            pass
        return

    for pod in pods.items:
        pod_name = pod.metadata.name
        task_label = pod.metadata.labels.get("tkf/task", pod.spec.containers[0].name) if pod.metadata.labels else pod.spec.containers[0].name
        console.print(f"[bold cyan]--- Logs for {task_label} (Pod: {pod_name}, Namespace: {ns}) ---[/bold cyan]\n")
        try:
            if follow:
                from kubernetes import watch
                w = watch.Watch()
                for line in w.stream(core_v1.read_namespaced_pod_log, name=pod_name, namespace=ns, container=task_label):
                    print(line)
            else:
                log_content = core_v1.read_namespaced_pod_log(name=pod_name, namespace=ns, container=task_label)
                console.print(log_content)
        except client.exceptions.ApiException as e:
            console.print(f"[red]Error fetching logs: {e}[/red]")


@app.command()
def delete(
    name: str = typer.Argument(..., help="PipelineRun name"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Kubernetes namespace"),
):
    """Delete a PipelineRun and its associated resources."""
    init_k8s()
    ns = namespace or get_default_namespace()
    custom_api = client.CustomObjectsApi()
    try:
        custom_api.delete_namespaced_custom_object(
            group="tkf.dev",
            version="v1alpha1",
            namespace=ns,
            plural="pipelineruns",
            name=name,
        )
        console.print(f"[green]Deleted PipelineRun '{name}'.[/green]")
    except client.exceptions.ApiException as e:
        console.print(f"[red]Error deleting PipelineRun: {e}[/red]")


def main():
    app()


if __name__ == "__main__":
    main()

@app.command()
def clean(
    run_id: Optional[str] = typer.Option(None, "--run", "-r", help="Clean only resources for a specific run ID", autocompletion=complete_tasks),
    user: Optional[str] = typer.Option(None, "--user", "-u", help="Target specific user (defaults to current user)"),
    all_users: bool = typer.Option(False, "--all-users", help="Clean resources across all users in namespace"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Kubernetes namespace"),
    all_jobs: bool = typer.Option(True, "--all", help="Delete all completed/failed tkf jobs and pods"),
    pvcs: bool = typer.Option(False, "--pvcs", help="Also delete temporary tkf PVCs labeled with tkf/run"),
):
    """Safely clean up completed, failed, or leftover tkf jobs, pods, and temporary PVCs."""
    init_k8s()
    ns = namespace or get_default_namespace()
    batch_v1 = client.BatchV1Api()
    core_v1 = client.CoreV1Api()
    
    from tkf.pipeline import get_current_user
    target_user = user or (None if (all_users or run_id) else get_current_user())
    
    if run_id:
        label_selector = f"tkf/run={run_id}"
    elif target_user:
        label_selector = f"tkf/user={target_user}"
    else:
        label_selector = "tkf/run"

    user_info = f" for user '{target_user}'" if target_user else " across all users"
    console.print(f"[yellow]Cleaning up tkf resources in namespace '{ns}'{user_info} (selector: {label_selector})...[/yellow]")
    
    # 1. Delete tkf Jobs (and their pods)
    jobs = batch_v1.list_namespaced_job(namespace=ns, label_selector=label_selector)
    # Also include launcher jobs
    if not run_id:
        launcher_sel = f"tkf/launcher,tkf/user={target_user}" if target_user else "tkf/launcher"
        launcher_jobs = batch_v1.list_namespaced_job(namespace=ns, label_selector=launcher_sel)
        seen_names = {j.metadata.name for j in jobs.items}
        for lj in launcher_jobs.items:
            if lj.metadata.name not in seen_names:
                jobs.items.append(lj)

    deleted_jobs = 0
    for j in jobs.items:
        jname = j.metadata.name
        try:
            batch_v1.delete_namespaced_job(name=jname, namespace=ns, propagation_policy="Foreground")
            console.print(f"  - Deleted Job: [dim]{jname}[/dim]")
            deleted_jobs += 1
        except Exception:
            pass

    if deleted_jobs == 0:
        console.print("  [dim]No tkf jobs found to clean.[/dim]")

    # 2. Optionally Delete only tkf temporary PVCs
    if pvcs:
        pvc_list = core_v1.list_namespaced_persistent_volume_claim(namespace=ns, label_selector=label_selector)
        deleted_pvcs = 0
        for p in pvc_list.items:
            pname = p.metadata.name
            try:
                core_v1.delete_namespaced_persistent_volume_claim(name=pname, namespace=ns)
                console.print(f"  - Deleted tkf PVC: [red]{pname}[/red]")
                deleted_pvcs += 1
            except Exception:
                pass
        if deleted_pvcs == 0:
            console.print("  [dim]No tkf PVCs found to clean.[/dim]")

    console.print(f"[green]✔ tkf cleanup in namespace '{ns}' complete![/green]")

