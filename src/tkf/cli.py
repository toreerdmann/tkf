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


@app.command()
def list(
    namespace: str = typer.Option("tkf-dev", "--namespace", "-n", help="Kubernetes namespace"),
):
    """List PipelineRuns in the cluster."""
    init_k8s()
    custom_api = client.CustomObjectsApi()
    
    try:
        res = custom_api.list_namespaced_custom_object(
            group="tkf.dev",
            version="v1alpha1",
            namespace=namespace,
            plural="pipelineruns",
        )
    except client.exceptions.ApiException as e:
        console.print(f"[red]Failed to list pipelineruns: {e}[/red]")
        raise typer.Exit(code=1)

    items = res.get("items", [])
    if not items:
        console.print(f"[yellow]No PipelineRuns found in namespace '{namespace}'.[/yellow]")
        return

    table = Table(title=f"PipelineRuns ({namespace})")
    table.add_column("Name", style="bold cyan")
    table.add_column("Phase", style="bold")
    table.add_column("Tasks (Done/Total)")
    table.add_column("PVC")
    table.add_column("Start Time")

    for item in items:
        name = item["metadata"]["name"]
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

        table.add_row(name, f"[{phase_color}]{phase}[/{phase_color}]", task_str, pvc, str(start_time))

    console.print(table)


@app.command()
def status(
    name: str = typer.Argument(..., help="PipelineRun name"),
    namespace: str = typer.Option("tkf-dev", "--namespace", "-n", help="Kubernetes namespace"),
):
    """Show detailed status of a PipelineRun and its DAG tasks."""
    init_k8s()
    custom_api = client.CustomObjectsApi()
    
    try:
        obj = custom_api.get_namespaced_custom_object(
            group="tkf.dev",
            version="v1alpha1",
            namespace=namespace,
            plural="pipelineruns",
            name=name,
        )
    except client.exceptions.ApiException as e:
        console.print(f"[red]PipelineRun '{name}' not found: {e}[/red]")
        raise typer.Exit(code=1)

    spec = obj.get("spec", {})
    status = obj.get("status", {})
    phase = status.get("phase", "Pending")
    
    console.print(f"\n[bold]PipelineRun:[/bold] [cyan]{name}[/cyan] (Namespace: {namespace})")
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


@app.command()
def logs(
    pipeline_name: str = typer.Argument(..., help="PipelineRun name"),
    task_name: str = typer.Argument(..., help="Task name"),
    namespace: str = typer.Option("tkf-dev", "--namespace", "-n", help="Kubernetes namespace"),
):
    """View container logs for a specific task job."""
    init_k8s()
    core_v1 = client.CoreV1Api()
    
    label_selector = f"tkf.dev/pipeline={pipeline_name},tkf.dev/task={task_name}"
    pods = core_v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
    
    if not pods.items:
        console.print(f"[yellow]No pods found for task '{task_name}' in pipeline '{pipeline_name}'.[/yellow]")
        return

    pod_name = pods.items[0].metadata.name
    console.print(f"[bold cyan]--- Logs for {task_name} (Pod: {pod_name}) ---[/bold cyan]\n")
    try:
        log_content = core_v1.read_namespaced_pod_log(name=pod_name, namespace=namespace, container=task_name)
        console.print(log_content)
    except client.exceptions.ApiException as e:
        console.print(f"[red]Error fetching logs: {e}[/red]")


@app.command()
def delete(
    name: str = typer.Argument(..., help="PipelineRun name"),
    namespace: str = typer.Option("tkf-dev", "--namespace", "-n", help="Kubernetes namespace"),
):
    """Delete a PipelineRun and its associated resources."""
    init_k8s()
    custom_api = client.CustomObjectsApi()
    try:
        custom_api.delete_namespaced_custom_object(
            group="tkf.dev",
            version="v1alpha1",
            namespace=namespace,
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
    namespace: str = typer.Option("tkf-dev", "--namespace", "-n", help="Kubernetes namespace"),
    all_jobs: bool = typer.Option(True, "--all", help="Delete all completed/failed jobs and pods"),
):
    """Clean up all completed, failed, or leftover jobs and pods in the namespace."""
    init_k8s()
    batch_v1 = client.BatchV1Api()
    core_v1 = client.CoreV1Api()
    
    console.print(f"[yellow]Cleaning up all Jobs and Pods in namespace '{namespace}'...[/yellow]")
    
    # 1. Delete Jobs
    jobs = batch_v1.list_namespaced_job(namespace=namespace)
    for j in jobs.items:
        jname = j.metadata.name
        try:
            batch_v1.delete_namespaced_job(name=jname, namespace=namespace, propagation_policy="Foreground")
            console.print(f"  - Deleted Job: [dim]{jname}[/dim]")
        except Exception:
            pass

    console.print(f"[green]✔ Namespace '{namespace}' cleaned up successfully![/green]")
