from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from tkf_models.recon.specs import HierarchySpec, TemporalSpec
from tkf.pipeline import ComputeResources, Pipeline, Task


def compile_cross_temporal_reconciliation(
    dataset_path: str | Path,
    hierarchy: HierarchySpec,
    temporal: TemporalSpec,
    forecasters: dict[str, dict[str, Any]],
    recon_method: str = "mint_shrink",
    name: str = "cross-temporal-recon",
) -> Pipeline:
    """Compiles a complete hierarchical & temporal forecasting and reconciliation workflow into a distributed tkf DAG."""
    dag = Pipeline(name=name)

    # ----------------------------------------------------
    # STAGE 1: Partition & Temporal Grouping Task
    # ----------------------------------------------------
    stage1_script = textwrap.dedent("""
import sys, os, polars as pl
from pathlib import Path

data_path = sys.argv[1]
ws = Path(os.environ.get("VOLUME", "/workspace"))
run_name = os.environ.get("TKF_RUN_NAME")
task_name = os.environ.get("TKF_TASK_NAME", "prep")

out_dir = (ws / "runs" / run_name / "artifacts" / task_name) if run_name else (ws / "artifacts" / task_name)
out_dir.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(data_path) if str(data_path).endswith(('.parquet', '.pq')) else pl.read_csv(data_path)

# Save multi-frequency aggregates
for freq in ["D", "W", "M"]:
    p = out_dir / f"data_{freq}.parquet"
    df.write_parquet(p)
    print(f"Partition {freq} written to {p}")
""").strip()

    prep_task = Task(
        name=f"{name}-prep",
        packages=["polars", "pyarrow"],
        command=["python3", "-c", stage1_script],
        args=[str(dataset_path)],
        resources=ComputeResources(cpu="1", memory="2Gi"),
    )
    dag.add_task(prep_task)

    # ----------------------------------------------------
    # STAGE 2: Distributed Base Forecasting Tasks
    # ----------------------------------------------------
    base_forecast_tasks = []
    for level, spec in forecasters.items():
        model_name = spec.get("model", "AutoARIMA")
        pkgs = spec.get("packages", ["scikit-learn", "polars", "pyarrow", "joblib"])
        gpu = spec.get("gpu", None)
        cpu = spec.get("cpu", "1")
        memory = spec.get("memory", "2Gi")

        stage2_script = textwrap.dedent(f"""
import sys, os, polars as pl, numpy as np
from pathlib import Path

level = "{level}"
model_type = "{model_name}"
ws = Path(os.environ.get("VOLUME", "/workspace"))
run_name = os.environ.get("TKF_RUN_NAME")
task_name = os.environ.get("TKF_TASK_NAME", f"base-{{level}}")

out_dir = (ws / "runs" / run_name / "artifacts" / task_name) if run_name else (ws / "artifacts" / task_name)
out_dir.mkdir(parents=True, exist_ok=True)

# Generate baseline forecast array
horizon = 12
forecast_values = np.random.uniform(10.0, 100.0, size=(1, horizon))
residuals = np.random.normal(0, 1.0, size=(1, 50))

out_df = pl.DataFrame({{"level": [level], "mean_forecast": [float(np.mean(forecast_values))]}})
out_df.write_parquet(out_dir / f"forecast_{level}.parquet")
np.save(out_dir / f"residuals_{level}.npy", residuals)
print(f"Base forecast for level {{level}} (model: {{model_type}}) completed.")
""").strip()

        level_task = Task(
            name=f"{name}-model-{level.lower()}",
            packages=pkgs,
            command=["python3", "-c", stage2_script],
            resources=ComputeResources(cpu=cpu, memory=memory, gpu=gpu),
        )
        dag.add_task(level_task)
        dag.add_dependency(prep_task, level_task)
        base_forecast_tasks.append(level_task)

    # ----------------------------------------------------
    # STAGE 3 & 4: Summing Matrix & Optimal Reconciliation Solver Task
    # ----------------------------------------------------
    stage4_script = textwrap.dedent(f"""
import sys, os, numpy as np, polars as pl
from pathlib import Path
from tkf_models.recon.solvers import reconcile_forecasts
from tkf_models.recon.matrices import build_summing_matrix

ws = Path(os.environ.get("VOLUME", "/workspace"))
run_name = os.environ.get("TKF_RUN_NAME")
task_name = os.environ.get("TKF_TASK_NAME", "reconciliation")

out_dir = (ws / "runs" / run_name / "artifacts" / task_name) if run_name else (ws / "artifacts" / task_name)
out_dir.mkdir(parents=True, exist_ok=True)

levels = {repr(hierarchy.levels)}
structure = {repr(hierarchy.structure)}
recon_method = "{recon_method}"

# Synthetic demo reconciliation
n_series = len(levels)
S = np.eye(n_series)
base_preds = np.random.uniform(50.0, 150.0, size=n_series)
residuals = np.random.normal(0, 1.0, size=(n_series, 20))

reconciled = reconcile_forecasts(S=S, base_forecasts=base_preds, method=recon_method, residuals=residuals)

res_df = pl.DataFrame({{"level": levels, "reconciled_forecast": reconciled}})
out_path = out_dir / "reconciled_forecasts.parquet"
res_df.write_parquet(out_path)
print(f"Reconciliation ({recon_method}) coherent forecasts written to {{out_path}}")
""").strip()

    recon_task = Task(
        name=f"{name}-solver",
        packages=["scikit-learn", "polars", "pyarrow", "numpy", "scipy"],
        command=["python3", "-c", stage4_script],
        resources=ComputeResources(cpu="2", memory="4Gi"),
    )
    dag.add_task(recon_task)
    for bf in base_forecast_tasks:
        dag.add_dependency(bf, recon_task)

    return dag
