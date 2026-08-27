from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from tkf.pipeline import ComputeResources, Pipeline, Task


def compile_sklearn_pipeline(
    pipeline: Any,
    dataset_path: str | Path,
    target_column: str = "target",
    name: str = "sklearn-pipeline",
    packages: list[str] | None = None,
    resources_map: dict[str, ComputeResources] | None = None,
    docker_image: str = "python:3.12-slim",
) -> Pipeline:
    """Compiles a standard scikit-learn Pipeline into a modular, distributed tkf.Pipeline DAG.

    Each transformer step saves intermediate transformed features to Parquet and its fitted state to Joblib.
    The final estimator step fits on the final transformed feature matrix.
    """
    pkgs = packages or ["scikit-learn", "polars", "pyarrow", "joblib"]
    res_map = resources_map or {}

    dag = Pipeline(name=name)

    if not hasattr(pipeline, "steps"):
        raise TypeError("Expected an object with a 'steps' attribute (e.g. sklearn.pipeline.Pipeline).")

    steps = pipeline.steps
    prev_task: Task | None = None
    prev_data_ref: Any = str(dataset_path)

    for i, (step_name, transformer) in enumerate(steps):
        is_last_step = (i == len(steps) - 1)
        res = res_map.get(step_name, ComputeResources())

        import base64
        import pickle
        serialized_step = base64.b64encode(pickle.dumps(transformer)).decode("ascii")

        if is_last_step and (hasattr(transformer, "predict") and not hasattr(transformer, "transform")):
            # Final Estimator / Model (Fit only)
            script = textwrap.dedent(f"""
import sys, os, joblib, polars as pl, base64, pickle
from pathlib import Path

data_path = sys.argv[1]
target_col = sys.argv[2]
step_name = "{step_name}"

ws = Path(os.environ.get("VOLUME", "/workspace"))
run_name = os.environ.get("TKF_RUN_NAME")
task_name = os.environ.get("TKF_TASK_NAME", step_name)

if run_name:
    out_dir = ws / "runs" / run_name / "artifacts" / task_name
else:
    out_dir = ws / "artifacts" / task_name
out_dir.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(data_path) if str(data_path).endswith(('.parquet', '.pq')) else pl.read_csv(data_path)
if target_col in df.columns:
    X = df.drop(target_col).to_numpy()
    y = df[target_col].to_numpy()
else:
    X = df.to_numpy()
    y = None

# Unpack serialized estimator
model = pickle.loads(base64.b64decode("{serialized_step}"))
if y is not None:
    model.fit(X, y)
else:
    model.fit(X)

model_path = out_dir / "fitted_model.pkl"
joblib.dump(model, model_path)
print(f"[{step_name}] Fitted final model saved to {{model_path}}")
""").strip()

            step_task = Task(
                name=f"{name}-{step_name}",
                docker_image=docker_image,
                packages=pkgs,
                command=["python3", "-c", script],
                args=[prev_data_ref, target_column],
                resources=res,
            )
            dag.add_task(step_task)
            if prev_task:
                dag.add_dependency(prev_task, step_task)
            prev_task = step_task
        else:
            # Transformer Step (Fit & Transform)
            script = textwrap.dedent(f"""
import sys, os, joblib, polars as pl, numpy as np, base64, pickle
from pathlib import Path

data_path = sys.argv[1]
target_col = sys.argv[2]
step_name = "{step_name}"

ws = Path(os.environ.get("VOLUME", "/workspace"))
run_name = os.environ.get("TKF_RUN_NAME")
task_name = os.environ.get("TKF_TASK_NAME", step_name)

if run_name:
    out_dir = ws / "runs" / run_name / "artifacts" / task_name
else:
    out_dir = ws / "artifacts" / task_name
out_dir.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(data_path) if str(data_path).endswith(('.parquet', '.pq')) else pl.read_csv(data_path)
has_target = target_col in df.columns
if has_target:
    X = df.drop(target_col).to_numpy()
    y = df[target_col].to_numpy()
else:
    X = df.to_numpy()
    y = None

transformer = pickle.loads(base64.b64decode("{serialized_step}"))
if hasattr(transformer, "fit_transform"):
    X_trans = transformer.fit_transform(X, y) if y is not None else transformer.fit_transform(X)
else:
    transformer.fit(X, y) if y is not None else transformer.fit(X)
    X_trans = transformer.transform(X)

# Save transformer
joblib.dump(transformer, out_dir / "fitted_transformer.pkl")

# Save transformed dataset as Parquet
if isinstance(X_trans, np.ndarray):
    cols = [f"f_{{j}}" for j in range(X_trans.shape[1])] if X_trans.ndim > 1 else ["f_0"]
    out_df = pl.DataFrame(X_trans, schema=cols)
else:
    out_df = pl.DataFrame(X_trans)

if has_target:
    out_df = out_df.with_columns(pl.Series(target_col, y))

out_parquet = out_dir / "X_transformed.parquet"
out_df.write_parquet(out_parquet)
print(f"[{step_name}] Transformed data saved to {{out_parquet}} (shape: {{out_df.shape}})")
""").strip()

            step_task = Task(
                name=f"{name}-{step_name}",
                docker_image=docker_image,
                packages=pkgs,
                command=["python3", "-c", script],
                args=[prev_data_ref, target_column],
                resources=res,
            )
            dag.add_task(step_task)
            if prev_task:
                dag.add_dependency(prev_task, step_task)

            prev_task = step_task
            prev_data_ref = step_task.artifact("X_transformed.parquet")

    return dag
