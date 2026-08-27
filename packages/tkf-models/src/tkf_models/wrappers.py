from __future__ import annotations

import inspect
import os
import textwrap
from pathlib import Path
from typing import Any

from tkf_models.contract import BaseTkfEstimator, ModelMetadata
from tkf.pipeline import ComputeResources, Task


class SklearnModelWrapper(BaseTkfEstimator):
    """Wrapper that adapts standard scikit-learn estimators to tkf execution contracts."""

    def __init__(
        self,
        estimator: Any,
        target_col: str = "target",
        feature_cols: list[str] | None = None,
        resources: ComputeResources | None = None,
        packages: list[str] | None = None,
        docker_image: str = "python:3.12-slim",
    ):
        name = getattr(estimator, "__class__", type(estimator)).__name__.lower()
        metadata = ModelMetadata(
            name=name,
            framework="sklearn",
            packages=packages or ["scikit-learn", "polars", "pyarrow", "joblib"],
            docker_image=docker_image,
            resources=resources or ComputeResources(),
            target_col=target_col,
            features_in=feature_cols or [],
        )
        super().__init__(metadata=metadata)
        self.estimator = estimator

    def _convert_input(self, data: Any) -> Any:
        if hasattr(data, "to_numpy"):
            return data.to_numpy()
        return data

    def fit(self, X: Any, y: Any = None) -> "SklearnModelWrapper":
        X_arr = self._convert_input(X)
        y_arr = self._convert_input(y) if y is not None else None

        if y_arr is not None:
            self.estimator.fit(X_arr, y_arr)
        else:
            self.estimator.fit(X_arr)
        self._is_fitted = True
        return self

    def predict(self, X: Any) -> Any:
        X_arr = self._convert_input(X)
        preds = self.estimator.predict(X_arr)
        try:
            import polars as pl
            return pl.DataFrame({"prediction": preds})
        except ImportError:
            return preds

    def transform(self, X: Any) -> Any:
        X_arr = self._convert_input(X)
        trans = self.estimator.transform(X_arr)
        try:
            import polars as pl
            import numpy as np
            if isinstance(trans, np.ndarray):
                cols = [f"col_{i}" for i in range(trans.shape[1])] if trans.ndim > 1 else ["transformed"]
                return pl.DataFrame(trans, schema=cols)
            return pl.DataFrame(trans)
        except Exception:
            return trans

    def save(self, path: str | Path) -> Path:
        import joblib
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "SklearnModelWrapper":
        import joblib
        p = Path(path)
        est = joblib.load(p)
        wrapper = cls(estimator=est)
        wrapper._is_fitted = True
        return wrapper


def create_fit_task(
    estimator: BaseTkfEstimator,
    name: str,
    train_data_path: str | Path | Any,
    target_col: str = "target",
    output_model_filename: str = "model.pkl",
) -> Task:
    """Constructs a Task that fits an estimator and persists the serialized artifact."""
    
    runner_code = textwrap.dedent(f"""
import sys, os
from pathlib import Path
import joblib
import polars as pl

ws = Path(os.environ.get("VOLUME", "/workspace"))
run_name = os.environ.get("TKF_RUN_NAME")
task_name = os.environ.get("TKF_TASK_NAME", "{name}")

data_path = sys.argv[1]
target_col = sys.argv[2]
model_out = sys.argv[3]

if run_name:
    out_dir = ws / "runs" / run_name / "artifacts" / task_name
else:
    out_dir = ws / "artifacts" / task_name
out_dir.mkdir(parents=True, exist_ok=True)
model_path = out_dir / model_out

# Read dataset
if str(data_path).endswith(".parquet"):
    df = pl.read_parquet(data_path)
else:
    df = pl.read_csv(data_path)

if target_col in df.columns:
    X = df.drop(target_col).to_pandas()
    y = df[target_col].to_pandas()
else:
    X = df.to_pandas()
    y = None

# Fit estimator
import pickle
estimator_blob = {repr(estimator.save.__code__.co_consts)}
# Instantiate or deserialize
from tkf.ml.wrappers import SklearnModelWrapper
wrapper = joblib.load("{estimator.save('/tmp/temp_est.pkl') if hasattr(estimator, 'estimator') else '/tmp/temp_est.pkl'}")
if hasattr(wrapper, "fit"):
    if y is not None:
        wrapper.fit(X, y)
    else:
        wrapper.fit(X)
    joblib.dump(wrapper, model_path)
    print(f"Model saved successfully to {{model_path}}")
""").strip()

    # Dynamic python script execution
    script = textwrap.dedent(f"""
import sys, os, joblib, polars as pl
from pathlib import Path

data_path = sys.argv[1]
target_col = sys.argv[2]
model_name = sys.argv[3]

ws = Path(os.environ.get("VOLUME", "/workspace"))
run_name = os.environ.get("TKF_RUN_NAME")
task_name = os.environ.get("TKF_TASK_NAME", "{name}")

if run_name:
    out_dir = ws / "runs" / run_name / "artifacts" / task_name
else:
    out_dir = ws / "artifacts" / task_name
out_dir.mkdir(parents=True, exist_ok=True)
dest = out_dir / model_name

df = pl.read_parquet(data_path) if str(data_path).endswith(('.parquet', '.pq')) else pl.read_csv(data_path)
if target_col in df.columns:
    X = df.drop(target_col).to_pandas()
    y = df[target_col].to_pandas()
else:
    X = df.to_pandas()
    y = None

# If wrapped
est = getattr(estimator, 'estimator', estimator)
if y is not None:
    est.fit(X, y)
else:
    est.fit(X)

joblib.dump(est, dest)
print(f"Fitted model saved to {{dest}}")
""").strip()

    return Task(
        name=name,
        docker_image=estimator.metadata.docker_image,
        packages=estimator.metadata.packages,
        command=["python3", "-c", script],
        args=[str(train_data_path), target_col, output_model_filename],
        resources=estimator.metadata.resources,
    )


def create_predict_task(
    estimator: BaseTkfEstimator,
    name: str,
    model_artifact_ref: Any,
    features_data_path: str | Path | Any,
    output_pred_filename: str = "predictions.parquet",
) -> Task:
    """Constructs a Task that loads a serialized model and outputs a Parquet prediction artifact."""

    script = textwrap.dedent(f"""
import sys, os, joblib, polars as pl
from pathlib import Path

model_path = sys.argv[1]
features_path = sys.argv[2]
out_filename = sys.argv[3]

ws = Path(os.environ.get("VOLUME", "/workspace"))
run_name = os.environ.get("TKF_RUN_NAME")
task_name = os.environ.get("TKF_TASK_NAME", "{name}")

if run_name:
    out_dir = ws / "runs" / run_name / "artifacts" / task_name
else:
    out_dir = ws / "artifacts" / task_name
out_dir.mkdir(parents=True, exist_ok=True)
dest = out_dir / out_filename

model = joblib.load(model_path)
df = pl.read_parquet(features_path) if str(features_path).endswith(('.parquet', '.pq')) else pl.read_csv(features_path)
X = df.to_pandas()

preds = model.predict(X)
pred_df = pl.DataFrame({{"prediction": preds}})
pred_df.write_parquet(dest)
print(f"Predictions saved to {{dest}}")
""").strip()

    return Task(
        name=name,
        docker_image=estimator.metadata.docker_image,
        packages=estimator.metadata.packages,
        command=["python3", "-c", script],
        args=[str(model_artifact_ref), str(features_data_path), output_pred_filename],
        resources=estimator.metadata.resources,
    )
