from __future__ import annotations

import copy
import textwrap
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

from tkf_models.contract import BaseTkfEstimator, ModelMetadata
from tkf.pipeline import ComputeResources, Pipeline, Task


def make_lagged_features(
    y: np.ndarray,
    lags: list[int],
    horizon: int = 1,
    X: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds autoregressive lagged feature matrix and direct step-h target.

    For horizon h, target at index t is y[t + h].
    Features at index t are [y[t - lag + 1] for lag in lags] (+ exogenous X[t] if provided).
    """
    max_lag = max(lags)
    n_samples = len(y) - max_lag - horizon + 1
    if n_samples <= 0:
        raise ValueError(f"Series length ({len(y)}) too short for max_lag={max_lag} and horizon={horizon}")

    X_lags = []
    for lag in lags:
        start = max_lag - lag
        end = start + n_samples
        X_lags.append(y[start:end, np.newaxis])

    X_feat = np.hstack(X_lags)
    if X is not None:
        X_exog = X[max_lag - 1 : max_lag - 1 + n_samples]
        X_feat = np.hstack([X_feat, X_exog])

    # Direct target at horizon step h
    target_start = max_lag + horizon - 1
    y_target = y[target_start : target_start + n_samples]

    return X_feat, y_target


class DirectForecaster(BaseTkfEstimator):
    """Direct Multi-Step Time Series Forecaster that trains an independent estimator per forecast horizon.

    Supports wrapping any scikit-learn / LightGBM regressor and can compile training
    into H parallel distributed Kubernetes tasks.
    """

    def __init__(
        self,
        estimator: Any,
        horizon: int = 12,
        lags: list[int] | None = None,
        target_col: str = "target",
        resources: ComputeResources | None = None,
        packages: list[str] | None = None,
        docker_image: str = "python:3.12-slim",
    ):
        name = f"direct-{getattr(estimator, '__class__', type(estimator)).__name__.lower()}"
        metadata = ModelMetadata(
            name=name,
            framework="sklearn",
            packages=packages or ["scikit-learn", "polars", "pyarrow", "joblib"],
            docker_image=docker_image,
            resources=resources or ComputeResources(cpu="1", memory="2Gi"),
            target_col=target_col,
        )
        super().__init__(metadata=metadata)

        self.base_estimator = estimator
        self.horizon = horizon
        self.lags = sorted(lags or [1, 2, 3, 6, 12])
        self.models: dict[int, Any] = {}
        self._last_window: np.ndarray | None = None

    def fit(self, X: Any = None, y: Any = None) -> "DirectForecaster":
        """Fits H independent models sequentially (interactive/local mode)."""
        y_arr = self._convert_input(y if y is not None else X)
        if y_arr is None:
            raise ValueError("Target series y must be provided.")

        y_flat = np.asarray(y_arr).squeeze()
        self._last_window = y_flat[-max(self.lags) :]

        for h in range(1, self.horizon + 1):
            X_h, y_h = make_lagged_features(y_flat, self.lags, horizon=h)
            model_h = copy.deepcopy(self.base_estimator)
            model_h.fit(X_h, y_h)
            self.models[h] = model_h

        self._is_fitted = True
        return self

    def predict(self, fh: int | None = None) -> pl.DataFrame:
        """Forecast the next `fh` steps (defaults to self.horizon)."""
        if not self._is_fitted:
            raise RuntimeError("DirectForecaster is not fitted yet. Call .fit() or load a fitted model.")

        h_to_pred = fh or self.horizon
        if self._last_window is None:
            raise RuntimeError("Missing historical context window for autoregressive forecasting.")

        preds = []
        # Build feature vector from last window
        # Features are: [last_window[-lag] for lag in lags]
        feat_vector = np.array([[self._last_window[-lag] for lag in self.lags]])

        for h in range(1, h_to_pred + 1):
            model = self.models.get(h)
            if model is None:
                raise ValueError(f"No trained sub-model found for horizon h={h}")
            pred_h = float(model.predict(feat_vector)[0])
            preds.append(pred_h)

        return pl.DataFrame({
            "horizon_step": list(range(1, h_to_pred + 1)),
            "forecast": preds,
        })

    def _convert_input(self, data: Any) -> np.ndarray:
        if hasattr(data, "to_numpy"):
            return data.to_numpy()
        return np.asarray(data)

    def save(self, path: str | Path) -> Path:
        import joblib
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "DirectForecaster":
        import joblib
        p = Path(path)
        return joblib.load(p)

    def compile_distributed_pipeline(
        self,
        dataset_path: str | Path,
        name: str = "direct-forecaster",
        target_column: str = "target",
    ) -> Pipeline:
        """Compiles the H horizon models into H parallel distributed tkf.Pipeline tasks."""
        import base64
        import pickle

        dag = Pipeline(name=name)

        # STAGE 1: Feature generation / lag extraction task
        prep_script = textwrap.dedent(f"""
import sys, os, numpy as np, polars as pl
from pathlib import Path
from tkf_models.forecasting import make_lagged_features

data_path = sys.argv[1]
target_col = sys.argv[2]
lags = {repr(self.lags)}
horizon = {self.horizon}

ws = Path(os.environ.get("VOLUME", "/workspace"))
run_name = os.environ.get("TKF_RUN_NAME")
task_name = os.environ.get("TKF_TASK_NAME", "prep-lags")

out_dir = (ws / "runs" / run_name / "artifacts" / task_name) if run_name else (ws / "artifacts" / task_name)
out_dir.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(data_path) if str(data_path).endswith(('.parquet', '.pq')) else pl.read_csv(data_path)
y_series = df[target_col].to_numpy()

# Save last window for inference
last_window = y_series[-max(lags):]
np.save(out_dir / "last_window.npy", last_window)

# Build & save datasets for each horizon h
for h in range(1, horizon + 1):
    X_h, y_h = make_lagged_features(y_series, lags, horizon=h)
    np.savez_compressed(out_dir / f"train_h{{h}}.npz", X=X_h, y=y_h)

print(f"Lagged matrices prepared for horizons 1..{{horizon}} at {{out_dir}}")
""").strip()

        prep_task = Task(
            name=f"{name}-prep-lags",
            docker_image=self.metadata.docker_image,
            packages=self.metadata.packages,
            command=["python3", "-c", prep_script],
            args=[str(dataset_path), target_column],
            resources=ComputeResources(cpu="1", memory="1Gi"),
        )
        dag.add_task(prep_task)

        # STAGE 2: H parallel fitting tasks (one per horizon step)
        fit_tasks = []
        serialized_base = base64.b64encode(pickle.dumps(self.base_estimator)).decode("ascii")

        for h in range(1, self.horizon + 1):
            fit_script = textwrap.dedent(f"""
import sys, os, numpy as np, joblib, base64, pickle
from pathlib import Path

h = {h}
ws = Path(os.environ.get("VOLUME", "/workspace"))
run_name = os.environ.get("TKF_RUN_NAME")
task_name = os.environ.get("TKF_TASK_NAME", f"fit-h{{h}}")

# Locate prep artifact
prep_dir = (ws / "runs" / run_name / "artifacts" / "{name}-prep-lags") if run_name else (ws / "artifacts" / "{name}-prep-lags")
data_file = prep_dir / f"train_h{{h}}.npz"
data = np.load(data_file)
X_h, y_h = data["X"], data["y"]

model_h = pickle.loads(base64.b64decode("{serialized_base}"))
model_h.fit(X_h, y_h)

out_dir = (ws / "runs" / run_name / "artifacts" / task_name) if run_name else (ws / "artifacts" / task_name)
out_dir.mkdir(parents=True, exist_ok=True)
model_out = out_dir / f"model_h{{h}}.pkl"
joblib.dump(model_h, model_out)
print(f"[Horizon h={{h}}] Model fitted and saved to {{model_out}}")
""").strip()

            h_task = Task(
                name=f"{name}-fit-h{h}",
                docker_image=self.metadata.docker_image,
                packages=self.metadata.packages,
                command=["python3", "-c", fit_script],
                resources=self.metadata.resources,
            )
            dag.add_task(h_task)
            dag.add_dependency(prep_task, h_task)
            fit_tasks.append(h_task)

        # STAGE 3: Aggregation & Multi-Step Prediction Task
        agg_script = textwrap.dedent(f"""
import sys, os, numpy as np, polars as pl, joblib
from pathlib import Path

horizon = {self.horizon}
lags = {repr(self.lags)}
ws = Path(os.environ.get("VOLUME", "/workspace"))
run_name = os.environ.get("TKF_RUN_NAME")
task_name = os.environ.get("TKF_TASK_NAME", "aggregate-forecasts")

prep_dir = (ws / "runs" / run_name / "artifacts" / "{name}-prep-lags") if run_name else (ws / "artifacts" / "{name}-prep-lags")
last_window = np.load(prep_dir / "last_window.npy")
feat_vector = np.array([[last_window[-lag] for lag in lags]])

predictions = []
for h in range(1, horizon + 1):
    h_dir = (ws / "runs" / run_name / "artifacts" / f"{name}-fit-h{{h}}") if run_name else (ws / "artifacts" / f"{name}-fit-h{{h}}")
    model_path = h_dir / f"model_h{{h}}.pkl"
    model = joblib.load(model_path)
    pred_val = float(model.predict(feat_vector)[0])
    predictions.append(pred_val)

out_df = pl.DataFrame({{
    "horizon_step": list(range(1, horizon + 1)),
    "forecast": predictions,
}})

out_dir = (ws / "runs" / run_name / "artifacts" / task_name) if run_name else (ws / "artifacts" / task_name)
out_dir.mkdir(parents=True, exist_ok=True)
out_parquet = out_dir / "direct_forecasts.parquet"
out_df.write_parquet(out_parquet)
print(f"Direct multi-step forecast (H={{horizon}}) assembled successfully at {{out_parquet}}:")
print(out_df)
""").strip()

        agg_task = Task(
            name=f"{name}-aggregate",
            docker_image=self.metadata.docker_image,
            packages=self.metadata.packages,
            command=["python3", "-c", agg_script],
            resources=ComputeResources(cpu="1", memory="1Gi"),
        )
        dag.add_task(agg_task)
        for ht in fit_tasks:
            dag.add_dependency(ht, agg_task)

        return dag
