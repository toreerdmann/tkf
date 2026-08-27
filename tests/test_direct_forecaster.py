import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from tkf_models.forecasting import DirectForecaster, make_lagged_features
from tkf.pipeline import Pipeline


def test_make_lagged_features():
    # y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = np.arange(10, dtype=float)
    lags = [1, 2]
    # For horizon h=1:
    # max_lag=2 -> n_samples = 10 - 2 - 1 + 1 = 8
    # at sample 0 (index 2): lags are y[1], y[0], target is y[2]
    X, y_target = make_lagged_features(y, lags=lags, horizon=1)
    assert X.shape == (8, 2)
    assert len(y_target) == 8
    assert np.array_equal(X[0], [1.0, 0.0])
    assert y_target[0] == 2.0


def test_direct_forecaster_fit_and_predict(tmp_path):
    # Create synthetic autoregressive series (e.g. trend + noise)
    t = np.arange(60, dtype=float)
    y = 5.0 + 0.5 * t + np.sin(t / 2.0)

    base_est = HistGradientBoostingRegressor(max_iter=20, random_state=42)
    forecaster = DirectForecaster(
        estimator=base_est,
        horizon=12,
        lags=[1, 2, 3, 6, 12],
    )

    forecaster.fit(y=y)
    assert forecaster._is_fitted
    assert len(forecaster.models) == 12

    # Predict 12 months ahead
    preds = forecaster.predict(fh=12)
    assert isinstance(preds, pl.DataFrame)
    assert len(preds) == 12
    assert "horizon_step" in preds.columns
    assert "forecast" in preds.columns
    assert preds["horizon_step"].to_list() == list(range(1, 13))

    # Test serialization
    save_file = tmp_path / "direct_forecaster.pkl"
    forecaster.save(save_file)
    assert save_file.exists()

    loaded = DirectForecaster.load(save_file)
    assert loaded._is_fitted
    loaded_preds = loaded.predict(fh=12)
    assert np.allclose(preds["forecast"].to_numpy(), loaded_preds["forecast"].to_numpy())


def test_direct_forecaster_distributed_dag_generation():
    base_est = RandomForestRegressor(n_estimators=10, random_state=42)
    forecaster = DirectForecaster(
        estimator=base_est,
        horizon=12,
        lags=[1, 2, 3],
    )

    dag = forecaster.compile_distributed_pipeline(
        dataset_path="/workspace/data/monthly_sales.parquet",
        name="sales-12m-direct",
        target_column="sales",
    )

    assert isinstance(dag, Pipeline)
    tasks = dag.topological_sort()
    # 1 prep task + 12 parallel fit tasks + 1 aggregate task = 14 tasks
    assert len(tasks) == 14
    assert tasks[0].name == "sales-12m-direct-prep-lags"
    assert tasks[-1].name == "sales-12m-direct-aggregate"

    # Verify that all 12 fit tasks depend on prep task
    manifest = dag.to_manifest()
    task_specs = {t["name"]: t for t in manifest["spec"]["tasks"]}
    for h in range(1, 13):
        h_name = f"sales-12m-direct-fit-h{h}"
        assert task_specs[h_name]["dependsOn"] == ["sales-12m-direct-prep-lags"]

    # Verify aggregate depends on all 12 fit tasks
    expected_parents = [f"sales-12m-direct-fit-h{h}" for h in range(1, 13)]
    assert sorted(task_specs["sales-12m-direct-aggregate"]["dependsOn"]) == sorted(expected_parents)
