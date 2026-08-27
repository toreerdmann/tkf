import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from tkf_models.contract import BaseTkfEstimator, ModelMetadata
from tkf_models.wrappers import SklearnModelWrapper
from tkf.pipeline import ComputeResources


def test_sklearn_model_wrapper_fit_predict(tmp_path):
    X = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    y = pl.Series("target", [10.0, 20.0, 30.0, 40.0])

    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    wrapper = SklearnModelWrapper(
        estimator=rf,
        target_col="target",
        resources=ComputeResources(cpu="1", memory="1Gi"),
    )

    # Fit
    wrapper.fit(X, y)
    assert wrapper._is_fitted

    # Predict
    preds = wrapper.predict(X)
    assert isinstance(preds, pl.DataFrame)
    assert "prediction" in preds.columns
    assert len(preds) == 4

    # Save & Load
    save_file = tmp_path / "rf_model.pkl"
    wrapper.save(save_file)
    assert save_file.exists()

    loaded = SklearnModelWrapper.load(save_file)
    assert loaded._is_fitted
    loaded_preds = loaded.predict(X)
    assert np.allclose(preds["prediction"].to_numpy(), loaded_preds["prediction"].to_numpy())


def test_sklearn_transformer_wrapper():
    X = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    scaler = StandardScaler()
    wrapper = SklearnModelWrapper(estimator=scaler)
    wrapper.fit(X)
    trans = wrapper.transform(X)
    assert isinstance(trans, pl.DataFrame)
    assert trans.shape == (4, 2)
