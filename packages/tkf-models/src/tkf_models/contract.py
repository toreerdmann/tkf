from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Literal, TypeVar

from tkf.pipeline import ComputeResources, Task

T = TypeVar("T")


@dataclass
class ModelMetadata:
    """Metadata describing model capabilities, hardware requirements, and package dependencies."""

    name: str
    version: str = "0.1.0"
    framework: Literal["sklearn", "sktime", "torch", "lightgbm", "custom"] = "sklearn"
    packages: list[str] = field(default_factory=lambda: ["scikit-learn", "polars", "pyarrow", "joblib"])
    docker_image: str = "python:3.12-slim"
    resources: ComputeResources = field(default_factory=ComputeResources)
    features_in: list[str] = field(default_factory=list)
    target_col: str = "target"
    tags: dict[str, Any] = field(default_factory=dict)


class BaseTkfEstimator(abc.ABC):
    """Abstract Base Class defining the contract between ML Estimators and tkf execution engine."""

    def __init__(
        self,
        metadata: ModelMetadata | None = None,
        resources: ComputeResources | None = None,
        packages: list[str] | None = None,
        docker_image: str = "python:3.12-slim",
    ):
        self.metadata = metadata or ModelMetadata(name=self.__class__.__name__.lower())
        if resources:
            self.metadata.resources = resources
        if packages:
            self.metadata.packages = packages
        if docker_image:
            self.metadata.docker_image = docker_image
        self._is_fitted: bool = False

    @abc.abstractmethod
    def fit(self, X: Any, y: Any = None) -> "BaseTkfEstimator":
        """Fit estimator to data."""
        pass

    @abc.abstractmethod
    def predict(self, X: Any) -> Any:
        """Generate predictions given input features."""
        pass

    def transform(self, X: Any) -> Any:
        """Transform input data (optional for forecasters/classifiers, required for transformers)."""
        raise NotImplementedError("Transform is not supported by this estimator.")

    @abc.abstractmethod
    def save(self, path: str | Path) -> Path:
        """Serialize model state to disk (joblib, safetensors, onnx, etc.)."""
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str | Path) -> "BaseTkfEstimator":
        """Deserialize model state from disk."""
        pass

    def as_fit_task(
        self,
        name: str,
        train_data_path: str | Path | Any,
        target_col: str | None = None,
        output_model_filename: str = "model.pkl",
    ) -> Task:
        """Compile this estimator's fitting process into a containerized tkf Task."""
        from tkf_models.wrappers import create_fit_task
        return create_fit_task(
            estimator=self,
            name=name,
            train_data_path=train_data_path,
            target_col=target_col or self.metadata.target_col,
            output_model_filename=output_model_filename,
        )

    def as_predict_task(
        self,
        name: str,
        model_artifact_ref: Any,
        features_data_path: str | Path | Any,
        output_pred_filename: str = "predictions.parquet",
    ) -> Task:
        """Compile this estimator's inference process into a containerized tkf Task."""
        from tkf_models.wrappers import create_predict_task
        return create_predict_task(
            estimator=self,
            name=name,
            model_artifact_ref=model_artifact_ref,
            features_data_path=features_data_path,
            output_pred_filename=output_pred_filename,
        )
