from __future__ import annotations

import concurrent.futures
import functools
import inspect
import json
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar, Union

from tkf.io import OutputRef, set_output, artifact_path, dataset_path, model_path
from tkf.pipeline import ComputeResources, Task

T = TypeVar("T")


def is_local() -> bool:
    """Returns True if executing interactively on the local machine / notebook, False inside Kubernetes."""
    return os.environ.get("TKF_EXECUTION_MODE") != "cluster" and os.environ.get("KUBERNETES_SERVICE_HOST") is None


def get_default_workspace() -> Path:
    """Get the active workspace directory, falling back to local_pipeline_volume on host."""
    vol = os.environ.get("VOLUME")
    if vol:
        return Path(vol)
    p = Path("local_pipeline_volume").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# High-Level ML Artifact Wrappers
# ---------------------------------------------------------------------------


class Artifact(Generic[T]):
    """Represents a file or directory artifact produced or consumed by a task."""

    def __init__(self, value: T | None = None, filename: str = "artifact.bin", path: str | Path | None = None):
        self._value = value
        self.filename = filename
        self._path = Path(path) if path else None

    @property
    def path(self) -> Path:
        if self._path:
            return self._path
        vol = get_default_workspace()
        run_name = os.environ.get("TKF_RUN_NAME")
        task_name = os.environ.get("TKF_TASK_NAME", "default")
        if run_name:
            p = vol / "runs" / run_name / "artifacts" / task_name / self.filename
        else:
            p = vol / "artifacts" / task_name / self.filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def load(self) -> T:
        """Load the artifact content into memory."""
        if self._value is not None:
            return self._value
        if self.path.exists():
            return self._load_from_path(self.path)
        raise FileNotFoundError(f"Artifact path '{self.path}' does not exist.")

    def _load_from_path(self, p: Path) -> Any:
        return p.read_bytes()

    def save(self, target_path: Path | None = None) -> Path:
        """Serialize the in-memory value to disk."""
        dest = target_path or self.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(self._value, (bytes, bytearray)):
            dest.write_bytes(self._value)
        elif isinstance(self._value, str):
            dest.write_text(self._value)
        return dest

    def __str__(self) -> str:
        return str(self.path)


class Dataset(Artifact[Any]):
    """Specialized Artifact for Tabular / Time-Series Datasets."""

    def __init__(self, value: Any = None, filename: str = "dataset.parquet", path: str | Path | None = None):
        super().__init__(value=value, filename=filename, path=path)

    def _load_from_path(self, p: Path) -> Any:
        suffix = p.suffix.lower()
        if suffix in (".parquet", ".pq"):
            try:
                import polars as pl
                return pl.read_parquet(p)
            except ImportError:
                import pandas as pd
                return pd.read_parquet(p)
        elif suffix in (".csv", ".txt"):
            try:
                import polars as pl
                return pl.read_csv(p)
            except ImportError:
                import pandas as pd
                return pd.read_csv(p)
        return p.read_text()

    def save(self, target_path: Path | None = None) -> Path:
        dest = target_path or self.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        val = self._value

        if hasattr(val, "write_parquet") and dest.suffix == ".parquet":
            val.write_parquet(dest)
        elif hasattr(val, "to_parquet") and dest.suffix == ".parquet":
            val.to_parquet(dest)
        elif hasattr(val, "write_csv") and dest.suffix == ".csv":
            val.write_csv(dest)
        elif hasattr(val, "to_csv") and dest.suffix == ".csv":
            val.to_csv(dest, index=False)
        else:
            super().save(dest)
        return dest


class Model(Artifact[Any]):
    """Specialized Artifact for ML Models."""

    def __init__(self, value: Any = None, filename: str = "model.pkl", path: str | Path | None = None):
        super().__init__(value=value, filename=filename, path=path)

    def _load_from_path(self, p: Path) -> Any:
        import joblib
        return joblib.load(p)

    def save(self, target_path: Path | None = None) -> Path:
        dest = target_path or self.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(self._value, dest)
        return dest


# ---------------------------------------------------------------------------
# Task Decorator & Callable Wrapper (with .map() support!)
# ---------------------------------------------------------------------------


class TaskCallable:
    """Wraps a Python function to support interactive execution, .map() fan-out, and K8s compilation."""

    def __init__(
        self,
        fn: Callable,
        name: str | None = None,
        packages: list[str] | None = None,
        docker_image: str = "python:3.12-slim",
        resources: ComputeResources | None = None,
        helpers: list[Callable] | None = None,
    ):
        self.fn = fn
        self.name = name or fn.__name__.replace("_", "-")
        self.packages = packages or []
        self.docker_image = docker_image
        self.resources = resources or ComputeResources()
        self.helpers = helpers or []
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs) -> Any:
        """Direct execution mode (e.g. interactive notebook cell)."""
        return self.fn(*args, **kwargs)

    def map(self, iterable: list[Any], **kwargs) -> list[Any]:
        """Dynamic fan-out: runs across an iterable in parallel or in loop."""
        if is_local():
            # Local Interactive / Marimo Mode: run directly
            results = []
            for item in iterable:
                res = self.fn(item, **kwargs)
                results.append(res)
            return results
        else:
            # Compiled DAG Mode: returns list of Tasks
            return [self.to_task(item, **kwargs, name_suffix=str(i)) for i, item in enumerate(iterable)]

    def to_task(self, *args, name_suffix: str | None = None, **kwargs) -> Task:
        """Compile this function into a containerized Kubernetes Task."""
        # 1. Extract helper functions
        helper_sources = []
        # Auto-discover helpers in same module if not explicitly given
        target_mod = inspect.getmodule(self.fn)
        all_helpers = list(self.helpers)
        if not all_helpers and target_mod:
            for name, val in self.fn.__globals__.items():
                if callable(val) and inspect.isfunction(val) and val != self.fn:
                    if inspect.getmodule(val) == target_mod and not name.startswith("_"):
                        all_helpers.append(val)

        for h in all_helpers:
            try:
                h_src = inspect.getsource(h)
                h_lines = [l for l in h_src.splitlines() if not l.strip().startswith("@task")]
                helper_sources.append(textwrap.dedent("\n".join(h_lines)).strip())
            except Exception:
                pass

        # 2. Extract main function source (skipping all decorator lines)
        src = inspect.getsource(self.fn)
        lines = src.splitlines()
        start_idx = 0
        for idx, l in enumerate(lines):
            if l.strip().startswith(("def ", "async def ")):
                start_idx = idx
                break
        clean_src = textwrap.dedent("\n".join(lines[start_idx:])).strip()

        combined_src = "\n\n".join(helper_sources + [clean_src])

        tname = f"{self.name}-{name_suffix}" if name_suffix else self.name

        preamble = textwrap.dedent("""
import sys, os, json
from pathlib import Path

try:
    import polars as pl
except ImportError:
    pass

try:
    import pandas as pd
except ImportError:
    pass

try:
    import numpy as np
except ImportError:
    pass

def _get_ws():
    return Path(os.environ.get("VOLUME", "/workspace"))

def is_local():
    return False

def _set_output(name, val):
    out = {name: str(val)}
    try:
        cur = {}
        tlog = Path("/dev/termination-log")
        if tlog.exists() and tlog.read_text().strip():
            cur = json.loads(tlog.read_text())
        cur.update(out)
        tlog.write_text(json.dumps(cur))
    except Exception:
        pass
    rname = os.environ.get("TKF_RUN_NAME")
    if rname:
        p = _get_ws() / "runs" / rname / "outputs" / os.environ.get("TKF_TASK_NAME", "default") / name
    else:
        p = _get_ws() / ".tkf" / "outputs" / os.environ.get("TKF_TASK_NAME", "default") / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(val))

class Artifact:
    def __init__(self, value=None, filename="artifact.bin", path=None):
        self._value = value
        self.filename = filename
        self._path = Path(path) if path else None

    @property
    def path(self):
        if self._path: return self._path
        tname = os.environ.get("TKF_TASK_NAME", "default")
        rname = os.environ.get("TKF_RUN_NAME")
        if rname:
            p = _get_ws() / "runs" / rname / "artifacts" / tname / self.filename
        else:
            p = _get_ws() / "artifacts" / tname / self.filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save(self, target_path=None):
        dest = target_path or self.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(self._value, (bytes, bytearray)): dest.write_bytes(self._value)
        elif isinstance(self._value, str): dest.write_text(self._value)
        return dest

class Dataset(Artifact):
    def __init__(self, value=None, filename="dataset.parquet", path=None):
        super().__init__(value=value, filename=filename, path=path)

    def load(self):
        p = self.path
        if p.suffix.lower() in (".parquet", ".pq"):
            try:
                import polars as pl
                return pl.read_parquet(p)
            except ImportError:
                import pandas as pd
                return pd.read_parquet(p)
        return p.read_text()

    def save(self, target_path=None):
        dest = target_path or self.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        v = self._value
        if hasattr(v, "write_parquet") and dest.suffix == ".parquet": v.write_parquet(dest)
        elif hasattr(v, "to_parquet") and dest.suffix == ".parquet": v.to_parquet(dest)
        elif hasattr(v, "write_csv") and dest.suffix == ".csv": v.write_csv(dest)
        elif hasattr(v, "to_csv") and dest.suffix == ".csv": v.to_csv(dest, index=False)
        else: super().save(dest)
        return dest

class Model(Artifact):
    def __init__(self, value=None, filename="model.pkl", path=None):
        super().__init__(value=value, filename=filename, path=path)

    def load(self):
        import joblib
        return joblib.load(self.path)

    def save(self, target_path=None):
        dest = target_path or self.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(self._value, dest)
        return dest
""").strip()

        runner_script = "\n".join([
            preamble,
            "",
            combined_src,
            "",
            "raw_args = sys.argv[1:]",
            f"result = {self.fn.__name__}(*raw_args)",
            "",
            "if isinstance(result, tuple):",
            "    items = result",
            "elif result is not None:",
            "    items = [result]",
            "else:",
            "    items = []",
            "",
            "for i, item in enumerate(items):",
            "    if isinstance(item, Artifact):",
            "        item.save()",
            "    elif isinstance(item, (int, float, str, bool)):",
            "        _set_output(f'out_{i}', item)",
        ])

        task_args = []
        for arg in args:
            if isinstance(arg, OutputRef):
                task_args.append(str(arg))
            elif isinstance(arg, Artifact):
                task_args.append(str(arg.path))
            else:
                task_args.append(str(arg))

        return Task(
            name=tname,
            docker_image=self.docker_image,
            packages=self.packages,
            command=["python3", "-c", runner_script],
            args=task_args,
            resources=self.resources,
        )


def task(
    name: str | None = None,
    packages: list[str] | None = None,
    docker_image: str = "python:3.12-slim",
    resources: ComputeResources | None = None,
    helpers: list[Callable] | None = None,
) -> Callable[[Callable], TaskCallable]:
    """Decorator to convert a standard Python function into a dual-mode tkf Task."""
    def decorator(fn: Callable) -> TaskCallable:
        return TaskCallable(
            fn=fn,
            name=name,
            packages=packages,
            docker_image=docker_image,
            resources=resources,
            helpers=helpers,
        )
    return decorator
