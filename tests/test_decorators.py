from pathlib import Path
from tkf import Pipeline, VolumeConfig, task, Dataset, Model


@task(name="prep-step", packages=["polars"])
def prep_data(raw_text: str) -> tuple[Dataset, int]:
    # In interactive or container execution:
    lines = raw_text.strip().split("\n")
    return Dataset(raw_text, filename="raw_clean.txt"), len(lines)


@task(name="train-step", packages=["scikit-learn"])
def train_model(data_path: str, count: str) -> tuple[Model, float]:
    return Model({"weights": [1.0, 2.0], "count": int(count)}, filename="model.pkl"), 0.95


def test_interactive_in_memory_call():
    """Verify that @task can be called directly in a Jupyter notebook cell with Python return values."""
    dataset, count = prep_data("line1\nline2\nline3")
    assert count == 3
    assert dataset.load() == "line1\nline2\nline3"

    model_art, acc = train_model("dummy_path", count)
    assert acc == 0.95
    assert model_art.load()["count"] == 3


def test_compile_decorated_tasks_to_dag(tmp_path):
    """Verify compiling @task functions to a tkf Pipeline and executing the DAG."""
    vol = VolumeConfig(local_path=str(tmp_path / "vol"))

    t1 = prep_data.to_task("row1\nrow2")
    t2 = train_model.to_task(t1.dataset("raw_clean.txt"), t1.output("out_1"))

    p = Pipeline("decorated-pipeline", volume=vol)
    p.add_task(t1).add_task(t2)

    order = [t.name for t in p.topological_sort()]
    assert order == ["prep-step", "train-step"]

    success = p.run(local=True)
    assert success is True
