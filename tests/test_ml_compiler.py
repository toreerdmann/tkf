from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from tkf_models.compiler import compile_sklearn_pipeline
from tkf.pipeline import Pipeline


def test_compile_sklearn_pipeline():
    sk_pipe = SkPipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2)),
        ("rf", RandomForestRegressor(n_estimators=10)),
    ])

    dag = compile_sklearn_pipeline(
        pipeline=sk_pipe,
        dataset_path="/workspace/data/train.parquet",
        target_column="target",
        name="test-ml-dag",
    )

    assert isinstance(dag, Pipeline)
    sorted_tasks = dag.topological_sort()
    assert len(sorted_tasks) == 3
    assert sorted_tasks[0].name == "test-ml-dag-scaler"
    assert sorted_tasks[1].name == "test-ml-dag-pca"
    assert sorted_tasks[2].name == "test-ml-dag-rf"

    # Verify manifest generation
    manifest = dag.to_manifest()
    assert len(manifest["spec"]["tasks"]) == 3
    assert manifest["spec"]["tasks"][1]["dependsOn"] == ["test-ml-dag-scaler"]
    assert manifest["spec"]["tasks"][2]["dependsOn"] == ["test-ml-dag-pca"]
