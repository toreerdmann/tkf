import pytest
from tkf import Pipeline, Task, VolumeConfig


def test_linear_pipeline():
    t1 = Task(name="step-1", command=["echo", "1"])
    t2 = Task(name="step-2", command=["echo", "2"])

    p = Pipeline("linear")
    p.add_task(t1).add_task(t2).add_dependency(t1, t2)

    order = p.topological_sort()
    assert [t.name for t in order] == ["step-1", "step-2"]


def test_parallel_pipeline_build_from_list():
    t1 = Task(name="prep", command=["echo", "prep"])
    t2a = Task(name="branch-a", command=["echo", "a"])
    t2b = Task(name="branch-b", command=["echo", "b"])
    t3 = Task(name="join", command=["echo", "join"])

    p = Pipeline("branching")
    p.build_from_list([
        t1,
        [[t2a], [t2b]],
        t3,
    ])

    order = [t.name for t in p.topological_sort()]
    assert order[0] == "prep"
    assert order[-1] == "join"
    assert set(order[1:3]) == {"branch-a", "branch-b"}


def test_cycle_detection():
    t1 = Task(name="t1", command=["echo", "1"])
    t2 = Task(name="t2", command=["echo", "2"])

    p = Pipeline("cyclic")
    p.add_task(t1).add_task(t2)
    p.add_dependency(t1, t2)
    p.add_dependency(t2, t1)

    with pytest.raises(RuntimeError, match="cycle"):
        p.topological_sort()


def test_to_manifest():
    t1 = Task(name="task1", command=["python", "main.py"], docker_image="python:3.12")
    p = Pipeline("manifest-test", volume=VolumeConfig(size="2Gi"))
    p.add_task(t1)

    manifest = p.to_manifest()
    assert manifest["apiVersion"] == "tkf.dev/v1alpha1"
    assert manifest["kind"] == "PipelineRun"
    assert manifest["metadata"]["name"] == "manifest-test"
    assert len(manifest["spec"]["tasks"]) == 1
    assert manifest["spec"]["volume"]["size"] == "2Gi"
