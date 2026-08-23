import textwrap
from pathlib import Path
from tkf import Pipeline, Task, VolumeConfig, set_output


def test_auto_dependency_from_output_ref():
    prep = Task(name="preprocess", command=["echo", "prep"])
    train = Task(
        name="train",
        command=["python", "train.py"],
        args=["--data", prep.dataset("data.csv"), "--samples", prep.output("num_samples")],
    )

    p = Pipeline("auto-dep-test")
    p.add_task(prep)
    p.add_task(train)

    order = [t.name for t in p.topological_sort()]
    assert order == ["preprocess", "train"]


def test_local_run_with_dynamic_params(tmp_path):
    vol = VolumeConfig(local_path=str(tmp_path / "vol"))

    prep_script = textwrap.dedent("""
        import tkf, os
        tkf.set_output('best_lr', 0.05)
        p = tkf.dataset_path('train.csv')
        p.write_text('id,val\\n1,10\\n')
    """).strip()

    prep_task = Task(
        name="step1",
        command=["python3", "-c", prep_script],
    )

    train_script = textwrap.dedent("""
        import sys, os
        lr = sys.argv[1]
        data_file = sys.argv[2]
        print(f"LR: {lr}")
        assert lr == "0.05"
        assert os.path.exists(data_file)
    """).strip()

    train_task = Task(
        name="step2",
        command=["python3", "-c", train_script],
        args=[prep_task.output("best_lr"), prep_task.dataset("train.csv")],
    )

    p = Pipeline("dynamic-test", volume=vol)
    p.add_task(prep_task).add_task(train_task)

    success = p.run(local=True)
    assert success is True
