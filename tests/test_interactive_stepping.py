from pathlib import Path
from tkf.pipeline import Pipeline, Task, VolumeConfig


def test_run_until_and_resume(tmp_path):
    vol = VolumeConfig(local_path=str(tmp_path / "volume"))

    t1 = Task(name="step-1", command=["python3", "-c", "import sys; print('Running Step 1')"])
    t2 = Task(name="step-2", command=["python3", "-c", "import sys; print('Running Step 2')"])
    t3 = Task(name="step-3", command=["python3", "-c", "import sys; print('Running Step 3')"])

    p = Pipeline(name="test-stepping", volume=vol)
    p.add_dependency(t1, t2)
    p.add_dependency(t2, t3)

    # 1. Run only up to step-2
    success = p.run(until="step-2", local=True)
    assert success

    # Verify ancestors
    ancestors = p.get_upstream_tasks("step-2")
    assert ancestors == {"step-1", "step-2"}

    # 2. Resume from step-3
    descendants = p.get_downstream_tasks("step-3")
    assert descendants == {"step-3"}
    success_resume = p.run(from_task="step-3", local=True)
    assert success_resume


def test_interactive_pipeline_stepper(tmp_path):
    vol = VolumeConfig(local_path=str(tmp_path / "volume"))

    t1 = Task(name="prep", command=["python3", "-c", "import sys; print('Prep Done')"])
    t2 = Task(name="train", command=["python3", "-c", "import sys; print('Train Done')"])
    t3 = Task(name="eval", command=["python3", "-c", "import sys; print('Eval Done')"])

    p = Pipeline(name="test-stepper-dag", volume=vol)
    p.add_dependency(t1, t2)
    p.add_dependency(t2, t3)

    stepper = p.stepper(local=True)
    assert not stepper.is_finished
    assert stepper.current_task.name == "prep"

    # Step 1: prep
    executed_1 = stepper.step()
    assert executed_1.name == "prep"
    assert stepper.current_task.name == "train"

    # Run until eval
    stepper.run_until("eval")
    assert stepper.is_finished
    assert "train" in stepper.executed_tasks
    assert "eval" in stepper.executed_tasks
