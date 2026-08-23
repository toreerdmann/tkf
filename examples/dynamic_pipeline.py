"""Sample Pipeline demonstrating dynamic parameter passing and dataset/model artifact paths."""

import textwrap
from tkf import Pipeline, Task, VolumeConfig


def create_pipeline(namespace: str = "tkf-dev") -> Pipeline:
    volume = VolumeConfig(size="1Gi", mount_path="/workspace")

    # Step 1: Preprocess & Dataset creation + Parameter output
    # Uses standard /dev/termination-log to emit dynamic parameters
    prep_script = textwrap.dedent("""
        import json, os, time
        from pathlib import Path

        print("Creating dataset...")
        vol = os.environ.get("VOLUME", "/workspace")
        data_path = Path(vol) / "artifacts" / "preprocess" / "raw_data.csv"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, "w") as f:
            f.write("id,feature_x,feature_y\\n1,10.5,20.1\\n2,12.3,22.4\\n")

        # Dynamically emit parameters to /dev/termination-log
        outputs = {"num_samples": "150", "feature_count": "2"}
        try:
            Path("/dev/termination-log").write_text(json.dumps(outputs))
        except Exception:
            pass

        print(f"Dataset saved to {data_path} with 150 samples.")
        time.sleep(2)
    """).strip()

    prep = Task(
        name="preprocess",
        docker_image="python:3.12-slim",
        command=["python3", "-c", prep_script],
    )

    # Step 2: Train Model consuming Dataset artifact AND parameter outputs
    train_script = textwrap.dedent("""
        import json, os, sys, time
        from pathlib import Path

        dataset_file = sys.argv[1]
        sample_count = sys.argv[2]
        print(f"Training on dataset: {dataset_file}")
        print(f"Total samples passed dynamically from parent: {sample_count}")

        vol = os.environ.get("VOLUME", "/workspace")
        model_file = Path(vol) / "artifacts" / "train" / "random_forest.bin"
        model_file.parent.mkdir(parents=True, exist_ok=True)
        with open(model_file, "w") as f:
            f.write("MODEL_WEIGHTS_VERSION_1")

        # Emit training metrics dynamically
        outputs = {"val_accuracy": "0.985"}
        try:
            Path("/dev/termination-log").write_text(json.dumps(outputs))
        except Exception:
            pass

        print(f"Model saved to {model_file} with accuracy 0.985")
        time.sleep(2)
    """).strip()

    train = Task(
        name="train",
        docker_image="python:3.12-slim",
        command=["python3", "-c", train_script],
        args=[
            prep.dataset("raw_data.csv"),        # Injects artifact path
            prep.output("num_samples"),          # Injects dynamic parameter value
        ],
    )

    # Step 3: Evaluate Model consuming Model artifact and accuracy parameter
    eval_script = textwrap.dedent("""
        import sys, os
        model_file = sys.argv[1]
        accuracy = sys.argv[2]
        print(f"Evaluating Model from: {model_file}")
        print(f"Reported Validation Accuracy from parent: {accuracy}")
        assert float(accuracy) > 0.90
        print("Model approved for deployment!")
    """).strip()

    evaluate = Task(
        name="evaluate",
        docker_image="python:3.12-slim",
        command=["python3", "-c", eval_script],
        args=[
            train.model("random_forest.bin"),
            train.output("val_accuracy"),
        ],
    )

    # Dependencies are automatically inferred from .output() and .dataset() references!
    p = Pipeline(name="dynamic-demo", volume=volume, namespace=namespace)
    p.add_task(prep).add_task(train).add_task(evaluate)
    return p


if __name__ == "__main__":
    p = create_pipeline()
    p.print()
    p.submit(namespace="tkf-dev", wait=False)
