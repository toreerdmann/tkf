"""Sample ML Pipeline demonstrating tkf DAG execution with shared PVC storage."""

import textwrap
from tkf import Pipeline, Task, VolumeConfig


def create_pipeline(namespace: str = "tkf-dev") -> Pipeline:
    # Shared volume mounted at /workspace (accessed via $VOLUME)
    volume = VolumeConfig(
        size="1Gi",
        mount_path="/workspace",
        temp=False,  # Set to True to auto-delete after pipeline run
    )

    # 1. Root task: Data preprocessing
    prep_script = textwrap.dedent("""
        import os
        import time

        vol = os.environ.get("VOLUME", "/workspace")
        print(f"Preprocessing data in {vol}...")
        with open(f"{vol}/data.txt", "w") as f:
            f.write("sample_id,feature_1,feature_2\\n1,0.5,1.2\\n2,0.9,2.4\\n")
        print("Dataset created successfully.")
        time.sleep(2)
    """).strip()

    prep_task = Task(
        name="preprocess",
        docker_image="python:3.12-slim",
        command=["python3", "-c", prep_script],
    )

    # 2a. Parallel task: Train Model A
    train_a_script = textwrap.dedent("""
        import os
        import time

        vol = os.environ.get("VOLUME", "/workspace")
        print("Training Model A (Random Forest)...")
        with open(f"{vol}/data.txt") as f:
            data = f.read()
        print(f"Read {len(data)} bytes of data.")
        time.sleep(3)

        with open(f"{vol}/model_a.txt", "w") as f:
            f.write("model_a_accuracy=0.92\\n")
        print("Model A trained and saved.")
    """).strip()

    train_a_task = Task(
        name="train-model-a",
        docker_image="python:3.12-slim",
        command=["python3", "-c", train_a_script],
    )

    # 2b. Parallel task: Train Model B
    train_b_script = textwrap.dedent("""
        import os
        import time

        vol = os.environ.get("VOLUME", "/workspace")
        print("Training Model B (Gradient Boosting)...")
        with open(f"{vol}/data.txt") as f:
            data = f.read()
        print(f"Read {len(data)} bytes of data.")
        time.sleep(4)

        with open(f"{vol}/model_b.txt", "w") as f:
            f.write("model_b_accuracy=0.95\\n")
        print("Model B trained and saved.")
    """).strip()

    train_b_task = Task(
        name="train-model-b",
        docker_image="python:3.12-slim",
        command=["python3", "-c", train_b_script],
    )

    # 3. Final evaluation task (depends on both Model A and Model B)
    eval_script = textwrap.dedent("""
        import os

        vol = os.environ.get("VOLUME", "/workspace")
        print("Evaluating trained models...")
        with open(f"{vol}/model_a.txt") as f:
            res_a = f.read().strip()
        with open(f"{vol}/model_b.txt") as f:
            res_b = f.read().strip()

        print(f"Evaluation Results:\\n- {res_a}\\n- {res_b}")
        print("Best Model: Model B (95% accuracy)")
    """).strip()

    eval_task = Task(
        name="evaluate",
        docker_image="python:3.12-slim",
        command=["python3", "-c", eval_script],
    )

    # Build DAG using nested list syntax (similar to kubeflow-utils!)
    # [prep_task, [[train_a_task], [train_b_task]], eval_task]
    pipeline = Pipeline(name="ml-experiment", volume=volume, namespace=namespace)
    pipeline.build_from_list([
        prep_task,
        [
            [train_a_task],
            [train_b_task],
        ],
        eval_task,
    ])

    return pipeline


if __name__ == "__main__":
    p = create_pipeline()
    p.print()
    p.run(local=True)
