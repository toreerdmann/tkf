"""Demonstrates tkf Direct Runner without CRDs and dynamic package installation with uv."""

import textwrap
from tkf import Pipeline, Task, VolumeConfig


def create_pipeline(namespace: str = "tkf-dev") -> Pipeline:
    volume = VolumeConfig(size="1Gi", mount_path="/workspace", temp=False)

    # 1. Task using Polars (no custom Docker image build needed!)
    prep_script = textwrap.dedent("""
        import json, os, time
        from pathlib import Path
        import polars as pl

        print("Creating dataset using Polars...")
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "feature_a": [1.1, 2.2, 3.3, 4.4, 5.5],
            "label": [0, 1, 0, 1, 0],
        })

        vol = os.environ.get("VOLUME", "/workspace")
        data_file = Path(vol) / "artifacts" / "prep" / "dataset.parquet"
        data_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(data_file)

        # Emit output parameters
        try:
            Path("/dev/termination-log").write_text(json.dumps({"total_rows": str(len(df))}))
        except Exception:
            pass

        print(f"Polars dataset saved to {data_file} with {len(df)} rows.")
    """).strip()

    prep = Task(
        name="prep",
        packages=["polars", "pyarrow"],  # Injected via uv run --with
        command=["python3", "-c", prep_script],
    )

    # 2. Task using Scikit-Learn (installed dynamically by uv in seconds)
    train_script = textwrap.dedent("""
        import json, os, sys, time
        from pathlib import Path
        import polars as pl
        from sklearn.linear_model import LogisticRegression

        dataset_path = sys.argv[1]
        row_count = sys.argv[2]
        print(f"Loading dataset from: {dataset_path}")
        print(f"Rows reported by parent: {row_count}")

        df = pl.read_parquet(dataset_path)
        X = df.select("feature_a").to_numpy()
        y = df.select("label").to_numpy().ravel()

        clf = LogisticRegression()
        clf.fit(X, y)

        vol = os.environ.get("VOLUME", "/workspace")
        model_file = Path(vol) / "artifacts" / "train" / "model.txt"
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_text(f"coef={clf.coef_[0][0]}")

        try:
            Path("/dev/termination-log").write_text(json.dumps({"coef": str(clf.coef_[0][0])}))
        except Exception:
            pass

        print(f"LogisticRegression model trained and saved to {model_file}!")
    """).strip()

    train = Task(
        name="train",
        packages=["polars", "pyarrow", "scikit-learn"],  # Injected via uv run --with
        command=["python3", "-c", train_script],
        args=[
            prep.dataset("dataset.parquet"),
            prep.output("total_rows"),
        ],
    )

    # Build DAG
    p = Pipeline(name="direct-uv-demo", volume=volume, namespace=namespace)
    p.add_task(prep).add_task(train)
    return p


if __name__ == "__main__":
    p = create_pipeline(namespace="tkf-dev")
    p.print()
    # Runs directly against k8s Jobs with live log streaming (NO CRD required!)
    p.run(direct=True)
