import numpy as np

from tkf_models.recon.compiler import compile_cross_temporal_reconciliation
from tkf_models.recon.matrices import build_summing_matrix, build_temporal_matrix
from tkf_models.recon.solvers import reconcile_forecasts
from tkf_models.recon.specs import HierarchySpec, TemporalSpec


def test_build_summing_matrix():
    all_nodes = ["Total", "Region_A", "Region_B", "Store_1", "Store_2", "Store_3"]
    bottom_nodes = ["Store_1", "Store_2", "Store_3"]
    structure = {
        "Total": ["Region_A", "Region_B"],
        "Region_A": ["Store_1", "Store_2"],
        "Region_B": ["Store_3"],
    }

    S = build_summing_matrix(all_nodes, bottom_nodes, structure)
    assert S.shape == (6, 3)

    # Total = Store_1 + Store_2 + Store_3
    assert np.array_equal(S[0, :], [1, 1, 1])
    # Region_A = Store_1 + Store_2
    assert np.array_equal(S[1, :], [1, 1, 0])
    # Region_B = Store_3
    assert np.array_equal(S[2, :], [0, 0, 1])
    # Bottom nodes are identity
    assert np.array_equal(S[3:, :], np.eye(3))


def test_reconcile_forecasts_bottom_up():
    S = np.array([
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    # Total, Bottom1, Bottom2
    base_forecasts = np.array([100.0, 40.0, 70.0])

    reconciled = reconcile_forecasts(S, base_forecasts, method="bottom_up")
    # Bottom-up total should be 40 + 70 = 110
    assert np.allclose(reconciled, [110.0, 40.0, 70.0])


def test_reconcile_forecasts_ols_and_mint():
    S = np.array([
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    base_forecasts = np.array([100.0, 40.0, 70.0])

    # OLS
    reconciled_ols = reconcile_forecasts(S, base_forecasts, method="ols")
    assert reconciled_ols.shape == (3,)
    # Check coherence: Total == Bottom1 + Bottom2
    assert np.isclose(reconciled_ols[0], reconciled_ols[1] + reconciled_ols[2])

    # MinT with shrinkage
    residuals = np.random.normal(0, 1.0, size=(3, 30))
    reconciled_mint = reconcile_forecasts(S, base_forecasts, method="mint_shrink", residuals=residuals)
    assert np.isclose(reconciled_mint[0], reconciled_mint[1] + reconciled_mint[2])


def test_compile_cross_temporal_reconciliation():
    hierarchy = HierarchySpec(
        levels=["Total", "Region", "Store"],
        structure={"Total": ["Region"], "Region": ["Store"]},
    )
    temporal = TemporalSpec(frequencies=["D", "W", "M"])

    forecasters = {
        "Total": {"model": "AutoARIMA", "cpu": "1"},
        "Region": {"model": "LightGBM", "cpu": "2"},
        "Store": {"model": "Chronos", "gpu": 1},
    }

    dag = compile_cross_temporal_reconciliation(
        dataset_path="/workspace/data/sales.parquet",
        hierarchy=hierarchy,
        temporal=temporal,
        forecasters=forecasters,
        recon_method="mint_shrink",
        name="sales-recon",
    )

    tasks = dag.topological_sort()
    assert len(tasks) == 5  # prep + 3 model levels + 1 reconciliation solver
    assert tasks[0].name == "sales-recon-prep"
    assert tasks[-1].name == "sales-recon-solver"
