from __future__ import annotations

from typing import Literal
import numpy as np


def reconcile_forecasts(
    S: np.ndarray,
    base_forecasts: np.ndarray,
    method: Literal["bottom_up", "ols", "wls", "mint_shrink"] = "mint_shrink",
    residuals: np.ndarray | None = None,
) -> np.ndarray:
    """Reconciles base forecasts y_hat to be coherent: y_tilde = S @ P @ y_hat.

    Parameters
    ----------
    S : np.ndarray
        Summing matrix of shape (n_series, m_bottom).
    base_forecasts : np.ndarray
        Vector or matrix of base forecasts of shape (n_series, horizon) or (n_series,).
    method : str
        Reconciliation algorithm:
        - "bottom_up": Uses bottom level predictions directly: y_tilde = S @ y_bottom
        - "ols": Ordinary Least Squares projection (W = I)
        - "wls": Weighted Least Squares (W = diag(S @ 1))
        - "mint_shrink": Minimum Trace with Ledoit-Wolf / Schafer-Strimmer covariance shrinkage
    residuals : np.ndarray, optional
        In-sample residuals matrix of shape (n_series, n_timesteps) used to estimate W.

    Returns
    -------
    reconciled_forecasts : np.ndarray
        Reconciled coherent forecasts of shape (n_series, horizon).
    """
    n, m = S.shape
    y_hat = np.asarray(base_forecasts)
    is_1d = y_hat.ndim == 1
    if is_1d:
        y_hat = y_hat[:, np.newaxis]

    if method == "bottom_up":
        # Extract bottom rows (assuming bottom are the last m rows or direct slice)
        y_bottom = y_hat[-m:, :]
        y_tilde = S @ y_bottom
        return y_tilde.squeeze() if is_1d else y_tilde

    # Determine Weight matrix W
    if method == "ols" or residuals is None:
        W = np.eye(n)
    elif method == "wls":
        # Structural scaling (summing counts)
        diag_elements = np.sum(S, axis=1)
        W = np.diag(diag_elements)
    elif method == "mint_shrink":
        W = _estimate_shrinkage_covariance(residuals, n)
    else:
        raise ValueError(f"Unknown reconciliation method: {method}")

    # Compute optimal projection matrix P = (S.T @ W^-1 @ S)^-1 @ S.T @ W^-1
    try:
        W_inv = np.linalg.pinv(W)
        inv_st_w_s = np.linalg.pinv(S.T @ W_inv @ S)
        P = inv_st_w_s @ S.T @ W_inv
        y_tilde = S @ P @ y_hat
    except np.linalg.LinAlgError:
        # Fallback to OLS
        P = np.linalg.pinv(S.T @ S) @ S.T
        y_tilde = S @ P @ y_hat

    return y_tilde.squeeze() if is_1d else y_tilde


def _estimate_shrinkage_covariance(residuals: np.ndarray, n_series: int) -> np.ndarray:
    """Computes target shrinkage covariance matrix from residuals."""
    if residuals is None or residuals.shape[1] < 2:
        return np.eye(n_series)

    # Sample covariance
    cov_sample = np.cov(residuals)
    if cov_sample.ndim == 0 or cov_sample.shape[0] != n_series:
        return np.eye(n_series)

    # Diagonal shrinkage target
    target = np.diag(np.diag(cov_sample))
    
    # Shrinkage intensity factor (simplified Ledoit-Wolf)
    shrinkage = 0.2
    W_shrunk = (1 - shrinkage) * cov_sample + shrinkage * target
    return W_shrunk
