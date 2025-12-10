"""Test that sparse and dense matrix implementations produce identical results."""
import polars as pl
import numpy as np
import pytest
from leanfe.compress import (
    build_design_matrix,
    solve_wls,
    compute_rss_grouped,
    compute_se_compress,
    compress_polars,
)


def test_sparse_dense_coefficients_identical():
    """Verify sparse and dense implementations produce bit-identical coefficients."""
    np.random.seed(42)
    n = 1000
    
    df = pl.DataFrame({
        'y': np.random.normal(10, 2, n),
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(0, 1, n),
        'fe1': np.random.randint(0, 50, n),
        'fe2': np.random.randint(0, 20, n),
    })
    
    # Compress data
    compressed, n_obs = compress_polars(df, 'y', ['x1', 'x2'], ['fe1', 'fe2'])
    
    # Build design matrix - sparse
    X_sparse, Y, wts, cols_sparse, _ = build_design_matrix(
        compressed, ['x1', 'x2'], ['fe1', 'fe2'], backend='polars', use_sparse=True
    )
    
    # Build design matrix - dense
    X_dense, Y_dense, wts_dense, cols_dense, _ = build_design_matrix(
        compressed, ['x1', 'x2'], ['fe1', 'fe2'], backend='polars', use_sparse=False
    )
    
    # Solve WLS
    beta_sparse, XtX_inv_sparse = solve_wls(X_sparse, Y, wts)
    beta_dense, XtX_inv_dense = solve_wls(X_dense, Y_dense, wts_dense)
    
    # Coefficients should be nearly identical (within numerical precision)
    np.testing.assert_allclose(beta_sparse[:2], beta_dense[:2], rtol=1e-10)
    
    # XtX_inv should also match
    np.testing.assert_allclose(XtX_inv_sparse, XtX_inv_dense, rtol=1e-10)


def test_sparse_dense_se_identical():
    """Verify sparse and dense SE computations produce identical results."""
    np.random.seed(123)
    n = 500
    
    df = pl.DataFrame({
        'y': np.random.normal(10, 2, n),
        'x': np.random.normal(0, 1, n),
        'fe1': np.random.randint(0, 30, n),
        'cluster': np.random.randint(0, 20, n),
    })
    
    # Compress with cluster
    compressed, n_obs = compress_polars(df, 'y', ['x'], ['fe1'], cluster_col='cluster')
    
    # Build design matrices
    X_sparse, Y, wts, cols, _ = build_design_matrix(
        compressed, ['x'], ['fe1'], backend='polars', use_sparse=True
    )
    X_dense, Y_d, wts_d, cols_d, _ = build_design_matrix(
        compressed, ['x'], ['fe1'], backend='polars', use_sparse=False
    )
    
    # Solve WLS
    beta_sparse, XtX_inv_sparse = solve_wls(X_sparse, Y, wts)
    beta_dense, XtX_inv_dense = solve_wls(X_dense, Y_d, wts_d)
    
    # Compute RSS
    rss_total_s, rss_g_s = compute_rss_grouped(compressed, X_sparse, beta_sparse, 'polars')
    rss_total_d, rss_g_d = compute_rss_grouped(compressed, X_dense, beta_dense, 'polars')
    
    # RSS should match (allow small numerical differences from floating point)
    np.testing.assert_allclose(rss_total_s, rss_total_d, rtol=1e-7)
    np.testing.assert_allclose(rss_g_s, rss_g_d, rtol=1e-7)
    
    # Test IID SEs
    df_resid = n_obs - len(cols)
    se_iid_s, _ = compute_se_compress(
        XtX_inv_sparse, rss_total_s, rss_g_s, n_obs, df_resid, 'iid', X_sparse, ['x']
    )
    se_iid_d, _ = compute_se_compress(
        XtX_inv_dense, rss_total_d, rss_g_d, n_obs, df_resid, 'iid', X_dense, ['x']
    )
    np.testing.assert_allclose(se_iid_s, se_iid_d, rtol=1e-10)
    
    # Test HC1 SEs
    se_hc1_s, _ = compute_se_compress(
        XtX_inv_sparse, rss_total_s, rss_g_s, n_obs, df_resid, 'HC1', X_sparse, ['x']
    )
    se_hc1_d, _ = compute_se_compress(
        XtX_inv_dense, rss_total_d, rss_g_d, n_obs, df_resid, 'HC1', X_dense, ['x']
    )
    np.testing.assert_allclose(se_hc1_s, se_hc1_d, rtol=1e-10)


def test_sparse_dense_cluster_se_identical():
    """Verify sparse and dense cluster SE computations produce identical results."""
    np.random.seed(456)
    n = 600
    
    df = pl.DataFrame({
        'y': np.random.normal(10, 2, n),
        'x': np.random.normal(0, 1, n),
        'fe1': np.random.randint(0, 30, n),
        'cluster': np.repeat(np.arange(30), 20),  # 30 clusters, 20 obs each
    })
    
    # Compress with cluster
    compressed, n_obs = compress_polars(df, 'y', ['x'], ['fe1'], cluster_col='cluster')
    
    # Build design matrices
    X_sparse, Y, wts, cols, _ = build_design_matrix(
        compressed, ['x'], ['fe1'], backend='polars', use_sparse=True
    )
    X_dense, Y_d, wts_d, cols_d, _ = build_design_matrix(
        compressed, ['x'], ['fe1'], backend='polars', use_sparse=False
    )
    
    # Solve WLS
    beta_sparse, XtX_inv_sparse = solve_wls(X_sparse, Y, wts)
    beta_dense, XtX_inv_dense = solve_wls(X_dense, Y_d, wts_d)
    
    # Compute RSS
    rss_total_s, rss_g_s = compute_rss_grouped(compressed, X_sparse, beta_sparse, 'polars')
    rss_total_d, rss_g_d = compute_rss_grouped(compressed, X_dense, beta_dense, 'polars')
    
    # Get cluster IDs and e0_g
    cluster_ids = compressed['cluster'].to_numpy()
    n_g = compressed['_n'].to_numpy()
    sum_y_g = compressed['_sum_y'].to_numpy()
    yhat_g_s = np.asarray(X_sparse @ beta_sparse).flatten()
    yhat_g_d = X_dense @ beta_dense
    e0_g_s = sum_y_g - n_g * yhat_g_s
    e0_g_d = sum_y_g - n_g * yhat_g_d
    
    df_resid = n_obs - len(cols)
    
    # Test cluster SEs
    se_cluster_s, n_clusters_s = compute_se_compress(
        XtX_inv_sparse, rss_total_s, rss_g_s, n_obs, df_resid, 'cluster', X_sparse, ['x'],
        cluster_ids=cluster_ids, e0_g=e0_g_s
    )
    se_cluster_d, n_clusters_d = compute_se_compress(
        XtX_inv_dense, rss_total_d, rss_g_d, n_obs, df_resid, 'cluster', X_dense, ['x'],
        cluster_ids=cluster_ids, e0_g=e0_g_d
    )
    
    np.testing.assert_equal(n_clusters_s, n_clusters_d)
    np.testing.assert_allclose(se_cluster_s, se_cluster_d, rtol=1e-10)
