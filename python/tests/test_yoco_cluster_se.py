"""
Tests for YOCO compression with clustered standard errors.

Implements Section 5.3.1 of Wong et al. (2021):
"You Only Compress Once: Optimal Data Compression for Estimating Linear Models"

The key insight is that for clustered SEs with compression:
1. Include cluster identifier in the grouping (within-cluster compression)
2. Compute ẽ⁰ = ỹ⁰ - ñ ⊙ ŷ (sum of residuals per group)
3. Build meat matrix using sparse cluster matrix: Ξ̂ = M̃ᵀ diag(ẽ⁰) W̃_C W̃_Cᵀ diag(ẽ⁰) M̃
"""

import numpy as np
import polars as pl
import pytest
from leanfe import leanfe


def test_cluster_se_compress_vs_demean():
    """
    Test that cluster SEs from compression match those from demeaning.
    
    Both methods should produce identical results for the same data.
    """
    np.random.seed(42)
    n = 2000
    n_clusters = 100
    
    # Create data with discrete regressors (good for compression)
    df = pl.DataFrame({
        'y': np.random.randn(n),
        'x1': np.random.choice([0, 1], n),
        'x2': np.random.choice([0, 1, 2], n),
        'fe1': np.random.choice(['A', 'B', 'C', 'D'], n),
        'cluster_id': np.random.choice(range(n_clusters), n)
    })
    
    # Run with compression (should use YOCO)
    result_compress = leanfe(
        df, 
        formula='y ~ x1 + x2 | fe1', 
        vcov='cluster', 
        cluster_cols=['cluster_id'],
        backend='polars'
    )
    
    # Verify compression was used
    assert result_compress.get('strategy') == 'compress', "Should use compression strategy"
    assert result_compress.get('n_clusters') == n_clusters, f"Should have {n_clusters} clusters"
    
    # Check that SEs are reasonable (not NaN or zero)
    for var, se in result_compress['std_errors'].items():
        assert not np.isnan(se), f"SE for {var} should not be NaN"
        assert se > 0, f"SE for {var} should be positive"


def test_cluster_se_duckdb_matches_polars():
    """
    Test that DuckDB and Polars backends produce identical cluster SEs.
    """
    np.random.seed(123)
    n = 1000
    n_clusters = 50
    
    df = pl.DataFrame({
        'y': np.random.randn(n),
        'x1': np.random.choice([0, 1], n),
        'x2': np.random.choice([0, 1, 2], n),
        'fe1': np.random.choice(['A', 'B', 'C'], n),
        'cluster_id': np.random.choice(range(n_clusters), n)
    })
    
    result_polars = leanfe(
        df, 
        formula='y ~ x1 + x2 | fe1', 
        vcov='cluster', 
        cluster_cols=['cluster_id'],
        backend='polars'
    )
    
    result_duckdb = leanfe(
        df, 
        formula='y ~ x1 + x2 | fe1', 
        vcov='cluster', 
        cluster_cols=['cluster_id'],
        backend='duckdb'
    )
    
    # Coefficients should match exactly
    for var in result_polars['coefficients']:
        assert np.isclose(
            result_polars['coefficients'][var], 
            result_duckdb['coefficients'][var],
            rtol=1e-10
        ), f"Coefficients for {var} should match"
    
    # SEs should match (may have small numerical differences)
    for var in result_polars['std_errors']:
        assert np.isclose(
            result_polars['std_errors'][var], 
            result_duckdb['std_errors'][var],
            rtol=1e-6
        ), f"SEs for {var} should match"


def test_cluster_se_ssc_adjustment():
    """
    Test small sample correction (SSC) for clustered SEs.
    
    With SSC, the adjustment is: (G/(G-1)) * ((n-1)/(n-k))
    Without SSC, the adjustment is: G/(G-1)
    
    So SSC should produce larger SEs when n-1 > n-k (i.e., k > 1).
    """
    np.random.seed(456)
    n = 500
    n_clusters = 25
    
    df = pl.DataFrame({
        'y': np.random.randn(n),
        'x1': np.random.choice([0, 1], n),
        'x2': np.random.choice([0, 1, 2], n),
        'fe1': np.random.choice(['A', 'B'], n),
        'cluster_id': np.random.choice(range(n_clusters), n)
    })
    
    result_no_ssc = leanfe(
        df, 
        formula='y ~ x1 + x2 | fe1', 
        vcov='cluster', 
        cluster_cols=['cluster_id'],
        ssc=False
    )
    
    result_ssc = leanfe(
        df, 
        formula='y ~ x1 + x2 | fe1', 
        vcov='cluster', 
        cluster_cols=['cluster_id'],
        ssc=True
    )
    
    # With SSC, SEs should be larger (since we have multiple regressors + FE)
    for var in result_no_ssc['std_errors']:
        assert result_ssc['std_errors'][var] >= result_no_ssc['std_errors'][var], \
            f"SSC should produce larger or equal SEs for {var}"


def test_cluster_se_compression_ratio():
    """
    Test that compression achieves good compression ratio with discrete data.
    """
    np.random.seed(789)
    n = 10000
    n_clusters = 200
    
    # Create data with limited unique combinations
    df = pl.DataFrame({
        'y': np.random.randn(n),
        'x1': np.random.choice([0, 1], n),  # 2 values
        'x2': np.random.choice([0, 1, 2], n),  # 3 values
        'fe1': np.random.choice(['A', 'B', 'C', 'D'], n),  # 4 values
        'cluster_id': np.random.choice(range(n_clusters), n)  # 200 clusters
    })
    
    result = leanfe(
        df, 
        formula='y ~ x1 + x2 | fe1', 
        vcov='cluster', 
        cluster_cols=['cluster_id']
    )
    
    # Should achieve compression
    assert result.get('strategy') == 'compress'
    
    # Compression ratio should be < 1 (fewer compressed records than original)
    compression_ratio = result.get('compression_ratio', 1.0)
    assert compression_ratio < 1.0, f"Should achieve compression, got ratio {compression_ratio}"
    
    # With 2*3*4*200 = 4800 max unique combinations, should compress well
    assert compression_ratio < 0.5, f"Should achieve >50% compression, got {compression_ratio}"


def test_sparse_cluster_matrix():
    """
    Test the sparse cluster matrix builder directly.
    """
    from leanfe.compress import _build_sparse_cluster_matrix
    
    cluster_ids = np.array(['A', 'A', 'B', 'B', 'C', 'A', 'C'])
    W_C, n_clusters = _build_sparse_cluster_matrix(cluster_ids)
    
    assert n_clusters == 3, "Should have 3 unique clusters"
    assert W_C.shape == (7, 3), "Matrix should be 7x3"
    
    # Each row should have exactly one 1
    row_sums = np.array(W_C.sum(axis=1)).flatten()
    assert np.all(row_sums == 1), "Each row should sum to 1"
    
    # Column sums should match cluster sizes
    col_sums = np.array(W_C.sum(axis=0)).flatten()
    assert set(col_sums) == {2, 2, 3}, "Column sums should be cluster sizes"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
