"""Basic tests for fast-hdfe-reg package."""

import polars as pl
import numpy as np
from leanfe import leanfe, leanfe_polars, leanfe_duckdb


def test_polars_basic():
    """Test basic Polars functionality."""
    # Create simple test data
    np.random.seed(42)
    n = 1000
    df = pl.DataFrame({
        "y": np.random.randn(n),
        "x": np.random.randn(n),
        "fe1": np.random.randint(0, 10, n),
        "fe2": np.random.randint(0, 5, n),
    })
    
    result = leanfe_polars(df, "y", ["x"], ["fe1", "fe2"])
    
    assert "coefficients" in result
    assert "std_errors" in result
    assert "x" in result["coefficients"]
    assert result["n_obs"] > 0


def test_formula_api():
    """Test formula parsing API."""
    np.random.seed(42)
    n = 1000
    df = pl.DataFrame({
        "y": np.random.randn(n),
        "x": np.random.randn(n),
        "fe1": np.random.randint(0, 10, n),
        "fe2": np.random.randint(0, 5, n),
    })
    
    result = leanfe_polars(df, formula="y ~ x | fe1 + fe2")
    
    assert "coefficients" in result
    assert "x" in result["coefficients"]


def test_standard_errors():
    """Test different standard error types."""
    np.random.seed(42)
    n = 1000
    df = pl.DataFrame({
        "y": np.random.randn(n),
        "x": np.random.randn(n),
        "fe1": np.random.randint(0, 10, n),
        "cluster": np.random.randint(0, 20, n),
    })
    
    # IID
    result_iid = leanfe_polars(df, "y", ["x"], ["fe1"], vcov="iid")
    assert result_iid["vcov_type"] == "iid"
    
    # HC1
    result_hc1 = leanfe_polars(df, "y", ["x"], ["fe1"], vcov="HC1")
    assert result_hc1["vcov_type"] == "HC1"
    
    # Clustered
    result_cluster = leanfe_polars(
        df, "y", ["x"], ["fe1"], 
        vcov="cluster", cluster_cols=["cluster"]
    )
    assert result_cluster["vcov_type"] == "cluster"
    assert result_cluster["n_clusters"] > 0


def test_unified_api_polars():
    """Test unified leanfe() with Polars backend."""
    np.random.seed(42)
    n = 1000
    df = pl.DataFrame({
        "y": np.random.randn(n),
        "x": np.random.randn(n),
        "fe1": np.random.randint(0, 10, n),
        "fe2": np.random.randint(0, 5, n),
    })
    
    result = leanfe(df, formula="y ~ x | fe1 + fe2", backend="polars")
    
    assert "coefficients" in result
    assert "x" in result["coefficients"]


def test_unified_api_duckdb():
    """Test unified leanfe() with DuckDB backend."""
    np.random.seed(42)
    n = 1000
    df = pl.DataFrame({
        "y": np.random.randn(n),
        "x": np.random.randn(n),
        "fe1": np.random.randint(0, 10, n),
        "fe2": np.random.randint(0, 5, n),
    })
    
    result = leanfe(df, formula="y ~ x | fe1 + fe2", backend="duckdb")
    
    assert "coefficients" in result
    assert "x" in result["coefficients"]


def test_unified_api_default_backend():
    """Test unified leanfe() uses Polars by default."""
    np.random.seed(42)
    n = 1000
    df = pl.DataFrame({
        "y": np.random.randn(n),
        "x": np.random.randn(n),
        "fe1": np.random.randint(0, 10, n),
        "fe2": np.random.randint(0, 5, n),
    })
    
    # No backend specified - should use polars
    result = leanfe(df, formula="y ~ x | fe1 + fe2")
    
    assert "coefficients" in result
    assert "x" in result["coefficients"]


def test_unified_api_invalid_backend():
    """Test unified leanfe() raises error for invalid backend."""
    np.random.seed(42)
    n = 100
    df = pl.DataFrame({
        "y": np.random.randn(n),
        "x": np.random.randn(n),
        "fe1": np.random.randint(0, 10, n),
    })
    
    import pytest
    with pytest.raises(ValueError, match="backend must be"):
        leanfe(df, formula="y ~ x | fe1", backend="invalid")
