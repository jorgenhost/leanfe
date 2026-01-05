"""
Optimized standard error computation for Polars backend.

This module implements HC1 and clustered standard errors using Polars expressions
for efficient computation that avoids scipy.sparse matrices.

The key optimization is to use Polars native operations (group_by, expressions)
rather than converting to numpy arrays and using scipy.sparse matrices, similar
to how DuckDB uses SQL aggregation for performance.
"""

import numpy as np
import polars as pl
from typing import Literal
from itertools import combinations
from duckdb import DuckDBPyConnection

# ============================================================================
# CONSTANTS
# ============================================================================

MIN_CLUSTERS_FOR_ADJUSTMENT = 2
FIRST_ORDER_SUBSET_SIZE = 1


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def compute_standard_errors_polars(
    df: pl.DataFrame,
    x_cols: list[str],
    XtX_inv: np.ndarray,
    resid: np.ndarray,
    weights: str | None,
    cluster_cols: list[str] | None,
    n_obs: int,
    df_resid: int,
    vcov: Literal["iid", "HC1", "cluster"],
    ssc: bool = True,
    X: np.ndarray | None = None,
    is_iv: bool = False
) -> tuple[np.ndarray, int | tuple | None]:
    """
    Compute standard errors using Polars expressions for efficient computation.
    
    This implementation uses Polars native operations to compute standard errors,
    avoiding scipy.sparse matrices and leveraging Polars' query optimization.
    
    Parameters
    ----------
    df : pl.DataFrame
        Data containing regressors
    x_cols : list[str]
        Names of regressor columns in df
    XtX_inv : np.ndarray
        (X'X)^-1 matrix
    resid : np.ndarray
        Residuals array
    weights : str, optional
        Name of weights column in df
    vcov : {"iid", "HC1", "cluster"}
        Variance-covariance estimator type
    cluster_cols : list[str], optional
        Clustering variable names
    n_obs : int
        Number of observations
    df_resid : int
        Residual degrees of freedom
    ssc : bool, default True
        Apply small sample correction
    X : np.ndarray, optional
        Design matrix (required for IV/2SLS, uses X_hat from first stage)
    is_iv : bool, default False
        Whether this is IV/2SLS estimation
        
    Returns
    -------
    tuple
        (standard_errors, n_clusters)
        where n_clusters is int for one-way or tuple for multi-way
    """
    if vcov == "iid":
        return _compute_se_iid_polars(
            XtX_inv=XtX_inv,
            resid=resid,
            weights=weights,
            df=df,
            df_resid=df_resid
        )
    
    elif vcov == "HC1":
        if is_iv and X is not None:
            # For IV, need to use X_hat from first stage
            return _compute_se_hc1_iv(
                XtX_inv=XtX_inv,
                resid=resid,
                X=X,
                weights_array=df.select(weights).to_numpy().flatten() if weights else None,
                n_obs=n_obs,
                df_resid=df_resid
            )
        else:
            # For OLS, use Polars expressions
            return _compute_se_hc1_polars(
                df=df,
                x_cols=x_cols,
                XtX_inv=XtX_inv,
                resid=resid,
                weights=weights,
                n_obs=n_obs,
                df_resid=df_resid
            )
    
    elif vcov == "cluster":
        if cluster_cols is None:
            raise ValueError("cluster_cols required for vcov='cluster'")
        
        if is_iv and X is not None:
            # For IV, use numpy-based computation
            cluster_ids = df.select(cluster_cols).to_numpy()
            if cluster_ids.ndim == 1:
                cluster_ids = cluster_ids.reshape(-1, 1)
            weights_array = df.select(weights).to_numpy().flatten() if weights else None
            
            if len(cluster_cols) == 1:
                return _compute_se_cluster_oneway_iv(
                    XtX_inv=XtX_inv,
                    resid=resid,
                    X=X,
                    weights=weights_array,
                    cluster_ids=cluster_ids.flatten(),
                    n_obs=n_obs,
                    df_resid=df_resid,
                    ssc=ssc
                )
            else:
                return _compute_se_cluster_multiway_iv(
                    XtX_inv=XtX_inv,
                    resid=resid,
                    X=X,
                    weights=weights_array,
                    cluster_ids=cluster_ids,
                    n_obs=n_obs,
                    df_resid=df_resid,
                    ssc=ssc
                )
        else:
            # For OLS, use Polars expressions
            if len(cluster_cols) == 1:
                return _compute_se_cluster_oneway_polars(
                    df=df,
                    x_cols=x_cols,
                    XtX_inv=XtX_inv,
                    resid=resid,
                    weights=weights,
                    cluster_col=cluster_cols[0],
                    n_obs=n_obs,
                    df_resid=df_resid,
                    ssc=ssc
                )
            else:
                return _compute_se_cluster_multiway_polars(
                    df=df,
                    x_cols=x_cols,
                    XtX_inv=XtX_inv,
                    resid=resid,
                    weights=weights,
                    cluster_cols=cluster_cols,
                    n_obs=n_obs,
                    df_resid=df_resid,
                    ssc=ssc
                )
    
    else:
        raise ValueError(f"Unknown vcov type: {vcov}")


# ============================================================================
# IID STANDARD ERRORS
# ============================================================================

def _compute_se_iid_polars(
    XtX_inv: np.ndarray,
    resid: np.ndarray,
    weights: str | None,
    df: pl.DataFrame,
    df_resid: int
) -> tuple[np.ndarray, None]:
    """
    Compute IID (homoskedastic) standard errors.
    
    Uses Polars expression for weighted sum to compute sigma^2.
    """
    # Add residuals to dataframe temporarily
    df_with_resid = df.with_columns(pl.lit(resid).alias("_resid"))
    
    if weights is not None:
        # sigma^2 = sum(w * resid^2) / df_resid
        sigma2 = df_with_resid.select(
            (pl.col(weights) * pl.col("_resid").pow(2)).sum() / df_resid
        ).item()
    else:
        # sigma^2 = sum(resid^2) / df_resid
        sigma2 = df_with_resid.select(
            pl.col("_resid").pow(2).sum() / df_resid
        ).item()
    
    se = np.sqrt(np.maximum(sigma2 * np.diag(XtX_inv), 0.0))
    return se, None


# ============================================================================
# HC1 ROBUST STANDARD ERRORS
# ============================================================================

def _compute_se_hc1_polars(
    df: pl.DataFrame,
    x_cols: list[str],
    XtX_inv: np.ndarray,
    resid: np.ndarray,
    weights: str | None,
    n_obs: int,
    df_resid: int
) -> tuple[np.ndarray, None]:
    """
    Compute heteroskedasticity-robust (HC1) standard errors using Polars expressions.
    
    Meat matrix: X'diag(resid^2)X
    
    This implementation uses Polars expressions to compute all upper triangle
    elements of the meat matrix in a single select statement.
    """
    k = len(x_cols)
    
    # Add residuals to dataframe
    df_with_resid = df.with_columns(pl.lit(resid).alias("_resid"))
    
    # Build expressions for upper triangle of meat matrix
    meat_exprs = []
    for i in range(k):
        for j in range(i, k):
            col_i, col_j = x_cols[i], x_cols[j]
            
            if weights is not None:
                # Sum(w * x_i * x_j * resid^2)
                expr = (
                    pl.col(weights) * 
                    pl.col(col_i) * 
                    pl.col(col_j) * 
                    pl.col("_resid").pow(2)
                ).sum().alias(f"m_{i}_{j}")
            else:
                # Sum(x_i * x_j * resid^2)
                expr = (
                    pl.col(col_i) * 
                    pl.col(col_j) * 
                    pl.col("_resid").pow(2)
                ).sum().alias(f"m_{i}_{j}")
            
            meat_exprs.append(expr)
    
    # Compute all meat elements in one pass
    meat_result = df_with_resid.select(meat_exprs)
    
    # Extract values and build symmetric meat matrix
    meat = np.zeros((k, k))
    idx = 0
    for i in range(k):
        for j in range(i, k):
            val = meat_result[f"m_{i}_{j}"].item()
            meat[i, j] = meat[j, i] = val
            idx += 1
    
    # Compute variance-covariance matrix
    vcov_matrix = XtX_inv @ meat @ XtX_inv
    
    # HC1 adjustment: n / (n - k)
    adjustment = n_obs / df_resid
    se = np.sqrt(np.maximum(adjustment * np.diag(vcov_matrix), 0.0))
    
    return se, None


# ============================================================================
# ONE-WAY CLUSTERED STANDARD ERRORS
# ============================================================================

def _compute_se_cluster_oneway_polars(
    df: pl.DataFrame,
    x_cols: list[str],
    XtX_inv: np.ndarray,
    resid: np.ndarray,
    weights: str | None,
    cluster_col: str,
    n_obs: int,
    df_resid: int,
    ssc: bool
) -> tuple[np.ndarray, int]:
    """
    Compute one-way clustered standard errors using Polars group_by.
    
    This implementation:
    1. Adds residuals to dataframe
    2. Groups by cluster
    3. Computes cluster-level scores using expressions
    4. Builds meat matrix from scores
    
    This approach is much faster than scipy.sparse for large datasets.
    """
    k = len(x_cols)
    
    # Add residuals to dataframe
    df_with_resid = df.with_columns(pl.lit(resid).alias("_resid"))
    
    # Build expressions for cluster-level scores
    if weights is not None:
        score_exprs = [
            (pl.col(col) * pl.col("_resid") * pl.col(weights)).sum().alias(f"score_{i}")
            for i, col in enumerate(x_cols)
        ]
    else:
        score_exprs = [
            (pl.col(col) * pl.col("_resid")).sum().alias(f"score_{i}")
            for i, col in enumerate(x_cols)
        ]
    
    # Group by cluster and compute scores
    scores_df = df_with_resid.group_by(cluster_col).agg(score_exprs)
    n_clusters = len(scores_df)
    
    # Extract scores as numpy array
    scores = scores_df.select([f"score_{i}" for i in range(k)]).to_numpy()
    
    # Meat matrix: S'S where S is cluster scores
    meat = scores.T @ scores
    
    # Small sample correction
    if ssc:
        adjustment = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / df_resid)
    else:
        adjustment = n_clusters / (n_clusters - 1)
    
    vcov_matrix = adjustment * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.maximum(np.diag(vcov_matrix), 0.0))
    
    return se, n_clusters


# ============================================================================
# MULTI-WAY CLUSTERED STANDARD ERRORS
# ============================================================================

def _compute_se_cluster_multiway_polars(
    df: pl.DataFrame,
    x_cols: list[str],
    XtX_inv: np.ndarray,
    resid: np.ndarray,
    weights: str | None,
    cluster_cols: list[str],
    n_obs: int,
    df_resid: int,
    ssc: bool
) -> tuple[np.ndarray, tuple]:
    """
    Compute multi-way clustered standard errors using Cameron-Gelbach-Miller (2011).
    
    Uses Polars expressions to compute cluster intersections and scores efficiently.
    
    Implementation matches fixest defaults:
    - G.df = "min": Single G_min/(G_min-1) adjustment at the end
    - Components accumulated WITHOUT per-component G/(G-1) adjustment
    - K adjustment (n-1)/(n-K) applied separately if ssc=True
    """
    k = len(x_cols)
    n_ways = len(cluster_cols)
    vcov_matrix = np.zeros_like(XtX_inv)
    n_clusters_list = []
    
    # Add residuals to dataframe
    df_with_resid = df.with_columns(pl.lit(resid).alias("_resid"))
    
    # Build expressions for scores
    if weights is not None:
        score_exprs = [
            (pl.col(col) * pl.col("_resid") * pl.col(weights)).sum().alias(f"score_{i}")
            for i, col in enumerate(x_cols)
        ]
    else:
        score_exprs = [
            (pl.col(col) * pl.col("_resid")).sum().alias(f"score_{i}")
            for i, col in enumerate(x_cols)
        ]
    
    # Iterate over all non-empty subsets
    for subset_size in range(FIRST_ORDER_SUBSET_SIZE, n_ways + 1):
        sign = (-1) ** (subset_size - 1)
        
        for cluster_subset in combinations(range(n_ways), subset_size):
            # Determine grouping columns
            if subset_size == 1:
                group_cols = [cluster_cols[cluster_subset[0]]]
            else:
                # For multi-way intersections, group by all columns
                group_cols = [cluster_cols[i] for i in cluster_subset]
            
            # Group by cluster intersection and compute scores
            scores_df = df_with_resid.group_by(group_cols).agg(score_exprs)
            n_clust = len(scores_df)
            
            # Track first-order cluster counts
            if subset_size == 1:
                n_clusters_list.append(n_clust)
            
            # Skip if too few clusters
            if n_clust <= 1:
                continue
            
            # Extract scores
            scores = scores_df.select([f"score_{i}" for i in range(k)]).to_numpy()
            
            # Compute meat matrix for this subset
            meat = scores.T @ scores
            
            # Accumulate WITHOUT per-component G/(G-1) adjustment
            vcov_matrix += sign * (XtX_inv @ meat @ XtX_inv)
    
    # Apply single G_min/(G_min-1) adjustment (fixest default with G.df="min")
    if len(n_clusters_list) > 0:
        G_min = min(n_clusters_list)
        if G_min > MIN_CLUSTERS_FOR_ADJUSTMENT:
            vcov_matrix *= (G_min / (G_min - 1))
    
    # Apply K small-sample correction if requested
    if ssc:
        vcov_matrix *= ((n_obs - 1) / df_resid)
    
    se = np.sqrt(np.maximum(np.diag(vcov_matrix), 0.0))
    n_clusters = tuple(n_clusters_list)
    
    return se, n_clusters


# ============================================================================
# IV/2SLS STANDARD ERRORS (numpy-based for X_hat compatibility)
# ============================================================================

def _compute_se_hc1_iv(
    XtX_inv: np.ndarray,
    resid: np.ndarray,
    X: np.ndarray,
    weights_array: np.ndarray | None,
    n_obs: int,
    df_resid: int
) -> tuple[np.ndarray, None]:
    """
    Compute HC1 standard errors for IV/2SLS using X_hat from first stage.
    
    For IV, we need X_hat (fitted values from first stage), not the original X.
    This requires numpy-based computation rather than Polars expressions.
    """
    if weights_array is not None:
        meat = X.T @ (X * (weights_array * resid**2)[:, np.newaxis])
    else:
        meat = X.T @ (X * (resid**2)[:, np.newaxis])
    
    vcov_matrix = XtX_inv @ meat @ XtX_inv
    adjustment = n_obs / df_resid
    se = np.sqrt(np.maximum(np.diag(vcov_matrix) * adjustment, 0.0))
    return se, None


def _compute_se_cluster_oneway_iv(
    XtX_inv: np.ndarray,
    resid: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray | None,
    cluster_ids: np.ndarray,
    n_obs: int,
    df_resid: int,
    ssc: bool
) -> tuple[np.ndarray, int]:
    """
    Compute one-way clustered standard errors for IV/2SLS.
    
    Uses scipy.sparse for cluster aggregation since we need X_hat matrix.
    """
    from scipy import sparse
    
    unique_clusters, cluster_map = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(unique_clusters)
    
    # Build sparse cluster indicator matrix
    cluster_indicator_matrix = sparse.csr_matrix(
        (np.ones(n_obs), (np.arange(n_obs), cluster_map)),
        shape=(n_obs, n_clusters)
    )
    
    # Compute scores
    if weights is not None:
        X_resid = X * (resid * weights)[:, np.newaxis]
    else:
        X_resid = X * resid[:, np.newaxis]
    
    scores = cluster_indicator_matrix.T @ X_resid
    meat = scores.T @ scores
    
    # Adjustment
    if ssc:
        adjustment = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / df_resid)
    else:
        adjustment = n_clusters / (n_clusters - 1)
    
    vcov_matrix = XtX_inv @ meat @ XtX_inv * adjustment
    se = np.sqrt(np.maximum(np.diag(vcov_matrix), 0.0))
    
    return se, n_clusters


def _compute_se_cluster_multiway_iv(
    XtX_inv: np.ndarray,
    resid: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray | None,
    cluster_ids: np.ndarray,
    n_obs: int,
    df_resid: int,
    ssc: bool
) -> tuple[np.ndarray, tuple]:
    """
    Compute multi-way clustered standard errors for IV/2SLS.
    
    Uses scipy.sparse for cluster aggregation since we need X_hat matrix.
    Implements Cameron-Gelbach-Miller (2011) with fixest-compatible SSC.
    """
    from scipy import sparse
    
    n_ways = cluster_ids.shape[1]
    vcov_matrix = np.zeros_like(XtX_inv)
    n_clusters_list = []
    
    # Sum over all non-empty subsets with alternating signs
    for subset_size in range(FIRST_ORDER_SUBSET_SIZE, n_ways + 1):
        sign = (-1) ** (subset_size - 1)
 
        for cluster_subset_indices in combinations(range(n_ways), subset_size):
            # Build intersection cluster IDs
            if subset_size == 1:
                intersect_ids = cluster_ids[:, cluster_subset_indices[0]]
            else:
                arr = np.column_stack([cluster_ids[:, j] for j in cluster_subset_indices])
                _, cluster_map = np.unique(arr, axis=0, return_inverse=True)
                intersect_ids = cluster_map

            unique_clusters, cluster_map = np.unique(intersect_ids, return_inverse=True)
            n_clusters = len(unique_clusters)
            
            # Skip subsets with too few clusters
            if n_clusters <= 1:
                if subset_size == 1:
                    n_clusters_list.append(n_clusters)
                continue

            if subset_size == 1:
                n_clusters_list.append(n_clusters)

            # Build cluster indicator matrix
            cluster_indicator_matrix = sparse.csr_matrix(
                (np.ones(n_obs), (np.arange(n_obs), cluster_map)),
                shape=(n_obs, n_clusters)
            )
            
            # Compute scores
            if weights is not None:
                X_resid = X * (resid * weights)[:, np.newaxis]
            else:
                X_resid = X * resid[:, np.newaxis]

            scores = cluster_indicator_matrix.T @ X_resid
            meat = scores.T @ scores

            # Accumulate WITHOUT per-component G/(G-1) adjustment
            vcov_matrix += sign * (XtX_inv @ meat @ XtX_inv)

    # Apply single G_min/(G_min-1) adjustment (fixest default with G.df="min")
    if len(n_clusters_list) > 0:
        G_min = min(n_clusters_list)
        if G_min > MIN_CLUSTERS_FOR_ADJUSTMENT:
            vcov_matrix *= (G_min / (G_min - 1))

    # Apply K small-sample correction if requested
    if ssc:
        vcov_matrix *= ((n_obs - 1) / df_resid)

    se = np.sqrt(np.maximum(np.diag(vcov_matrix), 0.0))
    n_clusters = tuple(n_clusters_list)
    
    return se, n_clusters


# ============================================================================
# DUCKDB STANDARD ERRORS (SQL-based aggregation)
# ============================================================================

def compute_standard_errors_duckdb(
    con: 'DuckDBPyConnection',
    tmp_table: str,
    x_cols: list[str],
    XtX_inv: np.ndarray,
    weights: str | None,
    vcov: Literal["iid", "HC1", "cluster"],
    cluster_cols: list[str] | None,
    n_obs: int,
    df_resid: int,
    ssc: bool = True
) -> tuple[np.ndarray, int | tuple | None]:
    """
    Compute standard errors using DuckDB SQL aggregation.
    
    This implementation uses SQL queries for efficient aggregation,
    which is the key to DuckDB's performance advantage.
    
    Parameters
    ----------
    con : DuckDBPyConnection
        DuckDB connection
    tmp_table : str
        Name of temporary table containing demeaned data and residuals
    x_cols : list[str]
        Names of regressor columns (with _dm suffix)
    XtX_inv : np.ndarray
        (X'X)^-1 matrix
    weights : str, optional
        Name of weights column
    vcov : {"iid", "HC1", "cluster"}
        Variance-covariance estimator type
    cluster_cols : list[str], optional
        Clustering variable names
    n_obs : int
        Number of observations
    df_resid : int
        Residual degrees of freedom
    ssc : bool, default True
        Apply small sample correction
        
    Returns
    -------
    tuple
        (standard_errors, n_clusters)
        
    Notes
    -----
    This function assumes residuals are already stored in the table
    as '_resid' column and regressors have '_dm' suffix.
    """
    k = len(x_cols)
    
    if vcov == "iid":
        return _compute_se_iid_duckdb(
            con=con,
            tmp_table=tmp_table,
            XtX_inv=XtX_inv,
            weights=weights,
            df_resid=df_resid
        )
    
    elif vcov == "HC1":
        return _compute_se_hc1_duckdb(
            con=con,
            tmp_table=tmp_table,
            x_cols=x_cols,
            XtX_inv=XtX_inv,
            weights=weights,
            n_obs=n_obs,
            df_resid=df_resid
        )
    
    elif vcov == "cluster":
        if cluster_cols is None:
            raise ValueError("cluster_cols required for vcov='cluster'")
        
        if len(cluster_cols) == 1:
            return _compute_se_cluster_oneway_duckdb(
                con=con,
                tmp_table=tmp_table,
                x_cols=x_cols,
                XtX_inv=XtX_inv,
                weights=weights,
                cluster_col=cluster_cols[0],
                n_obs=n_obs,
                df_resid=df_resid,
                ssc=ssc
            )
        else:
            return _compute_se_cluster_multiway_duckdb(
                con=con,
                tmp_table=tmp_table,
                x_cols=x_cols,
                XtX_inv=XtX_inv,
                weights=weights,
                cluster_cols=cluster_cols,
                n_obs=n_obs,
                df_resid=df_resid,
                ssc=ssc
            )
    
    else:
        raise ValueError(f"Unknown vcov type: {vcov}")


def _compute_se_iid_duckdb(
    con: 'DuckDBPyConnection',
    tmp_table: str,
    XtX_inv: np.ndarray,
    weights: str | None,
    df_resid: int
) -> tuple[np.ndarray, None]:
    """Compute IID standard errors using DuckDB SQL."""
    if weights is not None:
        sigma2 = con.execute(
            f"SELECT SUM({weights} * _resid * _resid) / {df_resid} FROM {tmp_table}"
        ).fetchone()
    else:
        sigma2 = con.execute(
            f"SELECT SUM(_resid * _resid) / {df_resid} FROM {tmp_table}"
        ).fetchone()
    
    if sigma2 is None:
        raise ValueError('Could not compute sigma²')
    
    sigma2 = sigma2[0]
    se = np.sqrt(np.maximum(sigma2 * np.diag(XtX_inv), 0.0))
    return se, None


def _compute_se_hc1_duckdb(
    con: 'DuckDBPyConnection',
    tmp_table: str,
    x_cols: list[str],
    XtX_inv: np.ndarray,
    weights: str | None,
    n_obs: int,
    df_resid: int
) -> tuple[np.ndarray, None]:
    """
    Compute HC1 standard errors using DuckDB SQL.
    
    Computes all upper triangle elements of meat matrix in single query.
    """
    k = len(x_cols)
    meat_elements = []
    
    for i in range(k):
        for j in range(i, k):
            col_i, col_j = x_cols[i], x_cols[j]
            if weights is not None:
                expr = f"SUM({weights} * {col_i}_dm * {col_j}_dm * _resid * _resid)"
            else:
                expr = f"SUM({col_i}_dm * {col_j}_dm * _resid * _resid)"
            meat_elements.append(f"{expr} AS m_{i}_{j}")
    
    meat_row = con.execute(
        f"SELECT {', '.join(meat_elements)} FROM {tmp_table}"
    ).fetchone()
    
    if meat_row is None:
        raise ValueError('Could not compute meat matrix')
    
    # Build symmetric meat matrix
    vals = list(meat_row)
    meat = np.zeros((k, k))
    idx = 0
    for i in range(k):
        for j in range(i, k):
            val = float(vals[idx] or 0.0)
            meat[i, j] = meat[j, i] = val
            idx += 1
    
    vcov_matrix = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum((n_obs / df_resid) * np.diag(vcov_matrix), 0.0))
    return se, None


def _compute_se_cluster_oneway_duckdb(
    con: 'DuckDBPyConnection',
    tmp_table: str,
    x_cols: list[str],
    XtX_inv: np.ndarray,
    weights: str | None,
    cluster_col: str,
    n_obs: int,
    df_resid: int,
    ssc: bool
) -> tuple[np.ndarray, int]:
    """
    Compute one-way clustered standard errors using DuckDB SQL.
    
    Uses GROUP BY for cluster-level aggregation.
    """
    k = len(x_cols)
    
    # Build score expressions
    if weights is not None:
        score_exprs = [
            f"SUM({col}_dm * _resid * {weights}) AS score_{i}"
            for i, col in enumerate(x_cols)
        ]
    else:
        score_exprs = [
            f"SUM({col}_dm * _resid) AS score_{i}"
            for i, col in enumerate(x_cols)
        ]
    
    cluster_query = f"""
        SELECT {cluster_col} AS cluster_id, {', '.join(score_exprs)}
        FROM {tmp_table}
        GROUP BY 1
    """
    
    score_df = con.execute(cluster_query).pl()
    n_clusters = len(score_df)
    
    # Extract scores and compute meat matrix
    S = score_df.select([f"score_{i}" for i in range(k)]).to_numpy()
    meat = S.T @ S
    
    # Small sample correction
    if ssc:
        adjustment = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / df_resid)
    else:
        adjustment = n_clusters / (n_clusters - 1)
    
    vcov_matrix = adjustment * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.maximum(np.diag(vcov_matrix), 0.0))
    
    return se, n_clusters


def _compute_se_cluster_multiway_duckdb(
    con: 'DuckDBPyConnection',
    tmp_table: str,
    x_cols: list[str],
    XtX_inv: np.ndarray,
    weights: str | None,
    cluster_cols: list[str],
    n_obs: int,
    df_resid: int,
    ssc: bool
) -> tuple[np.ndarray, tuple]:
    """
    Compute multi-way clustered standard errors using DuckDB SQL.
    
    Uses CONCAT_WS for cluster intersections and GROUP BY for aggregation.
    Implements Cameron-Gelbach-Miller (2011) with fixest-compatible SSC.
    """
    k = len(x_cols)
    n_ways = len(cluster_cols)
    vcov_matrix = np.zeros_like(XtX_inv)
    n_clusters_list = []
    
    for subset_size in range(FIRST_ORDER_SUBSET_SIZE, n_ways + 1):
        sign = (-1) ** (subset_size - 1)
        
        for cluster_subset in combinations(range(n_ways), subset_size):
            # Build cluster ID SQL expression
            if subset_size == 1:
                cluster_id_sql = cluster_cols[cluster_subset[0]]
            else:
                cols_to_concat = [cluster_cols[i] for i in cluster_subset]
                cluster_id_sql = f"CONCAT_WS('_', {', '.join(cols_to_concat)})"
            
            # Build score expressions
            if weights is not None:
                score_exprs = [
                    f'SUM("{col}_dm" * _resid * {weights}) AS score_{i}'
                    for i, col in enumerate(x_cols)
                ]
            else:
                score_exprs = [
                    f'SUM("{col}_dm" * _resid) AS score_{i}'
                    for i, col in enumerate(x_cols)
                ]
            
            cluster_query = f"""
                SELECT {cluster_id_sql} AS cluster_id, {', '.join(score_exprs)}
                FROM {tmp_table}
                GROUP BY 1
            """
            
            score_df = con.execute(cluster_query).pl()
            n_clust = len(score_df)
            
            if subset_size == 1:
                n_clusters_list.append(n_clust)
            
            if n_clust <= 1:
                continue
            
            # Extract scores and compute meat matrix
            S = score_df.select([f"score_{i}" for i in range(k)]).to_numpy()
            meat = S.T @ S
            
            # Accumulate WITHOUT per-component G/(G-1) adjustment
            vcov_matrix += sign * (XtX_inv @ meat @ XtX_inv)
    
    # Apply single G_min/(G_min-1) adjustment (fixest default)
    if len(n_clusters_list) > 0:
        G_min = min(n_clusters_list)
        if G_min > MIN_CLUSTERS_FOR_ADJUSTMENT:
            vcov_matrix *= (G_min / (G_min - 1))
    
    # Apply K small-sample correction if requested
    if ssc:
        vcov_matrix *= ((n_obs - 1) / df_resid)
    
    se = np.sqrt(np.maximum(np.diag(vcov_matrix), 0.0))
    n_clusters = tuple(n_clusters_list)
    
    return se, n_clusters