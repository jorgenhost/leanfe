"""
YOCO (You Only Compress Once) compression strategy for leanfe.

Implements the optimal data compression strategy from Wong et al. (2021):
"You Only Compress Once: Optimal Data Compression for Estimating Linear Models"

This strategy compresses data by grouping on (regressors + fixed effects),
computing sufficient statistics (n, sum_y, sum_y_sq), and running weighted
least squares on the compressed data.

Used automatically when vcov is "iid" or "HC1" (not cluster) and no IV.
"""
from leanfe.result import LeanFEResult

import numpy as np
import polars as pl
from scipy import sparse
from duckdb import DuckDBPyConnection
from typing import Any, Callable, TypeAlias
ArrayLike: TypeAlias = np.ndarray | sparse.sparray | sparse.csr_matrix

def determine_strategy(
    vcov: str,
    has_instruments: bool,
    fe_cardinality: dict[str, int] | None = None,
    max_fe_levels: int = 10_000,
    n_obs: int | None = None,
    n_x_cols: int | None = None,
    estimated_compression_ratio: float | None = None,
) -> str:
    """
    Determine if compression strategy should be used.
    
    The decision is based on estimating which approach is faster:
    
    YOCO Compression:
    - Cost ≈ O(n_obs) for GROUP BY + O(n_compressed * total_fe_levels) for sparse matrix
    - Fast when: good compression ratio AND low total FE levels
    
    FWL Demeaning:
    - Cost ≈ O(n_obs * n_fe * n_iterations) for iterative demeaning
    - Fast when: high-cardinality FEs (avoids huge sparse matrix)
    
    Parameters
    ----------
    vcov : str
        Variance-covariance type
    has_instruments : bool
        Whether IV/2SLS is being used
    fe_cardinality : dict, optional
        Dictionary mapping FE column names to their cardinality.
    max_fe_levels : int, default 10_000
        Maximum FE cardinality to allow compression.
    n_obs : int, optional
        Number of observations (for cost estimation)
    n_x_cols : int, optional
        Number of X columns (for cost estimation)
    estimated_compression_ratio : float, optional
        Estimated compression ratio (n_compressed / n_obs). If provided, used
        for more accurate cost estimation.
        
    Returns
    -------
    bool
        True if compression should be used
    """
    # Basic checks
    vcov_ok = vcov.lower() in ("iid", "hc1", "cluster")
    if not vcov_ok or has_instruments:
        return 'alt_proj'
    
    if fe_cardinality is None:
        return 'compress'  # Default to compression if no info
    
    # Calculate total FE levels
    total_fe_levels = sum(fe_cardinality.values())
    max_single_fe = max(fe_cardinality.values()) if fe_cardinality else 0
    
    # Rule 1: If any single FE is very high-cardinality, use FWL
    # Building a sparse matrix with 100K+ columns is slow
    if max_single_fe > max_fe_levels:
        return 'alt_proj'
    
    # Rule 2: If total FE levels is very high, use FWL
    # Even if no single FE is huge, many medium FEs can add up
    if total_fe_levels > max_fe_levels * 2:
        return 'alt_proj'
    
    # Rule 3: If we have compression ratio estimate, use cost model
    if estimated_compression_ratio is not None and n_obs is not None:
        n_compressed = int(n_obs * estimated_compression_ratio)
        
        # Estimate YOCO cost: GROUP BY + sparse matrix build + WLS solve
        # Sparse matrix: n_compressed rows × total_fe_levels columns
        yoco_cost = n_obs + n_compressed * total_fe_levels + total_fe_levels ** 2
        
        # Estimate FWL cost: ~10 iterations × n_fe × n_obs GROUP BY operations
        n_fe = len(fe_cardinality)
        fwl_cost = 10 * n_fe * n_obs
        
        # Use YOCO if it's estimated to be faster
        if yoco_cost < fwl_cost:
            return 'compress'
        return 'alt_proj' 
    

    # Default: use compression for low-cardinality FEs
    return 'compress'


def compress_polars(
    lf: pl.LazyFrame,
    y_col: str,
    x_cols: list[str],
    fe_cols: list[str],
    weights: str | None = None,
    cluster_col: str | None = None
) -> tuple[pl.DataFrame, int]:
    """
    Compress data using GROUP BY on regressors + fixed effects.
    
    For cluster SEs (Section 5.3.1 of YOCO paper), include cluster_col
    in the grouping to ensure each compressed record belongs to one cluster.
    
    Parameters
    ----------
    lf : pl.LazyFrame
        Input data
    y_col : str
        Dependent variable column
    x_cols : list of str
        Regressor columns
    fe_cols : list of str
        Fixed effect columns
    weights : str, optional
        Weight column name
    cluster_col : str, optionalestimate_compression_ratio
        Cluster column for within-cluster compression
        
    Returns
    -------
    tuple
        (compressed_df, n_obs_original)
    """
    group_cols = x_cols + fe_cols
    # For cluster SEs, add cluster to grouping (Section 5.3.1)
    if cluster_col is not None and cluster_col not in group_cols:
        group_cols = group_cols + [cluster_col]
    
    n_obs_original = lf.select(pl.len()).collect().item()
    
    if weights is not None:
        # Weighted compression
        agg_exprs = [
            pl.col(weights).sum().alias("_n"),
            (pl.col(y_col) * pl.col(weights)).sum().alias("_sum_y"),
            (pl.col(y_col).pow(2) * pl.col(weights)).sum().alias("_sum_y_sq"),
        ]
    else:
        agg_exprs = [
            pl.len().alias("_n"),
            pl.col(y_col).sum().alias("_sum_y"),
            pl.col(y_col).pow(2).sum().alias("_sum_y_sq"),
        ]
    
    compressed = lf.group_by(group_cols).agg(agg_exprs).collect()
    
    # Add mean_y and sqrt weights for WLS
    compressed = compressed.with_columns([
        (pl.col("_sum_y") / pl.col("_n")).alias("_mean_y"),
        pl.col("_n").sqrt().alias("_wts"),
    ])
    
    return compressed, n_obs_original


class DuckDBResult:
    """Wrapper for DuckDB numpy result to provide dict-like access."""
    def __init__(self, data: dict[str, Any]):
        self._data = data
    
    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]
    
    def __len__(self) -> int:
        # Return length of first array
        return len(next(iter(self._data.values())))


def estimate_compression_ratio(
    x_cols: list[str],
    fe_cols: list[str],
    data: str | pl.LazyFrame | None = None,
    con: DuckDBPyConnection | None = None,
) -> float:
    """
    Estimate the compression ratio of the model design.

    Calculates the number of unique groups formed by the combination of 
    regressors and fixed effects, divided by the total number of observations. 
    This serves as a heuristic for degrees of freedom and data sparsity.

    Parameters
    ----------
    x_cols : list of str
        List of regressor column names.
    fe_cols : list of str
        List of fixed effect column names.
    data : str or pl.LazyFrame, optional
        The dataset to analyze. If a string is provided, it is treated as a 
        table name within the DuckDB connection.
    con : DuckDBPyConnection, optional
        An active DuckDB connection. Required if `data` is a table name 
        or if executing via SQL.

    Returns
    -------
    float
        The compression ratio (between 0 and 1). A value closer to 0 indicates 
        significant grouping/redundancy.

    Notes
    -----
    Inspired by the `dbreg` R package:
    https://github.com/grantmcdermott/dbreg/blob/main/R/dbreg.R#L587-L654
    """
    
    key_cols = list(set(x_cols + fe_cols))
    if not key_cols:
        return 1.0

    if con is not None:
        # If data is a path, we query the file directly; otherwise use 'raw_data' view
        table_ref = f"read_parquet('{data}')" if isinstance(data, str) else "raw_data"
        
        # 1. Total rows
        res = con.execute(f"SELECT COUNT(*)::BIGINT FROM {table_ref}").fetchone()
        if res is None:
            raise RuntimeError("Query failed to return a count.")
        total_n = int(res[0])
        
        # 2. Count distinct tuples (compressed size)
        cols_expr = ", ".join(key_cols)
        distinct_sql = f"SELECT COUNT(*)::BIGINT FROM (SELECT DISTINCT {cols_expr} FROM {table_ref}) t"
        distinct_res = con.execute(distinct_sql).fetchone()
        if distinct_res is None:
            raise RuntimeError("Query failed to return a distinct count.")
        n_groups_total = int(distinct_res[0])
                
    else:
        if isinstance(data, str):
            lf = pl.scan_parquet(data)
        if isinstance(data, pl.LazyFrame):
            lf = data
            total_n = lf.select(pl.len()).collect().item() 
            # Unique combinations of all relevant columns
            n_groups_total = lf.select(key_cols).unique().select(pl.len()).collect().item()

    comp_rat = n_groups_total / max(total_n, 1)
    return comp_rat

def compress_duckdb(
    con: DuckDBPyConnection,
    y_col: str,
    x_cols: list[str],
    fe_cols: list[str],
    weights: str | None = None,
    cluster_col: str | None = None
) -> tuple[DuckDBResult, int]:
    """
    Compress data using SQL GROUP BY.
    
    For cluster SEs (Section 5.3.1 of YOCO paper), include cluster_col
    in the grouping to ensure each compressed record belongs to one cluster.
    
    Parameters
    ----------
    con : duckdb.Connection
        DuckDB connection with 'data' table
    y_col : str
        Dependent variable column
    x_cols : list of str
        Regressor columns
    fe_cols : list of str
        Fixed effect columns
    weights : str, optional
        Weight column name
    cluster_col : str, optional
        Cluster column for within-cluster compression
        
    Returns
    -------
    tuple
        (compressed_data as DuckDBResult, n_obs_original)
    """
    group_cols = x_cols + fe_cols
    # For cluster SEs, add cluster to grouping (Section 5.3.1)
    if cluster_col is not None and cluster_col not in group_cols:
        group_cols = group_cols + [cluster_col]
    
    group_cols_sql = ", ".join(group_cols)
    res = con.execute("SELECT COUNT(*) FROM data").fetchone()
    if res is None:
        raise RuntimeError("Query failed to return a count.")
    n_obs_original = int(res[0])

    if weights is not None:
        query = f"""
        SELECT
            {group_cols_sql},
            SUM({weights}) AS _n,
            SUM({y_col} * {weights}) AS _sum_y,
            SUM(POWER({y_col}, 2) * {weights}) AS _sum_y_sq,
            SUM({y_col} * {weights}) / SUM({weights}) AS _mean_y,
            SQRT(SUM({weights})) AS _wts
        FROM data
        GROUP BY {group_cols_sql}
        """
    else:
        query = f"""
        SELECT
            {group_cols_sql},
            COUNT(*) AS _n,
            SUM({y_col}) AS _sum_y,
            SUM(POWER({y_col}, 2)) AS _sum_y_sq,
            SUM({y_col}) / COUNT(*) AS _mean_y,
            SQRT(COUNT(*)) AS _wts
        FROM data
        GROUP BY {group_cols_sql}
        """
    
    # Use fetchnumpy() to avoid pandas dependency
    result = con.execute(query).fetchnumpy()
    compressed = DuckDBResult(result)
    return compressed, n_obs_original


def _extract_numpy_arrays(
    compressed_df: pl.DataFrame | DuckDBResult, 
    x_cols: list[str], 
) -> tuple[
    np.ndarray,                       # X_reg
    np.ndarray,                       # Y
    np.ndarray,                       # wts
    Callable[[str | list[str]], Any]  # get_fe_values
]:
    """Extract numpy arrays from compressed dataframe without pandas dependency."""
    if isinstance(compressed_df, pl.DataFrame):
        # Use Polars native to_numpy() - no pandas needed
        X_reg = compressed_df.select(pl.ones(pl.len()), pl.col(x_cols)).to_numpy()
        Y = compressed_df.select(pl.col("_mean_y")).to_numpy().flatten()
        wts = compressed_df.select(pl.col("_wts")).to_numpy().flatten()
        
        def get_fe_values(fe: str | list[str]):
            return compressed_df.select(fe).to_numpy()
    else:
        # DuckDB returns DuckDBResult (dict-like with numpy arrays)
        X_reg = np.column_stack([compressed_df[col] for col in x_cols])
        X_reg = np.column_stack((np.ones(shape=(X_reg.shape[0], 1)), X_reg))
        Y = compressed_df["_mean_y"]
        wts = compressed_df["_wts"]
        
        def get_fe_values(fe: str | list[str]):
            return compressed_df[fe]
    
    return X_reg, Y, wts, get_fe_values  

def build_design_matrix(
    compressed_df: pl.DataFrame | DuckDBResult,
    x_cols: list[str],
    fe_cols: list[str],
    use_sparse: bool = True
) -> tuple[ArrayLike, np.ndarray, np.ndarray,    list[str], int]:
    """
    Build design matrix from compressed data with FE dummies.
    
    Returns
    -------
    tuple
        (X, Y, wts, all_col_names, n_fe_levels)
        all_col_names now includes '(Intercept)' as first element
    """
    # Extract numpy arrays without pandas dependency
    X_reg, Y, wts, get_fe_values = _extract_numpy_arrays(compressed_df, x_cols)
    n_rows = len(Y)

    # Create column names with intercept
    all_x_cols = ['(Intercept)'] + list(x_cols)
    
    if not fe_cols:
        return X_reg, Y, wts, all_x_cols, 0
    
    if use_sparse:        # Build sparse FE dummies efficiently using COO format
        fe_col_names = []
        n_fe_levels = 0
        col_offset = 0
        
        # Collect sparse matrix components
        rows_list = []
        cols_list = []
        data_list = []
        
        for fe in fe_cols:
            fe_values = get_fe_values(fe).flatten()
            categories = np.unique(fe_values)
            n_cats = len(categories)
            n_fe_levels += n_cats
            
            # Create mapping from category to column index (drop first for identification)
            cat_to_col = {cat: i + col_offset for i, cat in enumerate(categories[1:])}
            
            # Add column names
            for cat in categories[1:]:
                fe_col_names.append(f"{fe}_{cat}")
            
            # Build sparse entries for this FE
            for row_idx, val in enumerate(fe_values):
                if val in cat_to_col:
                    rows_list.append(row_idx)
                    cols_list.append(cat_to_col[val])
                    data_list.append(1.0)
            
            col_offset += n_cats - 1  # -1 because we drop first category
        
        # Create sparse FE matrix
        n_fe_cols = col_offset
        if n_fe_cols > 0:
            X_fe_sparse = sparse.coo_matrix(
                (data_list, (rows_list, cols_list)),
                shape=(n_rows, n_fe_cols)
            ).tocsr()
            
            # Combine: [X_reg | X_fe] as sparse
            X_reg_sparse = sparse.csr_matrix(X_reg)
            X = sparse.hstack([X_reg_sparse, X_fe_sparse], format='csr')
        else:
            X = sparse.csr_matrix(X_reg)
        
        all_col_names = all_x_cols + fe_col_names  # Use all_x_cols instead of x_cols
        return X, Y, wts, all_col_names, n_fe_levels
    
    else:
        # Original dense implementation
        fe_dummies = []
        fe_col_names = []
        n_fe_levels = 0
        
        for fe in fe_cols:
            fe_values = get_fe_values(fe)
            categories = np.unique(fe_values)
            n_cats = len(categories)
            n_fe_levels += n_cats
            
            # Create dummy matrix (drop first category for identification)
            for cat in categories[1:]:
                col_name = f"{fe}_{cat}"
                dummy = (fe_values == cat).astype(float)
                fe_dummies.append(dummy)
                fe_col_names.append(col_name)
        
        # Combine regressors and FE dummies
        if fe_dummies:
            X_fe = np.column_stack(fe_dummies)
            X = np.hstack([X_reg, X_fe])
            all_col_names = all_x_cols + fe_col_names  # Use all_x_cols
        else:
            X = X_reg
            all_col_names = all_x_cols  # Use all_x_cols
        
        return X, Y, wts, all_col_names, n_fe_levels


def solve_wls(
    X: ArrayLike,
    Y: np.ndarray,
    wts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve weighted least squares.
    
    Parameters
    ----------
    X : np.ndarray or scipy.sparse matrix
        Design matrix (G x p)
    Y : np.ndarray
        Response vector (G,) - group means
    wts : np.ndarray
        Weights (G,) - sqrt(n_g)
        
    Returns
    -------
    tuple
        (beta, XtX_inv)
    """
    is_sparse = sparse.issparse(X)
    
    if is_sparse:
        # Sparse weighted least squares
        # Weight the design matrix: diag(wts) @ X
        Xw = X * wts[:, np.newaxis]
        Yw = Y * wts
        
        # X'X is dense (p x p is typically small)
        XtX = (Xw.T @ Xw).toarray()
        Xty = Xw.T @ Yw
        
        # Solve using dense methods (XtX is small)
        try:
            L = np.linalg.cholesky(XtX)
            beta = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
            XtX_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(XtX.shape[0])))
        except np.linalg.LinAlgError:
            beta, _, _, _ = np.linalg.lstsq(XtX, Xty, rcond=None)
            XtX_inv = np.linalg.pinv(XtX)
    else:
        # Dense weighted least squares
        Xw = X * wts[:, np.newaxis]
        Yw = Y * wts
        
        XtX = Xw.T @ Xw
        Xty = Xw.T @ Yw
        
        try:
            L = np.linalg.cholesky(XtX)
            beta = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
            XtX_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(XtX.shape[0])))
        except np.linalg.LinAlgError:
            beta, _, _, _ = np.linalg.lstsq(XtX, Xty, rcond=None)
            XtX_inv = np.linalg.pinv(XtX)
    
    return beta, XtX_inv


def compute_rss_grouped(
    compressed_df: pl.DataFrame | DuckDBResult,
    X: ArrayLike, 
    beta: np.ndarray,
    backend: str = "polars"
) -> tuple[float, np.ndarray]:
    """
    Compute RSS from sufficient statistics.
    
    RSS = sum_g (sum_y_sq_g - 2 * yhat_g * sum_y_g + n_g * yhat_g^2)
    
    Parameters
    ----------
    compressed_df : DataFrame
        Compressed data with _n, _sum_y, _sum_y_sq
    X : np.ndarray or scipy.sparse matrix
        Design matrix
    beta : np.ndarray
        Coefficient vector
    backend : str
        "polars" or "duckdbbuild_design_matrix"
        
    Returns
    -------
    tuple
        (rss_total, rss_g) - total RSS and per-group RSS
    """
    if backend == "polars":
        n_g = compressed_df["_n"].to_numpy()
        sum_y_g = compressed_df["_sum_y"].to_numpy()
        sum_y_sq_g = compressed_df["_sum_y_sq"].to_numpy()
    else:
        # DuckDB returns DuckDBResult (dict-like with numpy arrays)
        n_g = compressed_df["_n"]
        sum_y_g = compressed_df["_sum_y"]
        sum_y_sq_g = compressed_df["_sum_y_sq"]
    
    # Fitted values for each group (handle sparse)
    if sparse.issparse(X):
        yhat_g = np.asarray(X @ beta).flatten()
    else:
        yhat_g = X @ beta
    
    # Per-group RSS
    rss_g = sum_y_sq_g - 2 * yhat_g * sum_y_g + n_g * (yhat_g ** 2)
    rss_total = np.sum(rss_g)
    
    return rss_total, rss_g


def _build_sparse_cluster_matrix(cluster_ids: np.ndarray) -> tuple[sparse.csr_matrix, int]:
    """
    Build sparse cluster indicator matrix W̃_C from Section 5.3.1.
    
    W̃_C ∈ R^{G×C} where entry (g, c) = 1 if group g belongs to cluster c.
    
    Parameters
    ----------
    cluster_ids : np.ndarray
        Cluster ID for each compressed group (length G)
        
    Returns
    -------
    tuple
        (W_C sparse matrix, n_clusters)
    """
    unique_clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(unique_clusters)
    n_groups = len(cluster_ids)
    
    # Build sparse matrix in COO format: row=group_idx, col=cluster_idx, data=1
    W_C = sparse.csr_matrix(
        (np.ones(n_groups), (np.arange(n_groups), inverse)),
        shape=(n_groups, n_clusters)
    )
    return W_C, n_clusters


def compute_se_compress(
    XtX_inv: np.ndarray,
    rss_total: float,
    rss_g: np.ndarray,
    n_obs: int,
    df_resid: int,
    vcov: str,
    X,  # np.ndarray or sparse matrix
    x_cols: list[str],
    cluster_ids: np.ndarray | None = None,
    e0_g: np.ndarray | None = None,
    ssc: bool = False
) -> tuple[np.ndarray, int | None]:
    """
    Compute standard errors from compressed data.
    
    For cluster SEs, implements Section 5.3.1 of YOCO paper using sparse matrices:
    Ξ̂ = M̃ᵀ diag(ẽ⁰) W̃_C W̃_Cᵀ diag(ẽ⁰) M̃
    
    The sparse cluster matrix W̃_C efficiently aggregates scores within clusters,
    avoiding explicit loops and enabling vectorized computation.
    
    Parameters
    ----------
    XtX_inv : np.ndarray
        Inverse of X'X
    rss_total : float
        Total residual sum of squares
    rss_g : np.ndarray
        Per-group RSS
    n_obs : int
        Original number of observations
    df_resid : int
        Residual degrees of freedom
    vcov : str
        "iid", "HC1", or "cluster"
    X : np.ndarray or scipy.sparse matrix
        Design matrix (compressed)
    x_cols : list of str
        Names of regressor columns (to extract subset of SEs)
    cluster_ids : np.ndarray, optional
        Cluster ID for each compressed group (required for cluster SEs)
    e0_g : np.ndarray, optional
        Sum of residuals per group: ẽ⁰ = ỹ⁰ - ñ ⊙ ŷ (required for cluster SEs)
    ssc : bool
        Small sample correction for cluster SEs
        
    Returns
    -------
    tuple
        (se for x_cols only, n_clusters or None)
    """
    k_x = len(x_cols)
    n_clusters = None
    
    if vcov == "iid":
        sigma2 = rss_total / df_resid
        se_full = np.sqrt(np.diag(XtX_inv) * sigma2)
        
    elif vcov.upper() == "HC1":
        # Meat matrix: X' diag(rss_g) X
        if sparse.issparse(X):
            Xw = X.multiply(rss_g[:, np.newaxis])
            meat = (X.T @ Xw).toarray()
        else:
            meat = X.T @ (X * rss_g[:, np.newaxis])
        vcov_matrix = XtX_inv @ meat @ XtX_inv
        # HC1 adjustment
        adjustment = n_obs / df_resid
        # Use np.maximum to handle numerical precision issues (tiny negative values)
        se_full = np.sqrt(np.maximum(np.diag(vcov_matrix) * adjustment, 0.0))
        
    elif vcov.lower() == "cluster":
        if cluster_ids is None or e0_g is None:
            raise ValueError("cluster_ids and e0_g required for cluster SEs")
        
        # Section 5.3.1: Ξ̂ = M̃ᵀ diag(ẽ⁰) W̃_C W̃_Cᵀ diag(ẽ⁰) M̃
        # Using sparse cluster matrix for efficient aggregation
        
        # Build sparse cluster indicator matrix W̃_C
        W_C, n_clusters = _build_sparse_cluster_matrix(cluster_ids)
        
        # Compute scores per group: diag(ẽ⁰) @ M̃ = X * e0_g[:, None]
        if sparse.issparse(X):
            # X is sparse: multiply each row by corresponding e0_g
            scores_g = X.multiply(e0_g[:, np.newaxis])  # G x p sparse
        else:
            scores_g = sparse.csr_matrix(X * e0_g[:, np.newaxis])  # G x p sparse
        
        # Aggregate scores within clusters using sparse matrix multiplication:
        # cluster_scores = W̃_Cᵀ @ scores_g  (C x p)
        # This is equivalent to summing scores_g rows within each cluster
        cluster_scores = W_C.T @ scores_g  # C x p
        
        # Meat matrix: cluster_scoresᵀ @ cluster_scores = Σ_c (s_c @ s_c')
        if sparse.issparse(cluster_scores):
            meat = (cluster_scores.T @ cluster_scores).toarray()
        else:
            meat = cluster_scores.T @ cluster_scores
        
        # Small sample correction
        if ssc:
            adjustment = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / df_resid)
        else:
            adjustment = n_clusters / (n_clusters - 1)
        
        vcov_matrix = adjustment * XtX_inv @ meat @ XtX_inv
        se_full = np.sqrt(np.diag(vcov_matrix))
        
    else:
        raise ValueError(f"vcov must be 'iid', 'HC1', or 'cluster', got '{vcov}'")
    
    # Return only SEs for x_cols (not FE dummies)
    return se_full[:k_x], n_clusters


def leanfe_compress_polars(
    lf: pl.LazyFrame,
    y_col: str,
    x_cols: list[str],
    fe_cols: list[str],
    weights: str | None = None,
    vcov: str = "iid",
    cluster_col: str | None = None,
    ssc: bool = False
) -> LeanFEResult:
    """
    Execute high-dimensional fixed effects regression via data compression.
    Backend: Polars.

    Compresses data into unique groups of (regressors + fixed effects) to solve 
    WLS efficiently. Returns estimates.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input dataset.
    y_col : str
        Dependent variable name.
    x_cols : list[str]
        Independent variable names.
    fe_cols : list[str]
        Fixed effect column names.
    weights : str, optional
        Weight column name.
    vcov : {"iid", "cluster"}, default "iid"
        Variance-covariance type.
    cluster_col : str, optional
        Column for clustered SEs; required if vcov="cluster".
    ssc : bool, default False
        Apply small-sample correction.

    Returns
    -------
    dict
        Results containing:
        - coefficients, std_errors: Mappings for x_cols.
        - n_obs, n_compressed, compression_ratio: Data scale metrics.
        - df_resid, rss, n_clusters: Model fit statistics.
    """
    # Compress data
    compressed, n_obs = compress_polars(
        lf, y_col, x_cols, fe_cols, weights, 
        cluster_col=cluster_col if vcov.lower() == "cluster" else None
    )
    n_compressed = len(compressed)
    
    # Build design matrix - all_cols now includes '(Intercept)'
    X, Y, wts, all_cols, n_fe_levels = build_design_matrix(
        compressed, x_cols, fe_cols
    )
    
    # Solve WLS
    beta, XtX_inv = solve_wls(X, Y, wts)
    
    # Compute RSS
    rss_total, rss_g = compute_rss_grouped(compressed, X, beta, backend="polars")
    
    # Degrees of freedom
    p = len(all_cols)
    df_resid = n_obs - p
    
    # For cluster SEs, compute e0_g = sum_y - n * yhat (sum of residuals per group)
    cluster_ids = None
    e0_g = None
    if vcov.lower() == "cluster" and cluster_col is not None:
        cluster_ids = compressed[cluster_col].to_numpy()
        n_g = compressed["_n"].to_numpy()
        sum_y_g = compressed["_sum_y"].to_numpy()
        if sparse.issparse(X):
            yhat_g = np.asarray(X @ beta).flatten()
        else:
            yhat_g = X @ beta
        e0_g = sum_y_g - n_g * yhat_g  # ẽ⁰ = ỹ⁰ - ñ ⊙ ŷ
    
    # Extract number of x columns INCLUDING intercept
    k_x = len(x_cols) + 1
    x_cols_with_intercept = all_cols[:k_x]


    # Standard errors - pass x_cols_with_intercept
    se, n_clusters = compute_se_compress(
        XtX_inv, rss_total, rss_g, n_obs, df_resid, vcov, X, 
        x_cols_with_intercept,  # Now includes '(Intercept)'
        cluster_ids=cluster_ids, e0_g=e0_g, ssc=ssc
    )
    
    # Extract coefficients for x_cols (excluding intercept from results)
    beta_x = beta[1:k_x]  # Skip intercept
    se_x = se[1:]  # Skip intercept
    
    
    return LeanFEResult(
        coefficients=dict(zip(x_cols, beta_x)),
        std_errors=dict(zip(x_cols, se_x)),
        n_compressed = n_compressed,
        n_obs=n_obs,
        vcov_type=vcov,
        df_resid=df_resid,
        rss=rss_total,
        n_clusters=n_clusters
    )

def leanfe_compress_duckdb(
    con: DuckDBPyConnection,
    y_col: str,
    x_cols: list[str],
    fe_cols: list[str],
    weights: str | None = None,
    vcov: str = "iid",
    cluster_col: str | None = None,
    ssc: bool = False
) -> LeanFEResult:
    """
    Execute high-dimensional fixed effects regression via data compression.
    Backend: DuckDB.

    Compresses data into unique groups of (regressors + fixed effects) to solve 
    WLS efficiently. Returns estimates.

    Parameters
    ----------
    con : DuckDBPyConnection
        Connection to a DuckDB (either persistent or in-memory).
    y_col : str
        Dependent variable name.
    x_cols : list[str]
        Independent variable names.
    fe_cols : list[str]
        Fixed effect column names.
    weights : str, optional
        Weight column name.
    vcov : {"iid", "cluster"}, default "iid"
        Variance-covariance type.
    cluster_col : str, optional
        Column for clustered SEs; required if vcov="cluster".
    ssc : bool, default False
        Apply small-sample correction.

    Returns
    -------
    dict
        Results containing:
        - coefficients, std_errors: Mappings for x_cols.
        - n_obs, n_compressed, compression_ratio: Data scale metrics.
        - df_resid, rss, n_clusters: Model fit statistics.
    """
    # Compress data
    compressed, n_obs = compress_duckdb(
        con, y_col, x_cols, fe_cols, weights,
        cluster_col=cluster_col if vcov.lower() == "cluster" else None
    )
    n_compressed = len(compressed)
    
    # Build design matrix - all_cols now includes '(Intercept)'
    X, Y, wts, all_cols, n_fe_levels = build_design_matrix(
        compressed, x_cols, fe_cols
    )
    
    # Solve WLS
    beta, XtX_inv = solve_wls(X, Y, wts)
    
    # Compute RSS
    rss_total, rss_g = compute_rss_grouped(compressed, X, beta, backend="duckdb")
    
    # Degrees of freedom
    p = len(all_cols)
    df_resid = n_obs - p
    
    # For cluster SEs, compute e0_g = sum_y - n * yhat (sum of residuals per group)
    cluster_ids = None
    e0_g = None
    n_clusters = None
    if vcov.lower() == "cluster" and cluster_col is not None:
        cluster_ids = compressed[cluster_col]
        n_g = compressed["_n"]
        sum_y_g = compressed["_sum_y"]
        if sparse.issparse(X):
            yhat_g = np.asarray(X @ beta).flatten()
        else:
            yhat_g = X @ beta
        e0_g = sum_y_g - n_g * yhat_g
    
    # Extract number of x columns INCLUDING intercept
    k_x = len(x_cols) + 1
    x_cols_with_intercept = all_cols[:k_x]
    
    # Standard errors - pass x_cols_with_intercept
    se, n_clusters = compute_se_compress(
        XtX_inv, rss_total, rss_g, n_obs, df_resid, vcov, X,
        x_cols_with_intercept,  # Now includes '(Intercept)'
        cluster_ids=cluster_ids, e0_g=e0_g, ssc=ssc
    )
    
    # Extract coefficients for x_cols (excluding intercept from results)
    beta_x = beta[1:k_x]  # Skip intercept
    se_x = se[1:]  # Skip intercept
    
    return LeanFEResult(
        coefficients=dict(zip(x_cols, beta_x)),
        std_errors=dict(zip(x_cols, se_x)),
        n_compressed = n_compressed,
        n_obs=n_obs,
        vcov_type=vcov,
        df_resid=df_resid,
        rss=rss_total,
        n_clusters=n_clusters
    )