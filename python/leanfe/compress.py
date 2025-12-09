"""
YOCO (You Only Compress Once) compression strategy for leanfe.

Implements the optimal data compression strategy from Wong et al. (2021):
"You Only Compress Once: Optimal Data Compression for Estimating Linear Models"

This strategy compresses data by grouping on (regressors + fixed effects),
computing sufficient statistics (n, sum_y, sum_y_sq), and running weighted
least squares on the compressed data.

Used automatically when vcov is "iid" or "HC1" (not cluster) and no IV.
"""

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import List, Optional, Dict, Tuple


def should_use_compress(
    vcov: str,
    has_instruments: bool,
    fe_cardinality: Optional[dict] = None,
    max_fe_levels: int = 10000,
    n_obs: Optional[int] = None,
    n_x_cols: Optional[int] = None,
    estimated_compression_ratio: Optional[float] = None
) -> bool:
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
    max_fe_levels : int, default 10000
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
        return False
    
    if fe_cardinality is None:
        return True  # Default to compression if no info
    
    # Calculate total FE levels
    total_fe_levels = sum(fe_cardinality.values())
    max_single_fe = max(fe_cardinality.values()) if fe_cardinality else 0
    
    # Rule 1: If any single FE is very high-cardinality, use FWL
    # Building a sparse matrix with 100K+ columns is slow
    if max_single_fe > max_fe_levels:
        return False
    
    # Rule 2: If total FE levels is very high, use FWL
    # Even if no single FE is huge, many medium FEs can add up
    if total_fe_levels > max_fe_levels * 2:
        return False
    
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
        return yoco_cost < fwl_cost
    
    # Default: use compression for low-cardinality FEs
    return True
    return True


def compress_polars(
    df: pl.DataFrame,
    y_col: str,
    x_cols: List[str],
    fe_cols: List[str],
    weights: Optional[str] = None,
    cluster_col: Optional[str] = None
) -> Tuple[pl.DataFrame, int]:
    """
    Compress data using GROUP BY on regressors + fixed effects.
    
    For cluster SEs (Section 5.3.1 of YOCO paper), include cluster_col
    in the grouping to ensure each compressed record belongs to one cluster.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input data
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
        (compressed_df, n_obs_original)
    """
    group_cols = x_cols + fe_cols
    # For cluster SEs, add cluster to grouping (Section 5.3.1)
    if cluster_col is not None and cluster_col not in group_cols:
        group_cols = group_cols + [cluster_col]
    
    n_obs_original = len(df)
    
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
    
    compressed = df.group_by(group_cols).agg(agg_exprs)
    
    # Add mean_y and sqrt weights for WLS
    compressed = compressed.with_columns([
        (pl.col("_sum_y") / pl.col("_n")).alias("_mean_y"),
        pl.col("_n").sqrt().alias("_wts"),
    ])
    
    return compressed, n_obs_original


class DuckDBResult:
    """Wrapper for DuckDB numpy result to provide dict-like access."""
    def __init__(self, data: Dict[str, np.ndarray]):
        self._data = data
    
    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]
    
    def __len__(self) -> int:
        # Return length of first array
        return len(next(iter(self._data.values())))


def compress_duckdb(
    con,
    y_col: str,
    x_cols: List[str],
    fe_cols: List[str],
    weights: Optional[str] = None,
    cluster_col: Optional[str] = None
) -> Tuple[DuckDBResult, int]:
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
    
    n_obs_original = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    
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


def _extract_numpy_arrays(compressed_df, x_cols: List[str], backend: str):
    """Extract numpy arrays from compressed dataframe without pandas dependency."""
    if backend == "polars":
        # Use Polars native to_numpy() - no pandas needed
        X_reg = np.column_stack([compressed_df[col].to_numpy() for col in x_cols])
        Y = compressed_df["_mean_y"].to_numpy()
        wts = compressed_df["_wts"].to_numpy()
        
        def get_fe_values(fe):
            return compressed_df[fe].to_numpy()
    else:
        # DuckDB returns DuckDBResult (dict-like with numpy arrays)
        X_reg = np.column_stack([compressed_df[col] for col in x_cols])
        Y = compressed_df["_mean_y"]
        wts = compressed_df["_wts"]
        
        def get_fe_values(fe):
            return compressed_df[fe]
    
    return X_reg, Y, wts, get_fe_values


def build_design_matrix(
    compressed_df,
    x_cols: List[str],
    fe_cols: List[str],
    backend: str = "polars",
    use_sparse: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], int]:
    """
    Build design matrix from compressed data with FE dummies.
    
    Parameters
    ----------
    compressed_df : DataFrame
        Compressed data (polars or pandas)
    x_cols : list of str
        Regressor columns
    fe_cols : list of str
        Fixed effect columns
    backend : str
        "polars" or "duckdb"
    use_sparse : bool
        If True, use sparse matrices for FE dummies (much faster)
        
    Returns
    -------
    tuple
        (X, Y, wts, all_col_names, n_fe_levels)
        X is sparse.csr_matrix if use_sparse=True and fe_cols exist
    """
    # Extract numpy arrays without pandas dependency
    X_reg, Y, wts, get_fe_values = _extract_numpy_arrays(compressed_df, x_cols, backend)
    n_rows = len(Y)
    
    if not fe_cols:
        return X_reg, Y, wts, list(x_cols), 0
    
    if use_sparse:
        # Build sparse FE dummies efficiently using COO format
        fe_col_names = []
        n_fe_levels = 0
        col_offset = 0
        
        # Collect sparse matrix components
        rows_list = []
        cols_list = []
        data_list = []
        
        for fe in fe_cols:
            fe_values = get_fe_values(fe)
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
        
        all_col_names = list(x_cols) + fe_col_names
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
            all_col_names = list(x_cols) + fe_col_names
        else:
            X = X_reg
            all_col_names = list(x_cols)
        
        return X, Y, wts, all_col_names, n_fe_levels


def solve_wls(
    X,  # np.ndarray or sparse matrix
    Y: np.ndarray,
    wts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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
        Xw = X.multiply(wts[:, np.newaxis])
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
    compressed_df,
    X,  # np.ndarray or sparse matrix
    beta: np.ndarray,
    backend: str = "polars"
) -> Tuple[float, np.ndarray]:
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
        "polars" or "duckdb"
        
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


def _build_sparse_cluster_matrix(cluster_ids: np.ndarray) -> Tuple[sparse.csr_matrix, int]:
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
    x_cols: List[str],
    cluster_ids: Optional[np.ndarray] = None,
    e0_g: Optional[np.ndarray] = None,
    ssc: bool = False
) -> Tuple[np.ndarray, Optional[int]]:
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
    df: pl.DataFrame,
    y_col: str,
    x_cols: List[str],
    fe_cols: List[str],
    weights: Optional[str] = None,
    vcov: str = "iid",
    cluster_col: Optional[str] = None,
    ssc: bool = False
) -> Dict:
    """
    Run compressed regression using Polars backend.
    
    For cluster SEs, implements Section 5.3.1 of YOCO paper (within-cluster compression).
    
    Parameters
    ----------
    df : pl.DataFrame
        Input data
    y_col : str
        Dependent variable
    x_cols : list of str
        Regressors
    fe_cols : list of str
        Fixed effects
    weights : str, optional
        Weight column
    vcov : str
        "iid", "HC1", or "cluster"
    cluster_col : str, optional
        Cluster column (required if vcov="cluster")
    ssc : bool
        Small sample correction for cluster SEs
        
    Returns
    -------
    dict
        Regression results
    """
    # Compress data (include cluster in grouping for cluster SEs)
    compressed, n_obs = compress_polars(
        df, y_col, x_cols, fe_cols, weights, 
        cluster_col=cluster_col if vcov.lower() == "cluster" else None
    )
    n_compressed = len(compressed)
    
    # Build design matrix
    X, Y, wts, all_cols, n_fe_levels = build_design_matrix(
        compressed, x_cols, fe_cols, backend="polars"
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
    
    # Standard errors
    se, n_clusters = compute_se_compress(
        XtX_inv, rss_total, rss_g, n_obs, df_resid, vcov, X, x_cols,
        cluster_ids=cluster_ids, e0_g=e0_g, ssc=ssc
    )
    
    # Extract coefficients for x_cols only
    beta_x = beta[:len(x_cols)]
    
    return {
        "coefficients": dict(zip(x_cols, beta_x)),
        "std_errors": dict(zip(x_cols, se)),
        "n_obs": n_obs,
        "n_compressed": n_compressed,
        "compression_ratio": n_compressed / n_obs,
        "vcov_type": vcov,
        "strategy": "compress",
        "df_resid": df_resid,
        "rss": rss_total,
        "n_clusters": n_clusters,
    }


def leanfe_compress_duckdb(
    con,
    y_col: str,
    x_cols: List[str],
    fe_cols: List[str],
    weights: Optional[str] = None,
    vcov: str = "iid",
    cluster_col: Optional[str] = None,
    ssc: bool = False
) -> Dict:
    """
    Run compressed regression using DuckDB backend.
    
    For cluster SEs, implements Section 5.3.1 of YOCO paper (within-cluster compression)
    using sparse matrices for efficient cluster score aggregation.
    
    Parameters
    ----------
    con : duckdb.Connection
        DuckDB connection with 'data' table
    y_col : str
        Dependent variable
    x_cols : list of str
        Regressors
    fe_cols : list of str
        Fixed effects
    weights : str, optional
        Weight column
    vcov : str
        "iid", "HC1", or "cluster"
    cluster_col : str, optional
        Cluster column (required if vcov="cluster")
    ssc : bool
        Small sample correction for cluster SEs
        
    Returns
    -------
    dict
        Regression results
    """
    # Compress data (include cluster in grouping for cluster SEs)
    compressed, n_obs = compress_duckdb(
        con, y_col, x_cols, fe_cols, weights,
        cluster_col=cluster_col if vcov.lower() == "cluster" else None
    )
    n_compressed = len(compressed)
    
    # Build design matrix
    X, Y, wts, all_cols, n_fe_levels = build_design_matrix(
        compressed, x_cols, fe_cols, backend="duckdb"
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
        e0_g = sum_y_g - n_g * yhat_g  # ẽ⁰ = ỹ⁰ - ñ ⊙ ŷ
    
    # Standard errors
    se, n_clusters = compute_se_compress(
        XtX_inv, rss_total, rss_g, n_obs, df_resid, vcov, X, x_cols,
        cluster_ids=cluster_ids, e0_g=e0_g, ssc=ssc
    )
    
    # Extract coefficients for x_cols only
    beta_x = beta[:len(x_cols)]
    
    return {
        "coefficients": dict(zip(x_cols, beta_x)),
        "std_errors": dict(zip(x_cols, se)),
        "n_obs": n_obs,
        "n_compressed": n_compressed,
        "compression_ratio": n_compressed / n_obs,
        "vcov_type": vcov,
        "strategy": "compress",
        "df_resid": df_resid,
        "rss": rss_total,
        "n_clusters": n_clusters,
    }
