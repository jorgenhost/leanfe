"""
YOCO (You Only Compress Once) compression strategy for leanfe.

ALGORITHM OVERVIEW:
1. Group observations by (X, FE) combinations → compressed dataset
2. Compute sufficient statistics (n, sum_y, sum_y²) per group
3. Build design matrix with FE dummies on compressed data
4. Solve weighted least squares using group sizes as weights
5. Compute standard errors from grouped residuals

WHY THIS WORKS:
- Identical (X, FE) observations contribute identically to OLS solution
- Sufficient statistics capture all information needed for estimation
- Memory scales with unique combinations, not total observations

REFERENCE:
Wong et al. (2021): "You Only Compress Once: Optimal Data Compression 
for Estimating Linear Models"

Used automatically when vcov is "iid" or "HC1" (not cluster) and no IV.
"""
from leanfe.result import LeanFEResult

import numpy as np
import polars as pl
from scipy import sparse
from duckdb import DuckDBPyConnection
from typing import Any, Callable, TypeAlias
from dataclasses import dataclass
from enum import Enum
from itertools import combinations

ArrayLike: TypeAlias = np.ndarray | sparse.spmatrix

# ============================================================================
# CONSTANTS
# ============================================================================

# Strategy selection thresholds
DEFAULT_MAX_FE_LEVELS = 10_000
DEFAULT_DEMEANING_ITERATIONS = 10

# Multi-way clustering constants
MIN_CLUSTERS_FOR_ADJUSTMENT = 2
FIRST_ORDER_SUBSET_SIZE = 1
CLUSTER_INTERSECTION_DELIMITER = '_'

# Cost estimation factors
SPARSE_MATRIX_COST_FACTOR = 1.0
GROUP_BY_COST_FACTOR = 1.0
WLS_SOLVE_COST_EXPONENT = 2


# ============================================================================
# ENUMS
# ============================================================================

class VcovType(Enum):
    """Variance-covariance estimator types."""
    IID = "iid"
    HC1 = "HC1"
    CLUSTER = "cluster"


class Strategy(Enum):
    """Regression strategy types."""
    AUTO = "auto"
    COMPRESS = "compress"
    ALT_PROJ = "alt_proj"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CompressionContext:
    """Context for computing standard errors from compressed data."""
    XtX_inv: np.ndarray
    rss_total: float
    rss_per_group: np.ndarray
    n_obs: int
    df_resid: int
    vcov: str
    design_matrix: ArrayLike
    x_cols: list[str]
    cluster_ids: np.ndarray | None = None
    residual_sums_per_group: np.ndarray | None = None
    apply_small_sample_correction: bool = False


# ============================================================================
# STRATEGY SELECTION
# ============================================================================

def determine_strategy(
    vcov: str,
    has_instruments: bool,
    fe_cardinality: dict[str, int] | None = None,
    max_fe_levels: int = DEFAULT_MAX_FE_LEVELS,
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
        Dictionary mapping FE column names to their cardinality
    max_fe_levels : int, default 10_000
        Maximum FE cardinality to allow compression
    n_obs : int, optional
        Number of observations (for cost estimation)
    n_x_cols : int, optional
        Number of X columns (for cost estimation)
    estimated_compression_ratio : float, optional
        Estimated compression ratio (n_compressed / n_obs)
        
    Returns
    -------
    str
        Strategy name: 'compress' or 'alt_proj'
    """
    if has_instruments:
        return Strategy.ALT_PROJ.value  # IV requires alternating projections
    
    vcov_lower = vcov.lower()
    if vcov_lower not in (VcovType.IID.value, VcovType.HC1.value.lower(), VcovType.CLUSTER.value):
        return Strategy.ALT_PROJ.value  # Unsupported vcov for compression
    
    if fe_cardinality is None:
        return Strategy.COMPRESS.value  # Default when no cardinality info
    
    # Calculate total FE levels
    total_fe_levels = sum(fe_cardinality.values())
    max_single_fe = max(fe_cardinality.values()) if fe_cardinality else 0
    
    # Rule 1: Very high-cardinality single FE → use FWL
    # Building a sparse matrix with 100K+ columns is slow
    if max_single_fe > max_fe_levels:
        return Strategy.ALT_PROJ.value
    
    # Rule 2: Very high total FE levels → use FWL
    # Even if no single FE is huge, many medium FEs add up
    if total_fe_levels > max_fe_levels * 2:
        return Strategy.ALT_PROJ.value
    
    # Rule 3: Use cost model if compression ratio available
    if estimated_compression_ratio is not None and n_obs is not None:
        n_compressed = int(n_obs * estimated_compression_ratio)
        
        # Estimate YOCO cost: GROUP BY + sparse matrix build + WLS solve
        yoco_cost = (
            GROUP_BY_COST_FACTOR * n_obs +
            SPARSE_MATRIX_COST_FACTOR * n_compressed * total_fe_levels +
            total_fe_levels ** WLS_SOLVE_COST_EXPONENT
        )
        
        # Estimate FWL cost: ~10 iterations × n_fe × n_obs GROUP BY operations
        n_fe = len(fe_cardinality)
        fwl_cost = DEFAULT_DEMEANING_ITERATIONS * n_fe * n_obs
        
        if yoco_cost < fwl_cost:
            return Strategy.COMPRESS.value
        return Strategy.ALT_PROJ.value
    
    # Default: use compression for low-cardinality FEs
    return Strategy.COMPRESS.value


def estimate_compression_ratio(
    x_cols: list[str],
    fe_cols: list[str] = [],
    table_ref: str | None = None,
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
        List of regressor column names
    fe_cols : list of str
        List of fixed effect column names
    data : str or pl.LazyFrame, optional
        The dataset to analyze. If a string is provided, it is treated as a 
        table name within the DuckDB connection
    con : DuckDBPyConnection, optional
        An active DuckDB connection. Required if `data` is a table name 
        or if executing via SQL

    Returns
    -------
    float
        The compression ratio (between 0 and 1). A value closer to 0 indicates 
        significant grouping/redundancy

    Examples
    --------
    >>> lf = pl.LazyFrame({'x': [1,1,2], 'fe': [1,1,2]})
    >>> ratio = estimate_compression_ratio(['x'], ['fe'], data=lf)
    >>> ratio  # 2 unique groups / 3 total obs = 0.667
    0.6666666666666666

    Notes
    -----
    Inspired by the `dbreg` R package:
    https://github.com/grantmcdermott/dbreg/blob/main/R/dbreg.R#L587-L654
    """
    key_cols = list(set(x_cols + fe_cols))
    if not key_cols:
        return 1.0

    if con is not None and table_ref is not None:
        # DuckDB backend
        total_n = _count_rows_duckdb(con, table_ref)
        n_unique_groups = _count_unique_groups_duckdb(con, table_ref, key_cols)
    else:
        # Polars backend
        if isinstance(data, str):
            lf = pl.scan_parquet(data)
        elif isinstance(data, pl.LazyFrame):
            lf = data
        else:
            raise ValueError("data must be a string path or pl.LazyFrame")
        
        total_n = lf.select(pl.len()).collect().item()
        n_unique_groups = lf.select(key_cols).unique().select(pl.len()).collect().item()

    compression_ratio = n_unique_groups / max(total_n, 1)
    return compression_ratio


def _count_rows_duckdb(con: DuckDBPyConnection, table_ref: str) -> int:
    """Count total rows in DuckDB table."""
    result = con.execute(f"SELECT COUNT(*)::BIGINT FROM {table_ref}").fetchone()
    if result is None:
        raise RuntimeError("Failed to count rows")
    return int(result[0])


def _count_unique_groups_duckdb(
    con: DuckDBPyConnection,
    table_ref: str,
    key_cols: list[str]
) -> int:
    """Count unique groups in DuckDB table."""
    cols_expr = ", ".join(key_cols)
    sql = f"SELECT COUNT(*)::BIGINT FROM (SELECT DISTINCT {cols_expr} FROM {table_ref}) t"
    result = con.execute(sql).fetchone()
    if result is None:
        raise RuntimeError("Failed to count unique groups")
    return int(result[0])


# ============================================================================
# DATA COMPRESSION
# ============================================================================

def compress_polars(
    lf: pl.LazyFrame,
    y_col: str,
    x_cols: list[str],
    fe_cols: list[str],
    weights: str | None = None,
    cluster_col: str | list[str] | None = None
) -> tuple[pl.DataFrame, int]:
    """
    Compress data using GROUP BY on regressors + fixed effects.
    
    For cluster SEs (Section 5.3.1 of YOCO paper), include cluster_col(s)
    in the grouping to ensure each compressed record belongs to one cluster.
    
    Parameters
    ----------
    lf : pl.LazyFrame
        Input data
    y_col : str
        Dependent variable column name
    x_cols : list of str
        Independent variable column names
    fe_cols : list of str
        Fixed effect column names
    weights : str, optional
        Weight column name
    cluster_col : str or list of str, optional
        Cluster column name(s) for multi-way clustering
        
    Returns
    -------
    tuple
        (compressed_df, n_obs_original)
        
    Examples
    --------
    >>> lf = pl.LazyFrame({'y': [1,2,3], 'x': [1,1,2], 'fe': [1,1,2]})
    >>> compressed, n_obs = compress_polars(lf, 'y', ['x'], ['fe'])
    >>> len(compressed)
    2
    >>> n_obs
    3
    """
    group_cols = x_cols + fe_cols
    
    # For cluster SEs, add cluster columns to grouping
    if cluster_col is not None:
        cluster_cols_list = [cluster_col] if isinstance(cluster_col, str) else cluster_col
        for col in cluster_cols_list:
            if col not in group_cols:
                group_cols.append(col)
    
    n_obs_original = lf.select(pl.len()).collect().item()
    
    # Build aggregation expressions
    if weights is not None:
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
    
    # Add derived columns for WLS
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
        """Return length of first array."""
        return len(next(iter(self._data.values())))


def compress_duckdb(
    con: DuckDBPyConnection,
    y_col: str,
    x_cols: list[str],
    fe_cols: list[str],
    table_ref: str,
    weights: str | None = None,
    cluster_col: str | list[str] | None = None
) -> tuple[DuckDBResult, int]:
    """
    Compress data using SQL GROUP BY.
    
    For cluster SEs (Section 5.3.1 of YOCO paper), include cluster_col(s)
    in the grouping to ensure each compressed record belongs to one cluster.
    
    Parameters
    ----------
    con : DuckDBPyConnection
        Active DuckDB connection
    y_col : str
        Dependent variable column name
    x_cols : list of str
        Independent variable column names
    fe_cols : list of str
        Fixed effect column names
    table_ref : str
        Name of table to compress
    weights : str, optional
        Weight column name
    cluster_col : str or list of str, optional
        Cluster column name(s) for multi-way clustering
        
    Returns
    -------
    tuple
        (compressed_result, n_obs_original)
    """
    group_cols = x_cols + fe_cols
    
    # For cluster SEs, add cluster columns to grouping
    if cluster_col is not None:
        cluster_cols_list = [cluster_col] if isinstance(cluster_col, str) else cluster_col
        for col in cluster_cols_list:
            if col not in group_cols:
                group_cols.append(col)
    
    group_cols_sql = ", ".join(group_cols)
    n_obs_original = _count_rows_duckdb(con, table_ref)

    # Build SQL query
    if weights is not None:
        query = f"""
        SELECT
            {group_cols_sql},
            SUM({weights}) AS _n,
            SUM({y_col} * {weights}) AS _sum_y,
            SUM(POWER({y_col}, 2) * {weights}) AS _sum_y_sq,
            SUM({y_col} * {weights}) / SUM({weights}) AS _mean_y,
            SQRT(SUM({weights})) AS _wts
        FROM {table_ref}
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
        FROM {table_ref}
        GROUP BY {group_cols_sql}
        """
    
    result = con.execute(query).fetchnumpy()
    compressed = DuckDBResult(result)
    return compressed, n_obs_original


# ============================================================================
# DESIGN MATRIX CONSTRUCTION
# ============================================================================

def _extract_numpy_arrays(
    compressed_df: pl.DataFrame | DuckDBResult,
    x_cols: list[str],
) -> tuple[
    np.ndarray,                       # X_regressors
    np.ndarray,                       # Y
    np.ndarray,                       # weights
    Callable[[str | list[str]], Any]  # get_fe_values
]:
    """
    Extract numpy arrays from compressed dataframe.

    Works with both Polars DataFrame and DuckDB Result without pandas dependency.
    """
    if isinstance(compressed_df, pl.DataFrame):
        # Polars backend
        n_rows = compressed_df.height
        if x_cols:
            X_reg_cols = compressed_df.select([pl.col(c) for c in x_cols]).to_numpy()
        else:
            X_reg_cols = np.empty((n_rows, 0))
        # prepend intercept
        X_regressors = np.hstack([np.ones((n_rows, 1)), X_reg_cols])
        Y = compressed_df.select(pl.col("_mean_y")).to_numpy().flatten()
        weights = compressed_df.select(pl.col("_wts")).to_numpy().flatten()

        def get_fe_values(fe: str | list[str]):
            return compressed_df.select(fe).to_numpy()
    else:
        # DuckDB backend
        n_rows = len(next(iter(compressed_df._data.values())))
        if x_cols:
            X_cols_stack = np.column_stack([compressed_df[col] for col in x_cols])
            X_regressors = np.column_stack((np.ones(shape=(X_cols_stack.shape[0], 1)), X_cols_stack))
        else:
            X_regressors = np.ones((n_rows, 1))
        Y = compressed_df["_mean_y"]
        weights = compressed_df["_wts"]

        def get_fe_values(fe: str | list[str]):
            return compressed_df[fe]

    return X_regressors, Y, weights, get_fe_values

def build_design_matrix(
    compressed_df: pl.DataFrame | DuckDBResult,
    x_cols: list[str],
    fe_cols: list[str],
    use_sparse: bool = True
) -> tuple[ArrayLike, np.ndarray, np.ndarray, list[str], int]:
    """
    Build design matrix from compressed data with FE dummies.
    
    Parameters
    ----------
    compressed_df : pl.DataFrame or DuckDBResult
        Compressed data
    x_cols : list of str
        Regressor column names
    fe_cols : list of str
        Fixed effect column names
    use_sparse : bool, default True
        Whether to use sparse matrices for FE dummies
        
    Returns
    -------
    tuple
        (design_matrix, Y, weights, all_col_names, n_fe_levels)
        all_col_names includes '(Intercept)' as first element
        
    Examples
    --------
    >>> compressed = pl.DataFrame({
    ...     'x': [1, 2], 'fe': [1, 2],
    ...     '_mean_y': [1.5, 2.5], '_wts': [1.4, 1.7]
    ... })
    >>> X, Y, wts, cols, n_fe = build_design_matrix(compressed, ['x'], ['fe'])
    >>> cols[0]
    '(Intercept)'
    """
    # Extract numpy arrays
    X_regressors, Y, weights, get_fe_values = _extract_numpy_arrays(compressed_df, x_cols)
    n_rows = len(Y)

    # Validate dimensions
    assert len(Y) == len(weights), "Y and weights must have same length"
    assert X_regressors.shape[0] == len(Y), "X rows must match Y length"

    # Create column names with intercept
    all_x_cols = ['(Intercept)'] + list(x_cols)
    
    if not fe_cols:
        return X_regressors, Y, weights, all_x_cols, 0
    
    if use_sparse:
        design_matrix, fe_col_names, n_fe_levels = _build_sparse_fe_dummies(
            X_regressors, fe_cols, get_fe_values, n_rows
        )
        all_col_names = all_x_cols + fe_col_names
        return design_matrix, Y, weights, all_col_names, n_fe_levels
    else:
        design_matrix, fe_col_names, n_fe_levels = _build_dense_fe_dummies(
            X_regressors, fe_cols, get_fe_values
        )
        all_col_names = all_x_cols + fe_col_names
        return design_matrix, Y, weights, all_col_names, n_fe_levels


def _build_sparse_fe_dummies(
    X_regressors: np.ndarray,
    fe_cols: list[str],
    get_fe_values: Callable,
    n_rows: int
) -> tuple[sparse.csr_matrix, list[str], int]:
    """Build sparse FE dummy matrix efficiently using COO format."""
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
        
        col_offset += n_cats - 1  # Drop first category
    
    # Create sparse FE matrix
    n_fe_cols = col_offset
    if n_fe_cols > 0:
        fe_dummy_matrix = sparse.coo_matrix(
            (data_list, (rows_list, cols_list)),
            shape=(n_rows, n_fe_cols)
        ).tocsr()
        
        # Combine: [X_regressors | fe_dummies] as sparse
        X_regressors_sparse = sparse.csr_matrix(X_regressors)
        design_matrix = sparse.hstack([X_regressors_sparse, fe_dummy_matrix], format='csr')
    else:
        design_matrix = sparse.csr_matrix(X_regressors)
    
    return design_matrix, fe_col_names, n_fe_levels


def _build_dense_fe_dummies(
    X_regressors: np.ndarray,
    fe_cols: list[str],
    get_fe_values: Callable
) -> tuple[np.ndarray, list[str], int]:
    """Build dense FE dummy matrix."""
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
        fe_dummy_matrix = np.column_stack(fe_dummies)
        design_matrix = np.hstack([X_regressors, fe_dummy_matrix])
    else:
        design_matrix = X_regressors
    
    return design_matrix, fe_col_names, n_fe_levels


# ============================================================================
# WEIGHTED LEAST SQUARES
# ============================================================================

def solve_wls(
    design_matrix: ArrayLike,
    Y: np.ndarray,
    weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve weighted least squares.
    
    Parameters
    ----------
    design_matrix : np.ndarray or scipy.sparse matrix
        Design matrix (G x p)
    Y : np.ndarray
        Response vector (G,) - group means
    weights : np.ndarray
        Weights (G,) - sqrt(n_g)
        
    Returns
    -------
    tuple
        (beta, XtX_inv)
        
    Notes
    -----
    Uses Cholesky decomposition when possible for numerical stability.
    Falls back to least squares for singular matrices.
    """
    if isinstance(design_matrix, sparse.csr_matrix):
        beta, XtX_inv = _solve_wls_sparse(design_matrix, Y, weights)
    elif isinstance(design_matrix, np.ndarray):
        beta, XtX_inv = _solve_wls_dense(design_matrix, Y, weights)
    
    return beta, XtX_inv


def _solve_wls_sparse(
    X: sparse.csr_matrix,
    Y: np.ndarray,
    weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solve WLS with sparse design matrix."""
    # Weight the design matrix: diag(weights) @ X
    X_weighted = X.multiply(weights[:, np.newaxis])
    Y_weighted = Y * weights
    
    # X'X is dense (p x p is typically small)
    XtX = (X_weighted.T @ X_weighted).toarray()
    Xty = X_weighted.T @ Y_weighted
    
    # Solve using dense methods (XtX is small)
    return _solve_normal_equations(XtX, Xty)


def _solve_wls_dense(
    X: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solve WLS with dense design matrix."""
    X_weighted = X * weights[:, np.newaxis]
    Y_weighted = Y * weights
    
    XtX = X_weighted.T @ X_weighted
    Xty = X_weighted.T @ Y_weighted
    
    return _solve_normal_equations(XtX, Xty)


def _solve_normal_equations(
    XtX: np.ndarray,
    Xty: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve normal equations X'X beta = X'y.
    
    Uses Cholesky decomposition for speed and numerical stability.
    Falls back to least squares for singular matrices.
    """
    try:
        # Cholesky decomposition: XtX = L L'
        L = np.linalg.cholesky(XtX)
        beta = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
        XtX_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(XtX.shape[0])))
    except np.linalg.LinAlgError:
        # Fallback for singular matrices
        beta, _, _, _ = np.linalg.lstsq(XtX, Xty, rcond=None)
        XtX_inv = np.linalg.pinv(XtX)
    
    return beta, XtX_inv


# ============================================================================
# RESIDUAL SUM OF SQUARES
# ============================================================================

def compute_rss_grouped(
    compressed_df: pl.DataFrame | DuckDBResult,
    design_matrix: ArrayLike,
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
    design_matrix : np.ndarray or scipy.sparse matrix
        Design matrix
    beta : np.ndarray
        Coefficient vector
    backend : str
        "polars" or "duckdb"
        
    Returns
    -------
    tuple
        (rss_total, rss_per_group)
        
    Notes
    -----
    This computes RSS without reconstructing individual residuals,
    using only the sufficient statistics from each group.
    """
    if backend == "polars" and isinstance(compressed_df, pl.DataFrame):
        n_per_group = compressed_df["_n"].to_numpy()
        sum_y_per_group = compressed_df["_sum_y"].to_numpy()
        sum_y_sq_per_group = compressed_df["_sum_y_sq"].to_numpy()
    elif backend == 'duckdb' and isinstance(compressed_df, DuckDBResult):
        # DuckDB backend
        n_per_group = compressed_df["_n"]
        sum_y_per_group = compressed_df["_sum_y"]
        sum_y_sq_per_group = compressed_df["_sum_y_sq"]
    
    # Fitted values for each group (handle sparse)
    if sparse.issparse(design_matrix):
        fitted_values_per_group = np.asarray(design_matrix @ beta).flatten()
    else:
        fitted_values_per_group = design_matrix @ beta
    
    # Per-group RSS using sufficient statistics
    rss_per_group = (
        sum_y_sq_per_group -
        2 * fitted_values_per_group * sum_y_per_group +
        n_per_group * (fitted_values_per_group ** 2)
    )
    rss_total = np.sum(rss_per_group)
    
    return rss_total, rss_per_group


# ============================================================================
# STANDARD ERROR COMPUTATION
# ============================================================================

def _build_sparse_cluster_matrix(cluster_ids: np.ndarray) -> tuple[sparse.csr_matrix, int]:
    """
    Build sparse cluster indicator matrix W_C from Section 5.3.1.
    
    W_C ∈ R^{G×C} where entry (g, c) = 1 if group g belongs to cluster c.
    
    Parameters
    ----------
    cluster_ids : np.ndarray
        Cluster ID for each compressed group (length G)
        
    Returns
    -------
    tuple
        (cluster_indicator_matrix, n_clusters)
        
    Examples
    --------
    >>> cluster_ids = np.array([1, 1, 2, 2, 3])
    >>> W_C, n_clusters = _build_sparse_cluster_matrix(cluster_ids)
    >>> W_C.shape
    (5, 3)
    >>> n_clusters
    3
    """
    unique_clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(unique_clusters)
    n_groups = len(cluster_ids)
    
    # Build sparse matrix in COO format: row=group_idx, col=cluster_idx, data=1
    cluster_indicator_matrix = sparse.csr_matrix(
        (np.ones(n_groups), (np.arange(n_groups), inverse)),
        shape=(n_groups, n_clusters)
    )
    return cluster_indicator_matrix, n_clusters


def compute_se_compress(ctx: CompressionContext) -> tuple[np.ndarray, int | tuple[int, ...] | None]:
    """
    Compute standard errors from compressed data.
    
    For cluster SEs, implements Section 5.3.1 of YOCO paper using sparse matrices.
    
    Parameters
    ----------
    ctx : CompressionContext
        Context containing all necessary data for SE computation
        
    Returns
    -------
    tuple
        (standard_errors, n_clusters)
        - standard_errors: SE for x_cols only (excluding intercept/FE)
        - n_clusters: int for single-way, tuple for multi-way, None otherwise
    """
    k_x = len(ctx.x_cols)
    n_clusters = None
    
    vcov_lower = ctx.vcov.lower()
    
    if vcov_lower == VcovType.IID.value:
        se_full = _compute_se_iid(ctx)
        
    elif vcov_lower == VcovType.HC1.value.lower():
        se_full = _compute_se_hc1(ctx)
        
    elif vcov_lower == VcovType.CLUSTER.value:
        if ctx.cluster_ids is None or ctx.residual_sums_per_group is None:
            raise ValueError("cluster_ids and residual_sums_per_group required for cluster SEs")
        
        # Single-way vs multi-way clustering
        if ctx.cluster_ids.ndim == 1:
            se_full, n_clusters = _compute_se_cluster_oneway(ctx)
        elif ctx.cluster_ids.ndim == 2 and ctx.cluster_ids.shape[1] > 1:
            se_full, n_clusters = _compute_se_cluster_multiway(ctx)
        else:
            raise ValueError("cluster_ids must be 1D for single-way or 2D for multi-way clustering")
    else:
        raise ValueError(f"vcov must be 'iid', 'HC1', or 'cluster', got '{ctx.vcov}'")
    
    return se_full[:k_x], n_clusters


def _compute_se_iid(ctx: CompressionContext) -> np.ndarray:
    """Compute IID standard errors."""
    sigma_squared = ctx.rss_total / ctx.df_resid
    se = np.sqrt(np.maximum(np.diag(ctx.XtX_inv) * sigma_squared, 0.0))
    return se


def _compute_se_hc1(ctx: CompressionContext) -> np.ndarray:
    """Compute heteroskedasticity-robust (HC1) standard errors."""
    # Meat matrix: X' diag(rss_g) X
    if isinstance(ctx.design_matrix, sparse.csr_matrix):
        X_weighted = ctx.design_matrix.multiply(ctx.rss_per_group[:, np.newaxis])
        meat = (ctx.design_matrix.T @ X_weighted).toarray()
    elif isinstance(ctx.design_matrix, np.ndarray):
        meat = ctx.design_matrix.T @ (ctx.design_matrix * ctx.rss_per_group[:, np.newaxis])
    
    vcov_matrix = ctx.XtX_inv @ meat @ ctx.XtX_inv
    adjustment = ctx.n_obs / ctx.df_resid
    se = np.sqrt(np.maximum(np.diag(vcov_matrix) * adjustment, 0.0))
    return se


def _compute_se_cluster_oneway(ctx: CompressionContext) -> tuple[np.ndarray, int]:
    """Compute one-way clustered standard errors."""
    if isinstance(ctx.cluster_ids, np.ndarray):
        cluster_indicator_matrix, n_clusters = _build_sparse_cluster_matrix(ctx.cluster_ids)
    
    # Compute scores per group
    if isinstance(ctx.design_matrix, sparse.csr_matrix) and isinstance(ctx.residual_sums_per_group, np.ndarray):
        scores_per_group = ctx.design_matrix.multiply(ctx.residual_sums_per_group[:, np.newaxis])
    elif isinstance(ctx.design_matrix, np.ndarray) and isinstance(ctx.residual_sums_per_group, np.ndarray):
        scores_per_group = sparse.csr_matrix(
            ctx.design_matrix * ctx.residual_sums_per_group[:, np.newaxis]
        )
    
    # Aggregate scores by cluster
    cluster_scores = cluster_indicator_matrix.T @ scores_per_group
    
    # Compute meat matrix
    if sparse.issparse(cluster_scores):
        meat = (cluster_scores.T @ cluster_scores).toarray()
    else:
        meat = cluster_scores.T @ cluster_scores
    
    # Apply adjustment
    if ctx.apply_small_sample_correction:
        adjustment = (n_clusters / (n_clusters - 1)) * ((ctx.n_obs - 1) / ctx.df_resid)
    else:
        adjustment = n_clusters / (n_clusters - 1)
    
    vcov_matrix = adjustment * (ctx.XtX_inv @ meat @ ctx.XtX_inv)
    se = np.sqrt(np.maximum(np.diag(vcov_matrix), 0.0))
    
    return se, n_clusters


def _compute_se_cluster_multiway(ctx: CompressionContext) -> tuple[np.ndarray, tuple | None]:
    """
    Compute multi-way clustered standard errors.
    
    Uses Cameron-Gelbach-Miller (2011) approach with fixest-compatible SSC.
    Implementation matches fixest defaults:
    - G.df = "min": Single G_min/(G_min-1) adjustment at the end
    - Components are accumulated WITHOUT per-component G/(G-1) adjustment
    - K adjustment (n-1)/(n-K) applied separately if ssc=True
    """

    assert ctx.cluster_ids is not None, "No cluster ids passed to multiway func"

    n_groups = ctx.cluster_ids.shape[0]
    n_ways = ctx.cluster_ids.shape[1]
    vcov_matrix = np.zeros_like(ctx.XtX_inv)
    first_order_cluster_counts = []
    
    # Precompute scores per group
    if isinstance(ctx.design_matrix, sparse.csr_matrix) and isinstance(ctx.residual_sums_per_group, np.ndarray):
        scores_per_group = ctx.design_matrix.multiply(ctx.residual_sums_per_group[:, np.newaxis])
    elif not isinstance(ctx.design_matrix, sparse.csr_matrix) and isinstance(ctx.residual_sums_per_group, np.ndarray):
        scores_per_group = sparse.csr_matrix(
            ctx.design_matrix * ctx.residual_sums_per_group[:, np.newaxis]
        )
    
    # Accumulate components with alternating signs
    for subset_size in range(FIRST_ORDER_SUBSET_SIZE, n_ways + 1):
        sign = (-1) ** (subset_size - 1)
        
        for cluster_subset in combinations(range(n_ways), subset_size):
            # Build intersection cluster ID
            if subset_size == 1:
                col = ctx.cluster_ids[:, cluster_subset[0]]
                unique_vals, inverse = np.unique(col, return_inverse=True)
                n_clusters = len(unique_vals)
            else:
                concat = np.array([
                    CLUSTER_INTERSECTION_DELIMITER.join(
                        str(ctx.cluster_ids[i, j]) for j in cluster_subset
                    )
                    for i in range(n_groups)
                ])
                unique_vals, inverse = np.unique(concat, return_inverse=True)
                n_clusters = len(unique_vals)
            
            # Skip subsets with too few clusters
            if n_clusters <= 1:
                if subset_size == 1:
                    first_order_cluster_counts.append(n_clusters)
                continue
            
            if subset_size == 1:
                first_order_cluster_counts.append(n_clusters)
            
            # Build cluster indicator matrix for this subset
            cluster_indicator_matrix = sparse.csr_matrix(
                (np.ones(len(inverse)), (np.arange(len(inverse)), inverse)),
                shape=(len(inverse), n_clusters)
            )
            
            # Aggregate scores by cluster
            cluster_scores = cluster_indicator_matrix.T @ scores_per_group
            
            # Compute meat matrix
            if sparse.issparse(cluster_scores):
                meat = (cluster_scores.T @ cluster_scores).toarray()
            else:
                meat = cluster_scores.T @ cluster_scores
            
            # Accumulate WITHOUT per-component G/(G-1) adjustment
            vcov_matrix += sign * (ctx.XtX_inv @ meat @ ctx.XtX_inv)
    
    # Apply single G_min/(G_min-1) adjustment at the end (fixest default with G.df="min")
    if len(first_order_cluster_counts) > 0:
        G_min = min(first_order_cluster_counts)
        if G_min > MIN_CLUSTERS_FOR_ADJUSTMENT:
            vcov_matrix *= (G_min / (G_min - 1))
    
    # Apply K small-sample correction if requested
    if ctx.apply_small_sample_correction:
        vcov_matrix *= ((ctx.n_obs - 1) / ctx.df_resid)
    
    se = np.sqrt(np.maximum(np.diag(vcov_matrix), 0.0))
    n_clusters = tuple(first_order_cluster_counts) if len(first_order_cluster_counts) > 0 else None
    
    return se, n_clusters


# ============================================================================
# HIGH-LEVEL COMPRESSION REGRESSION FUNCTIONS
# ============================================================================

def leanfe_compress_polars(
    lf: pl.LazyFrame,
    y_col: str,
    x_cols: list[str],
    fe_cols: list[str],
    weights: str | None = None,
    vcov: str = "iid",
    cluster_col: str | list[str] | None = None,
    ssc: bool = False
) -> LeanFEResult:
    """
    Execute high-dimensional fixed effects regression via data compression.
    Backend: Polars.
    
    Parameters
    ----------
    lf : pl.LazyFrame
        Input data
    y_col : str
        Dependent variable
    x_cols : list of str
        Independent variables
    fe_cols : list of str
        Fixed effects
    weights : str, optional
        Weight column
    vcov : str, default "iid"
        Variance estimator type
    cluster_col : str or list of str, optional
        Cluster column(s)
    ssc : bool, default False
        Apply small sample correction
        
    Returns
    -------
    LeanFEResult
        Regression results
    """
    # Normalize cluster_col to list
    if cluster_col is not None and not isinstance(cluster_col, list):
        cluster_col = [cluster_col]
    
    # Compress data
    compressed, n_obs = compress_polars(
        lf, y_col, x_cols, fe_cols, weights,
        cluster_col=cluster_col
    )
    n_compressed = len(compressed)
    
    # Build design matrix
    design_matrix, Y, wts, all_cols, n_fe_levels = build_design_matrix(
        compressed, x_cols, fe_cols
    )
    
    # Solve WLS
    beta, XtX_inv = solve_wls(design_matrix, Y, wts)
    
    # Compute RSS
    rss_total, rss_per_group = compute_rss_grouped(
        compressed, design_matrix, beta, backend="polars"
    )
    
    # Degrees of freedom
    p = len(all_cols)
    df_resid = n_obs - p
    
    # For cluster SEs, compute residual_sums_per_group = sum_y - n * yhat
    cluster_ids = None
    residual_sums_per_group = None
    if vcov.lower() == VcovType.CLUSTER.value and cluster_col is not None:
        if len(cluster_col) == 1:
            cluster_ids = compressed[cluster_col[0]].to_numpy()
        else:
            # Multi-way: stack all cluster columns into 2D array
            cluster_ids = compressed.select(cluster_col).to_numpy()
        
        n_per_group = compressed["_n"].to_numpy()
        sum_y_per_group = compressed["_sum_y"].to_numpy()
        if sparse.issparse(design_matrix):
            fitted_values_per_group = np.asarray(design_matrix @ beta).flatten()
        else:
            fitted_values_per_group = design_matrix @ beta
        residual_sums_per_group = sum_y_per_group - n_per_group * fitted_values_per_group
    
    # Create context for SE computation
    k_x = len(x_cols) + 1  # Include intercept
    x_cols_with_intercept = all_cols[:k_x]
    
    ctx = CompressionContext(
        XtX_inv=XtX_inv,
        rss_total=rss_total,
        rss_per_group=rss_per_group,
        n_obs=n_obs,
        df_resid=df_resid,
        vcov=vcov,
        design_matrix=design_matrix,
        x_cols=x_cols_with_intercept,
        cluster_ids=cluster_ids,
        residual_sums_per_group=residual_sums_per_group,
        apply_small_sample_correction=ssc
    )
    
    # Compute standard errors
    se, n_clusters = compute_se_compress(ctx)
    
    # Extract coefs for x_cols (excluding intercept from results)
    beta_x = beta[1:k_x]
    se_x = se[1:]
    
    return LeanFEResult(
        coefs=dict(zip(x_cols, beta_x)),
        std_errors=dict(zip(x_cols, se_x)),
        n_compressed=n_compressed,
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
    table_ref: str,
    weights: str | None = None,
    vcov: str = "iid",
    cluster_col: str | list[str] | None = None,
    ssc: bool = False
) -> LeanFEResult:
    """
    Execute high-dimensional fixed effects regression via data compression.
    Backend: DuckDB.
    
    Parameters
    ----------
    con : DuckDBPyConnection
        Active DuckDB connection
    y_col : str
        Dependent variable
    x_cols : list of str
        Independent variables
    fe_cols : list of str
        Fixed effects
    table_ref : str
        Table name
    weights : str, optional
        Weight column
    vcov : str, default "iid"
        Variance estimator type
    cluster_col : str or list of str, optional
        Cluster column(s)
    ssc : bool, default False
        Apply small sample correction
        
    Returns
    -------
    LeanFEResult
        Regression results
    """
    # Normalize cluster_col to list
    if cluster_col is not None and not isinstance(cluster_col, list):
        cluster_col = [cluster_col]
    
    # Compress data
    compressed, n_obs = compress_duckdb(
        con, y_col, x_cols, fe_cols, table_ref, weights,
        cluster_col=cluster_col
    )

    n_compressed = len(compressed)
    compression_ratio = (n_compressed / n_obs)

    # Build design matrix
    design_matrix, Y, wts, all_cols, n_fe_levels = build_design_matrix(
        compressed, x_cols, fe_cols
    )
    
    # Solve WLS
    beta, XtX_inv = solve_wls(design_matrix, Y, wts)
    
    # Compute RSS
    rss_total, rss_per_group = compute_rss_grouped(
        compressed, design_matrix, beta, backend="duckdb"
    )
    
    # Degrees of freedom
    p = len(all_cols)
    df_resid = n_obs - p
    
    # For cluster SEs, compute residual_sums_per_group
    cluster_ids = None
    residual_sums_per_group = None
    if vcov.lower() == VcovType.CLUSTER.value and cluster_col is not None:
        if len(cluster_col) == 1:
            cluster_ids = compressed[cluster_col[0]]
        else:
            # Multi-way: stack all cluster columns into 2D array
            cluster_ids = np.column_stack([compressed[col] for col in cluster_col])
        
        n_per_group = compressed["_n"]
        sum_y_per_group = compressed["_sum_y"]
        if sparse.issparse(design_matrix):
            fitted_values_per_group = np.asarray(design_matrix @ beta).flatten()
        else:
            fitted_values_per_group = design_matrix @ beta
        residual_sums_per_group = sum_y_per_group - n_per_group * fitted_values_per_group
    
    # Create context for SE computation
    k_x = len(x_cols) + 1  # Include intercept
    x_cols_with_intercept = all_cols[:k_x]
    
    ctx = CompressionContext(
        XtX_inv=XtX_inv,
        rss_total=rss_total,
        rss_per_group=rss_per_group,
        n_obs=n_obs,
        df_resid=df_resid,
        vcov=vcov,
        design_matrix=design_matrix,
        x_cols=x_cols_with_intercept,
        cluster_ids=cluster_ids,
        residual_sums_per_group=residual_sums_per_group,
        apply_small_sample_correction=ssc
    )
    
    # Compute standard errors
    se, n_clusters = compute_se_compress(ctx)
    
    # Extract coefs for x_cols (excluding intercept)
    beta_x = beta[1:k_x]
    se_x = se[1:]
    
    return LeanFEResult(
        coefs=dict(zip(x_cols, beta_x)),
        std_errors=dict(zip(x_cols, se_x)),
        n_compressed=n_compressed,
        n_obs=n_obs,
        compression_ratio=compression_ratio,
        vcov_type=vcov,
        df_resid=df_resid,
        rss=rss_total,
        n_clusters=n_clusters
    )