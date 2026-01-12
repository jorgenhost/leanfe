# aggs_polars.py
import itertools
import numpy as np
import polars as pl
from typing import List, Tuple, Dict, Any

def _aggregate_scores_polars(
    df: pl.DataFrame,
    x_cols: List[str],
    resid_col: str,
    cluster_subset: List[str],
    weights_col: str | None = None
) -> Tuple[np.ndarray, int]:
    """
    Aggregate per-cluster score sums for a given subset of cluster columns using Polars.

    Returns (meat, n_clusters) where meat = S.T @ S and S has shape (n_clusters, k).
    """
    # Build sum expressions: SUM(x * resid * weight?) for each regressor
    exprs = []
    for x in x_cols:
        if weights_col:
            expr = (pl.col(x) * pl.col(resid_col) * pl.col(weights_col)).sum().alias(f"score_{x}")
        else:
            expr = (pl.col(x) * pl.col(resid_col)).sum().alias(f"score_{x}")
        exprs.append(expr)

    # Group by the cluster subset
    grouped = df.group_by(cluster_subset).agg(exprs)

    # If no groups (empty), return zero meat
    if grouped.height == 0:
        k = len(x_cols)
        return np.zeros((k, k)), 0

    # Build S (G x k)
    scores_cols = [f"score_{x}" for x in x_cols]
    # Polars -> numpy: shape (G, k)
    S = np.vstack([grouped[col].to_numpy() for col in scores_cols]).T
    meat = S.T @ S
    return meat, S.shape[0]


def cgm_vcov_polars(
    df: pl.DataFrame,
    x_cols: List[str],
    resid_col: str,
    cluster_cols: List[str],
    XtX_inv: np.ndarray,
    n_obs: int,
    df_resid: int,
    weights_col: str | None = None,
    ssc: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute CGM multi-way clustered vcov using Polars aggregations.

    Parameters
    - df: Polars DataFrame containing demeaned X columns (use the same X that produced XtX_inv),
          and a column with residuals named resid_col (or compute resid and add it).
    - x_cols: list of regressor column names (these should be the demeaned regressor columns used)
    - resid_col: name of the residual column in df (dtype numeric)
    - cluster_cols: list of clustering columns (in user-specified order)
    - XtX_inv: (k x k) numpy array, inverse of X'X (or X_hat'X_hat for IV)
    - n_obs, df_resid: integers
    - weights_col: optional name of weights column if used
    - ssc: bool whether to apply small-sample correction ((N-1)/df_resid) at end

    Returns:
    - vcov_matrix: (k x k) numpy array
    - diagnostics: dict with per-subset meat matrices and cluster counts
    """
    k = len(x_cols)
    vcov_matrix = np.zeros((k, k))
    diagnostics = {"subsets": []}

    n_ways = len(cluster_cols)
    # Iterate over all non-empty subsets
    for subset_size in range(1, n_ways + 1):
        sign = (-1) ** (subset_size - 1)
        for subset in itertools.combinations(cluster_cols, subset_size):
            subset = list(subset)
            meat, G = _aggregate_scores_polars(df, x_cols, resid_col, subset, weights_col)
            diagnostics["subsets"].append({
                "subset": subset.copy(),
                "G": G,
                "meat": meat
            })
            if G <= 1:
                # skip subsets that do not produce multiple clusters
                continue
            adj_g = G / (G - 1)
            vcov_matrix += sign * adj_g * (XtX_inv @ meat @ XtX_inv)

    if ssc:
        vcov_matrix *= ((n_obs - 1) / df_resid)

    # Also return first-level cluster counts
    first_level_counts = tuple(
        df.group_by([c]).agg(pl.count()).height for c in cluster_cols
    )
    diagnostics["first_level_counts"] = first_level_counts
    diagnostics["n_obs"] = n_obs
    diagnostics["df_resid"] = df_resid

    return vcov_matrix, diagnostics



# aggs_duckdb.py
import itertools
import numpy as np
from typing import List, Tuple, Dict, Any
import duckdb

def _aggregate_scores_duckdb(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    x_cols: List[str],
    resid_col: str,
    cluster_subset: List[str],
    weights_col: str | None = None
) -> Tuple[np.ndarray, int]:
    """
    Aggregate per-cluster score sums for a given subset using DuckDB SQL.

    Returns (meat, n_clusters).
    """
    score_exprs = []
    for i, x in enumerate(x_cols):
        if weights_col:
            score_exprs.append(f"SUM({x} * {resid_col} * {weights_col}) AS score_{i}")
        else:
            score_exprs.append(f"SUM({x} * {resid_col}) AS score_{i}")

    # Build GROUP BY clause (multiple columns)
    group_by = ", ".join([f'"{c}"' for c in cluster_subset])
    select_group_cols = ", ".join([f'"{c}"' for c in cluster_subset])
    sql = f"""
        SELECT {select_group_cols}, {', '.join(score_exprs)}
        FROM "{table_name}"
        GROUP BY {group_by}
    """
    res = con.execute(sql).fetchall()
    if len(res) == 0:
        k = len(x_cols)
        return np.zeros((k, k)), 0

    # Convert to matrix: each row has cluster key then k score columns
    # Use DuckDB -> Arrow -> numpy via .pl() if you prefer polars; here we use fetchall -> numpy
    # fetchall returns list of tuples: (cl1, cl2, ..., score_0, score_1, ...)
    n_cols = len(res[0])
    k = len(x_cols)
    # Extract score columns from tuples
    scores = np.array([[row[n_cols - k + j] or 0.0 for j in range(k)] for row in res], dtype=float)  # shape (G, k)
    meat = scores.T @ scores
    G = scores.shape[0]
    return meat, G


def cgm_vcov_duckdb(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    x_cols: List[str],
    resid_col: str,
    cluster_cols: List[str],
    XtX_inv: np.ndarray,
    n_obs: int,
    df_resid: int,
    weights_col: str | None = None,
    ssc: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute CGM multi-way clustered vcov using DuckDB aggregations.

    table_name: name of the temporary table that contains demeaned {x_cols}, resid_col, and cluster_cols.
    """
    k = len(x_cols)
    vcov_matrix = np.zeros((k, k))
    diagnostics = {"subsets": []}
    n_ways = len(cluster_cols)

    for subset_size in range(1, n_ways + 1):
        sign = (-1) ** (subset_size - 1)
        for subset in itertools.combinations(cluster_cols, subset_size):
            subset = list(subset)
            meat, G = _aggregate_scores_duckdb(con, table_name, x_cols, resid_col, subset, weights_col)
            diagnostics["subsets"].append({
                "subset": subset.copy(),
                "G": G,
                "meat": meat
            })
            if G <= 1:
                continue
            adj_g = G / (G - 1)
            vcov_matrix += sign * adj_g * (XtX_inv @ meat @ XtX_inv)

    if ssc:
        vcov_matrix *= ((n_obs - 1) / df_resid)

    # first-level cluster counts
    first_counts = []
    for c in cluster_cols:
        q = f'SELECT COUNT(DISTINCT "{c}") FROM "{table_name}"'
        r = con.execute(q).fetchone()
        first_counts.append(int(r[0]) if r is not None else 0)
    diagnostics["first_level_counts"] = tuple(first_counts)
    diagnostics["n_obs"] = n_obs
    diagnostics["df_resid"] = df_resid

    return vcov_matrix, diagnostics