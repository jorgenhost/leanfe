"""
Common utilities shared between Polars and DuckDB backends.

Contains formula parsing, IV estimation, and other shared logic.
"""

import re
import numpy as np
from typing import NamedTuple
from leanfe.result import LeanFEResult
from itertools import combinations
from scipy import sparse

class FormulaComponents(NamedTuple):
    y_col: str
    x_cols: list[str]
    fe_cols: list[str]
    factor_vars: list[tuple[str, str | None]]
    interactions: list[tuple[str, str, str | None]]
    instruments: list[str]

def _oneway_cluster_vcov(
    XtX_inv: np.ndarray,
    resid: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray | None,
    cluster_ids: np.ndarray,
    n_obs: int,
    df_resid: int,
    ssc: bool
) -> np.ndarray:
    """Compute one-way clustered variance-covariance matrix."""
    
    unique_clusters, cluster_map = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(unique_clusters)
    
    # Build sparse cluster indicator matrix
    W_C = sparse.csr_matrix(
        (np.ones(n_obs), (np.arange(n_obs), cluster_map)),
        shape=(n_obs, n_clusters)
    )
    
    # Compute scores
    if weights is not None:
        X_resid = X * (resid * weights)[:, np.newaxis]
    else:
        X_resid = X * resid[:, np.newaxis]
    
    scores = W_C.T @ X_resid
    meat = scores.T @ scores
    
    # Adjustment
    if ssc:
        adjustment = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / df_resid)
    else:
        adjustment = n_clusters / (n_clusters - 1)
    
    return XtX_inv @ meat @ XtX_inv * adjustment

def _multiway_cluster_vcov(
    XtX_inv: np.ndarray,
    resid: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray | None,
    cluster_ids: np.ndarray,
    n_obs: int,
    df_resid: int,
    ssc: bool
) -> np.ndarray:
    """
    Compute multi-way clustered variance-covariance matrix.
    Uses Cameron-Gelbach-Miller (2011) approach.
    """

    n_ways = cluster_ids.shape[1]
    vcov_matrix = np.zeros_like(XtX_inv)

    # We'll collect first-order cluster counts to compute the "G_min" correction
    # (fixest default: use G_min/(G_min-1) once on the final VCOV).
    n_clusters_list = []

    # Sum over all non-empty subsets with alternating signs
    for k in range(1, n_ways + 1):
        sign = (-1) ** (k - 1)

        # Iterate over all k-way combinations
        for idx_combo in combinations(range(n_ways), k):
            # Create intersection cluster ID
            if k == 1:
                intersect_ids = cluster_ids[:, idx_combo[0]].astype(str)  # Ensure string type
            else:
                # Concatenate cluster IDs for intersection
                intersect_ids = np.array([
                    '_'.join(str(cluster_ids[i, j]) for j in idx_combo)
                    for i in range(n_obs)
                ])

            # Get unique clusters and build indicator matrix
            unique_clusters, cluster_map = np.unique(intersect_ids, return_inverse=True)
            n_clusters = len(unique_clusters)

            # Skip subsets that don't form multiple clusters (can't form cluster VCOV)
            if n_clusters <= 1:
                continue

            # Store first-order cluster counts for reporting / G_min if this is size-1 subset
            if k == 1:
                n_clusters_list.append(n_clusters)

            # Build sparse cluster indicator matrix
            W_C = sparse.csr_matrix(
                (np.ones(n_obs), (np.arange(n_obs), cluster_map)),
                shape=(n_obs, n_clusters)
            )

            # Compute scores
            if weights is not None:
                X_resid = X * (resid * weights)[:, np.newaxis]
            else:
                X_resid = X * resid[:, np.newaxis]

            scores = W_C.T @ X_resid
            meat = scores.T @ scores

            # NOTE: we do NOT apply the per-component G/(G-1) adjustment here.
            # The alternating-sum is computed on the unadjusted components and
            # a single G_min/(G_min-1) correction is applied once below (fixest default).
            vcov_matrix += sign * (XtX_inv @ meat @ XtX_inv)

    # Small-sample cluster correction: apply G_min/(G_min-1) once if requested (fixest default).
    if ssc and len(n_clusters_list) > 0:
        G_min = min(n_clusters_list)
        if G_min > 1:
            vcov_matrix *= (G_min / (G_min - 1))

    # Small-sample K adjustment ((n-1)/(n-K)) if requested — apply once at end.
    if ssc:
        # df_resid is expected to be n - K in caller code
        vcov_matrix *= (n_obs / df_resid)  # keep same scaling as HC1 handling (n / df_resid)
        # If you prefer the (n-1)/(n-K) variant, adjust to: vcov_matrix *= ((n_obs - 1) / df_resid)

    return vcov_matrix
            
def _parse_i_term(term: str) -> tuple[str, str | None]:
    """
    Parse i() term with optional reference category.
    
    Supports:
    - i(var) -> (var, None) - first category is reference
    - i(var, ref=value) -> (var, "value") - specified reference
    - i(var, ref="value") -> (var, "value") - quoted reference
    
    Returns
    -------
    tuple
        (variable_name, reference_category_or_None)
    """
    # Match i(var) or i(var, ref=value) or i(var, ref="value")
    match = re.match(r'i\((\w+)(?:\s*,\s*ref\s*=\s*["\']?([^"\')\s]+)["\']?)?\)', term)
    if match:
        return match.group(1), match.group(2)
    raise ValueError(f"Invalid i() syntax: {term}. Use i(var) or i(var, ref=value)")


def parse_formula(formula: str) -> FormulaComponents:
    """
        Parse R-style formula into components.
        
        Supports:
        - Basic: 'y ~ x1 + x2 | fe1 + fe2'
        - Factor variables: 'y ~ x1 + i(region) | fe1'
        - Factor with reference: 'y ~ x1 + i(region, ref=R1) | fe1'
        - Interactions: 'y ~ x1 + treatment:i(region) | fe1'
        - Interactions with reference: 'y ~ x1 + treatment:i(region, ref=R1) | fe1'
        - IV/2SLS: 'y ~ x1 + x2 | fe1 + fe2 | z1 + z2'
        
        Parameters
        ----------
        formula : str
            R-style formula string
            
        Returns
        -------
        tuple
            (y_col, x_cols, fe_cols, factor_vars, interactions, instruments)
            - y_col: dependent variable name
            - x_cols: list of independent variable names
            - fe_cols: list of fixed effect column names
            - factor_vars: list of (var, ref) tuples where ref is None or reference category
            - interactions: list of (var, factor, ref) tuples for var:i(factor) terms
            - instruments: list of instrument variable names (for IV)
    """
    parts = formula.split('|')
    if len(parts) < 2:
        raise ValueError("Formula must include fixed effects: 'y ~ x | fe1 + fe2'")
    if len(parts) > 3:
        raise ValueError("Formula has too many parts. Use: 'y ~ x | fe' or 'y ~ x | fe | z' (IV)")
    
    # Parse left-hand side (y ~ x terms)
    lhs_rhs = parts[0].split('~')
    if len(lhs_rhs) != 2:
        raise ValueError("Formula must have exactly one '~' separating y and x variables")
    
    y_col = lhs_rhs[0].strip()
    
    # Parse x terms
    x_terms = [x.strip() for x in lhs_rhs[1].split('+')]
    x_cols = []
    factor_vars = []
    interactions = []
    
    for term in x_terms:
        if ':i(' in term and term.endswith(')'):
            # Interaction term: var:i(factor) or var:i(factor, ref=value)
            match = re.match(r'(\w+):i\((\w+)(?:\s*,\s*ref\s*=\s*["\']?([^"\')\s]+)["\']?)?\)', term)
            if match:
                interactions.append((match.group(1), match.group(2), match.group(3)))
            else:
                raise ValueError(f"Invalid interaction syntax: {term}")
        elif term.startswith('i(') and term.endswith(')'):
            # Factor variable: i(var) or i(var, ref=value)
            var, ref = _parse_i_term(term)
            factor_vars.append((var, ref))
        else:
            # Regular variable
            x_cols.append(term)
    
    # Parse fixed effects
    fe_cols = [f.strip() for f in parts[1].split('+')]
    
    # Parse instruments (if present)
    instruments = []
    if len(parts) == 3:
        instruments = [z.strip() for z in parts[2].split('+')]

    return FormulaComponents(y_col, x_cols, fe_cols, factor_vars, interactions, instruments)

def iv_2sls(
    Y: np.ndarray, 
    X: np.ndarray, 
    Z: np.ndarray, 
    weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-Stage Least Squares (2SLS) estimation.
    
    Parameters
    ----------
    Y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Endogenous regressors (n, k)
    Z : np.ndarray
        Instruments (n, m) where m >= k
    weights : np.ndarray, optional
        Regression weights (n,)
        
    Returns
    -------
    tuple
        (beta, X_hat) where beta is coefficient vector and X_hat is fitted values from first stage
    """
    if Z.shape[1] < X.shape[1]:
        raise ValueError(f"Under-identified: {Z.shape[1]} instruments for {X.shape[1]} endogenous variables")
    
    if weights is not None:
        sqrt_w = np.sqrt(weights)
        Z_w = Z * sqrt_w[:, np.newaxis]
        X_w = X * sqrt_w[:, np.newaxis]
        Y_w = Y * sqrt_w
        
        # First stage: X = Z @ gamma
        ZtZ = Z_w.T @ Z_w
        ZtX = Z_w.T @ X_w
        gamma = np.linalg.solve(ZtZ, ZtX)
        X_hat = Z @ gamma
        
        # Second stage: Y = X_hat @ beta
        X_hat_w = X_hat * sqrt_w[:, np.newaxis]
        XhtXh = X_hat_w.T @ X_hat_w
        XhtY = X_hat_w.T @ Y_w
        beta = np.linalg.solve(XhtXh, XhtY)
    else:
        # First stage
        ZtZ = Z.T @ Z
        ZtX = Z.T @ X
        gamma = np.linalg.solve(ZtZ, ZtX)
        X_hat = Z @ gamma
        
        # Second stage
        XhtXh = X_hat.T @ X_hat
        XhtY = X_hat.T @ Y
        beta = np.linalg.solve(XhtXh, XhtY)
    
    return beta, X_hat


def compute_standard_errors(
    XtX_inv: np.ndarray,
    resid: np.ndarray,
    n_obs: int,
    df_resid: int,
    vcov: str,
    X: np.ndarray,
    weights: np.ndarray | None = None,
    cluster_ids: np.ndarray | None = None,
    ssc: bool = False
) -> tuple[np.ndarray, int | tuple | None]:
    """
    Compute standard errors for OLS/IV coefficients.
    
    Multi-way clustering uses the Cameron-Gelbach-Miller (2011) approach:
    For 2-way: V = V1 + V2 - V12
    For 3-way: V = V1 + V2 + V3 - V12 - V13 - V23 + V123
    
    Returns
    -------
    tuple
        (se, n_clusters) where n_clusters is:
        - None for iid/HC1
        - int for one-way clustering
        - tuple of ints for multi-way clustering
    """
    n_clusters = None
    
    if vcov == "iid":
        if weights is not None:
            sigma2 = np.sum(weights * resid**2) / df_resid
        else:
            sigma2 = np.sum(resid**2) / df_resid
        se = np.sqrt(np.diag(XtX_inv) * sigma2)
        
    elif vcov == "HC1":
        if weights is not None:
            meat = X.T @ (X * (weights * resid**2)[:, np.newaxis])
        else:
            meat = X.T @ (X * (resid**2)[:, np.newaxis])
        vcov_matrix = XtX_inv @ meat @ XtX_inv
        adjustment = n_obs / df_resid
        se = np.sqrt(np.diag(vcov_matrix) * adjustment)
        
    elif vcov == "cluster":
        if cluster_ids is None:
            raise ValueError("cluster_ids required for vcov='cluster'")
        
        # Check if multi-way clustering (cluster_ids is 2D array)
        if cluster_ids.ndim == 2 and cluster_ids.shape[1] > 1:
            vcov_matrix = _multiway_cluster_vcov(
                XtX_inv, resid, X, weights, cluster_ids, n_obs, df_resid, ssc
            )
            n_clusters = tuple(len(np.unique(cluster_ids[:, i])) for i in range(cluster_ids.shape[1]))
        else:
            # Single-way clustering - handle both 1D and 2D with single column
            if cluster_ids.ndim == 2:
                cluster_ids = cluster_ids.flatten()
            
            vcov_matrix = _oneway_cluster_vcov(
                XtX_inv, resid, X, weights, cluster_ids, n_obs, df_resid, ssc
            )
            n_clusters = len(np.unique(cluster_ids))
        
        se = np.sqrt(np.diag(vcov_matrix))
        
    else:
        raise ValueError(f"vcov must be 'iid', 'HC1', or 'cluster', got '{vcov}'")
    
    return se, n_clusters
    
def build_result(
    x_cols: list[str],
    beta: np.ndarray,
    se: np.ndarray,
    n_obs: int,
    iterations: int,
    vcov: str,
    is_iv: bool,
    n_instruments: int | None,
    n_clusters: int | None,
    df_resid: int | None = None,
    r_squared_within: float | None = None,
    formula: str | None = None,
    fe_cols: list[str] | None = None
) -> LeanFEResult:
    """
    Build LeanFEResult object.
    
    Returns
    -------
    LeanFEResult
        Result object with formatted output
    """
    return LeanFEResult(
        coefficients=dict(zip(x_cols, beta)),
        std_errors=dict(zip(x_cols, se)),
        n_obs=n_obs,
        iterations=iterations,
        vcov_type=vcov,
        is_iv=is_iv,
        n_instruments=n_instruments if is_iv else None,
        n_clusters=n_clusters,
        df_resid=df_resid,
        r_squared_within=r_squared_within,
        formula=formula,
        fe_cols=fe_cols
    )


# Keep old function for backwards compatibility
def build_result_dict(
    x_cols: list[str],
    beta: np.ndarray,
    se: np.ndarray,
    n_obs: int,
    iterations: int,
    vcov: str,
    is_iv: bool,
    n_instruments: int | None,
    n_clusters: int | None
) -> dict:
    """
    Build standardized result dictionary.
    
    DEPRECATED: Use build_result() instead which returns LeanFEResult.
    
    Returns
    -------
    dict
        Standardized output with coefficients, std_errors, n_obs, iterations,
        vcov_type, is_iv, n_instruments, n_clusters
    """
    return {
        'coefficients': dict(zip(x_cols, beta)),
        'std_errors': dict(zip(x_cols, se)),
        'n_obs': n_obs,
        'iterations': iterations,
        'vcov_type': vcov,
        'is_iv': is_iv,
        'n_instruments': n_instruments if is_iv else None,
        'n_clusters': n_clusters
    }
