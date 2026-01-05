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
from enum import Enum

# ============================================================================
# CONSTANTS
# ============================================================================

# Multi-way clustering constants
MIN_CLUSTERS_FOR_ADJUSTMENT = 2
FIRST_ORDER_SUBSET_SIZE = 1


# ============================================================================
# ENUMS
# ============================================================================

class VcovType(Enum):
    """Variance-covariance estimator types."""
    IID = "iid"
    HC1 = "HC1"
    CLUSTER = "cluster"


# ============================================================================
# DATA CLASSES
# ============================================================================

class FormulaComponents(NamedTuple):
    """Parsed components of a regression formula."""
    y_col: str
    x_cols: list[str]
    fe_cols: list[str]
    factor_vars: list[tuple[str, str | None]]
    interactions: list[tuple[str, str, str | None]]
    instruments: list[str]


# ============================================================================
# FORMULA PARSING
# ============================================================================

def parse_formula(formula: str) -> FormulaComponents:
    """
    Parse R-style formula into components.

    Supports:
    - Basic: 'y ~ x1 + x2' (no fixed effects)
    - With FE: 'y ~ x1 + x2 | fe1 + fe2'
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
    FormulaComponents
        Named tuple with: y_col, x_cols, fe_cols, factor_vars, interactions, instruments
    """
    parts = [p.strip() for p in formula.split('|')]
    if len(parts) > 3:
        raise ValueError("Formula has too many parts. Use: 'y ~ x' or 'y ~ x | fe' or 'y ~ x | fe | z' (IV)")

    # Parse left-hand side (y ~ x terms)
    lhs_rhs = parts[0].split('~')
    if len(lhs_rhs) != 2:
        raise ValueError("Formula must have exactly one '~' separating y and x variables")

    y_col = lhs_rhs[0].strip()

    # Parse x terms
    x_terms = [x.strip() for x in lhs_rhs[1].split('+') if x.strip() != ""]
    x_cols, factor_vars, interactions = _parse_x_terms(x_terms)

    # Parse fixed effects (optional)
    if len(parts) >= 2 and parts[1].strip() != "":
        fe_cols = [f.strip() for f in parts[1].split('+') if f.strip() != ""]
    else:
        fe_cols = []

    # Parse instruments (if present)
    instruments = []
    if len(parts) == 3 and parts[2].strip() != "":
        instruments = [z.strip() for z in parts[2].split('+') if z.strip() != ""]

    return FormulaComponents(y_col, x_cols, fe_cols, factor_vars, interactions, instruments)


def _parse_x_terms(x_terms: list[str]) -> tuple[list[str], list[tuple], list[tuple]]:
    """
    Parse X terms into regular variables, factor variables, and interactions.

    Parameters
    ----------
    x_terms : list of str
        List of term strings from formula

    Returns
    -------
    tuple
        (x_cols, factor_vars, interactions)
    """
    x_cols = []
    factor_vars = []
    interactions = []

    for term in x_terms:
        if ':i(' in term and term.endswith(')'):
            # Interaction term: var:i(factor) or var:i(factor, ref=value)
            parsed = _parse_interaction_term(term)
            interactions.append(parsed)
        elif term.startswith('i(') and term.endswith(')'):
            # Factor variable: i(var) or i(var, ref=value)
            var, ref = _parse_i_term(term)
            factor_vars.append((var, ref))
        else:
            # Regular variable
            if term != "":
                x_cols.append(term)

    return x_cols, factor_vars, interactions


def _parse_i_term(term: str) -> tuple[str, str | None]:
    """
    Parse i() term with optional reference category.

    Supports:
    - i(var) -> (var, None) - first category is reference
    - i(var, ref=value) -> (var, "value") - specified reference
    - i(var, ref="value") -> (var, "value") - quoted reference

    Parameters
    ----------
    term : str
        i() term to parse

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


def _parse_interaction_term(term: str) -> tuple[str, str, str | None]:
    """
    Parse interaction term: var:i(factor) or var:i(factor, ref=value).

    Parameters
    ----------
    term : str
        Interaction term to parse

    Returns
    -------
    tuple
        (variable, factor, reference_category_or_None)
    """
    match = re.match(r'(\w+):i\((\w+)(?:\s*,\s*ref\s*=\s*["\']?([^"\')\s]+)["\']?)?\)', term)
    if match:
        return match.group(1), match.group(2), match.group(3)
    raise ValueError(f"Invalid interaction syntax: {term}")


# ============================================================================
# IV/2SLS ESTIMATION
# ============================================================================

def iv_2sls(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-Stage Least Squares (2SLS) estimation.
    
    First stage:  X = Z @ gamma + error
    Second stage: Y = X_hat @ beta + error
    
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
        
    Raises
    ------
    ValueError
        If under-identified (fewer instruments than endogenous variables)
        
    Examples
    --------
    >>> Y = np.array([1, 2, 3, 4])
    >>> X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
    >>> Z = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
    >>> beta, X_hat = iv_2sls(Y, X, Z)
    >>> beta.shape
    (2,)
    """
    if Z.shape[1] < X.shape[1]:
        raise ValueError(
            f"Under-identified: {Z.shape[1]} instruments for {X.shape[1]} endogenous variables"
        )
    
    if weights is not None:
        beta, X_hat = _iv_2sls_weighted(Y, X, Z, weights)
    else:
        beta, X_hat = _iv_2sls_unweighted(Y, X, Z)
    
    return beta, X_hat


def _iv_2sls_weighted(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted 2SLS estimation."""
    sqrt_weights = np.sqrt(weights)
    Z_weighted = Z * sqrt_weights[:, np.newaxis]
    X_weighted = X * sqrt_weights[:, np.newaxis]
    Y_weighted = Y * sqrt_weights
    
    # First stage: X = Z @ gamma
    ZtZ = Z_weighted.T @ Z_weighted
    ZtX = Z_weighted.T @ X_weighted
    gamma = np.linalg.solve(ZtZ, ZtX)
    X_hat = Z @ gamma
    
    # Second stage: Y = X_hat @ beta
    X_hat_weighted = X_hat * sqrt_weights[:, np.newaxis]
    XhtXh = X_hat_weighted.T @ X_hat_weighted
    XhtY = X_hat_weighted.T @ Y_weighted
    beta = np.linalg.solve(XhtXh, XhtY)
    
    return beta, X_hat


def _iv_2sls_unweighted(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unweighted 2SLS estimation."""
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


# ============================================================================
# STANDARD ERROR COMPUTATION
# ============================================================================

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
    Compute standard errors for OLS/IV coefs.
    
    Multi-way clustering uses the Cameron-Gelbach-Miller (2011) approach:
    For 2-way: V = V1 + V2 - V12
    For 3-way: V = V1 + V2 + V3 - V12 - V13 - V23 + V123
    
    Parameters
    ----------
    XtX_inv : np.ndarray
        (X'X)^-1 matrix
    resid : np.ndarray
        Residuals
    n_obs : int
        Number of observations
    df_resid : int
        Residual degrees of freedom
    vcov : str
        Variance type: "iid", "HC1", or "cluster"
    X : np.ndarray
        Design matrix
    weights : np.ndarray, optional
        Regression weights
    cluster_ids : np.ndarray, optional
        Cluster IDs (1D for single-way, 2D for multi-way)
    ssc : bool, default False
        Apply small sample correction
        
    Returns
    -------
    tuple
        (standard_errors, n_clusters)
        - standard_errors: SE for each coefficient
        - n_clusters: None for iid/HC1, int for one-way, tuple for multi-way
        
    Examples
    --------
    >>> XtX_inv = np.eye(2)
    >>> resid = np.array([0.1, -0.1, 0.2, -0.2])
    >>> X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
    >>> se, _ = compute_standard_errors(XtX_inv, resid, 4, 2, "iid", X)
    >>> se.shape
    (2,)
    """
    n_clusters = None
    vcov_lower = vcov.lower()
    
    if vcov_lower == VcovType.IID.value:
        se = _compute_se_iid(XtX_inv, resid, df_resid, weights)
        
    elif vcov_lower == VcovType.HC1.value.lower():
        se = _compute_se_hc1(XtX_inv, resid, X, n_obs, df_resid, weights)
        
    elif vcov_lower == VcovType.CLUSTER.value:
        if cluster_ids is None:
            raise ValueError("cluster_ids required for vcov='cluster'")
        
        # Check if multi-way clustering
        if cluster_ids.ndim == 2 and cluster_ids.shape[1] > 1:
            se, n_clusters = _compute_se_cluster_multiway(
                XtX_inv, resid, X, weights, cluster_ids, n_obs, df_resid, ssc
            )
        else:
            # Single-way clustering - handle both 1D and 2D with single column
            if cluster_ids.ndim == 2:
                cluster_ids = cluster_ids.flatten()
            
            se, n_clusters = _compute_se_cluster_oneway(
                XtX_inv, resid, X, weights, cluster_ids, n_obs, df_resid, ssc
            )
    else:
        raise ValueError(f"vcov must be 'iid', 'HC1', or 'cluster', got '{vcov}'")
    
    return se, n_clusters


def _compute_se_iid(
    XtX_inv: np.ndarray,
    resid: np.ndarray,
    df_resid: int,
    weights: np.ndarray | None
) -> np.ndarray:
    """Compute IID (homoskedastic) standard errors."""
    if weights is not None:
        sigma_squared = np.sum(weights * resid**2) / df_resid
    else:
        sigma_squared = np.sum(resid**2) / df_resid
    
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma_squared, 0.0))
    return se


def _compute_se_hc1(
    XtX_inv: np.ndarray,
    resid: np.ndarray,
    X: np.ndarray,
    n_obs: int,
    df_resid: int,
    weights: np.ndarray | None
) -> np.ndarray:
    """Compute heteroskedasticity-robust (HC1) standard errors."""
    if weights is not None:
        meat = X.T @ (X * (weights * resid**2)[:, np.newaxis])
    else:
        meat = X.T @ (X * (resid**2)[:, np.newaxis])
    
    vcov_matrix = XtX_inv @ meat @ XtX_inv
    adjustment = n_obs / df_resid
    se = np.sqrt(np.maximum(np.diag(vcov_matrix) * adjustment, 0.0))
    return se


def _compute_se_cluster_oneway(
    XtX_inv: np.ndarray,
    resid: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray | None,
    cluster_ids: np.ndarray,
    n_obs: int,
    df_resid: int,
    ssc: bool
) -> tuple[np.ndarray, int]:
    """Compute one-way clustered variance-covariance matrix."""
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


def _compute_se_cluster_multiway(
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
    Compute multi-way clustered variance-covariance matrix.
    
    Uses Cameron-Gelbach-Miller (2011) approach with fixest-compatible SSC.
    
    Implementation matches fixest defaults:
    - G.df = "min": Single G_min/(G_min-1) adjustment at the end
    - Components are accumulated WITHOUT per-component G/(G-1) adjustment
    - K adjustment (n-1)/(n-K) applied separately if ssc=True
    """
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
# RESULT BUILDING (DEPRECATED - Use LeanFEResult directly)
# ============================================================================

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
    
    Parameters
    ----------
    x_cols : list of str
        Regressor names
    beta : np.ndarray
        Coefficient estimates
    se : np.ndarray
        Standard errors
    n_obs : int
        Number of observations
    iterations : int
        Number of demeaning iterations
    vcov : str
        Variance estimator type
    is_iv : bool
        Whether IV was used
    n_instruments : int, optional
        Number of instruments
    n_clusters : int, optional
        Number of clusters
    df_resid : int, optional
        Residual degrees of freedom
    r_squared_within : float, optional
        Within R-squared
    formula : str, optional
        Original formula
    fe_cols : list of str, optional
        Fixed effect column names
        
    Returns
    -------
    LeanFEResult
        Result object with formatted output
    """
    return LeanFEResult(
        coefs=dict(zip(x_cols, beta)),
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
    
    Parameters
    ----------
    x_cols : list of str
        Regressor names
    beta : np.ndarray
        Coefficient estimates
    se : np.ndarray
        Standard errors
    n_obs : int
        Number of observations
    iterations : int
        Number of demeaning iterations
    vcov : str
        Variance estimator type
    is_iv : bool
        Whether IV was used
    n_instruments : int, optional
        Number of instruments
    n_clusters : int, optional
        Number of clusters
        
    Returns
    -------
    dict
        Standardized output with coefs, std_errors, n_obs, iterations,
        vcov_type, is_iv, n_instruments, n_clusters
    """
    return {
        'coefs': dict(zip(x_cols, beta)),
        'std_errors': dict(zip(x_cols, se)),
        'n_obs': n_obs,
        'iterations': iterations,
        'vcov_type': vcov,
        'is_iv': is_iv,
        'n_instruments': n_instruments if is_iv else None,
        'n_clusters': n_clusters
    }