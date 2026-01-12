"""
Common utilities shared between Polars and DuckDB backends.

Contains formula parsing, IV estimation, and other shared logic.
"""

import re
import numpy as np
from typing import NamedTuple
from leanfe.result import LeanFEResult
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