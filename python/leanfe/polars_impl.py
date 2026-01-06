"""
Polars-based fixed effects regression implementation.

Optimized for speed using Polars LazyFrame/DataFrame operations.
Uses YOCO compression automatically for IID/HC1 standard errors.
"""
import polars as pl
import numpy as np
import polars.selectors as cs
from typing import Literal
from leanfe.result import LeanFEResult
from leanfe.common import (
    parse_formula,
    iv_2sls,
)
from leanfe.std_errors import compute_standard_errors_polars
from leanfe.compress import (
    determine_strategy,
    leanfe_compress_polars, 
    estimate_compression_ratio
)
pl.Config.set_engine_affinity('streaming')

MAX_FE_LEVELS = 10_000


def _expand_factors_polars(lf: pl.LazyFrame, factor_vars: list[tuple]) -> tuple:
    """
    Expand factor variables into dummy variables, dropping reference category.
    """
    if not factor_vars:
        return lf, []

    unique_cat_map = lf.select([
        pl.col(var).unique().sort().alias(var)
        for var, _  in factor_vars
    ]).collect()

    dummy_cols = []
    factor_exprs = []

    for var, ref in factor_vars:
        categories = unique_cat_map.select(pl.col(var)).to_series().to_list()
        if ref is None:
            ref_cat = categories[0]
        else:
            ref_cat = ref
            if categories and not isinstance(categories[0], type(ref)):
                try:
                    ref_cat(categories[0])(ref)
                except (ValueError, TypeError):
                    pass

            if ref_cat not in categories:
                raise ValueError(f"Reference category '{ref}' not found in {var}. Available: {categories}")
        # Build expressions for dummy columns
        for cat in categories:
            if cat == ref_cat:
                continue  # Skip reference category
            dummy_name = f"{var}_{cat}"
            expr = (pl.col(var) == cat).cast(pl.UInt8).alias(dummy_name)
            factor_exprs.append(expr)
            dummy_cols.append(dummy_name)

    if factor_exprs:
        lf = lf.with_columns(
            factor_exprs
        )
    return lf, dummy_cols


def _expand_interactions_polars(lf: pl.LazyFrame, interactions: list[tuple]) -> tuple:
    """
    Expand interaction terms: var:i(factor) -> var_cat1, var_cat2, ...
    """
    if not interactions:
        return lf, []

    # Collect all unique categories for all factors in ONE pass
    unique_factors = list(set(inter[1] for inter in interactions))
    unique_cat_map = lf.select([
        pl.col(f).unique().sort().alias(f) 
        for f in unique_factors
    ]).collect()

    interaction_cols = []
    interaction_exprs = []

    for var, factor, ref in interactions:
        categories = unique_cat_map.select(pl.col(factor)).to_series().to_list()
        if ref is None:
            # Default: drop first category
            ref_cat = categories[0]
        else:
            ref_cat = ref
            if categories and not isinstance(categories[0], type(ref)):
                try:
                    ref_cat = type(categories[0])(ref)
                except (ValueError, TypeError):
                    pass
            if ref_cat not in categories:
                raise ValueError(f"Reference category '{ref}' not found in {factor}. Available: {categories}")

        for cat in categories:
            if cat == ref_cat:
                continue  # Skip reference category
            col_name = f"{var}_{cat}"
            expr = (pl.col(var) * (pl.col(factor) == cat)).cast(pl.Float64).alias(col_name)
            interaction_exprs.append(expr)
            interaction_cols.append(col_name)
    if interaction_exprs:
        lf = lf.with_columns(
            interaction_exprs
        )
    return lf, interaction_cols


def _cats_to_int(lf: pl.LazyFrame, fe_cols: list[str] | None) -> pl.LazyFrame:
    """If FE cols are coded as pl.Categorical/pl.String, cast to pl.Int for numpy to read."""
    if fe_cols is None:
        return lf

    # select only FE columns that are string/categorical
    # get schema of selected columns
    try:
        fe_cols_cast = lf.select(pl.col(fe_cols)).select(cs.by_dtype(pl.String, pl.Categorical)).collect_schema().names()
    except Exception:
        fe_cols_cast = []
    EXPR = [
        pl.col(col).alias(col).cast(pl.Categorical).to_physical()
        for col in fe_cols_cast
    ]

    if EXPR:
        lf = lf.with_columns(
            EXPR
        )

    return lf

def _run_regression(
    df: pl.DataFrame,
    y_col: str,
    x_cols: list[str],
    instruments: list[str],
    weights: str | None,
    vcov: Literal["iid", "hc1", "cluster"],
    cluster_cols: list[str] | None,
    ssc: bool,
    n_obs: int,
    absorbed_df: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int | tuple | None, float | None]:
    """
    Common regression logic for both OLS and FE cases.
    
    Returns: (beta, se, resid, df_resid, n_clusters, r_squared)
    """
    assert vcov in ["iid", "hc1", "cluster"], f"Invalid vcov method: {vcov}"
    # Extract weights
    if weights is not None:
        w = df.select(weights).to_numpy().flatten()
    else:
        w = None
    
    # Build X including intercept
    if len(x_cols) > 0:
        X_reg = df.select([pl.col(c) for c in x_cols]).to_numpy()
        intercept = np.ones((X_reg.shape[0], 1))
        X = np.hstack([intercept, X_reg])
    else:
        X = np.ones((n_obs, 1))
    
    Y = df.select(y_col).to_numpy().flatten()
    
    # IV/2SLS or OLS
    is_iv = len(instruments) > 0
    if is_iv:
        Z = df.select(instruments).to_numpy()
        if X.shape[1] > Z.shape[1]:
            if not any(np.allclose(col, 1.0) for col in Z.T):
                Z = np.column_stack([np.ones(Z.shape[0]), Z])
        beta_full, X_hat_full = iv_2sls(Y, X, Z, w)
        beta = beta_full[1:] if X.shape[1] > len(x_cols) else beta_full
        X_hat = X_hat_full

        # Compute (X'X)^-1 for IV using Cholesky with fallback
        if w is not None:
            sqrt_w = np.sqrt(w)
            X_hat_w = X_hat * sqrt_w[:, np.newaxis]
            XtX_for_inv = X_hat_w.T @ X_hat_w
        else:
            XtX_for_inv = X_hat.T @ X_hat
        
        try:
            L = np.linalg.cholesky(XtX_for_inv)
            XtX_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(L.shape[0])))
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.inv(XtX_for_inv)
    else:
        # OLS path: compute XtX, solve for beta, and get XtX_inv in one go
        if w is not None:
            sqrt_w = np.sqrt(w)
            X_w = X * sqrt_w[:, np.newaxis]
            Y_w = Y * sqrt_w
            XtX = X_w.T @ X_w
            Xty = X_w.T @ Y_w
        else:
            XtX = X.T @ X
            Xty = X.T @ Y
        
        # Use Cholesky: solve for beta and compute inverse from same factorization
        try:
            L = np.linalg.cholesky(XtX)
            beta_full = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
            # Compute XtX_inv from same L factor
            XtX_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(L.shape[0])))
        except np.linalg.LinAlgError:
            # Fallback to direct methods
            beta_full = np.linalg.solve(XtX, Xty)
            XtX_inv = np.linalg.inv(XtX)
        
        if X.shape[1] == len(x_cols) + 1:
            beta = beta_full[1:]
        else:
            beta = beta_full
        X_hat = X

    # Residuals
    resid = Y - X_hat @ beta_full
    
    # Degrees of freedom
    df_resid = n_obs - (len(x_cols) + 1) - absorbed_df
    
    # Standard errors - use unified std_errors module for all cases
    if not is_iv:
        # For OLS with Polars expressions, pass XtX_inv without intercept dimension
        # since x_cols doesn't include intercept (data is demeaned)
        XtX_inv_no_intercept = XtX_inv[1:, 1:] if XtX_inv.shape[0] > len(x_cols) else XtX_inv
        
        se, n_clusters = compute_standard_errors_polars(
            df=df,
            x_cols=x_cols,
            XtX_inv=XtX_inv_no_intercept,
            resid=resid,
            weights=weights,
            vcov=vcov,
            cluster_cols=cluster_cols,
            n_obs=n_obs,
            df_resid=df_resid,
            ssc=ssc,
            X=None,  # Not needed for OLS
            is_iv=False
        )
    else:
        # For IV, use full XtX_inv and X_hat
        se_full, n_clusters = compute_standard_errors_polars(
            df=df,
            x_cols=x_cols,
            XtX_inv=XtX_inv,
            resid=resid,
            weights=weights,
            vcov=vcov,
            cluster_cols=cluster_cols,
            n_obs=n_obs,
            df_resid=df_resid,
            ssc=ssc,
            X=X_hat,
            is_iv=True
        )
        
        # Strip intercept from IV results
        if X.shape[1] == len(x_cols) + 1:
            se = se_full[1:]
        else:
            se = se_full
        
        # Skip the intercept stripping below for IV
        return beta, se, resid, df_resid, n_clusters, None
    
    # R-squared (only for OLS path)
    rss = np.sum(resid**2)
    tss = np.sum((Y - np.mean(Y))**2)
    r_squared = 1 - rss / tss if tss > 0 else None
    
    return beta, se, resid, df_resid, n_clusters, r_squared

def leanfe_polars(
    data: str | pl.DataFrame | pl.LazyFrame,
    demean_tol: float,
    y_col: str | None = None,
    x_cols: list[str] | None = None,
    fe_cols: list[str] | None = None,
    formula: str | None = None,
    strategy: str = 'auto',
    weights: str | None = None,
    max_iter: int = 25,
    vcov: Literal["iid", "hc1", "cluster"] = "iid",
    cluster_cols: list[str] | None = None,
    ssc: bool = True,
    sample_frac: float | None = None
) -> LeanFEResult:
    """
    Fast fixed effects regression using Polars and FWL theorem.

    Supports running OLS with no fixed effects (fe_cols=[] or formula without '|').
    """

    est_comp_ratio = None

    # Parse formula if provided
    if formula is not None:
        y_col, x_cols, fe_cols, factor_vars, interactions, instruments = parse_formula(formula)
    elif y_col is None or x_cols is None or fe_cols is None:
        raise ValueError("Must provide either 'formula' or (y_col, x_cols, fe_cols)")
    else:
        factor_vars = []
        interactions = []
        instruments = []

    # Ensure lists
    x_cols = list(x_cols)
    fe_cols = list(fe_cols) if fe_cols is not None else []

    # Build needed columns
    needed_cols = [y_col] + list(x_cols) + list(fe_cols) + instruments
    # Add factor variable columns
    for var, ref in factor_vars:
        if var not in needed_cols:
            needed_cols.append(var)
    # Add interaction variable columns
    for var, factor, ref in interactions:
        if var not in needed_cols:
            needed_cols.append(var)
        if factor not in needed_cols:
            needed_cols.append(factor)
    if cluster_cols is not None:
        needed_cols += [c for c in cluster_cols if c not in needed_cols]
    if weights is not None and weights not in needed_cols:
        needed_cols.append(weights)

    # Load data
    if isinstance(data, str):
        lf = pl.scan_parquet(data).select(needed_cols)
    elif isinstance(data, pl.LazyFrame):
        lf = data.select(needed_cols)
    else:
        lf = data.select(needed_cols).lazy()

    # Expand interactions
    if interactions:
        lf, interaction_cols = _expand_interactions_polars(lf, interactions)
        x_cols = x_cols + interaction_cols

    # Optimize types (always done for memory efficiency)
    factor_var_names = [var for var, ref in factor_vars]
    lf = _cats_to_int(lf, fe_cols + factor_var_names)

    # Sample data if requested
    if sample_frac is not None:
        lf = lf.collect().sample(fraction=sample_frac, seed=42).lazy()

    # Expand factor variables into dummies
    if factor_vars:
        lf, dummy_cols = _expand_factors_polars(lf, factor_vars)
        x_cols = x_cols + dummy_cols

    # Check if we should use compression strategy (faster for IID/HC1 without IV)
    is_iv = len(instruments) > 0
    
    fe_cardinality = None
    if fe_cols:
        # Strategy selection based on cost estimation:
        fe_cardinality = {fe: lf.select(pl.col(fe).n_unique()).collect().item() for fe in fe_cols} if fe_cols else {}

    if strategy == 'auto':

        n_obs_initial = lf.select(pl.len()).collect().item()

        est_comp_ratio = estimate_compression_ratio(
            data = lf,
            x_cols = x_cols,
            fe_cols = fe_cols
        )

        if not fe_cols:
            # No FE: decide between OLS and compress based on compression ratio
            if est_comp_ratio >= 0.8:
                inferred_strategy = 'ols'
            else:
                inferred_strategy = 'compress'

        elif len(fe_cols) == 1:
            inferred_strategy = 'demean'
        else:
            inferred_strategy = determine_strategy(
                vcov, is_iv, fe_cardinality,
                max_fe_levels=MAX_FE_LEVELS,
                n_obs=n_obs_initial,
                n_x_cols=len(x_cols),
                estimated_compression_ratio=est_comp_ratio
            )

        print(f'Auto selection: Inferring {inferred_strategy} strategy. N = {n_obs_initial:_}, est. compression ratio: {est_comp_ratio}')
        strategy = inferred_strategy

    if strategy == 'compress':
        print('Using compresssion strategy...')
        result = leanfe_compress_polars(
            lf=lf,
            y_col=y_col,
            x_cols=x_cols,
            fe_cols=fe_cols,
            weights=weights,
            vcov=vcov,
            cluster_col=cluster_cols,
            ssc=ssc
        )
        # Update attributes directly
        result.formula = formula
        result.fe_cols = fe_cols

        return result

    if strategy == 'demean':
        if not fe_cols or len(fe_cols) != 1:
            raise ValueError("Strategy 'demean' requires exactly one FE column.")
        assert fe_cardinality is not None, "No FE-cardinality computed for strategy='demean'"
        print("Using simple within-transform (demean) strategy for single FE...")

        fe = fe_cols[0]
        cols_to_demean = [y_col] + x_cols + instruments

        # Drop singletons (keep as LazyFrame)
        EXPRS = [pl.len().over(fe) > 1]
        df = lf.filter(pl.all_horizontal(EXPRS)).collect().lazy()

        # Weighted vs unweighted demeaning
        if weights is not None:
            demean_exprs = [
                (
                    pl.col(c)
                    - (pl.col(c) * pl.col(weights)).sum().over(fe)
                      / pl.col(weights).sum().over(fe)
                ).alias(c)
                for c in cols_to_demean
            ]
        else:
            demean_exprs = [
                (pl.col(c) - pl.col(c).mean().over(fe)).alias(c)
                for c in cols_to_demean
            ]

        # Apply within transform once and materialize
        df = (
            df.with_columns(demean_exprs)
              .select(needed_cols)
              .collect()
        )

        # FE dimension and absorbed DF
        unique_counts = df.select(pl.col(fe).n_unique())
        fe_dims = unique_counts.row(0)   # (n_levels,)
        absorbed_df = fe_dims[0] - 1
        n_obs = len(df)
        iterations = 1  # single-pass demeaning


    elif strategy == 'alt_proj':
        if not fe_cols:
            raise ValueError(
                "Strategy 'alt_proj' requires FE-cols. "
                "Use strategy='ols' instead for OLS without FE."
            )
        assert fe_cardinality is not None, "No FE-cardinality computed for strategy='alt_proj'"
        print('Using FWL/alternating projections strategy...')

        # Drop singletons (keep as LazyFrame)
        EXPRS = [
            pl.len().over(fe) > 1
            for fe in fe_cols
        ]
        df = lf.filter(pl.all_horizontal(EXPRS)).collect().lazy()

        # Order FEs by cardinality (low-card first) for faster convergence
        fe_cols_ordered = sorted(fe_cols, key=lambda fe: fe_cardinality.get(fe, 0))
        cols_to_demean = [y_col] + x_cols + instruments
        iterations = 0

        # Multi-FE FWL demeaning - all operations stay lazy between iterations
        for it in range(1, max_iter + 1):
            for fe in fe_cols_ordered:
                if weights is not None:
                    demean_exprs = [
                        (
                            pl.col(c)
                            - (pl.col(c) * pl.col(weights)).sum().over(fe)
                              / pl.col(weights).sum().over(fe)
                        ).alias(c)
                        for c in cols_to_demean
                    ]
                else:
                    demean_exprs = [
                        (pl.col(c) - pl.col(c).mean().over(fe)).alias(c)
                        for c in cols_to_demean
                    ]

                # Overwrite columns in place for this FE
                df = df.with_columns(demean_exprs)

            # Check convergence after cycling all FEs
            if it >= 3:
                check_exprs = [
                    pl.col(y_col).mean().over(fe).abs().alias(f"{y_col}_mean_abs_{fe}")
                    for fe in fe_cols
                ]
                max_mean = (
                    df.select(pl.max_horizontal(check_exprs))
                      .collect()
                      .max()
                      .item()
                )
                if max_mean < demean_tol:
                    iterations = it
                    df = df.select(needed_cols).collect()
                    break
            iterations = it

        if isinstance(df, pl.LazyFrame):
            df = df.select(needed_cols).collect()

        unique_counts = df.select([
            pl.col(fe).n_unique() for fe in fe_cols
        ])
        fe_dims = unique_counts.row(0)
        absorbed_df = sum(fe_dims) - len(fe_cols)
        n_obs = len(df)
        df_resid = n_obs - (len(x_cols) + 1) - absorbed_df


    # Handle OLS case (no FE)
    elif strategy == 'ols':
        print('Using simple OLS strategy (no fixed effects)...')
        iterations = 0
        absorbed_df = 0
        fe_dims = None
        df = lf.collect()    
        n_obs = len(df)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


    beta, se, resid, df_resid, n_clusters, r_squared = _run_regression(
    df=df,
    y_col=y_col,
    x_cols=x_cols,
    instruments=instruments,
    weights=weights,
    vcov=vcov,
    cluster_cols=cluster_cols,
    ssc=ssc,
    n_obs=n_obs,
    absorbed_df=absorbed_df)

    return LeanFEResult(
        coefs=dict(zip(x_cols, beta)),
        std_errors=dict(zip(x_cols, se)),
        n_obs=n_obs,
        iterations=iterations,
        vcov_type=vcov,
        is_iv=len(instruments) > 0,
        n_instruments=len(instruments) if instruments else None,
        n_clusters=n_clusters,
        df_resid=df_resid,
        formula=formula,
        fe_cols=fe_cols,
        fe_dims=fe_dims,
        r_squared=r_squared,
        compression_ratio = est_comp_ratio
    )