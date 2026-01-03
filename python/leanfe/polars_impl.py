"""
Polars-based fixed effects regression implementation.

Optimized for speed using Polars LazyFrame/DataFrame operations.
Uses YOCO compression automatically for IID/HC1 standard errors.
"""
from scipy.special import it2struve0, it2j0y0
import polars as pl
import numpy as np
import polars.selectors as cs
from leanfe.result import LeanFEResult
from leanfe.common import (
    parse_formula,
    iv_2sls,
    compute_standard_errors,
)
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
    
    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    factor_vars : list of tuples
        List of (var_name, ref_category) tuples. ref_category is None for first category,
        or a specific value to use as reference.
    
    Returns
    -------
    tuple
        (df, dummy_cols) - DataFrame with dummies added and list of dummy column names
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
    
    Parameters
    ----------
    lf : pl.LazyFrame
        Input LazyFrame
    interactions : list of tuples
        List of (var, factor, ref) tuples. ref is None for first category,
        or a specific value to use as reference.
    
    Returns
    -------
    tuple
        (lf, interaction_cols) - LazyFrame with interactions added and list of column names
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
        categories = unique_cat_map.select(pl.col(var)).to_list()
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

    fe_cols_cast = lf.select(pl.col(fe_cols)).select(cs.by_dtype(pl.String, pl.Categorical)).collect_schema().names()
    EXPR = [
        pl.col(col).alias(col).cast(pl.Categorical).to_physical()
        for col in fe_cols_cast
    ]

    lf = lf.with_columns(
        EXPR
    )

    return lf


def leanfe_polars(
    data: str | pl.DataFrame | pl.LazyFrame,
    y_col: str | None = None,
    x_cols: list[str] | None = None,
    fe_cols: list[str] | None = None,
    formula: str | None = None,
    strategy: str = 'auto',
    weights: str | None = None,
    demean_tol: float = 1e-10,
    max_iter: int = 500,
    vcov: str = "iid",
    cluster_cols: list[str] | None = None,
    ssc: bool = True,
    sample_frac: float | None = None
) -> LeanFEResult:
    """
    Fast fixed effects regression using Polars and FWL theorem.
    
    Parameters
    ----------
    data : str, pl.LazyFrame or pl.DataFrame
        Input data: either LazyFrame/DataFrame or path to Parquet file
    y_col : str, optional
        Dependent variable column name (optional if formula provided)
    x_cols : list of str, optional
        Independent variable column names (optional if formula provided)
    fe_cols : list of str, optional
        Fixed effect column names (optional if formula provided)
    formula : str, optional
        R-style formula: "y ~ x1 + x2 + x:i(factor) | fe1 + fe2 | z1 + z2" (IV)
    weights : str, optional
        Column name for regression weights
    demean_tol : float, default 1e-5
        Convergence tolerance for demeaning
    max_iter : int, default 500
        Maximum iterations for demeaning
    vcov : str, default "iid"
        Variance-covariance estimator: "iid", "HC1", or "cluster"
    cluster_cols : list of str, optional
        Clustering variables (required if vcov="cluster")
    ssc : bool, default False
        Small sample correction for clustered SEs
    sample_frac : float, optional
        Fraction of data to sample (e.g., 0.1 for 10%)
    
    Returns
    -------
    dict
        coefficients, std_errors, n_obs, iterations, vcov_type, is_iv, n_instruments, n_clusters
    """
    # Parse formula if provided
    if formula is not None:
        y_col, x_cols, fe_cols, factor_vars, interactions, instruments = parse_formula(formula)
    elif y_col is None or x_cols is None or fe_cols is None:
        raise ValueError("Must provide either 'formula' or (y_col, x_cols, fe_cols)")
    else:
        factor_vars = []
        interactions = []
        instruments = []
    
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
    
    # Make x_cols mutable
    x_cols = list(x_cols)
    
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
    # But only if FEs are low-cardinality (otherwise FWL demeaning is faster)
    is_iv = len(instruments) > 0
    
    # Strategy selection based on cost estimation:
    # - YOCO compression: fast when good compression ratio AND low total FE levels
    # - FWL demeaning: fast when high-cardinality FEs (avoids huge sparse matrix)
    #
    # NOTE: A "hybrid" approach (demean high-card FEs, then YOCO on low-card) was tested
    # but is mathematically incorrect. After partial demeaning, Y still has non-zero means
    # within low-card FE groups, so adding dummies doesn't give the same result as full FWL.
    # Compute FE cardinality to decide strategy
    fe_cardinality = {fe: lf.select(pl.col(fe).n_unique()).collect().item() for fe in fe_cols}

    if strategy == 'auto':

        n_obs_initial = lf.select(pl.len()).collect().item()

        est_comp_ratio = estimate_compression_ratio(
        data = lf,
        x_cols = x_cols,
        fe_cols = fe_cols
        )
        
        inferred_strategy = determine_strategy(
            vcov, is_iv, fe_cardinality,
            max_fe_levels=MAX_FE_LEVELS,
            n_obs=n_obs_initial,
            n_x_cols=len(x_cols),
            estimated_compression_ratio=est_comp_ratio
        )
        print(f'Auto strategy: Inferring {inferred_strategy} strategy')
        strategy = inferred_strategy

        
    if strategy == 'compress':
        print('Using compresssion strategy...')
        # Use YOCO compression strategy - much faster for discrete regressors
        # For cluster SEs, use first cluster column (Section 5.3.1 of YOCO paper)

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

    if strategy == 'alt_proj':
        print('Using FWL/alternating projections strategy...')

        # Drop singletons (keep as LazyFrame)
        EXPRS = [
            pl.len().over(fe) > 1
            for fe in fe_cols
        ]
        lf = lf.filter(pl.all_horizontal(EXPRS))

        # Order FEs by cardinality (low-card first) for faster convergence
        fe_cols_ordered = sorted(fe_cols, key=lambda fe: fe_cardinality.get(fe, 0))

        cols_to_demean = [y_col] + x_cols + instruments

        # FWL demeaning - all operations stay lazy
        for it in range(1, max_iter + 1):
            for fe in fe_cols_ordered:
                if weights is not None:
                    demean_exprs = [
                        (pl.col(c) - (pl.col(c) * pl.col(weights)).sum().over(fe) /
                        pl.col(weights).sum().over(fe)).alias(c)
                        for c in cols_to_demean
                    ]
                else:
                    demean_exprs = [
                        (pl.col(c) - pl.col(c).mean().over(fe)).alias(c)
                        for c in cols_to_demean
                    ]
                
                # Directly overwrite columns in place
                lf = lf.with_columns_seq(
                    demean_exprs
                )
            
            # Check convergence - only collect for this check if necessary
            if it >= 3:
                check_exprs = [
                    pl.col(y_col).mean().over(fe).abs().alias(f"{y_col}_mean_abs_{fe}")
                    for fe in fe_cols
                ]                
                max_mean = lf.select(pl.max_horizontal(check_exprs)).max().collect().item()
                if max_mean < demean_tol:
                    break
    df = lf.collect()
    # Extract weights AFTER collection
    if weights is not None:
        w = df.select(weights).to_numpy().flatten()
    else:
        w = None

    # Get n_obs from the collected DataFrame
    n_obs = len(df)

    # Build X including intercept if present in df
    X = df.select(pl.ones(pl.len()), pl.col(x_cols)).to_numpy()    
    Y = df.select(y_col).to_numpy().flatten()
    
    # IV/2SLS or OLS
    is_iv = len(instruments) > 0
    if is_iv:
        Z = df.select(instruments).to_numpy()
        # If X includes intercept but Z doesn't, ensure Z has an intercept column
        if X.shape[1] > Z.shape[1]:
            # check whether Z already contains a constant column
            if not any(np.allclose(col, 1.0) for col in Z.T):
                Z = np.column_stack([np.ones(Z.shape[0]), Z])
        beta_full, X_hat_full = iv_2sls(Y, X, Z, w)
        beta = beta_full[1:] if X.shape[1] > len(x_cols) else beta_full
        X_hat = X_hat_full
    else:
        if w is not None:
            sqrt_w = np.sqrt(w)
            X_w = X * sqrt_w[:, np.newaxis]
            Y_w = Y * sqrt_w
            XtX = X_w.T @ X_w
            Xty = X_w.T @ Y_w
        else:
            XtX = X.T @ X
            Xty = X.T @ Y
        beta_full = np.linalg.solve(XtX, Xty)
        # If we added intercept, strip intercept from returned coefficients
        if X.shape[1] == len(x_cols) + 1:
            beta = beta_full[1:]
        else:
            beta = beta_full
        X_hat = X
    
    # Compute (X'X)^-1 for standard errors
    if w is not None:
        sqrt_w = np.sqrt(w)
        X_hat_w = X_hat * sqrt_w[:, np.newaxis]
        XtX_inv = np.linalg.inv(X_hat_w.T @ X_hat_w)
    else:
        XtX_inv = np.linalg.inv(X_hat.T @ X_hat)
    
    # Residuals
    resid = Y - X_hat @ (beta_full if 'beta_full' in locals() else np.concatenate([[0], beta]))
    
    # Calculate absorbed degrees of freedom
    fe_counts = {fe: df.group_by(fe).agg(pl.len()).height for fe in fe_cols}
    absorbed_df = sum(fe_counts.values()) - len(fe_cols)
    df_resid = n_obs - (len(x_cols) + 1) - absorbed_df
    
    cluster_ids = None
    if vcov == "cluster":
        cluster_ids = df.select(cluster_cols).to_numpy()
        if cluster_cols is None:
            raise ValueError("cluster_cols must be provided when vcov='cluster'")
        if cluster_ids.ndim == 1:
            cluster_ids = cluster_ids.reshape(-1, 1)
        else:
            # Multi-way clustering: pass as 2D array
            cluster_ids = df.select(cluster_cols).to_numpy()
    # Then call the new compute_multiway_standard_errors function    
    # Compute standard errors
    se_full, n_clusters = compute_standard_errors(
        XtX_inv=XtX_inv,
        resid=resid,
        n_obs=n_obs,
        df_resid=df_resid,
        vcov=vcov,
        X=X_hat,
        weights=w,
        cluster_ids=cluster_ids,
        ssc=ssc
    )
    
    # If an intercept exists, strip intercept from reported SEs
    if X.shape[1] == len(x_cols) + 1:
        se = se_full[1:]
    else:
        se = se_full
    
    # Compute R-squared (within)
    rss = np.sum(resid**2)
    tss = np.sum((Y - np.mean(Y))**2)
    r_squared_within = 1 - rss / tss if tss > 0 else None
    
    return LeanFEResult(
        coefficients=dict(zip(x_cols, beta)), 
        std_errors=dict(zip(x_cols, se)),     
        n_obs=n_obs,
        iterations=it,
        vcov_type=vcov,                       
        is_iv=is_iv,
        n_instruments=len(instruments) if is_iv else None,
        n_clusters=n_clusters,
        df_resid=df_resid,
        formula=formula,
        fe_cols=fe_cols,
        r_squared = r_squared_within
    )