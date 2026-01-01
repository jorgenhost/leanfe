"""
DuckDB-based fixed effects regression implementation.

Optimized for memory efficiency using in-database operations.
Uses YOCO compression automatically for IID/HC1 standard errors.
"""
import duckdb
import numpy as np
import polars as pl

from leanfe.result import LeanFEResult
from leanfe.common import (
    parse_formula,
    iv_2sls,
    compute_standard_errors,
)
from leanfe.compress import (
    determine_strategy,
    leanfe_compress_duckdb, 
    estimate_compression_ratio
)
MAX_FE_LEVELS = 10_000

def _expand_factors_duckdb(
    con: duckdb.DuckDBPyConnection, 
    factor_vars: list[tuple[str, str | None]],
    table_name: str = 'data',     
    ) -> list[str]:
    # 1. Get unique categories for all factors in one go
    # Using a single query to fetch metadata is faster than one query per column.

    case_parts = []
    dummy_cols = []
    
    for var, ref in factor_vars:
        # Fetch categories
        categories = [r[0] for r in con.execute(f"SELECT DISTINCT {var} FROM {table_name} ORDER BY 1").fetchall()]
        ref_cat = ref if ref is not None else categories[0]
        
        for cat in categories:
            if cat == ref_cat: 
                continue
            if isinstance(cat, str):
                cat_sql = f"'{cat}'"
            else:
                cat_sql = str(cat)
            col_name = f"{var}_{cat}"            # Use CASE statements for bulk expansion
            case_parts.append(f"CASE WHEN {var} = '{cat_sql}' THEN 1 ELSE 0 END AS {col_name}")
            dummy_cols.append(col_name)
            
    # 2. Re-create the table once
    if case_parts:
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT *, {', '.join(case_parts)} FROM {table_name}")
    
    return dummy_cols

def _expand_interactions_duckdb(
    con: duckdb.DuckDBPyConnection, 
    interactions: list[tuple[str, str, str | None]],
    table_name: str = 'data', 
    ) -> list[str]:
    """
    Optimized DuckDB interaction expansion using a single SQL pass.
    """
    if not interactions:
        return []

    interaction_exprs = []
    all_new_cols = []

    for var, factor, ref in interactions:
        # 1. Fetch distinct categories once per factor
        cats = [r[0] for r in con.execute(
            f"SELECT DISTINCT {factor} FROM {table_name} ORDER BY {factor}"
        ).fetchall()]
        
        # 2. Determine reference category (logic same as original)
        if ref is None:
            ref_cat = cats[0]
        else:
            ref_cat = ref
            if cats and not isinstance(cats[0], str):
                try: ref_cat = type(cats[0])(ref)
                except (ValueError, TypeError):
                    ref_cat = ref        
        # 3. Build CASE statements for each non-reference category
        for cat in cats:
            if cat == ref_cat:
                continue
            if isinstance(cat, str):
                cat_sql = f"'{cat}'"
            else:
                cat_sql = str(cat)
            col_name = f"{var}_{cat}"
            # Use SQL CASE for the interaction calculation
            expr = f"CASE WHEN {factor} = '{cat_sql}' THEN {var} ELSE 0 END AS {col_name}"
            interaction_exprs.append(expr)
            all_new_cols.append(col_name)

    # 4. Apply all interactions in one single relational operation
    if interaction_exprs:
        sql = f"CREATE OR REPLACE TABLE {table_name} AS SELECT *, {', '.join(interaction_exprs)} FROM {table_name}"
        con.execute(sql)

    return all_new_cols

def leanfe_duckdb(
    data: str | pl.DataFrame | pl.LazyFrame | None,
    y_col: str | None = None,
    x_cols: list[str] | None = None,
    fe_cols: list[str] | None = None,
    formula: str | None = None,
    strategy: str = 'auto',
    weights: str | None = None,
    demean_tol: float = 1e-8,
    max_iter: int = 500,
    vcov: str = "iid",
    cluster_cols: list[str] | None = None,
    ssc: bool = False,
    sample_frac: float | None = None,
    con: duckdb.DuckDBPyConnection | None = None
) -> LeanFEResult:
    """
    Fixed effects regression using DuckDB with optimized memory usage.
    
    Parameters
    ----------
    data : str or polars.DataFrame
        Input data: either DataFrame or path to Parquet file
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
        coefficients, std_errors, n_obs, iterations, vcov_type, 
        is_iv, n_instruments, n_clusters
    """

    assert con is not None, "User must provide a duckdb.connect() object."

    # Parse formula if provided
    if formula is not None:
        y_col, x_cols, fe_cols, factor_vars, interactions, instruments = parse_formula(formula)
    elif y_col is None or x_cols is None or fe_cols is None:
        raise ValueError("Must provide either 'formula' or (y_col, x_cols, fe_cols)")
    else:
        factor_vars = []
        interactions = []
        instruments = []
    
    # Make lists mutable
    x_cols = list(x_cols)
    fe_cols = list(fe_cols)
    
    # Build needed columns
    needed_cols = [y_col] + x_cols + fe_cols + instruments
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
    
    # Register data source
    if isinstance(data, str):
        col_list = ', '.join(needed_cols)
        con.execute(f"CREATE OR REPLACE VIEW raw_data AS SELECT {col_list} FROM read_parquet('{data}')")
    elif isinstance(data, pl.DataFrame): 
        df = data.select(needed_cols)
        con.register("data", df)
    elif isinstance(data, pl.LazyFrame):
        df = data.select(needed_cols).collect()
        con.register('data', df)
    else:
        raise ValueError(
            'Please specify either data or con argument'
        )
    # Sample if requested
    if sample_frac is not None:
        frac_sql =  f"CREATE TABLE data AS SELECT * FROM raw_data USING SAMPLE {sample_frac * 100}%"        
        con.execute(frac_sql)
    else:
        con.execute("CREATE TABLE data AS SELECT * FROM raw_data")
    # Handle interactions
    if interactions:
        new_cols = _expand_interactions_duckdb(
            con = con,
            table_name = 'data',
            interactions = interactions
        )
        x_cols = x_cols + new_cols

    # Handle factor variables
    if factor_vars:
        new_cols = _expand_factors_duckdb(
            con = con,
            table_name = 'data',
            factor_vars = factor_vars
        )
        x_cols = x_cols + new_cols
    # Check if we should use compression strategy (faster for IID/HC1 without IV)
    # But only if FEs are low-cardinality (otherwise FWL demeaning is faster)
    is_iv = len(instruments) > 0
    
    # Compute FE cardinality to decide strategy
    fe_cardinality = {}
    for fe in fe_cols:
        card_query = f"SELECT COUNT(DISTINCT {fe}) FROM data"

        card = con.execute(card_query).fetchone()
        if card is None:
            raise ValueError(f'Error in computing FE ({fe}) cardinality, check dtype of {fe} or try different backend')
        card = int(card[0])
        fe_cardinality[fe] = card
    
    n_obs_initial_query = "SELECT COUNT(*) FROM data"
    n_obs_initial = con.execute(n_obs_initial_query).fetchone()

    if n_obs_initial is None:
        raise ValueError('Could not fetch number of obs/rows. Check for corrupted data')
    n_obs_initial = int(n_obs_initial[0])

    if strategy == 'auto':
        est_comp_ratio = estimate_compression_ratio(
            con = con,
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
        # Use YOCO compression strategy - much faster and lower memory
        # For cluster SEs, use first cluster column (Section 5.3.1 of YOCO paper)
        cluster_col = cluster_cols[0] if vcov == "cluster" and cluster_cols else None
        result = leanfe_compress_duckdb(
            con=con,
            y_col=y_col,
            x_cols=x_cols,
            fe_cols=fe_cols,
            weights=weights,
            vcov=vcov,
            cluster_col=cluster_col,
            ssc=ssc
        )
        # Update attributes directly
        result.formula = formula
        result.fe_cols = fe_cols
        
        return result

    if strategy == 'alt_proj':
        print('Using FWL/alternating projections strategy...')
        # Fall back to FWL demeaning for cluster SEs or IV
        # Drop singletons
        for fe in fe_cols:
            con.execute(f"DELETE FROM data WHERE {fe} IN (SELECT {fe} FROM data GROUP BY {fe} HAVING COUNT(*) = 1)")
        
        n_obs_query = "SELECT COUNT(*) FROM data"
        n_obs_res = con.execute(n_obs_query).fetchone()
        if n_obs_res is None:
            raise ValueError('Error in fetching no of obs/rows')
        n_obs = int(n_obs_res[0])
        cols_to_demean = [y_col] + x_cols + instruments
        
        # Order FEs by cardinality (low-card first) for faster convergence
        fe_cols_ordered = sorted(fe_cols, key=lambda fe: fe_cardinality.get(fe, 0))
        alter_statements = [f"ALTER TABLE data ADD COLUMN {col}_dm DOUBLE" for col in cols_to_demean]
        con.execute("; ".join(alter_statements))

        # Initialize demeaned columns in one UPDATE
        update_cols = ", ".join([f"{col}_dm = {col}" for col in cols_to_demean])
        con.execute(f"UPDATE data SET {update_cols}")

        dm_cols = [f"{col}_dm" for col in cols_to_demean]
        
        # Iterative demeaning
        for it in range(1, max_iter + 1):
            for fe in fe_cols_ordered:
                if weights is not None:
                    mean_cols = ", ".join([f"SUM({col} * {weights}) / SUM({weights}) as mean_{col}" 
                                        for col in dm_cols])
                    con.execute(f"""
                        CREATE OR REPLACE TEMP TABLE fe_means AS
                        SELECT {fe}, {mean_cols}
                        FROM data
                        GROUP BY {fe}
                    """)
                    
                    update_expr = ", ".join([f"{col} = {col} - fe_means.mean_{col}" for col in dm_cols])
                    con.execute(f"""
                        UPDATE data 
                        SET {update_expr}
                        FROM fe_means 
                        WHERE data.{fe} = fe_means.{fe}
                    """)
                else:
                    mean_cols = ", ".join([f"AVG({col}) as mean_{col}" for col in dm_cols])
                    con.execute(f"""
                        CREATE OR REPLACE TEMP TABLE fe_means AS
                        SELECT {fe}, {mean_cols}
                        FROM data
                        GROUP BY {fe}
                    """)
                    
                    # Update all columns in one statement
                    update_expr = ", ".join([f"{col} = {col} - fe_means.mean_{col}" for col in dm_cols])
                    con.execute(f"""
                        UPDATE data 
                        SET {update_expr}
                        FROM fe_means 
                        WHERE data.{fe} = fe_means.{fe}
                    """)
            
            if it >= 3:
                check_cols = " UNION ALL ".join([
                    f"SELECT {fe} as fe_col, '{col}' as dm_col, AVG({col}) as avg_val FROM data GROUP BY {fe}"
                    for fe in fe_cols for col in dm_cols
                ])
                result = con.execute(f"""
                    SELECT MAX(ABS(avg_val)) FROM ({check_cols})
                """).fetchone()
                if result is None:
                        raise ValueError('Error in iterative demeaning')
                result = result[0]
                if abs(result or 0) < demean_tol:
                    break
    
    k = len(x_cols)
    is_iv = len(instruments) > 0
    
    if is_iv:
        # Extract data for IV
        select_cols = [f"{col}_dm" for col in [y_col] + x_cols + instruments]
        if weights is not None:
            select_cols.append(weights)
        if cluster_cols is not None:
            select_cols.extend(cluster_cols)
        
        result_df = con.execute(f"SELECT {', '.join(select_cols)} FROM data").pl()
        Y = result_df[f"{y_col}_dm"].to_numpy()
        X = result_df[[f"{col}_dm" for col in x_cols]].to_numpy()
        Z = result_df[[f"{col}_dm" for col in instruments]].to_numpy()
        w = result_df[weights].to_numpy() if weights is not None else None
        
        beta, X_hat = iv_2sls(Y, X, Z, w)
        resid = Y - X_hat @ beta
    else:
        # OLS via SQL aggregates
        XtX = np.zeros((k, k))
        Xty = np.zeros(k)
        
        for i, col_i in enumerate(x_cols):
            col_i_dm = f"{col_i}_dm"
            query_t_y = f"SELECT SUM({col_i_dm} * {y_col}_dm) FROM data"
            res_t_y = con.execute(query_t_y).fetchone()
            if res_t_y is None:
                raise ValueError(f'Could not compute SUM({col_i_dm} * {y_col}_dm)')
            Xty[i] = res_t_y[0]
            for j in range(i, k):
                col_j_dm = f"{x_cols[j]}_dm"
                col_j_dm_query = f"SELECT SUM({col_i_dm} * {col_j_dm}) FROM data"
                val_j_dm = con.execute(col_j_dm_query).fetchone()
                if val_j_dm is None:
                    raise ValueError(f'Could not compute {col_j_dm_query}')
                XtX[i, j] = val_j_dm[0]
                XtX[j, i] = val_j_dm[0]
        
        beta = np.linalg.solve(XtX, Xty)
        
        resid_expr = f"{y_col}_dm - (" + " + ".join([f"{b} * {col}_dm" for b, col in zip(beta, x_cols)]) + ")"
        con.execute("ALTER TABLE data ADD COLUMN _resid DOUBLE")
        con.execute(f"UPDATE data SET _resid = {resid_expr}")
    
    # Degrees of freedom
    distinct_fes_query = ", ".join([f"COUNT(DISTINCT {fe})" for fe in fe_cols])
    distinct_fes = con.execute(f"SELECT {distinct_fes_query} FROM data").fetchone()
    if distinct_fes is not None:
        n_fe_groups = sum(distinct_fes)
    else:   
        n_fe_groups = 0
    absorbed_df = n_fe_groups - len(fe_cols)
    df_resid = n_obs - k - absorbed_df
    
    # Compute XtX_inv
    if is_iv:
        if w is not None:
            sqrt_w = np.sqrt(w)
            X_hat_w = X_hat * sqrt_w[:, np.newaxis]
            XtX_inv = np.linalg.inv(X_hat_w.T @ X_hat_w)
        else:
            XtX_inv = np.linalg.inv(X_hat.T @ X_hat)
    else:
        XtX_inv = np.linalg.inv(XtX)
    
    # Standard errors
    if is_iv:
        cluster_ids = None
        if vcov == "cluster":
            if cluster_cols is None:
                raise ValueError("cluster_cols required for vcov='cluster'")
            if len(cluster_cols) == 1:
                cluster_ids = result_df.select(pl.col(cluster_cols)).to_numpy()
            else:
                cluster_ids = (
                    result_df.select(
                        pl.concat_str([pl.col(c).cast(pl.String) for c in cluster_cols], separator="_").cast(pl.Categorical).to_physical()
                    )
                    .to_numpy()
                )        
        se, n_clusters = compute_standard_errors(
            XtX_inv=XtX_inv, resid=resid, n_obs=n_obs, df_resid=df_resid,
            vcov=vcov, X=X_hat, weights=w, cluster_ids=cluster_ids, ssc=ssc
        )
    else:
        n_clusters = None
        if vcov == "iid":
            if weights is not None:
                sigma2 = con.execute(f"SELECT SUM({weights} * _resid * _resid) / {df_resid} FROM data").fetchone()
                if sigma2 is None:
                    raise ValueError('Could not compute sigma²')
                sigma2 = sigma2[0]
            else:
                sigma2 = con.execute(f"SELECT SUM(_resid * _resid) / {df_resid} FROM data").fetchone()
                if sigma2 is None:
                    raise ValueError('Could not compute sigma²')
                sigma2 = sigma2[0]
            se = np.sqrt(sigma2 * np.diag(XtX_inv))
        elif vcov == "HC1":
            meat = np.zeros((k, k))
            for i, col_i in enumerate(x_cols):
                for j in range(i, k):
                    col_j = x_cols[j]
                    if weights is not None:
                        val = con.execute(f"SELECT SUM({weights} * {col_i}_dm * {col_j}_dm * _resid * _resid) FROM data").fetchone()
                        if val is None:
                            raise ValueError(f'Could not compute (weighted) meat matrix, vcov = {vcov}')
                        val = val[0]
                    else:
                        val = con.execute(f"SELECT SUM({col_i}_dm * {col_j}_dm * _resid * _resid) FROM data").fetchone()
                        if val is None:
                            raise ValueError(f'Could not compute meat matrix, vcov = {vcov}')
                        val = val[0]
                    meat[i, j] = val
                    meat[j, i] = val
            vcov_matrix = XtX_inv @ meat @ XtX_inv
            se = np.sqrt((n_obs / df_resid) * np.diag(vcov_matrix))
        elif vcov == "cluster":
            if cluster_cols is None:
                raise ValueError("cluster_cols required for vcov='cluster'")
            cluster_expr = cluster_cols[0] if len(cluster_cols) == 1 else f"CONCAT_WS('_', {', '.join(cluster_cols)})"
            if weights is not None:
                score_exprs = [f"SUM({col}_dm * _resid * {weights}) AS score_{i}" for i, col in enumerate(x_cols)]
            else:
                score_exprs = [f"SUM({col}_dm * _resid) AS score_{i}" for i, col in enumerate(x_cols)]
            cluster_scores = con.execute(f"SELECT {cluster_expr} AS cluster_id, {', '.join(score_exprs)} FROM data GROUP BY {cluster_expr}").pl()
            n_clusters = len(cluster_scores)
            S = cluster_scores[[f"score_{i}" for i in range(k)]].to_numpy()
            meat = S.T @ S
            adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / df_resid) if ssc else n_clusters / (n_clusters - 1)
            vcov_matrix = adj * XtX_inv @ meat @ XtX_inv
            se = np.sqrt(np.diag(vcov_matrix))
        else:
            raise ValueError(f"Unknown vcov: {vcov}")

    # Let's not kill the users connection, but only drop the views/tables we used
    def drop_all_objects(con):
        # 1. Drop all Views first (to avoid dependency issues)
        views = con.execute("SELECT view_name FROM duckdb_views WHERE NOT internal").fetchall()
        for (view_name,) in views:
            con.execute(f'DROP VIEW IF EXISTS "{view_name}"')
            
        # 2. Drop all Tables
        tables = con.execute("SELECT table_name FROM duckdb_tables WHERE NOT internal").fetchall()
        for (table_name,) in tables:
            con.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')    

    drop_all_objects(con)
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
        fe_cols=fe_cols
    )
