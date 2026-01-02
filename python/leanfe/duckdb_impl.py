"""
DuckDB-based fixed effects regression implementation.

Optimized for memory efficiency using in-database operations.
Uses YOCO compression automatically for IID/HC1 standard errors.
"""
import duckdb
import numpy as np
import polars as pl
import uuid

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
            
    if case_parts:
        # Materialize as TEMPORARY to ensure it stays in RAM/temp-storage
        con.execute(f"CREATE OR REPLACE TEMPORARY TABLE {table_name} AS SELECT *, {', '.join(case_parts)} FROM {table_name}")
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

    # create unique prefix for all temp objects for this call
    uid = uuid.uuid4().hex[:8]
    def mk_tmp(name: str) -> str:
        # safe SQL identifier (no hyphens). We'll quote when used.
        return f"leanfe_{name}_{uid}"

    tmp_table = mk_tmp("data")  # will become the working table name for this call
    created_tmp_tables = [tmp_table]  # track for cleanup

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
    
    col_list = ', '.join([f'"{c}"' for c in needed_cols])
    where_parts = [f'"{col}" IS NOT NULL' for col in needed_cols]
    if weights is not None:
        where_parts.append(f'"{weights}" > 0') # Weights must be positive
    where_clause = " WHERE " + " AND ".join(where_parts) if where_parts else ""
    # If sampling requested, do it in the same CREATE to avoid extra copies
    if isinstance(data, str):
        if sample_frac is not None:
            pct = sample_frac * 100
            sql = (
                f"CREATE TEMPORARY TABLE \"{tmp_table}\" AS "
                f"SELECT {col_list} FROM read_parquet('{data}') "
                f"{where_clause} USING SAMPLE {pct}%"
            )
        else:
            sql = (
                f"CREATE TEMPORARY TABLE \"{tmp_table}\" AS "
                f"SELECT {col_list} FROM read_parquet('{data}') "
                f"{where_clause}"
            )
        con.execute(sql)

    elif isinstance(data, pl.DataFrame):
        # Register the polars DF under a short ephemeral name, then materialize it
        src_name = mk_tmp("src")
        # register the DataFrame as a relation name and then create a TEMP TABLE from it
        con.register(src_name, data.select(needed_cols))
        con.execute(f'CREATE TEMPORARY TABLE "{tmp_table}" AS SELECT * FROM "{src_name}"')
        created_tmp_tables.append(src_name)  # we may try to drop this registered name later (best-effort)

    elif isinstance(data, pl.LazyFrame):
        df = data.select(needed_cols).collect()
        src_name = mk_tmp("src")
        con.register(src_name, df)
        con.execute(f'CREATE TEMPORARY TABLE "{tmp_table}" AS SELECT * FROM "{src_name}"')
        created_tmp_tables.append(src_name)
    else:
        raise ValueError('Please specify either data or con argument')

    # Handle interactions
    if interactions:
        new_cols = _expand_interactions_duckdb(
            con = con,
            table_name = tmp_table,
            interactions = interactions
        )
        x_cols = x_cols + new_cols

    # Handle factor variables
    if factor_vars:
        new_cols = _expand_factors_duckdb(
            con = con,
            table_name = tmp_table,
            factor_vars = factor_vars
        )
        x_cols = x_cols + new_cols
    # Check if we should use compression strategy (faster for IID/HC1 without IV)
    # But only if FEs are low-cardinality (otherwise FWL demeaning is faster)
    is_iv = len(instruments) > 0
    
    # Compute FE cardinality to decide strategy
    fe_cardinality = {}
    for fe in fe_cols:
        card_query = f"SELECT COUNT(DISTINCT {fe}) FROM {tmp_table}"

        card = con.execute(card_query).fetchone()
        if card is None:
            raise ValueError(f'Error in computing FE ({fe}) cardinality, check dtype of {fe} or try different backend')
        card = int(card[0])
        fe_cardinality[fe] = card
    
    n_obs_initial_query = f"SELECT COUNT(*) FROM {tmp_table}"
    n_obs_initial = con.execute(n_obs_initial_query).fetchone()

    if n_obs_initial is None:
        raise ValueError('Could not fetch number of obs/rows. Check for corrupted data')
    n_obs_initial = int(n_obs_initial[0])

    est_comp_ratio = estimate_compression_ratio(
            con = con,
            table_ref = f'{tmp_table}',
            x_cols = x_cols,
            fe_cols = fe_cols
    )

    if strategy == 'auto':

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
            table_ref = tmp_table,
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
        fe_filter = " OR ".join([f'"{fe}" IN (SELECT "{fe}" FROM "{tmp_table}" GROUP BY 1 HAVING COUNT(*) = 1)' for fe in fe_cols])
        con.execute(f'DELETE FROM "{tmp_table}" WHERE {fe_filter}')        
        n_obs_query = "SELECT COUNT(*) FROM {tmp_table}"
        n_obs_res = con.execute(n_obs_query).fetchone()

        if n_obs_res is None:
            raise ValueError('Error in fetching no of obs/rows')
        n_obs = int(n_obs_res[0])
        cols_to_demean = [y_col] + x_cols + instruments
        cols_to_keep = cols_to_demean + fe_cols
        
        dm_cols = [f"{col}_dm" for col in cols_to_demean]

        # We create the _dm columns as simple aliases of the originals
        select_dm = ", ".join([f"{col} AS {col}_dm" for col in cols_to_demean])
        all_cols = ", ".join(cols_to_keep)

        con.execute(f"CREATE OR REPLACE TEMP TABLE data AS SELECT {all_cols}, {select_dm} FROM {tmp_table}")        
        # Order FEs by cardinality (low-card first) for faster convergence
        fe_cols_ordered = sorted(fe_cols, key=lambda fe: fe_cardinality.get(fe, 0))

        avg_exprs = ", ".join([f"AVG({c}) AS {c}_avg" for c in dm_cols])
        avg_names = ", ".join([f"{c}_avg" for c in dm_cols])
        fe_list = ", ".join(fe_cols)
        convergence_sql = f"""
            SELECT MAX(ABS(val)) 
            FROM (
                SELECT {avg_exprs} FROM {tmp_table} GROUP BY GROUPING SETS ({fe_list})
            ) UNPIVOT(val FOR col IN ({avg_names}))
        """

        dm_cols = [f"{col}_dm" for col in cols_to_demean]
        
        # Iterative demeaning
        for it in range(1, max_iter + 1):
            for fe in fe_cols_ordered:
                if weights is not None:
                    agg_expr = ", ".join([f"SUM({c} * {weights}) / SUM({weights}) as {c}_mean" for c in dm_cols])
                else:
                    agg_expr = ", ".join([f"AVG({c}) as {c}_mean" for c in dm_cols])

                con.execute(f"CREATE OR REPLACE TEMP TABLE fe_means AS SELECT {fe}, {agg_expr} FROM {tmp_table} GROUP BY {fe}")

                subtract_expr = ", ".join([f"d.{c} - COALESCE(m.{c}_mean, 0) as {c}" for c in dm_cols])
                other_cols_str = ", ".join([f"d.{c}" for c in cols_to_keep])

                con.execute(f"""
                    CREATE OR REPLACE TEMP TABLE data AS 
                    SELECT {other_cols_str}, {subtract_expr}
                    FROM {tmp_table} d
                    LEFT JOIN fe_means m ON d.{fe} = m.{fe}
                """)
            if it >= 3:
                max_err = con.execute(f"""
                    {convergence_sql}
                """).fetchone()               
                if max_err is None:
                    raise ValueError('Error in iterative demeaning')
                max_err = max_err[0]
                if abs(max_err or 0) < demean_tol:
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
        
        result_df = con.execute(f"SELECT {', '.join(select_cols)} FROM {tmp_table}").pl()
        Y = result_df[f"{y_col}_dm"].to_numpy()
        X = result_df[[f"{col}_dm" for col in x_cols]].to_numpy()
        Z = result_df[[f"{col}_dm" for col in instruments]].to_numpy()
        w = result_df[weights].to_numpy() if weights is not None else None
        
        beta, X_hat = iv_2sls(Y, X, Z, w)
        resid = Y - X_hat @ beta
    else:
        # build expressions
        agg_exprs = []
        # X'y
        for i, col in enumerate(x_cols):
            agg_exprs.append(f"SUM({col}_dm * {y_col}_dm) AS xty_{i}")
        # XtX upper triangle
        for i, ci in enumerate(x_cols):
            for j in range(i, k):
                cj = x_cols[j]
                agg_exprs.append(f"SUM({ci}_dm * {cj}_dm) AS xtx_{i}_{j}")

        sql = f"SELECT {', '.join(agg_exprs)} FROM {tmp_table}"
        row = con.execute(sql).fetchone()
        if row is None:
            raise ValueError("Failed to compute XtX/X'y in a single query")
        vals = list(row)  # tuple -> list

        # Unpack
        idx = 0
        Xty = np.zeros(k)
        for i in range(k):
            Xty[i] = vals[idx]
            idx += 1

        XtX = np.zeros((k, k))
        for i in range(k):
            for j in range(i, k):
                XtX[i, j] = XtX[j, i] = vals[idx]
                idx += 1

        beta = np.linalg.solve(XtX, Xty)
        
        resid_expr = f'"{y_col}_dm" - (' + " + ".join([f"{b} * \"{col}_dm\"" for b, col in zip(beta, x_cols)]) + ")"
        con.execute(f'CREATE OR REPLACE TEMPORARY TABLE "{tmp_table}" AS SELECT *, {resid_expr} AS _resid FROM "{tmp_table}"')    
    # Degrees of freedom
    distinct_fes_query = ", ".join([f"COUNT(DISTINCT {fe})" for fe in fe_cols])
    n_fe_groups = con.execute(f"SELECT {distinct_fes_query} FROM {tmp_table}").fetchone()
    if n_fe_groups is not None:
        n_fe_groups = sum(n_fe_groups)
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
                sigma2 = con.execute(f"SELECT SUM({weights} * _resid * _resid) / {df_resid} FROM {tmp_table}").fetchone()
                if sigma2 is None:
                    raise ValueError('Could not compute sigma²')
                sigma2 = sigma2[0]
            else:
                sigma2 = con.execute(f"SELECT SUM(_resid * _resid) / {df_resid} FROM {tmp_table}").fetchone()
                if sigma2 is None:
                    raise ValueError('Could not compute sigma²')
                sigma2 = sigma2[0]
            se = np.sqrt(sigma2 * np.diag(XtX_inv))
        elif vcov == "HC1":
            k = len(x_cols)
            meat_elements = []
            for i in range(k):
                for j in range(i, k):
                    # Calculate the sum of squares/products weighted by squared residuals
                    col_i, col_j = x_cols[i], x_cols[j]
                    expr = f"SUM({col_i}_dm * {col_j}_dm * _resid * _resid)"
                    if weights is not None:
                        expr = f"SUM({weights} * {col_i}_dm * {col_j}_dm * _resid * _resid)"
                    meat_elements.append(f"{expr} AS m_{i}_{j}")
            meat_row = con.execute(f"SELECT {', '.join(meat_elements)} FROM {tmp_table}").fetchone()
            if meat_row is None:
                raise ValueError('Could not compute meat row')
            vals = list(meat_row)
            meat = np.zeros((k, k))
            idx = 0
            for i in range(k):
                for j in range(i, k):
                    val = float(vals[idx] or 0.0)
                    meat[i, j] = meat[j, i] = val
                    idx += 1
            vcov_matrix = XtX_inv @ meat @ XtX_inv
            se = np.sqrt((n_obs / df_resid) * np.diag(vcov_matrix))
        elif vcov == "cluster":
            if cluster_cols is None:
                raise ValueError("cluster_cols required for vcov='cluster'")
            
            if len(cluster_cols) == 1:
                # Single-way clustering
                cluster_id_sql = cluster_cols[0]
                
                if weights is not None:
                    score_exprs = [f"SUM({col}_dm * _resid * {weights}) AS score_{i}" 
                                for i, col in enumerate(x_cols)]
                else:
                    score_exprs = [f"SUM({col}_dm * _resid) AS score_{i}" 
                                for i, col in enumerate(x_cols)]
                
                cluster_query = f"""
                    SELECT {cluster_id_sql} AS cluster_id, {', '.join(score_exprs)} 
                    FROM {tmp_table} 
                    GROUP BY 1
                """
                score_df = con.execute(cluster_query).pl()
                n_clusters = len(score_df)
                
                S = score_df.select([f"score_{i}" for i in range(k)]).to_numpy()
                meat = S.T @ S
                
                adj = ((n_clusters / (n_clusters - 1)) * ((n_obs - 1) / df_resid) 
                    if ssc else n_clusters / (n_clusters - 1))
                vcov_matrix = adj * XtX_inv @ meat @ XtX_inv
                se = np.sqrt(np.diag(vcov_matrix))
                
            else:
                # Multi-way clustering using Cameron-Gelbach-Miller (2011)
                from itertools import combinations
                
                n_ways = len(cluster_cols)
                vcov_matrix = np.zeros_like(XtX_inv)
                n_clusters_list = []
                
                # Iterate over all non-empty subsets with alternating signs
                for subset_size in range(1, n_ways + 1):
                    sign = (-1) ** (subset_size - 1)
                    
                    for cluster_subset in combinations(range(n_ways), subset_size):
                        # Build intersection cluster ID
                        if subset_size == 1:
                            cluster_id_sql = cluster_cols[cluster_subset[0]]
                        else:
                            # Concatenate for intersection
                            cols_to_concat = [cluster_cols[i] for i in cluster_subset]
                            cluster_id_sql = f"CONCAT_WS('_', {', '.join(cols_to_concat)})"
                        
                        # Compute scores for this clustering dimension
                        if weights is not None:
                            score_exprs = [f"SUM({col}_dm * _resid * {weights}) AS score_{i}" 
                                        for i, col in enumerate(x_cols)]
                        else:
                            score_exprs = [f"SUM({col}_dm * _resid) AS score_{i}" 
                                        for i, col in enumerate(x_cols)]
                        
                        cluster_query = f"""
                            SELECT {cluster_id_sql} AS cluster_id, {', '.join(score_exprs)} 
                            FROM {tmp_table} 
                            GROUP BY 1
                        """
                        score_df = con.execute(cluster_query).pl()
                        n_clust = len(score_df)
                        
                        # Store cluster counts for first-level clusters
                        if subset_size == 1:
                            n_clusters_list.append(n_clust)
                        
                        S = score_df.select([f"score_{i}" for i in range(k)]).to_numpy()
                        meat = S.T @ S
                        
                        # Adjustment factor
                        adj = ((n_clust / (n_clust - 1)) * ((n_obs - 1) / df_resid) 
                            if ssc else n_clust / (n_clust - 1))
                        
                        # Add/subtract with alternating signs
                        vcov_matrix += sign * adj * XtX_inv @ meat @ XtX_inv
                
                se = np.sqrt(np.diag(vcov_matrix))
                n_clusters = tuple(n_clusters_list)  # Return as tuple for multi-way

        else:
            raise ValueError(f"Unknown vcov: {vcov}")
            
    def safe_cleanup(con, uid_prefix):
        """Only drops objects created by this specific LeanFE execution."""
        # Clean up tables/views starting with our unique ID
        for obj_type in ["TABLE", "VIEW"]:
            names = con.execute(f"SELECT {obj_type.lower()}_name FROM duckdb_{obj_type.lower()}s WHERE {obj_type.lower()}_name LIKE 'leanfe_%_{uid_prefix}'").fetchall()
            for (name,) in names:
                con.execute(f"DROP {obj_type} IF EXISTS \"{name}\" CASCADE")
        # Specifically drop the fixed names used in loops
        con.execute("DROP TABLE IF EXISTS fe_means")

    # Call this at the end
    safe_cleanup(con, uid)

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
