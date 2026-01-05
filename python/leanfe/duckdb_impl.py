"""
DuckDB-based fixed effects regression implementation.

Optimized for memory efficiency using in-database operations.
Uses YOCO compression automatically for IID/HC1 standard errors.
"""
import duckdb
from duckdb import DuckDBPyConnection
import numpy as np
import polars as pl
import uuid
from typing import Literal
from leanfe.result import LeanFEResult
from leanfe.common import (
    parse_formula,
    iv_2sls,
)
from leanfe.std_errors import compute_standard_errors_duckdb
from leanfe.compress import (
    determine_strategy,
    leanfe_compress_duckdb, 
    estimate_compression_ratio
)
MAX_FE_LEVELS = 10_000

def _safe_cleanup(con, uid_prefix):
    """Only drops objects created by this specific LeanFE execution."""
    # Clean up tables/views starting with our unique ID
    for obj_type in ["TABLE", "VIEW"]:
        names = con.execute(f"SELECT {obj_type.lower()}_name FROM duckdb_{obj_type.lower()}s WHERE {obj_type.lower()}_name LIKE 'leanfe_%_{uid_prefix}'").fetchall()
        for (name,) in names:
            con.execute(f"DROP {obj_type} IF EXISTS \"{name}\" CASCADE")
    # Specifically drop the fixed names used in loops
    con.execute("DROP TABLE IF EXISTS fe_means")

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

def _run_regression_duckdb(
    con: DuckDBPyConnection,
    tmp_table: str,
    y_col: str,
    x_cols: list[str],
    instruments: list[str],
    weights: str | None,
    vcov: Literal["iid", "HC1", "cluster"],
    cluster_cols: list[str] | None,
    ssc: bool,
    n_obs: int,
    absorbed_df: int = 0
) -> tuple[np.ndarray, np.ndarray, int, int | tuple | None]:
    """
    Common regression logic for both OLS and FE cases (DuckDB version).
    
    Returns: (beta, se, df_resid, n_clusters)
    """
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
        
        # Compute XtX_inv for IV using Cholesky with fallback
        if w is not None:
            sqrt_w = np.sqrt(w)
            X_for_inv = X_hat * sqrt_w[:, np.newaxis]
        else:
            X_for_inv = X_hat
        
        # Use Cholesky decomposition with fallback
        try:
            L = np.linalg.cholesky(X_for_inv.T @ X_for_inv)
            XtX_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(L.shape[0])))
        except np.linalg.LinAlgError:
            # Fallback to direct inverse
            XtX_inv = np.linalg.inv(X_for_inv.T @ X_for_inv)
            
        df_resid = n_obs - k - absorbed_df
        
        # For IV, we need to use the numpy-based IV functions from std_errors
        # since we have X_hat as a numpy array
        from leanfe.std_errors import (
            _compute_se_hc1_iv,
            _compute_se_cluster_oneway_iv, 
            _compute_se_cluster_multiway_iv
        )
        
        if vcov == "iid":
            # IID for IV
            sigma2 = np.sum((w * resid**2 if w is not None else resid**2)) / df_resid
            se = np.sqrt(np.maximum(sigma2 * np.diag(XtX_inv), 0.0))
            n_clusters = None
            
        elif vcov == "HC1":
            se, n_clusters = _compute_se_hc1_iv(
                XtX_inv=XtX_inv,
                resid=resid,
                X=X_hat,
                weights_array=w,
                n_obs=n_obs,
                df_resid=df_resid
            )
            
        elif vcov == "cluster":
            if cluster_cols is None:
                raise ValueError("cluster_cols required for vcov='cluster'")
            
            if len(cluster_cols) == 1:
                cluster_ids = result_df.select(pl.col(cluster_cols)).to_numpy().flatten()
            else:
                cluster_ids = result_df.select(pl.col(cluster_cols)).to_numpy()
            
            if len(cluster_cols) == 1:
                se, n_clusters = _compute_se_cluster_oneway_iv(
                    XtX_inv=XtX_inv,
                    resid=resid,
                    X=X_hat,
                    weights=w,
                    cluster_ids=cluster_ids,
                    n_obs=n_obs,
                    df_resid=df_resid,
                    ssc=ssc
                )
            else:
                se, n_clusters = _compute_se_cluster_multiway_iv(
                    XtX_inv=XtX_inv,
                    resid=resid,
                    X=X_hat,
                    weights=w,
                    cluster_ids=cluster_ids,
                    n_obs=n_obs,
                    df_resid=df_resid,
                    ssc=ssc
                )
        else:
            raise ValueError(f"Unknown vcov type: {vcov}")
        
    else:
        # Build expressions for OLS
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
        vals = list(row)

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

        # Use Cholesky decomposition with fallback for both beta and XtX_inv
        try:
            L = np.linalg.cholesky(XtX)
            beta = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
            # Compute XtX_inv from same L factor
            XtX_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(L.shape[0])))
        except np.linalg.LinAlgError:
            # Fallback to direct methods
            beta = np.linalg.solve(XtX, Xty)
            XtX_inv = np.linalg.inv(XtX)

        # Compute residuals and store in table
        resid_expr = f'"{y_col}_dm" - (' + " + ".join([f"CAST({b} AS DOUBLE) * \"{col}_dm\"" for b, col in zip(beta, x_cols)]) + ")"
        con.execute(f'CREATE OR REPLACE TEMPORARY TABLE "{tmp_table}" AS SELECT *, {resid_expr} AS _resid FROM "{tmp_table}"')
        
        df_resid = n_obs - k - absorbed_df
        
        # Compute standard errors using std_errors module
        se, n_clusters = compute_standard_errors_duckdb(
            con=con,
            tmp_table=tmp_table,
            x_cols=x_cols,
            XtX_inv=XtX_inv,
            weights=weights,
            vcov=vcov,
            cluster_cols=cluster_cols,
            n_obs=n_obs,
            df_resid=df_resid,
            ssc=ssc
        )
    
    return beta, se, df_resid, n_clusters

def leanfe_duckdb(
    data: str | pl.DataFrame | pl.LazyFrame | None,
    y_col: str | None = None,
    x_cols: list[str] | None = None,
    fe_cols: list[str] = [],
    formula: str | None = None,
    strategy: str = 'auto',
    weights: str | None = None,
    demean_tol: float = 1e-8,
    max_iter: int = 500,
    vcov: Literal["iid", "HC1", "cluster"] = "iid",
    cluster_cols: list[str] | None = None,
    ssc: bool = False,
    sample_frac: float | None = None,
    con: duckdb.DuckDBPyConnection | None = None
) -> LeanFEResult:
    """
    Fixed effects regression using DuckDB with optimized memory usage.

    Supports formulas without fixed effects (e.g., "y ~ x1 + x2").
    """
    assert con is not None, "User must provide a duckdb.connect() object."

    est_comp_ratio = None

    # create unique prefix for all temp objects for this call
    uid = uuid.uuid4().hex[:8]
    def mk_tmp(name: str) -> str:
        return f"leanfe_{name}_{uid}"

    tmp_table = mk_tmp("data")
    created_tmp_tables = [tmp_table]

    # Parse formula if provided
    if formula is not None:
        y_col, x_cols, fe_cols, factor_vars, interactions, instruments = parse_formula(formula)
    elif y_col is None or x_cols is None or fe_cols is None:
        raise ValueError("Must provide either 'formula' or (y_col, x_cols, fe_cols)")
    else:
        factor_vars = []
        interactions = []
        instruments = []

    x_cols = list(x_cols)
    fe_cols = list(fe_cols) if fe_cols is not None else []

    # Build needed columns
    needed_cols = [y_col] + x_cols + fe_cols + instruments
    for var, ref in factor_vars:
        if var not in needed_cols:
            needed_cols.append(var)
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
        where_parts.append(f'"{weights}" > 0')
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
        src_name = mk_tmp("src")
        con.register(src_name, data.select(needed_cols))
        con.execute(f'CREATE TEMPORARY TABLE "{tmp_table}" AS SELECT * FROM "{src_name}"')
        created_tmp_tables.append(src_name)

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

    is_iv = len(instruments) > 0

    if fe_cols:
        # Compute FE cardinality to decide strategy
        fe_cardinality = {}
        for fe in fe_cols:
            card_query = f"SELECT COUNT(DISTINCT {fe}) FROM {tmp_table}"
            card = con.execute(card_query).fetchone()
            if card is None:
                raise ValueError(f'Error in computing FE ({fe}) cardinality, check dtype of {fe} or try different backend')
            card = int(card[0])
            fe_cardinality[fe] = card
    else:
        fe_cardinality = {}

    n_obs_initial_query = f"SELECT COUNT(*) FROM {tmp_table}"
    n_obs_initial = con.execute(n_obs_initial_query).fetchone()
    if n_obs_initial is None:
        raise ValueError('Could not fetch number of obs/rows. Check for corrupted data')
    n_obs_initial = int(n_obs_initial[0])

    if strategy == 'auto':

        est_comp_ratio = estimate_compression_ratio(
            con = con,
            table_ref = f'{tmp_table}',
            x_cols = x_cols,
            fe_cols = fe_cols
        )

        if not fe_cols:
            # No FE: decide between OLS and compress based on compression ratio
            if est_comp_ratio >= 0.8:
                inferred_strategy = 'ols'
            else:
                inferred_strategy = 'compress'

        else: 
            inferred_strategy = determine_strategy(
            vcov, is_iv, fe_cardinality,
            max_fe_levels=MAX_FE_LEVELS,
            n_obs=n_obs_initial,
            n_x_cols=len(x_cols),
            estimated_compression_ratio=est_comp_ratio)

        print(f'Auto selection: Inferring {inferred_strategy} strategy. N = {n_obs_initial:_}, est. compression ratio: {est_comp_ratio}')
        strategy = inferred_strategy

    if strategy == 'compress':
        print('Using compression strategy...')
        result = leanfe_compress_duckdb(
            con=con,
            y_col=y_col,
            x_cols=x_cols,
            fe_cols=fe_cols,
            table_ref=tmp_table,
            weights=weights,
            vcov=vcov,
            cluster_col=cluster_cols,
            ssc=ssc
        )
        result.formula = formula
        result.fe_cols = fe_cols
        _safe_cleanup(con, uid)
        return result
    
    if strategy == 'alt_proj':
        if not fe_cols:
            raise ValueError("Strategy 'alt_proj' requires FE-cols. Use strategy='ols' instead for OLS without FE.")
        print('Using FWL/alternating projections strategy...')
        
        it = 0
        # Drop singletons
        if fe_cols:
            fe_filter = " OR ".join([f'"{fe}" IN (SELECT "{fe}" FROM "{tmp_table}" GROUP BY 1 HAVING COUNT(*) = 1)' for fe in fe_cols])
            con.execute(f'DELETE FROM "{tmp_table}" WHERE {fe_filter}')
            n_obs_query = f'SELECT COUNT(*) FROM "{tmp_table}"'
            n_obs_res = con.execute(n_obs_query).fetchone()
            if n_obs_res is None:
                raise ValueError('Error in fetching no of obs/rows')
            n_obs = int(n_obs_res[0])
        else:
            n_obs = n_obs_initial

        # If there is exactly one FE, perform exact one-pass demeaning (no iterations).
        if len(fe_cols) == 1:
            fe = fe_cols[0]
            cols_to_demean = [y_col] + x_cols + instruments
            # Build aggregation expressions (weighted or unweighted)
            if weights is not None:
                agg_exprs = ", ".join([
                    f"SUM({c} * {weights}) / SUM({weights}) AS {c}_mean"
                    for c in cols_to_demean
                ])
            else:
                agg_exprs = ", ".join([
                    f"AVG({c}) AS {c}_mean"
                    for c in cols_to_demean
                ])
            # Create small FE means table
            con.execute(f'CREATE OR REPLACE TEMP TABLE fe_means AS SELECT "{fe}", {agg_exprs} FROM "{tmp_table}" GROUP BY 1')
            # Left join and compute demeaned columns in one pass
            subtract_exprs = ", ".join([f'd."{c}" - COALESCE(m.{c}_mean, 0) as {c}_dm' for c in cols_to_demean])
            other_cols_str = ", ".join([f'd."{c}"' for c in (cols_to_demean + fe_cols + (cluster_cols or []) + ([weights] if weights else []))])
            # Build final temp table replacing original with _dm columns included
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE "{tmp_table}" AS
                SELECT {other_cols_str}, {subtract_exprs}
                FROM "{tmp_table}" d
                LEFT JOIN fe_means m ON d."{fe}" = m."{fe}"
            """)
            # Compute fe_dims and absorbed_df for the single FE
            fe_dims_result = con.execute(f'SELECT COUNT(DISTINCT "{fe}") FROM "{tmp_table}"').fetchone()
            if fe_dims_result is not None:
                fe_count = int(fe_dims_result[0])
                fe_dims = (fe_count,)
                absorbed_df = fe_count - 1
            else:
                fe_dims = None
                absorbed_df = 0

        else:
            # Create working table with _dm columns
            cols_to_demean = [y_col] + x_cols + instruments
            cols_to_keep = cols_to_demean + fe_cols
            if cluster_cols is not None:
                for c in cluster_cols:
                    if c not in cols_to_keep:
                        cols_to_keep.append(c)
            if weights is not None and weights not in cols_to_keep:
                cols_to_keep.append(weights)

            work_table = mk_tmp("work")
            select_dm = ", ".join([f"{col} AS {col}_dm" for col in cols_to_demean])
            all_cols = ", ".join([f'"{c}"' for c in cols_to_keep])
            con.execute(f'CREATE OR REPLACE TEMP TABLE "{work_table}" AS SELECT {all_cols}, {select_dm} FROM "{tmp_table}"')
            tmp_table = work_table

            fe_cols_ordered = sorted(fe_cols, key=lambda fe: fe_cardinality.get(fe, 0))
            dm_cols = [f"{col}_dm" for col in cols_to_demean]

            # Iterative demeaning
            for it in range(1, max_iter + 1):
                for fe in fe_cols_ordered:
                    if weights is not None:
                        agg_expr = ", ".join([f"SUM({c} * {weights}) / SUM({weights}) as {c}_mean" for c in dm_cols])
                    else:
                        agg_expr = ", ".join([f"AVG({c}) as {c}_mean" for c in dm_cols])

                    con.execute(f'CREATE OR REPLACE TEMP TABLE fe_means AS SELECT "{fe}", {agg_expr} FROM "{tmp_table}" GROUP BY 1')

                    other_cols_str = ", ".join([f'd."{c}"' for c in cols_to_keep])
                    subtract_expr = ", ".join([f'd.{c} - COALESCE(m.{c}_mean, 0) as {c}' for c in dm_cols])

                    con.execute(f"""
                        CREATE OR REPLACE TEMP TABLE "{tmp_table}" AS 
                        SELECT {other_cols_str}, {subtract_expr}
                        FROM "{tmp_table}" d
                        LEFT JOIN fe_means m ON d."{fe}" = m."{fe}"
                    """)
                    
                if it >= 3:
                    avg_exprs = ", ".join([f"AVG({c}) AS {c}_avg" for c in dm_cols])
                    avg_names = ", ".join([f"{c}_avg" for c in dm_cols])
                    fe_list = ", ".join([f'"{fe}"' for fe in fe_cols])
                    convergence_sql = f"""
                        SELECT MAX(ABS(val)) 
                        FROM (
                            SELECT {avg_exprs} FROM "{tmp_table}" GROUP BY GROUPING SETS ({fe_list})
                        ) UNPIVOT(val FOR col IN ({avg_names}))
                    """
                    max_err = con.execute(convergence_sql).fetchone()
                    if max_err is None:
                        raise ValueError('Error in iterative demeaning')
                    max_err = max_err[0]
                    if abs(max_err or 0) < demean_tol:
                        break
            
            # Calculate absorbed df and fe dimensions
            distinct_fes_query = ", ".join([f"COUNT(DISTINCT {fe})" for fe in fe_cols])
            fe_dims_result = con.execute(f"SELECT {distinct_fes_query} FROM {tmp_table}").fetchone()
            if fe_dims_result is not None:
                fe_dims = fe_dims_result  # This is already a tuple!
                absorbed_df = sum(fe_dims) - len(fe_cols)
            else:
                fe_dims = None
                absorbed_df = 0

    # Handle OLS case (no FE)
    elif strategy == 'ols':
        print('Using simple OLS strategy (no fixed effects)...')
        it = 0
        absorbed_df = 0
        fe_dims = None
        n_obs = n_obs_initial
        
        # Create working table with _dm columns (just copy the data)
        cols_to_demean = [y_col] + x_cols + instruments
        cols_to_keep = cols_to_demean
        if cluster_cols is not None:
            cols_to_keep += cluster_cols
        if weights is not None and weights not in cols_to_keep:
            cols_to_keep.append(weights)
        
        work_table = mk_tmp("work")
        select_dm = ", ".join([f"{col} AS {col}_dm" for col in cols_to_demean])
        all_cols = ", ".join([f'"{c}"' for c in cols_to_keep])
        con.execute(f'CREATE OR REPLACE TEMP TABLE "{work_table}" AS SELECT {all_cols}, {select_dm} FROM "{tmp_table}"')
        tmp_table = work_table

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Run regression (common for both OLS and alt_proj)
    beta, se, df_resid, n_clusters = _run_regression_duckdb(
        con=con,
        tmp_table=tmp_table,
        y_col=y_col,
        x_cols=x_cols,
        instruments=instruments,
        weights=weights,
        vcov=vcov,
        cluster_cols=cluster_cols,
        ssc=ssc,
        n_obs=n_obs,
        absorbed_df=absorbed_df
    )

    _safe_cleanup(con, uid)

    return LeanFEResult(
        coefs=dict(zip(x_cols, beta)),
        std_errors=dict(zip(x_cols, se)),
        n_obs=n_obs,
        iterations=it,
        vcov_type=vcov,
        is_iv=len(instruments) > 0,
        n_instruments=len(instruments) if instruments else None,
        n_clusters=n_clusters,
        df_resid=df_resid,
        formula=formula,
        fe_cols=fe_cols,
        fe_dims=fe_dims,
        compression_ratio = est_comp_ratio
    )