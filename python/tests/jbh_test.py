# %%

import os 
os.environ['POLARS_MAX_THREADS'] = '12'
import polars as pl
from leanfe import leanfe
import duckdb
con = duckdb.connect()
con.execute(f'SET memory_limit = "8GB"')
con.execute(f'SET threads TO "12"')

from pyfixest import feols

FE_DATA = 'data/data_fe.pq'
LDFE_DATA = 'data/data_ldfe.pq'
HDFE_DATA = 'data/data_hdfe.pq'
UHDFE_DATA = 'data/data_uhdfe.pq'

lf_fe = pl.scan_parquet(FE_DATA)
lf_ldfe = pl.scan_parquet(LDFE_DATA)
lf_hdfe = pl.scan_parquet(HDFE_DATA)
lf_uhdfe = pl.scan_parquet(UHDFE_DATA)

print(lf_ldfe.head().collect())

import utils


# %% "NORMAL"
res_duckdb = leanfe(

    data = lf_fe,
    formula='log_wage ~ age + age_sq | firm_id',
    vcov = 'cluster',
    cluster_cols=['firm_id', 'worker_id'],
    backend = 'polars',
    strategy = 'compress',
    con = con
)

out = res_duckdb.to_dict()
out

# %%
res_pyfix = feols(
    fml='log_wage ~ age + age_sq | firm_id',
    data = lf_fe.collect(),
    copy_data = False,
    store_data = False,
    vcov = {'CRV1': 'firm_id'},
    lean = True 
)
res_pyfix.coef(), res_pyfix.se()
# %%
# low FE: force compress
res_duckdb = leanfe(
    data = LDFE_DATA,
    formula='log_wage ~ age + age_sq | firm_id',
    vcov = 'cluster',
    cluster_cols=['firm_id'],
    backend = 'duckdb',
    strategy = 'compress',
    con = con
)

out = res_duckdb.to_dict()
out

# %%
# low FE: force alt_proj
res_duckdb = leanfe(
    data = LDFE_DATA,
    formula='log_wage ~ age + age_sq | firm_id',
    # vcov = 'cluster',
    # cluster_cols=['firm_id'],
    backend = 'duckdb',
    strategy = 'alt_proj',
    con = con
)
out = res_duckdb.to_dict()
out

# low FE: auto - should pick compress
res_duckdb = leanfe(
    data = LDFE_DATA,
    formula='log_wage ~ age + age_sq | firm_id',
    # vcov = 'cluster',
    # cluster_cols=['firm_id'],
    backend = 'duckdb',
    # strategy = 'compress',
    con = con
)
out = res_duckdb.to_dict()
out

# low FE: force compress
res_polars = leanfe(
    data = lf_ldfe,
    formula='log_wage ~ age + age_sq | firm_id',
    # vcov = 'cluster',
    # cluster_cols=['firm_id'],
    strategy = 'compress',
    backend = 'polars',
)

res_polars.to_dict()

# low FE: force alt_proj
res_polars = leanfe(
    data = lf_ldfe,
    formula='log_wage ~ age + age_sq | firm_id',
    # vcov = 'cluster',
    # cluster_cols=['firm_id'],
    strategy = 'alt_proj',
    backend = 'polars',
)

res_polars.to_dict()


# low FE: auto - should pick compress
res_polars = leanfe(
    data = lf_ldfe,
    formula='log_wage ~ age + age_sq | firm_id',
    # vcov = 'cluster',
    # cluster_cols=['firm_id'],
    # strategy = 'compress',
    backend = 'polars',
)

res_polars.to_dict()


# High FE: force compress # WARNING: INEFFICIENT!
res_duckdb = leanfe(
    data = HDFE_DATA,
    formula='log_wage ~ experience + education + age + age_sq | firm_id + worker_id + city_id',
    # vcov = 'cluster',
    # cluster_cols=['firm_id'],
    backend = 'duckdb',
    strategy = 'compress',
    con = con
)
out = res_duckdb.to_dict()
out


df = lf.collect()
res_polars = leanfe(
    data = df,
    formula='log_wage ~ age + age_sq | firm_id + worker_id',
    # vcov = 'cluster',
    # cluster_cols=['firm_id'],
    strategy = 'alt_proj',
    backend = 'polars',
)

dict(res_polars)




res_pyfixest = pf.feols(
    data = lf.collect(),
    fml='log_wage ~ age + age_sq | firm_id',
    fixef_tol = 1e-10,
    copy_data = False,
    store_data = False,
    lean = True,
    # vcov = {'CRV1': 'firm_id'}
)

res_pyfixest.summary()

coef_check = 'age'

diff_in_coef_pl_duckdb = res_duckdb['coefficients'][coef_check] - res_polars['coefficients'][coef_check]
print(f'Coef =={coef_check}== diff between DuckDB & Polars: {abs(diff_in_coef_pl_duckdb)}') 

diff_in_coef_pl_pyfixest = res_polars['coefficients'][coef_check] - res_pyfixest.coef()[coef_check]
print(f'Coef =={coef_check}== diff between Polars & pyFixest: {abs(diff_in_coef_pl_pyfixest)}') 

diff_in_std_pl_pyfixest = res_polars['std_errors'][coef_check] - res_pyfixest.se()[coef_check]
print(f'Std error =={coef_check}== diff between Polars & pyFixest: {abs(diff_in_std_pl_pyfixest)}') 

# %%
