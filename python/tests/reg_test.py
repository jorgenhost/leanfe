import time
import gc
import os
import polars as pl
import duckdb
from leanfe import leanfe
from memory_profiler import memory_usage

os.environ['POLARS_MAX_THREADS'] = '12'

LDFE_DATA = 'data/data_ldfe.pq'
HDFE_DATA = 'data/data_hdfe.pq'
UHDFE_DATA = 'data/data_uhdfe.pq'

datasets = {
    "LDFE_BASE": {
        "path": LDFE_DATA, 
        "formula": 'log_wage ~ age + age_sq | firm_id',
        'vcov': 'iid',
        'cluster_cols': None,
    },
    "LDFE_CLUSTER1" : {
        "path": LDFE_DATA, 
        "formula": 'log_wage ~ age + age_sq | firm_id',
        'vcov': 'cluster',
        'cluster_cols': ['firm_id'],
    },
    "HDFE_BASE": {
        "path": HDFE_DATA, 
        "formula": 'log_wage ~ experience + education + age + age_sq | firm_id + worker_id + city_id',
        'vcov': 'iid',
        'cluster_cols': None
        },
    "HDFE_CLUSTER1" : {
        "path": HDFE_DATA, 
        "formula": 'log_wage ~ experience + education + age + age_sq | firm_id + worker_id + city_id',
        'vcov': 'cluster',
        'cluster_cols': ['firm_id']        
    },    
    "HDFE_CLUSTER2": {
        "path": HDFE_DATA, 
        "formula": 'log_wage ~ experience + education + age + age_sq | firm_id + worker_id + city_id',
        'vcov': 'cluster',
        'cluster_cols': ['firm_id', 'worker_id']   
    },
    "UDHFE_BASE": {
        "path": UHDFE_DATA, 
        "formula": 'log_wage ~ experience + education + age + age_sq | firm_id + worker_id + city_id',
        'vcov': 'iid',
        'cluster_cols': None
    },
    "UHDFE_CLUSTER1": {
        "path": UHDFE_DATA, 
        "formula": 'log_wage ~ experience + education + age + age_sq | firm_id + worker_id + city_id',
        'vcov': 'cluster',
        'cluster_cols': ['firm_id']
    },
    "UHDFE_CLUSTER2": {
        "path": UHDFE_DATA, 
        "formula": 'log_wage ~ experience + education + age + age_sq | firm_id + worker_id + city_id',
        'vcov': 'cluster',
        'cluster_cols': ['firm_id', 'worker_id']
    }
}

backends = ['duckdb', 'polars-lazy', 'polars-in-memory']
strategies = ['auto', 'compress', 'alt_proj']

# 2. Benchmark Wrapper
def execute_leanfe(data, formula, backend, strategy, vcov, cluster_cols):
    actual_backend = 'polars' if 'polars' in backend else backend
    res = leanfe(
        data=data,
        formula=formula,
        backend=actual_backend,
        strategy=strategy,
        vcov=vcov,
        cluster_cols=cluster_cols,
        con=con if backend == 'duckdb' else None
    )
    _ = res.to_dict()

def run_benchmark(data, formula, backend, strategy, vcov, cluster_cols):
    gc.collect()

    start_time = time.perf_counter()

    # memory_usage returns a list of samples.
    # interval=0.1 means check every 100ms.
    # retval=True allows us to get the function result if we needed it.
    mem_samples = memory_usage(
        (execute_leanfe, (data, formula, backend, strategy, vcov, cluster_cols)),
        interval=0.1,
        timeout=None
    )

    end_time = time.perf_counter()

    peak_mem = max(mem_samples) if mem_samples else 0
    duration = round(end_time - start_time, 4)

    return duration, round(peak_mem, 2)

# 3. Execution Loop
results = []

for d_name, d_info in datasets.items():
    for b in backends:
        if b == 'polars-lazy':
            data_to_pass = pl.scan_parquet(d_info['path'])
        elif b == 'polars-in-memory':
            data_to_pass = pl.read_parquet(d_info['path'])
        else:
            data_to_pass = d_info['path']
            con = duckdb.connect()
            con.execute('SET memory_limit = "8GB"')
            con.execute('SET threads TO 12')

        for s in strategies:
            print(f"Running: {d_name} | {b} | {s}...")
            try:
                duration, peak_memory = run_benchmark(
                    data = data_to_pass, 
                    formula = d_info['formula'], 
                    vcov=d_info['vcov'],
                    cluster_cols = d_info['cluster_cols'],
                    strategy = s,
                    backend = b,
                    )
            except Exception as e:
                print(f"Error: {e}")
                duration, peak_memory = None, None

            results.append({
                "Dataset": d_name,
                "Backend": b,
                "Strategy": s,
                "Time (s)": duration,
                "Peak RAM (MB)": peak_memory
            })

# --- Display Table ---
df_results = pl.DataFrame(results)

display_df = df_results.with_columns([
    pl.col("Time (s)").fill_null("FAILED/OOM"),
    pl.col("Peak RAM (MB)").fill_null("-")
])

print("\nBenchmark Results Summary:")
print(display_df)

df_results.write_csv("benchmark_results.csv")
