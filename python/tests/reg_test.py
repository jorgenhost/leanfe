import time
import gc
import os
import polars as pl
import polars.selectors as cs
import json
import duckdb
from leanfe import leanfe
from memory_profiler import memory_usage
os.environ['POLARS_MAX_THREADS'] = '16'

LDFE_DATA = 'data/data_ldfe.pq'
HDFE_DATA = 'data/data_hdfe.pq'
UHDFE_DATA = 'data/data_uhdfe.pq'
MEGA_FE_DATA = 'data/data_mega_fe.pq'


lf_uhdfe = pl.scan_parquet(UHDFE_DATA)
lf_mega_fe = pl.scan_parquet(MEGA_FE_DATA)
UHDFE_X = lf_uhdfe.select(cs.starts_with('x')).collect_schema().names()
MEGA_FE_X = lf_mega_fe.select(cs.starts_with('x')).collect_schema().names()

UHDFE_X = ' + '.join(UHDFE_X)
MEGA_FE_X = ' + '.join(MEGA_FE_X)

BENCHMARKS = {
    # "LDFE_BASE": {
    #     "path": LDFE_DATA, 
    #     "formula": 'log_wage ~ age + age_sq + education | firm_id ',
    #     'vcov': 'iid',
    #     'cluster_cols': None,
    # },
    # "LDFE_CLUSTER1" : {
    #     "path": LDFE_DATA, 
    #     "formula": 'log_wage ~ age + age_sq + education | firm_id ',
    #     'vcov': 'cluster',
    #     'cluster_cols': ['firm_id'],
    # },
    # "LDFE_CLUSTER2" : {
    #     "path": LDFE_DATA, 
    #     "formula": 'log_wage ~ age + age_sq + education | firm_id ',
    #     'vcov': 'cluster',
    #     'cluster_cols': ['firm_id', 'worker_id'],
    # },
    "HDFE_BASE": {
        "path": HDFE_DATA, 
        "formula": 'log_wage ~ experience + education + age + age_sq | firm_id + worker_id ',
        'vcov': 'iid',
        'cluster_cols': None
        },
    "HDFE_CLUSTER1" : {
        "path": HDFE_DATA, 
        "formula": 'log_wage ~ experience + education + age + age_sq | firm_id + worker_id ',
        'vcov': 'cluster',
        'cluster_cols': ['firm_id']        
    },    
    "HDFE_CLUSTER2": {
        "path": HDFE_DATA, 
        "formula": 'log_wage ~ experience + education + age + age_sq | firm_id + worker_id ',
        'vcov': 'cluster',
        'cluster_cols': ['firm_id', 'worker_id']   
    },
    # "UHDFE_BASE": {
    #     "path": UHDFE_DATA, 
    #     "formula": f'log_wage ~ experience + education + age + age_sq + {UHDFE_X} | firm_id + worker_id + city_id',
    #     'vcov': 'iid',
    #     'cluster_cols': None
    # },
    # "UHDFE_CLUSTER1": {
    #     "path": UHDFE_DATA, 
    #     "formula": f'log_wage ~ experience + education + age + age_sq + {UHDFE_X} | firm_id + worker_id + city_id',
    #     'vcov': 'cluster',
    #     'cluster_cols': ['firm_id']
    # },
    # "UHDFE_CLUSTER2": {
    #     "path": UHDFE_DATA, 
    #     "formula": f'log_wage ~ experience + education + age + age_sq + {UHDFE_X} | firm_id + worker_id + city_id',
    #     'vcov': 'cluster',
    #     'cluster_cols': ['firm_id', 'worker_id']
    # },
    # "MEGA_HDFE_BASE": {
    #     "path": MEGA_FE_DATA, 
    #     "formula": f'log_wage ~ experience + education + age + age_sq + {MEGA_FE_X} | firm_id + worker_id + city_id',
    #     'vcov': 'iid',
    #     'cluster_cols': None
    # },
    # "MEGA_HDFE_CLUSTER1": {
    #     "path": MEGA_FE_DATA, 
    #     "formula": f'log_wage ~ experience + education + age + age_sq + {MEGA_FE_X} | firm_id + worker_id + city_id',
    #     'vcov': 'cluster',
    #     'cluster_cols': ['firm_id']
    # },
    # "MEGA_HDFE_CLUSTER2": {
    #     "path": MEGA_FE_DATA, 
    #     "formula": f'log_wage ~ experience + education + age + age_sq + {MEGA_FE_X} | firm_id + worker_id + city_id',
    #     'vcov': 'cluster',
    #     'cluster_cols': ['firm_id', 'worker_id']
    # }
}

backends = [
    'duckdb', 
    # 'polars-lazy', 
    # 'polars-in-memory'
]
strategies = [
    # 'auto', 
    # 'compress', 
    'alt_proj'
]

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
    return res.to_dict()

def run_benchmark(data, formula, backend, strategy, vcov, cluster_cols):
    gc.collect()

    start_time = time.perf_counter()

    # memory_usage returns a list of samples.
    # interval=0.1 means check every 100ms.
    # retval=True allows us to get the function result if we needed it.
    mem_samples, model_output = memory_usage(
        (execute_leanfe, (data, formula, backend, strategy, vcov, cluster_cols)),
        interval=0.1,
        timeout=None,
        retval=True
    )

    end_time = time.perf_counter()

    peak_mem = max(mem_samples) if mem_samples else 0
    duration = round(end_time - start_time, 4)

    return duration, round(peak_mem, 2), model_output



if __name__ == '__main__':
    # 3. Execution Loop
    results = []

    for d_name, d_info in BENCHMARKS.items():
        for b in backends:
            for s in strategies:

                # --- Logic: Skip 'compress' for HDFE_CLUSTER2 and all UHDFE instances ---
                
                compress_strats_to_skip = ['UHDFE_BASE', 'UDHFE_CLUSTER1', 'UHDFE_CLUSTER2', 'HDFE_CLUSTER1', 'HDFE_CLUSTER2']
                            
                if s == 'compress' and d_name in compress_strats_to_skip:
                    print(f"Skipping strategy '{s}' for dataset '{d_name}'")
                    continue
                print(f"Running: {d_name} | {b} | {s}...")
                


                # Initialize connection variable for scoping
                con = None
                data_to_pass = None
                
                try:
                    # 1. Setup Data/Connection inside the loop for isolation
                    if b == 'polars-lazy':
                        data_to_pass = pl.scan_parquet(d_info['path'])
                    elif b == 'polars-in-memory':
                        data_to_pass = pl.read_parquet(d_info['path'])
                    else:
                        data_to_pass = d_info['path']
                        con = duckdb.connect()
                        con.execute('SET memory_limit = "8GB"')
                        con.execute('SET threads TO 12')

                    # 2. Run the benchmark
                    duration, peak_memory, model_output = run_benchmark(
                        data=data_to_pass, 
                        formula=d_info['formula'], 
                        vcov=d_info['vcov'],
                        cluster_cols=d_info['cluster_cols'],
                        strategy=s,
                        backend=b,
                    )

                except Exception as e:
                    print(f"Error on {d_name} {b} {s}: {e}")
                    duration, peak_memory, model_output = None, None, {}

                # 3. Explicit Cleanup Sequence
                if con is not None:
                    con.close()
                    del con
                
                del data_to_pass
                
                # Collect garbage for all three generations
                gc.collect() 

                res = {
                    "Dataset": d_name,
                    "Backend": b,
                    "Strategy": s,
                    "Time (s)": duration,
                    "Peak RAM (MB)": peak_memory,
                    "coef": model_output['coefs']['age'],
                    "std.err": model_output['std_errors']['age'],
                    'FE dimensionality': f'{model_output['fe_dims']}',
                    'FEs': f'{model_output['fe_cols']}',
                    'N': model_output['n_obs']
                }

                outfile = f'{d_name}_{b}_{s}.json'

                with open(outfile, 'w') as f:
                    json.dump(res, f, indent=4)

                results.append(res)

    # --- Display Table ---
    df_results = pl.DataFrame(results)
    print("\nBenchmark Results Summary:")
    print(df_results)
    df_results.write_csv('benchmark_results4.csv')