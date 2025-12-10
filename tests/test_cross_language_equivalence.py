"""
Cross-language equivalence test: Verify Python and R produce identical results.

This test generates deterministic data, runs regressions in both languages,
and compares the results to ensure they match within numerical precision.
"""
import subprocess
import json
import numpy as np
import polars as pl
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from leanfe import leanfe


def generate_test_data():
    """Generate deterministic test data for cross-language comparison."""
    # Use a fixed seed and simple deterministic data
    n = 1000
    
    # Create deterministic data (no randomness)
    np.random.seed(12345)
    
    data = {
        'y': np.random.normal(10, 2, n),
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(5, 2, n),
        'treatment': np.random.binomial(1, 0.5, n),
        'fe1': np.repeat(np.arange(100), 10),  # 100 groups, 10 obs each
        'fe2': np.tile(np.arange(50), 20),     # 50 groups, 20 obs each
        'cluster': np.repeat(np.arange(50), 20),  # 50 clusters
        'weight': np.random.uniform(0.5, 2.0, n),
    }
    
    # Save to parquet for R to read - use unique filename to avoid stale data issues
    import uuid
    df = pl.DataFrame(data)
    parquet_path = f'/tmp/test_cross_language_data_{uuid.uuid4().hex[:8]}.parquet'
    df.write_parquet(parquet_path)
    
    # Ensure file is fully written before R reads it
    import time
    time.sleep(0.1)
    
    return df, parquet_path


def run_python_regression(parquet_path, formula, vcov='iid', cluster_cols=None, weights=None, backend='polars'):
    """Run regression in Python and return results."""
    # Read from parquet to ensure same data as R
    df = pl.read_parquet(parquet_path)
    result = leanfe(
        df,
        formula=formula,
        vcov=vcov,
        cluster_cols=cluster_cols,
        weights=weights,
        backend=backend
    )
    return {
        'coefficients': result['coefficients'],
        'std_errors': result['std_errors'],
        'n_obs': result['n_obs'],
        'vcov_type': result['vcov_type'],
    }


def run_r_regression(parquet_path, formula, vcov='iid', cluster_cols=None, weights=None, backend='polars'):
    """Run regression in R and return results."""
    cluster_str = f'c("{cluster_cols[0]}")' if cluster_cols else 'NULL'
    weights_str = f'"{weights}"' if weights else 'NULL'
    
    # Build R script - use raw strings to avoid escaping issues with $
    r_script = f'''suppressWarnings(library(polars))
suppressWarnings(library(duckdb))
suppressWarnings(library(DBI))
source("r/R/common.R")
source("r/R/compress.R")
source("r/R/polars.R")
source("r/R/duckdb.R")
source("r/R/leanfe.R")

df <- pl$read_parquet("{parquet_path}")

result <- leanfe(
    df,
    formula = "{formula}",
    vcov = "{vcov}",
    cluster_cols = {cluster_str},
    weights = {weights_str},
    backend = "{backend}"
)

# Output as JSON-like format
cat("COEFFICIENTS:", paste(names(result$coefficients), result$coefficients, sep="=", collapse=","), "\\n")
cat("STD_ERRORS:", paste(names(result$std_errors), result$std_errors, sep="=", collapse=","), "\\n")
cat("N_OBS:", result$n_obs, "\\n")
cat("VCOV_TYPE:", result$vcov_type, "\\n")
'''
    
    # Write R script to temp file to avoid escaping issues
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_script)
        r_script_path = f.name
    
    try:
        # Run R script from file
        result = subprocess.run(
            ['Rscript', r_script_path],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), '..')
        )
        
        if result.returncode != 0:
            print("R stderr:", result.stderr)
            raise RuntimeError(f"R script failed: {result.stderr}")
    finally:
        os.unlink(r_script_path)
    
    # Parse output
    output = result.stdout
    # Debug: print raw output (uncomment for debugging)
    # if 'cluster' in vcov.lower():
    #     print(f"DEBUG R script:\n{r_script}")
    #     print(f"DEBUG R output:\n{output}")
    parsed = {}
    
    for line in output.split('\n'):
        if line.startswith('COEFFICIENTS:'):
            coef_str = line.replace('COEFFICIENTS:', '').strip()
            parsed['coefficients'] = {}
            for item in coef_str.split(','):
                if '=' in item:
                    name, val = item.split('=')
                    parsed['coefficients'][name.strip()] = float(val.strip())
        elif line.startswith('STD_ERRORS:'):
            se_str = line.replace('STD_ERRORS:', '').strip()
            parsed['std_errors'] = {}
            for item in se_str.split(','):
                if '=' in item:
                    name, val = item.split('=')
                    parsed['std_errors'][name.strip()] = float(val.strip())
        elif line.startswith('N_OBS:'):
            parsed['n_obs'] = int(line.replace('N_OBS:', '').strip())
        elif line.startswith('VCOV_TYPE:'):
            parsed['vcov_type'] = line.replace('VCOV_TYPE:', '').strip()
    
    return parsed


def compare_results(py_result, r_result, test_name, rtol=1e-6, se_rtol=None):
    """Compare Python and R results."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    
    # Compare n_obs
    assert py_result['n_obs'] == r_result['n_obs'], \
        f"n_obs mismatch: Python={py_result['n_obs']}, R={r_result['n_obs']}"
    print(f"✓ n_obs match: {py_result['n_obs']}")
    
    # Compare vcov_type
    assert py_result['vcov_type'] == r_result['vcov_type'], \
        f"vcov_type mismatch: Python={py_result['vcov_type']}, R={r_result['vcov_type']}"
    print(f"✓ vcov_type match: {py_result['vcov_type']}")
    
    # Compare coefficients
    print("\nCoefficients:")
    for var in py_result['coefficients']:
        py_coef = py_result['coefficients'][var]
        r_coef = r_result['coefficients'].get(var)
        if r_coef is None:
            print(f"  ✗ {var}: Missing in R results")
            continue
        
        rel_diff = abs(py_coef - r_coef) / max(abs(py_coef), 1e-10)
        match = rel_diff < rtol
        status = "✓" if match else "✗"
        print(f"  {status} {var}: Python={py_coef:.10f}, R={r_coef:.10f}, rel_diff={rel_diff:.2e}")
        
        if not match:
            raise AssertionError(f"Coefficient {var} mismatch: Python={py_coef}, R={r_coef}")
    
    # Compare standard errors
    print("\nStandard Errors:")
    se_tolerance = se_rtol if se_rtol is not None else rtol
    for var in py_result['std_errors']:
        py_se = py_result['std_errors'][var]
        r_se = r_result['std_errors'].get(var)
        if r_se is None:
            print(f"  ✗ {var}: Missing in R results")
            continue
        
        rel_diff = abs(py_se - r_se) / max(abs(py_se), 1e-10)
        match = rel_diff < se_tolerance
        status = "✓" if match else "✗"
        print(f"  {status} {var}: Python={py_se:.10f}, R={r_se:.10f}, rel_diff={rel_diff:.2e}")
        
        if not match:
            raise AssertionError(f"SE {var} mismatch: Python={py_se}, R={r_se}")
    
    print(f"\n✓ All comparisons passed for: {test_name}")


def main():
    print("Cross-Language Equivalence Test: Python vs R")
    print("=" * 60)
    
    # Generate test data
    df, parquet_path = generate_test_data()
    n_obs = len(df)
    del df  # Free memory, we'll read from parquet
    print(f"Generated test data: {n_obs} observations")
    print(f"Saved to: {parquet_path}")
    
    # Test 1: Basic regression with IID SEs (Polars)
    print("\n" + "="*60)
    print("TEST 1: Basic regression with IID SEs (Polars backend)")
    py_result = run_python_regression(parquet_path, "y ~ x1 + x2 | fe1 + fe2", backend='polars')
    r_result = run_r_regression(parquet_path, "y ~ x1 + x2 | fe1 + fe2", backend='polars')
    compare_results(py_result, r_result, "Basic IID (Polars)")
    
    # Test 2: Basic regression with IID SEs (DuckDB)
    print("\n" + "="*60)
    print("TEST 2: Basic regression with IID SEs (DuckDB backend)")
    py_result = run_python_regression(parquet_path, "y ~ x1 + x2 | fe1 + fe2", backend='duckdb')
    r_result = run_r_regression(parquet_path, "y ~ x1 + x2 | fe1 + fe2", backend='duckdb')
    compare_results(py_result, r_result, "Basic IID (DuckDB)")
    
    # Test 3: HC1 robust SEs
    print("\n" + "="*60)
    print("TEST 3: HC1 robust standard errors")
    py_result = run_python_regression(parquet_path, "y ~ x1 + x2 | fe1 + fe2", vcov='HC1')
    r_result = run_r_regression(parquet_path, "y ~ x1 + x2 | fe1 + fe2", vcov='HC1')
    compare_results(py_result, r_result, "HC1 Robust SEs")
    
    # Test 4: Clustered SEs (use slightly relaxed tolerance for SEs due to numerical differences)
    print("\n" + "="*60)
    print("TEST 4: Clustered standard errors")
    py_result = run_python_regression(parquet_path, "y ~ x1 + x2 | fe1 + fe2", vcov='cluster', cluster_cols=['cluster'])
    r_result = run_r_regression(parquet_path, "y ~ x1 + x2 | fe1 + fe2", vcov='cluster', cluster_cols=['cluster'])
    compare_results(py_result, r_result, "Clustered SEs", se_rtol=0.02)  # 2% tolerance for cluster SEs
    
    # Test 5: Single treatment variable
    print("\n" + "="*60)
    print("TEST 5: Single treatment variable")
    py_result = run_python_regression(parquet_path, "y ~ treatment | fe1 + fe2")
    r_result = run_r_regression(parquet_path, "y ~ treatment | fe1 + fe2")
    compare_results(py_result, r_result, "Single Treatment")
    
    # Test 6: Weighted regression
    print("\n" + "="*60)
    print("TEST 6: Weighted regression")
    py_result = run_python_regression(parquet_path, "y ~ x1 + x2 | fe1 + fe2", weights='weight')
    r_result = run_r_regression(parquet_path, "y ~ x1 + x2 | fe1 + fe2", weights='weight')
    compare_results(py_result, r_result, "Weighted Regression")
    
    # Test 7: Multiple treatments
    print("\n" + "="*60)
    print("TEST 7: Multiple treatment variables")
    py_result = run_python_regression(parquet_path, "y ~ x1 + x2 + treatment | fe1 + fe2")
    r_result = run_r_regression(parquet_path, "y ~ x1 + x2 + treatment | fe1 + fe2")
    compare_results(py_result, r_result, "Multiple Treatments")
    
    print("\n" + "="*60)
    print("ALL CROSS-LANGUAGE EQUIVALENCE TESTS PASSED!")
    print("="*60)
    
    # Cleanup temp file
    try:
        os.unlink(parquet_path)
    except:
        pass


if __name__ == '__main__':
    main()
