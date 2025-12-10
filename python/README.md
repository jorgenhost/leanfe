# 🍃💨 leanfe

**Lean, Fast Fixed Effects Regression for Python**

[![PyPI](https://img.shields.io/pypi/v/leanfe.svg)](https://pypi.org/project/leanfe/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

High-dimensional fixed effects regression that's **fast** when you need speed, and **memory-efficient** when you're working with data larger than RAM.

## Installation

```bash
pip install leanfe
```

## Quick Start

```python
from leanfe import leanfe

# Two-way fixed effects with clustered standard errors
result = leanfe(
    formula="outcome ~ treatment + controls | unit_id + time_id",
    data=df,
    vcov="cluster",
    cluster_cols=["unit_id"],
    backend="polars"  # or "duckdb" for large datasets
)

print(result)
```

## Features

- ⚡ **Polars backend** - Blazing fast in-memory computation
- 💾 **DuckDB backend** - Process datasets larger than RAM
- 📊 **Full econometrics toolkit** - Clustered SEs, factor variables, IV/2SLS
- 🚀 **YOCO compression** - Optimal data compression with sparse matrices

## Two Backends

| Backend | Best For | Trade-off |
|---------|----------|-----------|
| **Polars** (default) | Maximum speed | Higher memory usage |
| **DuckDB** | Large datasets | Slightly slower, minimal memory |

```python
# Fast in-memory (default)
result = leanfe(data=df, formula="y ~ x | fe", backend="polars")

# Memory-efficient (data larger than RAM)
result = leanfe(data="huge_data.parquet", formula="y ~ x | fe", backend="duckdb")
```

## Standard Errors

```python
# IID (default)
result = leanfe(data=df, formula="y ~ x | fe", vcov="iid")

# Heteroskedasticity-robust
result = leanfe(data=df, formula="y ~ x | fe", vcov="HC1")

# Clustered
result = leanfe(data=df, formula="y ~ x | fe", vcov="cluster", cluster_cols=["firm_id"])

# Two-way clustered
result = leanfe(data=df, formula="y ~ x | fe", vcov="cluster", cluster_cols=["firm_id", "year"])
```

## Documentation

📖 **[Full Documentation](https://diegogentilepassaro.github.io/leanfe/)**

## License

MIT
