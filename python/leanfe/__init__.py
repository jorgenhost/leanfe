"""leanfe: Lean, fast fixed effects regression using Polars and DuckDB."""

from .leanfe import leanfe, fast_feols, fast_feols_polars, fast_feols_duckdb
from .polars_impl import leanfe_polars
from .duckdb_impl import leanfe_duckdb
from .common import parse_formula

__version__ = "0.1.0"
__all__ = [
    "leanfe",
    "leanfe_polars", 
    "leanfe_duckdb",
    # Backwards compatibility
    "fast_feols",
    "fast_feols_polars", 
    "fast_feols_duckdb",
    "parse_formula"
]
