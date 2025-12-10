# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-10

### Initial Public Release

**leanfe** is a lean, fast fixed effects regression library for Python and R.

### Features

- **Unified API**: `leanfe()` function with identical syntax in Python and R
- **Two backends**:
  - **Polars**: Blazing fast in-memory computation
  - **DuckDB**: Memory-efficient, can process datasets larger than RAM
- **Formula syntax**: R-style formulas like `"y ~ x1 + x2 | fe1 + fe2"`
- **Factor variables**: `i(region)` syntax for automatic dummy expansion
- **Custom reference categories**: `i(region, ref=R2)` syntax
- **Interaction terms**: `treatment:i(region)` for heterogeneous effects
- **IV/2SLS**: Instrumental variables via `"y ~ x | fe | z"` syntax
- **Standard errors**: IID, HC1 (robust), and clustered (one-way and multi-way)
- **Weighted regression**: WLS via `weights` parameter

### Performance

- **YOCO compression**: Implements [Wong et al. (2021)](https://arxiv.org/abs/2102.11297) optimal data compression
- **Sparse matrices**: FE dummies stored as sparse matrices (~100x memory savings)
- **Vectorized clustered SEs**: Sparse matrix multiplication instead of loops (~31x faster)
- **Smart FE ordering**: Low-cardinality FEs processed first (~14% faster convergence)
- **Automatic strategy selection**: Chooses YOCO vs FWL demeaning based on data characteristics

### Cross-Language Equivalence

Python and R implementations produce identical results (within ~1e-14 precision) for all features, verified by automated cross-language tests.

### Documentation

- Comprehensive Quarto documentation site
- Tutorials for standard errors, DiD, IV regression
- Benchmark comparisons with pyfixest and fixest
- API reference for both Python and R
