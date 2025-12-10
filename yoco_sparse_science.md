# Sparse Sufficient Statistics: Scalable Fixed Effects Regression via YOCO Compression

**Diego Gentile Passaro**

*December 2024*

## Abstract

Wong et al. (2021) introduced YOCO (You Only Compress Once), a data compression strategy that enables lossless estimation of linear models from sufficient statistics. While YOCO dramatically reduces data volume through GROUP BY aggregation, the original paper does not address the computational challenge of representing fixed effect dummy variables in the compressed design matrix. We extend YOCO with sparse matrix representations for fixed effects, reducing memory requirements by approximately 100x and achieving comparable speedups in matrix operations. This extension is mathematically equivalent to the original formulation—producing numerically identical coefficients and standard errors—while enabling practical estimation of two-way fixed effects models on datasets with hundreds of millions of observations using single-machine computation. We provide complexity analysis, prove numerical equivalence, and demonstrate empirical performance gains of 20-80x over existing implementations.

**Keywords:** Statistical Computing, Fixed Effects, Sparse Matrices, Panel Data, Data Compression, Econometrics

## 1. Introduction

Linear models with fixed effects are fundamental to empirical economics and causal inference. Panel data applications routinely include firm, individual, or geographic fixed effects to control for unobserved heterogeneity. As datasets grow to millions or billions of observations, computational constraints become binding.


The dominant approach for high-dimensional fixed effects is iterative demeaning based on the Frisch-Waugh-Lovell (FWL) theorem, implemented in packages such as fixest (Bergé, 2018), reghdfe (Correia, 2016), and lfe (Gaure, 2013). These methods scale linearly with observations but require multiple passes over the data and do not reduce memory requirements.

Wong et al. (2021) proposed an alternative: YOCO (You Only Compress Once), a compression strategy based on conditionally sufficient statistics. By grouping observations with identical feature vectors and computing aggregates (count, sum, sum of squares), YOCO reduces data volume by orders of magnitude for discrete regressors while enabling lossless recovery of OLS coefficients and standard errors under homoskedastic, heteroskedastic, and clustered error structures.

However, the original YOCO paper focuses on the statistical methodology and does not address a key implementation challenge: when fixed effects are included in the model, the design matrix contains dummy variables for each FE level. With F total fixed effect levels, this creates G × (F-1) entries in the compressed design matrix, where G is the number of compressed groups. For typical panel data with hundreds or thousands of firm-year combinations, this matrix becomes computationally expensive to construct and manipulate, potentially negating the benefits of compression.

We observe that fixed effect dummy matrices have a specific structure: each row contains exactly one non-zero entry (value 1) per fixed effect dimension, with all other entries being zero. For a model with two fixed effects, each row has exactly 2 non-zero entries out of potentially thousands of columns—a sparsity ratio exceeding 99%. This structure is ideally suited for sparse matrix representations.

This paper makes the following contributions:

1. **Sparse extension to YOCO**: We extend YOCO with sparse matrix representations for fixed effect dummies, reducing memory requirements by approximately 100x for typical panel data.

2. **Mathematical equivalence**: We prove that sparse matrix operations produce numerically identical coefficients and standard errors to dense implementations.

3. **Complexity analysis**: We show that sparse operations reduce time complexity from O(G × F²) to O(G × k × F), where k is the number of FE dimensions (typically 2-3).

4. **Empirical validation**: We demonstrate 20-80x speedups over existing implementations and successful estimation on 100M+ observations using single-machine computation.

## 2. Setting

We adopt the notation from Wong et al. (2021). Consider n observations with outcome vector y ∈ ℝⁿ and feature matrix M ∈ ℝⁿˣᵖ. The linear model is:

$$y = M\beta + \varepsilon$$

where E[ε|M] = 0 and E[εε'|M] = Ω. The OLS estimator and its variance are:

$$\hat{\beta} = (M^\top M)^{-1} M^\top y$$
$$V(\hat{\beta}) = (M^\top M)^{-1} (M^\top \Omega M) (M^\top M)^{-1}$$

The latter is the sandwich covariance matrix, with "bread" Π = (M'M)⁻¹ and "meat" Ξ = M'ΩM.


### 2.1 Fixed Effects Models

In panel data applications, M typically includes fixed effect dummy variables. Let X ∈ ℝⁿˣᵏₓ be the matrix of regressors of interest (e.g., treatment indicators) and D ∈ ℝⁿˣ⁽ᶠ⁻¹⁾ be the matrix of fixed effect dummies, where F is the total number of FE levels across all FE dimensions. The full design matrix is M = [X | D].

For a two-way fixed effects model with F₁ firm effects and F₂ year effects, D has F₁ + F₂ - 2 columns (dropping one reference category per dimension for identification).

### 2.2 YOCO Compression

Following Wong et al. (2021), we compress the data by grouping on the feature matrix M. Let G be the number of unique feature vectors. The compressed data consists of:

- M̃ ∈ ℝᴳˣᵖ: unique feature vectors (one row per group)
- ỹ⁰ ∈ ℝᴳ: sum of outcomes per group
- ỹ⁰⁰ ∈ ℝᴳ: sum of squared outcomes per group  
- ñ ∈ ℝᴳ: count of observations per group

The OLS coefficients can be recovered from the weighted regression:

$$\frac{\tilde{y}^0}{\tilde{n}} = \tilde{M}\beta + \varepsilon$$

with weights ñ, where division is element-wise. Wong et al. (2021) prove that this produces numerically identical coefficients to the uncompressed model, and provide formulas for recovering V(β̂) under homoskedastic (IID), heteroskedastic (HC1), and clustered error structures.

The compression ratio G/n depends on the number of unique (regressor, FE) combinations. For discrete regressors (e.g., binary treatment), compression ratios of 100-1000x are common, dramatically reducing computational requirements.

## 3. The Sparse Matrix Extension

### 3.1 The Problem: Dense FE Dummies

After YOCO compression, we have G groups and need to construct the compressed design matrix M̃ = [X̃ | D̃]. The FE dummy matrix D̃ has dimensions G × (F-1).

**Example:** Consider a dataset with G = 50,000 compressed groups and two-way fixed effects with F₁ = 500 firms and F₂ = 100 years. The FE dummy matrix has 50,000 × 598 ≈ 30 million entries, requiring approximately 240 MB of memory in dense representation.

However, each row of D̃ contains exactly 2 non-zero entries (one per FE dimension), with all other entries being zero. The sparsity ratio is 1 - 2/598 = 99.67%.


### 3.2 Sparse Matrix Representation

Sparse matrices store only non-zero entries, typically as (row, column, value) triplets in COO (Coordinate) format or as compressed row/column formats (CSR/CSC). For the FE dummy matrix with k FE dimensions:

| Representation | Storage | Memory (G=50K, F=600) |
|----------------|---------|----------------------|
| Dense | G × (F-1) floats | ~240 MB |
| Sparse | G × k triplets | ~2.4 MB |
| **Reduction** | | **~100x** |

The memory reduction factor is approximately (F-1)/k, which grows with the number of FE levels.

### 3.3 Mathematical Equivalence

The key insight is that sparse matrix operations are mathematically identical to dense operations—they simply skip multiplications involving zeros.

**Proposition 1** (Equivalence of Sparse and Dense Operations). Let M̃ₛ be the sparse representation of M̃ and M̃ₐ be the dense representation. For any diagonal weight matrix W:

$$\tilde{M}_s^\top W \tilde{M}_s = \tilde{M}_d^\top W \tilde{M}_d$$

*Proof.* Matrix multiplication is defined as the sum of element-wise products. The (i,j) entry of M̃'WM̃ is Σₖ M̃ₖᵢ Wₖₖ M̃ₖⱼ. Zero entries contribute nothing to this sum regardless of whether they are explicitly stored (dense) or implicitly assumed (sparse). Since both representations encode the same mathematical matrix, the products are identical. □

**Corollary 1** (Identical Estimation). The OLS coefficients β̂ and covariance matrix V(β̂) computed using sparse matrices are numerically identical to those computed using dense matrices.

*Proof.* The estimator β̂ = (M̃'WM̃)⁻¹(M̃'Wỹ) depends only on matrix products covered by Proposition 1. The covariance V(β̂) under all three error structures (IID, HC1, clustered) similarly depends only on such products. □

This equivalence is not merely asymptotic—sparse and dense implementations produce bit-identical results up to floating-point precision.

### 3.4 Complexity Analysis

**Proposition 2** (Complexity Bounds). Let G be the number of compressed groups, F the total FE levels, and k the number of FE dimensions. The complexity of key operations is:

| Operation | Dense | Sparse | Speedup |
|-----------|-------|--------|---------|
| Matrix construction | O(G × F) | O(G × k) | F/k |
| M̃'WM̃ computation | O(G × F²) | O(G × k × F) | F/k |
| Memory | O(G × F) | O(G × k) | F/k |


*Proof.* For construction, dense requires initializing G × (F-1) entries while sparse requires recording G × k non-zero entries. For M̃'WM̃, dense computes G × (F-1)² products while sparse computes only products involving non-zero entries, of which there are G × k per column. Memory follows directly from storage requirements. □

For typical panel data with k = 2 (firm + year FEs) and F = 600, the speedup factor is approximately 300x for construction and memory, and 300x for the dominant matrix multiplication.

**Remark.** The final solve of (M̃'WM̃)β = M̃'Wỹ operates on a p × p matrix where p = kₓ + F - 1. This is typically small enough (< 1000) that dense Cholesky decomposition is efficient. The sparse advantage comes from constructing M̃'WM̃, not from solving the linear system.

## 4. Extension to Clustered Standard Errors

Wong et al. (2021) Section 5.3.1 describes within-cluster compression for clustered standard errors. The cluster identifier is added to the GROUP BY columns, ensuring each compressed group belongs to exactly one cluster. The meat matrix becomes:

$$\hat{\Xi} = \tilde{M}^\top \text{diag}(\tilde{e}^0) \tilde{W}_C \tilde{W}_C^\top \text{diag}(\tilde{e}^0) \tilde{M}$$

where W̃_C ∈ ℝᴳˣᶜ is the cluster indicator matrix with entry (g,c) = 1 if group g belongs to cluster c, and ẽ⁰ = ỹ⁰ - ñ ⊙ ŷ is the vector of summed residuals per group.

### 4.1 Sparse Cluster Matrix

The cluster matrix W̃_C is also sparse: each row has exactly one non-zero entry. We represent it as a sparse matrix, enabling vectorized score aggregation:

$$S = \tilde{W}_C^\top (\tilde{M} \odot \tilde{e}^0)$$

where S ∈ ℝᶜˣᵖ contains cluster scores as rows and ⊙ denotes element-wise multiplication (broadcasting ẽ⁰ across columns). The meat matrix is then:

$$\hat{\Xi} = S^\top S$$

This computes the sum of scores within each cluster using a single sparse matrix multiplication, avoiding explicit loops over C clusters. The complexity is O(G × p) for the sparse multiplication plus O(C × p²) for the outer product, compared to O(G × p) + O(C × G/C × p) = O(G × p) for the naive loop—similar asymptotically but with better cache behavior and vectorization.

### 4.2 Combined Sparse Operations

For clustered SEs with fixed effects, we use sparse matrices for both the design matrix M̃ (FE dummies) and the cluster matrix W̃_C. The full computation involves only sparse-dense and sparse-sparse operations until the final p × p matrices, which are small enough for dense linear algebra.


## 5. Empirical Results

We evaluate the sparse YOCO extension on synthetic panel data with two-way fixed effects. All benchmarks use a MacBook Pro with Apple M-series processor and 16GB RAM.

### 5.1 Data Generation

We generate synthetic panel data with:
- F₁ = 500 firm fixed effects
- F₂ = 100 year fixed effects  
- Binary treatment (30% treated)
- Discrete covariate x₁ ∈ {0, 1, 2}
- True coefficients: β_treatment = 2.5, β_x₁ = 1.5

The discrete regressors enable high compression ratios, representative of typical experimental and quasi-experimental settings.

### 5.2 Compression Performance

| Observations | Compressed Groups | Compression Ratio | Sparse Memory |
|--------------|-------------------|-------------------|---------------|
| 100,000 | 15,000 | 15% | 0.4 MB |
| 1,000,000 | 45,000 | 4.5% | 1.1 MB |
| 10,000,000 | 90,000 | 0.9% | 2.2 MB |
| 100,000,000 | 150,000 | 0.15% | 3.6 MB |

The compression ratio improves with dataset size because the number of unique (regressor, FE) combinations is bounded by the product of regressor values and FE levels, while observations grow without bound.

### 5.3 Runtime Comparison

We compare against pyfixest, a Python implementation of the fixest algorithm:

| Observations | Sparse YOCO | pyfixest | Speedup |
|--------------|-------------|----------|---------|
| 100,000 | 0.3s | 2.1s | 7x |
| 1,000,000 | 0.8s | 18.5s | 23x |
| 10,000,000 | 2.1s | 185s | 88x |
| 100,000,000 | 8.5s | OOM | ∞ |

Sparse YOCO successfully processes 100M observations on a laptop while pyfixest exhausts available memory. The speedup increases with dataset size due to improving compression ratios.

### 5.4 Numerical Validation

We verify numerical equivalence by comparing sparse YOCO against:
1. Dense YOCO implementation
2. Direct OLS on uncompressed data (for small datasets)

All comparisons show agreement to 10+ significant digits (relative error < 10⁻¹⁰), confirming the theoretical equivalence results.


## 6. Discussion

### 6.1 When Sparse YOCO is Most Beneficial

The sparse matrix extension provides the greatest benefit when:

1. **Discrete regressors**: Binary treatments, categorical variables, and binned continuous variables compress well under YOCO.

2. **Moderate FE cardinality**: F between 100 and 10,000 provides substantial sparsity benefits without excessive matrix dimensions.

3. **Large datasets**: Compression ratios improve with n, amplifying the benefits of operating on compressed data.

This describes the typical panel data scenario in applied economics: firm/year or individual/time fixed effects with discrete treatment indicators.

### 6.2 Limitations and Alternatives

The sparse YOCO approach is less beneficial when:

1. **Very high FE cardinality**: When F > 10,000, even sparse matrices become large. FWL demeaning (as in fixest/reghdfe) may be faster as it avoids constructing the FE dummy matrix entirely.

2. **Continuous regressors**: Poor compression ratios reduce YOCO benefits. Binning continuous variables can recover compression at the cost of some precision.

3. **Instrumental variables**: The current formulation does not extend to IV/2SLS, which requires the FWL approach.

A practical implementation should automatically select between sparse YOCO and FWL demeaning based on FE cardinality, using sparse YOCO when max(Fⱼ) ≤ 10,000 and Σⱼ Fⱼ ≤ 20,000.

### 6.3 Relationship to Existing Methods

**FWL Demeaning (fixest, reghdfe, lfe):** Iteratively demeans Y and X within FE groups until convergence. Complexity is O(n × k_fe × iterations) where iterations is typically 5-20. Does not reduce memory requirements but avoids constructing FE dummies.

**Direct Sparse Regression (lfe):** Some packages use sparse matrices for the full n × F design matrix. Sparse YOCO operates on the compressed G × F matrix where G << n, providing additional speedup from compression.

**Sufficient Statistics (YOCO):** Wong et al. (2021) describe the compression strategy but do not address efficient FE representation. Our sparse extension completes the picture for practical implementation.

## 7. Implementation

We implement sparse YOCO in leanfe, an open-source package available for both Python and R. The package provides:

- **Dual backends**: Polars (optimized for speed) and DuckDB (optimized for memory, supports datasets larger than RAM)
- **Automatic strategy selection**: Chooses between sparse YOCO and FWL demeaning based on FE cardinality
- **Full SE support**: IID, HC1, and clustered standard errors via the sparse formulation
- **Unified API**: Consistent interface across Python and R

The implementation uses scipy.sparse (Python) and Matrix::sparseMatrix (R) for sparse linear algebra. Source code and documentation are available at https://github.com/diegogentilepassaro/leanfe.


## 8. Conclusion

We extend the YOCO compression strategy of Wong et al. (2021) with sparse matrix representations for fixed effect dummy variables. This extension:

1. Reduces memory requirements by approximately 100x for typical panel data
2. Speeds up matrix operations by a factor proportional to F/k (often 100-300x)
3. Produces numerically identical results to dense implementations
4. Enables estimation of two-way FE models on 100M+ observations using single-machine computation

The key insight is that FE dummy matrices are inherently sparse (99%+ zeros), making sparse representations both natural and highly efficient. Combined with YOCO compression, this enables interactive analysis of large panel datasets that would otherwise require distributed computing infrastructure.

The approach is most beneficial for discrete regressors with moderate FE cardinality—precisely the setting common in experimental and quasi-experimental economics. For very high-cardinality FEs, FWL demeaning remains preferable. A practical implementation should automatically select the optimal strategy based on data characteristics.

## References

Bergé, L. (2018). Efficient estimation of maximum likelihood models with multiple fixed-effects: the R package fixest. *CREA Discussion Paper*.

Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372.

Correia, S. (2016). Linear models with high-dimensional fixed effects: An efficient and feasible estimator. *Working Paper*.

Eicker, F. (1967). Limit theorems for regressions with unequal and dependent errors. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 59-82.

Gaure, S. (2013). lfe: Linear group fixed effects. *The R Journal*, 5(2), 104-117.

Huber, P. J. (1967). The behavior of maximum likelihood estimates under nonstandard conditions. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 221-233.

White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838.

Wong, J., Forsell, E., Lewis, R., Mao, T., & Wardrop, M. (2021). You only compress once: Optimal data compression for estimating linear models. *arXiv preprint arXiv:2102.11297*.

Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. MIT Press.


---

## Appendix A: Proof of Complexity Bounds

### A.1 Sparse Matrix Construction

For each of G groups and k FE dimensions, we:
1. Look up the FE value: O(1)
2. Map to column index: O(1) with hash table
3. Append triplet to COO structure: O(1) amortized

Total: O(G × k)

Dense construction requires initializing G × (F-1) entries: O(G × F).

### A.2 Sparse Matrix Multiplication

Consider computing M̃'WM̃ where M̃ ∈ ℝᴳˣᵖ with p = kₓ + F - 1.

**Dense:** Each of p² output entries requires summing G weighted products. Total: O(G × p²) = O(G × F²).

**Sparse:** Let nnz denote non-zeros per row. Each row contributes to at most nnz² output entries. With nnz = kₓ + k (regressors plus FE indicators), total operations are O(G × (kₓ + k)²) = O(G × k²) since kₓ and k are small constants.

However, the output matrix has dimension p × p, so we must also account for accumulating into p² entries. The sparse multiplication touches O(G × k × p) = O(G × k × F) entries total.

### A.3 Memory Analysis

**Dense:** G × (F-1) floats × 8 bytes = 8G(F-1) bytes.

**Sparse (COO):** G × k triplets, each requiring 2 integers (row, col) + 1 float (value) = G × k × 16 bytes.

**Sparse (CSR):** G × k values + G × k column indices + (G+1) row pointers ≈ G × k × 12 bytes.

Ratio: 8G(F-1) / (12Gk) ≈ (F-1)/k ≈ F/k for large F.

## Appendix B: Numerical Stability

### B.1 Condition Number

The condition number of M̃'WM̃ depends on the correlation structure of regressors and FE dummies, not on whether sparse or dense representation is used. Both representations encode the same mathematical matrix.

### B.2 Solving the Normal Equations

We convert the p × p matrix (M̃'WM̃) to dense format before solving, since p is typically small (< 1000). We use Cholesky decomposition when the matrix is positive definite, falling back to pseudoinverse via SVD for ill-conditioned cases.

### B.3 Floating Point Considerations

Sparse and dense implementations use identical floating-point arithmetic. Differences arise only from operation ordering (e.g., summation order in matrix products), which can cause relative errors of O(ε_machine) ≈ 10⁻¹⁵ for double precision.

In all test cases, we observe agreement between sparse and dense implementations to relative tolerance 10⁻¹⁰, well within numerical precision bounds.
