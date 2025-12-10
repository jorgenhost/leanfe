#!/usr/bin/env Rscript
# Test that sparse and dense matrix implementations produce identical results
#
# This verifies that our sparse matrix optimization doesn't change the numerical
# results compared to a dense matrix implementation.

library(polars)
library(Matrix)
source("R/common.R")
source("R/compress.R")

set.seed(42)

cat("Testing sparse vs dense matrix equivalence...\n\n")

# Test 1: Coefficients are identical
cat("Test 1: Sparse vs dense coefficients identical... ")

n <- 1000
df <- pl$DataFrame(
  y = rnorm(n, 10, 2),
  x1 = rnorm(n),
  x2 = rnorm(n, 5, 2),
  fe1 = rep(0:49, each = 20),
  fe2 = rep(0:19, 50)
)

# Compress data
comp_result <- .compress_polars(df, "y", c("x1", "x2"), c("fe1", "fe2"))
compressed <- comp_result$compressed

# Build design matrix - sparse
dm_sparse <- .build_design_matrix(compressed, c("x1", "x2"), c("fe1", "fe2"), use_sparse = TRUE)
X_sparse <- dm_sparse$X
Y <- dm_sparse$Y
wts <- dm_sparse$wts

# Build design matrix - dense
dm_dense <- .build_design_matrix(compressed, c("x1", "x2"), c("fe1", "fe2"), use_sparse = FALSE)
X_dense <- dm_dense$X

# Solve WLS
wls_sparse <- .solve_wls(X_sparse, Y, wts)
wls_dense <- .solve_wls(X_dense, Y, wts)

# Compare coefficients (first 2 are x1, x2)
beta_sparse <- wls_sparse$beta[1:2]
beta_dense <- wls_dense$beta[1:2]

stopifnot(all(abs(beta_sparse - beta_dense) < 1e-10))
cat("PASSED\n")

# Test 2: XtX_inv is identical
cat("Test 2: Sparse vs dense XtX_inv identical... ")

XtX_inv_sparse <- wls_sparse$XtX_inv
XtX_inv_dense <- wls_dense$XtX_inv

stopifnot(all(abs(XtX_inv_sparse - XtX_inv_dense) < 1e-10))
cat("PASSED\n")

# Test 3: RSS is identical
cat("Test 3: Sparse vs dense RSS identical... ")

rss_sparse <- .compute_rss_grouped(compressed, X_sparse, wls_sparse$beta)
rss_dense <- .compute_rss_grouped(compressed, X_dense, wls_dense$beta)

stopifnot(abs(rss_sparse$rss_total - rss_dense$rss_total) < 1e-8)
stopifnot(all(abs(rss_sparse$rss_g - rss_dense$rss_g) < 1e-8))
cat("PASSED\n")

# Test 4: IID SEs are identical
cat("Test 4: Sparse vs dense IID SEs identical... ")

n_obs <- comp_result$n_obs_original
df_resid <- n_obs - length(dm_sparse$all_col_names)

se_iid_sparse <- .compute_se_compress(
  XtX_inv_sparse, rss_sparse$rss_total, rss_sparse$rss_g,
  n_obs, df_resid, "iid", X_sparse, 2
)

se_iid_dense <- .compute_se_compress(
  XtX_inv_dense, rss_dense$rss_total, rss_dense$rss_g,
  n_obs, df_resid, "iid", X_dense, 2
)

stopifnot(all(abs(se_iid_sparse$se - se_iid_dense$se) < 1e-10))
cat("PASSED\n")

# Test 5: HC1 SEs are identical
cat("Test 5: Sparse vs dense HC1 SEs identical... ")

se_hc1_sparse <- .compute_se_compress(
  XtX_inv_sparse, rss_sparse$rss_total, rss_sparse$rss_g,
  n_obs, df_resid, "HC1", X_sparse, 2
)

se_hc1_dense <- .compute_se_compress(
  XtX_inv_dense, rss_dense$rss_total, rss_dense$rss_g,
  n_obs, df_resid, "HC1", X_dense, 2
)

stopifnot(all(abs(se_hc1_sparse$se - se_hc1_dense$se) < 1e-10))
cat("PASSED\n")

# Test 6: Cluster SEs are identical
cat("Test 6: Sparse vs dense cluster SEs identical... ")

# Create data with clusters
df_cluster <- pl$DataFrame(
  y = rnorm(600, 10, 2),
  x = rnorm(600),
  fe1 = rep(0:29, each = 20),
  cluster = rep(0:29, each = 20)
)

comp_cluster <- .compress_polars(df_cluster, "y", "x", "fe1", cluster_col = "cluster")
compressed_c <- comp_cluster$compressed

dm_sparse_c <- .build_design_matrix(compressed_c, "x", "fe1", use_sparse = TRUE)
dm_dense_c <- .build_design_matrix(compressed_c, "x", "fe1", use_sparse = FALSE)

wls_sparse_c <- .solve_wls(dm_sparse_c$X, dm_sparse_c$Y, dm_sparse_c$wts)
wls_dense_c <- .solve_wls(dm_dense_c$X, dm_dense_c$Y, dm_dense_c$wts)

rss_sparse_c <- .compute_rss_grouped(compressed_c, dm_sparse_c$X, wls_sparse_c$beta)
rss_dense_c <- .compute_rss_grouped(compressed_c, dm_dense_c$X, wls_dense_c$beta)

# Get cluster IDs and e0_g
pdf <- as.data.frame(compressed_c)
cluster_ids <- pdf[["cluster"]]
n_g <- pdf[["_n"]]
sum_y_g <- pdf[["_sum_y"]]
yhat_sparse <- as.vector(dm_sparse_c$X %*% wls_sparse_c$beta)
yhat_dense <- as.vector(dm_dense_c$X %*% wls_dense_c$beta)
e0_g_sparse <- sum_y_g - n_g * yhat_sparse
e0_g_dense <- sum_y_g - n_g * yhat_dense

n_obs_c <- comp_cluster$n_obs_original
df_resid_c <- n_obs_c - length(dm_sparse_c$all_col_names)

se_cluster_sparse <- .compute_se_compress(
  wls_sparse_c$XtX_inv, rss_sparse_c$rss_total, rss_sparse_c$rss_g,
  n_obs_c, df_resid_c, "cluster", dm_sparse_c$X, 1,
  cluster_ids = cluster_ids, e0_g = e0_g_sparse
)

se_cluster_dense <- .compute_se_compress(
  wls_dense_c$XtX_inv, rss_dense_c$rss_total, rss_dense_c$rss_g,
  n_obs_c, df_resid_c, "cluster", dm_dense_c$X, 1,
  cluster_ids = cluster_ids, e0_g = e0_g_dense
)

stopifnot(se_cluster_sparse$n_clusters == se_cluster_dense$n_clusters)
stopifnot(all(abs(se_cluster_sparse$se - se_cluster_dense$se) < 1e-10))
cat("PASSED\n")

cat("\n✅ All 6 sparse vs dense equivalence tests passed!\n")
