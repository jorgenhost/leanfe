# Tests for YOCO compression with clustered standard errors
#
# Implements Section 5.3.1 of Wong et al. (2021):
# "You Only Compress Once: Optimal Data Compression for Estimating Linear Models"
#
# The key insight is that for clustered SEs with compression:
# 1. Include cluster identifier in the grouping (within-cluster compression)
# 2. Compute e0_tilde = y0_tilde - n_tilde * yhat (sum of residuals per group)
# 3. Build meat matrix using sparse cluster matrix

library(testthat)
library(Matrix)

# Source the package files
source("R/common.R")
source("R/compress.R")

test_that("sparse cluster matrix builder works correctly", {
  cluster_ids <- c("A", "A", "B", "B", "C", "A", "C")
  result <- .build_sparse_cluster_matrix(cluster_ids)
  
  expect_equal(result$n_clusters, 3)
  expect_equal(dim(result$W_C), c(7, 3))
  
  # Each row should have exactly one 1
  row_sums <- Matrix::rowSums(result$W_C)
  expect_true(all(row_sums == 1))
  
  # Column sums should match cluster sizes
  col_sums <- Matrix::colSums(result$W_C)
  expect_equal(sort(col_sums), c(2, 2, 3))
})


test_that("cluster SE computation produces valid results", {
  set.seed(42)
  n_groups <- 100
  n_clusters <- 20
  p <- 3
  
  # Create mock compressed data
  X <- matrix(rnorm(n_groups * p), n_groups, p)
  Y <- rnorm(n_groups)
  wts <- sqrt(runif(n_groups, 1, 10))
  cluster_ids <- sample(1:n_clusters, n_groups, replace = TRUE)
  
  # Solve WLS
  Xw <- X * wts
  Yw <- Y * wts
  XtX <- crossprod(Xw)
  Xty <- crossprod(Xw, Yw)
  XtX_inv <- solve(XtX)
  beta <- XtX_inv %*% Xty
  
  # Compute RSS
  yhat <- X %*% beta
  rss_g <- (Y - yhat)^2 * wts^2
  rss_total <- sum(rss_g)
  
  # Compute e0_g (sum of residuals per group)
  n_g <- wts^2
  sum_y_g <- Y * n_g
  e0_g <- sum_y_g - n_g * as.vector(yhat)
  
  # Test cluster SE computation
  n_obs <- sum(n_g)
  df_resid <- n_obs - p
  
  se_result <- .compute_se_compress(
    XtX_inv = XtX_inv,
    rss_total = rss_total,
    rss_g = rss_g,
    n_obs = n_obs,
    df_resid = df_resid,
    vcov = "cluster",
    X = X,
    k_x = p,
    cluster_ids = cluster_ids,
    e0_g = e0_g,
    ssc = FALSE
  )
  
  # Check results
  expect_equal(length(se_result$se), p)
  expect_equal(se_result$n_clusters, n_clusters)
  expect_true(all(se_result$se > 0))
  expect_true(all(!is.na(se_result$se)))
})


test_that("SSC adjustment produces larger SEs", {
  set.seed(123)
  n_groups <- 50
  n_clusters <- 10
  p <- 2
  
  X <- matrix(rnorm(n_groups * p), n_groups, p)
  Y <- rnorm(n_groups)
  cluster_ids <- sample(1:n_clusters, n_groups, replace = TRUE)
  
  XtX <- crossprod(X)
  XtX_inv <- solve(XtX)
  beta <- XtX_inv %*% crossprod(X, Y)
  
  yhat <- X %*% beta
  rss_g <- (Y - yhat)^2
  rss_total <- sum(rss_g)
  e0_g <- Y - as.vector(yhat)
  
  n_obs <- n_groups
  df_resid <- n_obs - p
  
  # Without SSC
  se_no_ssc <- .compute_se_compress(
    XtX_inv = XtX_inv, rss_total = rss_total, rss_g = rss_g,
    n_obs = n_obs, df_resid = df_resid, vcov = "cluster",
    X = X, k_x = p, cluster_ids = cluster_ids, e0_g = e0_g, ssc = FALSE
  )
  
  # With SSC
  se_ssc <- .compute_se_compress(
    XtX_inv = XtX_inv, rss_total = rss_total, rss_g = rss_g,
    n_obs = n_obs, df_resid = df_resid, vcov = "cluster",
    X = X, k_x = p, cluster_ids = cluster_ids, e0_g = e0_g, ssc = TRUE
  )
  
  # SSC should produce larger SEs
  expect_true(all(se_ssc$se >= se_no_ssc$se))
})


test_that("IID and HC1 SEs still work with new function signature", {
  set.seed(456)
  n_groups <- 50
  p <- 2
  
  X <- matrix(rnorm(n_groups * p), n_groups, p)
  Y <- rnorm(n_groups)
  
  XtX <- crossprod(X)
  XtX_inv <- solve(XtX)
  beta <- XtX_inv %*% crossprod(X, Y)
  
  yhat <- X %*% beta
  rss_g <- (Y - yhat)^2
  rss_total <- sum(rss_g)
  
  n_obs <- n_groups
  df_resid <- n_obs - p
  
  # IID
  se_iid <- .compute_se_compress(
    XtX_inv = XtX_inv, rss_total = rss_total, rss_g = rss_g,
    n_obs = n_obs, df_resid = df_resid, vcov = "iid",
    X = X, k_x = p
  )
  expect_equal(length(se_iid$se), p)
  expect_true(all(se_iid$se > 0))
  expect_null(se_iid$n_clusters)
  
  # HC1
  se_hc1 <- .compute_se_compress(
    XtX_inv = XtX_inv, rss_total = rss_total, rss_g = rss_g,
    n_obs = n_obs, df_resid = df_resid, vcov = "HC1",
    X = X, k_x = p
  )
  expect_equal(length(se_hc1$se), p)
  expect_true(all(se_hc1$se > 0))
  expect_null(se_hc1$n_clusters)
})


# Tests complete - all test_that blocks above run automatically
