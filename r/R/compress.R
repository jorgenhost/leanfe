#' YOCO (You Only Compress Once) compression strategy for leanfe
#'
#' Implements the optimal data compression strategy from Wong et al. (2021):
#' "You Only Compress Once: Optimal Data Compression for Estimating Linear Models"
#'
#' Used automatically when vcov is "iid" or "HC1" (not cluster) and no IV.

#' @importFrom polars pl
#' @importFrom DBI dbGetQuery
#' @importFrom Matrix sparseMatrix crossprod
NULL


#' Check if compression strategy should be used
#'
#' @param vcov Variance-covariance type
#' @param has_instruments Whether IV/2SLS is being used
#' @return Logical
#' @keywords internal
.should_use_compress <- function(vcov, has_instruments) {
  vcov_ok <- tolower(vcov) %in% c("iid", "hc1")
  return(vcov_ok && !has_instruments)
}


#' Compress data using Polars GROUP BY
#'
#' @param df Polars DataFrame
#' @param y_col Dependent variable
#' @param x_cols Regressors
#' @param fe_cols Fixed effects
#' @param weights Weight column (optional)
#' @return List with compressed df and n_obs_original
#' @keywords internal
.compress_polars <- function(df, y_col, x_cols, fe_cols, weights = NULL) {
  group_cols <- c(x_cols, fe_cols)
  n_obs_original <- df$height
  
  if (!is.null(weights)) {
    agg_exprs <- list(
      pl$col(weights)$sum()$alias("_n"),
      (pl$col(y_col) * pl$col(weights))$sum()$alias("_sum_y"),
      (pl$col(y_col)$pow(2) * pl$col(weights))$sum()$alias("_sum_y_sq")
    )
  } else {
    agg_exprs <- list(
      pl$len()$alias("_n"),
      pl$col(y_col)$sum()$alias("_sum_y"),
      pl$col(y_col)$pow(2)$sum()$alias("_sum_y_sq")
    )
  }
  
  compressed <- df$group_by(group_cols)$agg(!!!agg_exprs)
  
  # Add mean_y and sqrt weights for WLS
  compressed <- compressed$with_columns(
    (pl$col("_sum_y") / pl$col("_n"))$alias("_mean_y"),
    pl$col("_n")$sqrt()$alias("_wts")
  )
  
  list(compressed = compressed, n_obs_original = n_obs_original)
}


#' Compress data using DuckDB SQL
#'
#' @param con DuckDB connection
#' @param y_col Dependent variable
#' @param x_cols Regressors
#' @param fe_cols Fixed effects
#' @param weights Weight column (optional)
#' @return List with compressed df and n_obs_original
#' @keywords internal
.compress_duckdb <- function(con, y_col, x_cols, fe_cols, weights = NULL) {
  group_cols <- c(x_cols, fe_cols)
  group_cols_sql <- paste(group_cols, collapse = ", ")
  
  n_obs_original <- dbGetQuery(con, "SELECT COUNT(*) FROM data")[[1]]
  
  if (!is.null(weights)) {
    query <- sprintf("
      SELECT
        %s,
        SUM(%s) AS _n,
        SUM(%s * %s) AS _sum_y,
        SUM(POWER(%s, 2) * %s) AS _sum_y_sq,
        SUM(%s * %s) / SUM(%s) AS _mean_y,
        SQRT(SUM(%s)) AS _wts
      FROM data
      GROUP BY %s
    ", group_cols_sql, weights, y_col, weights, y_col, weights, 
       y_col, weights, weights, weights, group_cols_sql)
  } else {
    query <- sprintf("
      SELECT
        %s,
        COUNT(*) AS _n,
        SUM(%s) AS _sum_y,
        SUM(POWER(%s, 2)) AS _sum_y_sq,
        SUM(%s) / COUNT(*) AS _mean_y,
        SQRT(COUNT(*)) AS _wts
      FROM data
      GROUP BY %s
    ", group_cols_sql, y_col, y_col, y_col, group_cols_sql)
  }
  
  compressed_df <- dbGetQuery(con, query)
  list(compressed = compressed_df, n_obs_original = n_obs_original)
}


#' Build design matrix from compressed data
#'
#' @param compressed_df Compressed data frame
#' @param x_cols Regressors
#' @param fe_cols Fixed effects
#' @param use_sparse Use sparse matrices for FE dummies (default TRUE)
#' @return List with X, Y, wts, all_col_names, n_fe_levels
#' @keywords internal
.build_design_matrix <- function(compressed_df, x_cols, fe_cols, use_sparse = TRUE) {
  # Convert to data.frame if polars
  if ("polars_data_frame" %in% class(compressed_df)) {
    pdf <- as.data.frame(compressed_df)
  } else {
    pdf <- compressed_df
  }
  
  n_rows <- nrow(pdf)
  
  # Extract regressors
  X_reg <- as.matrix(pdf[, x_cols, drop = FALSE])
  Y <- pdf[["_mean_y"]]
  wts <- pdf[["_wts"]]
  
  if (length(fe_cols) == 0) {
    return(list(X = X_reg, Y = Y, wts = wts, all_col_names = x_cols, n_fe_levels = 0))
  }
  
  if (use_sparse) {
    # Build sparse FE dummies efficiently using vectorized operations
    fe_col_names <- c()
    n_fe_levels <- 0
    col_offset <- 0
    
    # Pre-allocate lists for sparse matrix components
    all_rows <- vector("list", length(fe_cols))
    all_cols <- vector("list", length(fe_cols))
    
    for (idx in seq_along(fe_cols)) {
      fe <- fe_cols[idx]
      fe_values <- pdf[[fe]]
      categories <- sort(unique(fe_values))  # Sort for consistent ordering
      n_cats <- length(categories)
      n_fe_levels <- n_fe_levels + n_cats
      
      # Drop first category for identification
      cats_to_use <- categories[-1]
      
      # Add column names
      fe_col_names <- c(fe_col_names, paste0(fe, "_", cats_to_use))
      
      # Vectorized: create mapping and find matches
      # Map each value to its column index (0 if first category)
      cat_map <- match(fe_values, cats_to_use, nomatch = 0L)
      
      # Get rows where we have a match (not first category)
      valid_mask <- cat_map > 0L
      all_rows[[idx]] <- which(valid_mask)
      all_cols[[idx]] <- cat_map[valid_mask] + col_offset
      
      col_offset <- col_offset + n_cats - 1  # -1 because we drop first category
    }
    
    # Combine all sparse entries
    rows_list <- unlist(all_rows)
    cols_list <- unlist(all_cols)
    
    # Create sparse FE matrix
    n_fe_cols <- col_offset
    if (n_fe_cols > 0) {
      X_fe_sparse <- Matrix::sparseMatrix(
        i = rows_list,
        j = cols_list,
        x = rep(1, length(rows_list)),
        dims = c(n_rows, n_fe_cols)
      )
      
      # Combine: [X_reg | X_fe] as sparse
      X_reg_sparse <- Matrix::Matrix(X_reg, sparse = TRUE)
      X <- cbind(X_reg_sparse, X_fe_sparse)
    } else {
      X <- Matrix::Matrix(X_reg, sparse = TRUE)
    }
    
    all_col_names <- c(x_cols, fe_col_names)
    return(list(X = X, Y = Y, wts = wts, all_col_names = all_col_names, n_fe_levels = n_fe_levels))
    
  } else {
    # Original dense implementation
    fe_dummies <- list()
    fe_col_names <- c()
    n_fe_levels <- 0
    
    for (fe in fe_cols) {
      categories <- sort(unique(pdf[[fe]]))  # Sort for consistent ordering
      n_cats <- length(categories)
      n_fe_levels <- n_fe_levels + n_cats
      
      # Create dummies (drop first category for identification)
      for (cat in categories[-1]) {
        col_name <- paste0(fe, "_", cat)
        dummy <- as.numeric(pdf[[fe]] == cat)
        fe_dummies[[col_name]] <- dummy
        fe_col_names <- c(fe_col_names, col_name)
      }
    }
    
    # Combine regressors and FE dummies
    if (length(fe_dummies) > 0) {
      X_fe <- do.call(cbind, fe_dummies)
      X <- cbind(X_reg, X_fe)
      all_col_names <- c(x_cols, fe_col_names)
    } else {
      X <- X_reg
      all_col_names <- x_cols
    }
    
    list(X = X, Y = Y, wts = wts, all_col_names = all_col_names, n_fe_levels = n_fe_levels)
  }
}


#' Solve weighted least squares
#'
#' @param X Design matrix (dense or sparse)
#' @param Y Response vector (group means)
#' @param wts Weights (sqrt(n_g))
#' @return List with beta and XtX_inv
#' @keywords internal
.solve_wls <- function(X, Y, wts) {
  is_sparse <- inherits(X, "sparseMatrix") || inherits(X, "dgCMatrix")
  
  if (is_sparse) {
    # Sparse weighted least squares
    # Weight the design matrix: diag(wts) @ X
    Xw <- X * wts  # Matrix package handles this
    Yw <- Y * wts
    
    # X'X - convert to dense for solve (p x p is typically small)
    XtX <- as.matrix(Matrix::crossprod(Xw))
    Xty <- as.vector(Matrix::crossprod(Xw, Yw))
    
    # Solve using dense methods
    XtX_inv <- tryCatch({
      chol2inv(chol(XtX))
    }, error = function(e) {
      solve(XtX)
    })
    
    beta <- XtX_inv %*% Xty
  } else {
    # Dense weighted least squares
    Xw <- X * wts
    Yw <- Y * wts
    
    XtX <- crossprod(Xw)
    Xty <- crossprod(Xw, Yw)
    
    XtX_inv <- tryCatch({
      chol2inv(chol(XtX))
    }, error = function(e) {
      solve(XtX)
    })
    
    beta <- XtX_inv %*% Xty
  }
  
  list(beta = as.vector(beta), XtX_inv = XtX_inv)
}


#' Compute RSS from sufficient statistics
#'
#' @param compressed_df Compressed data
#' @param X Design matrix (dense or sparse)
#' @param beta Coefficients
#' @return List with rss_total and rss_g
#' @keywords internal
.compute_rss_grouped <- function(compressed_df, X, beta) {
  # Convert to data.frame if polars
  if ("polars_data_frame" %in% class(compressed_df)) {
    pdf <- as.data.frame(compressed_df)
  } else {
    pdf <- compressed_df
  }
  
  n_g <- pdf[["_n"]]
  sum_y_g <- pdf[["_sum_y"]]
  sum_y_sq_g <- pdf[["_sum_y_sq"]]
  
  # Fitted values for each group (handle sparse)
  yhat_g <- as.vector(X %*% beta)
  
  # Per-group RSS
  rss_g <- sum_y_sq_g - 2 * yhat_g * sum_y_g + n_g * (yhat_g^2)
  rss_total <- sum(rss_g)
  
  list(rss_total = rss_total, rss_g = rss_g)
}


#' Compute standard errors from compressed data
#'
#' @param XtX_inv Inverse of X'X
#' @param rss_total Total RSS
#' @param rss_g Per-group RSS
#' @param n_obs Original number of observations
#' @param df_resid Residual degrees of freedom
#' @param vcov "iid" or "HC1"
#' @param X Design matrix (dense or sparse)
#' @param k_x Number of regressor columns
#' @return Standard errors for x_cols only
#' @keywords internal
.compute_se_compress <- function(XtX_inv, rss_total, rss_g, n_obs, df_resid, vcov, X, k_x) {
  is_sparse <- inherits(X, "sparseMatrix") || inherits(X, "dgCMatrix")
  
  if (tolower(vcov) == "iid") {
    sigma2 <- rss_total / df_resid
    se_full <- sqrt(diag(XtX_inv) * sigma2)
  } else if (tolower(vcov) == "hc1") {
    # Meat matrix: X' diag(rss_g) X
    if (is_sparse) {
      Xw <- X * rss_g
      meat <- as.matrix(Matrix::crossprod(X, Xw))
    } else {
      meat <- crossprod(X, X * rss_g)
    }
    vcov_matrix <- XtX_inv %*% meat %*% XtX_inv
    # HC1 adjustment
    adjustment <- n_obs / df_resid
    se_full <- sqrt(diag(vcov_matrix) * adjustment)
  } else {
    stop(sprintf("vcov must be 'iid' or 'HC1' for compress strategy, got '%s'", vcov))
  }
  
  # Return only SEs for x_cols (not FE dummies)
  se_full[1:k_x]
}


#' Run compressed regression using Polars backend
#'
#' @param df Polars DataFrame
#' @param y_col Dependent variable
#' @param x_cols Regressors
#' @param fe_cols Fixed effects
#' @param weights Weight column (optional)
#' @param vcov "iid" or "HC1"
#' @return Regression results list
#' @keywords internal
.leanfe_compress_polars <- function(df, y_col, x_cols, fe_cols, weights = NULL, vcov = "iid") {
  # Compress data
  comp_result <- .compress_polars(df, y_col, x_cols, fe_cols, weights)
  compressed <- comp_result$compressed
  n_obs <- comp_result$n_obs_original
  n_compressed <- compressed$height
  
  # Build design matrix
  dm_result <- .build_design_matrix(compressed, x_cols, fe_cols)
  X <- dm_result$X
  Y <- dm_result$Y
  wts <- dm_result$wts
  all_cols <- dm_result$all_col_names
  
  # Solve WLS
  wls_result <- .solve_wls(X, Y, wts)
  beta <- wls_result$beta
  XtX_inv <- wls_result$XtX_inv
  
  # Compute RSS
  rss_result <- .compute_rss_grouped(compressed, X, beta)
  rss_total <- rss_result$rss_total
  rss_g <- rss_result$rss_g
  
  # Degrees of freedom
  p <- length(all_cols)
  df_resid <- n_obs - p
  
  # Standard errors
  se <- .compute_se_compress(XtX_inv, rss_total, rss_g, n_obs, df_resid, vcov, X, length(x_cols))
  
  # Extract coefficients for x_cols only
  beta_x <- beta[seq_along(x_cols)]
  
  list(
    coefficients = setNames(beta_x, x_cols),
    std_errors = setNames(se, x_cols),
    n_obs = n_obs,
    n_compressed = n_compressed,
    compression_ratio = n_compressed / n_obs,
    vcov_type = vcov,
    strategy = "compress",
    df_resid = df_resid,
    rss = rss_total
  )
}


#' Run compressed regression using DuckDB backend
#'
#' @param con DuckDB connection
#' @param y_col Dependent variable
#' @param x_cols Regressors
#' @param fe_cols Fixed effects
#' @param weights Weight column (optional)
#' @param vcov "iid" or "HC1"
#' @return Regression results list
#' @keywords internal
.leanfe_compress_duckdb <- function(con, y_col, x_cols, fe_cols, weights = NULL, vcov = "iid") {
  # Compress data
  comp_result <- .compress_duckdb(con, y_col, x_cols, fe_cols, weights)
  compressed <- comp_result$compressed
  n_obs <- comp_result$n_obs_original
  n_compressed <- nrow(compressed)
  
  # Build design matrix
  dm_result <- .build_design_matrix(compressed, x_cols, fe_cols)
  X <- dm_result$X
  Y <- dm_result$Y
  wts <- dm_result$wts
  all_cols <- dm_result$all_col_names
  
  # Solve WLS
  wls_result <- .solve_wls(X, Y, wts)
  beta <- wls_result$beta
  XtX_inv <- wls_result$XtX_inv
  
  # Compute RSS
  rss_result <- .compute_rss_grouped(compressed, X, beta)
  rss_total <- rss_result$rss_total
  rss_g <- rss_result$rss_g
  
  # Degrees of freedom
  p <- length(all_cols)
  df_resid <- n_obs - p
  
  # Standard errors
  se <- .compute_se_compress(XtX_inv, rss_total, rss_g, n_obs, df_resid, vcov, X, length(x_cols))
  
  # Extract coefficients for x_cols only
  beta_x <- beta[seq_along(x_cols)]
  
  list(
    coefficients = setNames(beta_x, x_cols),
    std_errors = setNames(se, x_cols),
    n_obs = n_obs,
    n_compressed = n_compressed,
    compression_ratio = n_compressed / n_obs,
    vcov_type = vcov,
    strategy = "compress",
    df_resid = df_resid,
    rss = rss_total
  )
}
