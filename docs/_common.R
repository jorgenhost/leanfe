# Common R setup for leanfe documentation
# Source this file at the beginning of R code blocks to load leanfe functions

# Configure reticulate to use system Python (for CI environments)
if (Sys.getenv("RETICULATE_PYTHON") != "") {
  library(reticulate)
  use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)
}

# Get the package R directory relative to docs
pkg_dir <- normalizePath(file.path(dirname(getwd()), "r", "R"), mustWork = FALSE)
if (!dir.exists(pkg_dir)) {
  # Try alternative path (when running from docs directory)
  pkg_dir <- normalizePath("../r/R", mustWork = FALSE)
}

# Source all R files
if (dir.exists(pkg_dir)) {
  source(file.path(pkg_dir, "common.R"))
  source(file.path(pkg_dir, "compress.R"))
  source(file.path(pkg_dir, "duckdb.R"))
  source(file.path(pkg_dir, "polars.R"))
  source(file.path(pkg_dir, "leanfe.R"))
} else {
  warning("Could not find leanfe R package directory")
}
