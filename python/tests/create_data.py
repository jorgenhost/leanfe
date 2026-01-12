import polars as pl
import polars_ds as pds
import polars.selectors as cs

import polars as pl
import polars_ds as pds
import polars.selectors as cs

def generate_fe_data(
    n_obs: int,
    n_firms: int,
    n_workers: int,
    n_cities: int,
    n_additional_regressors: int = 0,
    sigma_e: float = 2.0,          # Idiosyncratic error SD
    cluster_error_sd: float = 2.0, # Cluster-level error SD (High = High ICC)
    cluster_x_sd: float = 1.0,     # Cluster-level regressor SD (Crucial for Clustered SE)
    outlier_frac: float = 0.05,    
    seed: int = 42,
    filename: str = ""
):
    """
    Generates data where Clustered Standard Errors are necessary.
    
    Mechanics:
    1. Generates Cluster-level shocks for both ERRORS and REGRESSORS.
    2. Joins them to observation level.
    3. Ensures Cov(X_ig, X_jg) > 0 and Cov(e_ig, e_jg) > 0.
    """
    
    # 1. Setup IDs and Base Demographics
    # -------------------------------------------------------------------------
    lf = pl.LazyFrame().select(
        pl.int_range(0, n_obs, eager=False).alias("id")
    ).with_columns(
        firm_id = pds.random_int(0, n_firms),
        worker_id = pds.random_int(0, n_workers),
        city_id = pds.random_int(0, n_cities),
        age = pds.random_int(20, 65),
        education = pds.random_int(8, 20),
    ).with_columns(
        experience = (pl.col("age") - pl.col("education") - 6).clip(0, 50),
        age_sq = pl.col("age")**2
    )

    # 2. Generate Cluster-Level Components (The "Moulton" Factor)
    # -------------------------------------------------------------------------
    # To test clustered SEs, X and E must both be correlated within cluster.
    # We create a lookup table for firms to assign a fixed 'shock' to every worker in that firm.
    
    # Firm-level components
    lf_firms = pl.LazyFrame().select(
        pl.int_range(0, n_firms, eager=False).alias("firm_id")
    ).with_columns(
        # This part of the error is SHARED by everyone in the firm
        firm_error_component = pds.random_normal(0.0, cluster_error_sd),
        # This part of the regressor is SHARED (e.g., firm management quality)
        firm_x_component = pds.random_normal(0.0, cluster_x_sd) 
    )

    # City-level components
    lf_cities = pl.LazyFrame().select(
        pl.int_range(0, n_cities, eager=False).alias("city_id")
    ).with_columns(
        city_error_component = pds.random_normal(0.0, cluster_error_sd),
        city_x_component = pds.random_normal(0.0, cluster_x_sd)
    )

    # 3. Join Cluster Components to Main Data
    # -------------------------------------------------------------------------
    lf = lf.join(lf_firms, on="firm_id", how="left")
    lf = lf.join(lf_cities, on="city_id", how="left")

    # 4. Generate Regressors (Mixing Individual + Cluster Variation)
    # -------------------------------------------------------------------------
    # We generate X such that X_ij = X_individual + X_firm + X_city
    # This ensures X is correlated within clusters.
    
    if n_additional_regressors > 0:
        # Generate raw individual noise for Xs
        x_exprs = [
            (pds.random_normal(0.0, 1.0) + pl.col("firm_x_component") + pl.col("city_x_component")).alias(f"x{i}") 
            for i in range(n_additional_regressors)
        ]
        lf = lf.with_columns(x_exprs)

    # 5. Generate Total Error Term (Mixing Individual + Cluster Variation)
    # -------------------------------------------------------------------------
    # epsilon_ij = v_firm + v_city + e_ij
    
    lf = lf.with_columns(
        idiosyncratic_error = pds.random_normal(0.0, sigma_e)
    ).with_columns(
        total_error = pl.col("firm_error_component") + 
                      pl.col("city_error_component") + 
                      pl.col("idiosyncratic_error")
    )

    # Add outliers (inflate the idiosyncratic part only)
    if outlier_frac > 0:
        lf = lf.with_columns(
            is_outlier = pds.random_int(0, 1)
        ).with_columns(
            total_error = pl.when(pl.col("is_outlier")==1)
                            .then(pl.col("total_error") * 10.0)
                            .otherwise(pl.col("total_error"))
        ).drop("is_outlier")

    # 6. Construct Outcome (Log Wage)
    # -------------------------------------------------------------------------
    # Y = alpha + beta * X + Error
    
    # Base wage equation
    wage_eqn = (
        10.0 
        + 1.05 * pl.col("experience") 
        + 2.08 * pl.col("education")
        + 1.7 * pl.col("age")
        - 0.5 * pl.col("age_sq")
        + pl.col("total_error") # Contains the cluster shocks
    )
    
    # Add contribution of additional regressors (Beta = 1.0 for all x)
    if n_additional_regressors > 0:
        wage_eqn = wage_eqn + pl.sum_horizontal(cs.starts_with("x"))

    lf = lf.with_columns(
        log_wage = wage_eqn
    )

    # Cleanup: Drop intermediate calculation columns to keep file size realistic
    lf = lf.drop([
        "firm_error_component", "city_error_component", "idiosyncratic_error", 
        "firm_x_component", "city_x_component", "total_error"
    ])

    # 7. Execution
    # -------------------------------------------------------------------------
    lf.sink_parquet(filename)
    print(f"Saved to {filename}")

# --- RUN CONFIGURATION ---
if __name__ == "__main__":
    
    # 1. High Intra-Cluster Correlation (Good for testing Cluster SEs)
    # Low number of clusters (50) makes clustering critical.
    generate_fe_data(
    n_obs=15_000_000,
    n_firms=10_000,   # High dimensionality
    n_workers=2_000,  # High dimensionality
    n_cities=500,
        sigma_e=1.0,           # Low individual noise
        cluster_error_sd=2.5,  # High cluster noise (High ICC)
        cluster_x_sd=1.5,      # High cluster regressor correlation
    filename='data/data_hdfe.pq'
    )

    # 2. Huge dataset
    generate_fe_data(
        n_obs=80_000_000, 
        n_firms=50,        # Low dimensionality
        n_workers=500,     # Low dimensionality
        n_cities=20,
        filename='data/data_ldfe.pq'
    )
    # 3. ULTRA HDFE
    generate_fe_data(
        n_obs=15_000_000,
        n_firms=10_000,
        n_workers=2_000,
        n_cities=500,
        n_additional_regressors=16,  # Creates x5, x6, ..., x20
        filename='data/data_uhdfe.pq'
    )

    # 3. MEGA HDFE
    generate_fe_data(
        n_obs=50_000_000,
        n_firms=20_000,
        n_workers=4_000,
        n_cities=1_000,
        n_additional_regressors=10,  # Creates x5, x6, ..., x20
        filename='data/data_mega_fe.pq'
    )


    # 4. "NORMAL" for in-memory 
    generate_fe_data(
        n_obs=5_000_000,
        n_firms=100,
        n_workers=20,
        n_cities=10,
        filename='data/data_fe.pq'
    )