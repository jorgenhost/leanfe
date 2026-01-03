import numpy as np
import polars as pl

def generate_fe_data(n_obs, n_firms, n_workers, n_cities, n_additional_regressors=0, filename = None):
    """
    Generate ultra (potentially high-dimensional) fixed effects panel data with deterministic coefficients.

    Base model has 4 regressors (experience, education, age, age_sq) with original coefficients.
    Additional regressors x5, x6, ... have coefficients 1, 2, 3, ... respectively.

    Parameters
    ----------
    n_obs : int
        Number of observations
    n_firms : int
        Number of unique firms
    n_workers : int
        Number of unique workers
    n_cities : int
        Number of unique cities
    n_additional_regressors : int, default 0
        Number of additional regressors beyond the base 4
    filename : str, optional
        Output parquet filename
    """
    np.random.seed(42)

    # 1. Generate IDs
    firm_ids = np.random.randint(0, n_firms, n_obs, dtype=np.int32)
    worker_ids = np.random.randint(0, n_workers, n_obs, dtype=np.int32)
    city_ids = np.random.randint(0, n_cities, n_obs, dtype=np.int32)

    # 2. Generate Effects
    firm_effects = np.random.normal(0, 1, n_firms).astype(np.float32)
    worker_effects = np.random.normal(0, 1, n_workers).astype(np.float32)
    city_effects = np.random.normal(0, 1, n_cities).astype(np.float32)

    # 3. Generate Base Covariates (original 4)
    X1 = np.random.normal(2, 1, n_obs).astype(np.float32)  # Experience
    X2 = np.random.normal(0, 1, n_obs).astype(np.float32)  # Education
    X3 = np.random.randint(30, 60, n_obs, dtype=np.int32)  # Age
    X4 = (X3**2).astype(np.int32)                          # Age^2

    # 4. Generate Additional Covariates with deterministic coefficients
    X_dict = {
        'experience': X1,
        'education': X2,
        'age': X3,
        'age_sq': X4
    }

    # Base effects
    X_effects = (0.05 * X1) + (0.10 * X2) + (1.5 * X3) + (-0.2 * X4)

    # Additional regressors: x5 has coef=1, x6 has coef=2, etc.
    for i in range(n_additional_regressors):
        X_col = np.random.normal(0, 1, n_obs).astype(np.float32)
        X_dict[f'x{i+5}'] = X_col  # Start from x5
        coef = i + 1  # Coefficient is 1, 2, 3, ...
        X_effects += coef * X_col

    # 5. Calculate Dependent Variable
    log_wage = (
        X_effects +
        firm_effects[firm_ids] +
        worker_effects[worker_ids] +
        city_effects[city_ids] +
        np.random.normal(0, 1, n_obs).astype(np.float32)
    )

    # 6. Create DataFrame
    data_dict = {
        'log_wage': log_wage,
        **X_dict,
        'firm_id': firm_ids,
        'worker_id': worker_ids,
        'city_id': city_ids
    }

    lf = pl.LazyFrame(data_dict)

    lf.sink_parquet(filename)

    print(f"Saved {filename}.")


# --- EXECUTION ---


# 1. Low-FE Version: Small number of groups 
generate_fe_data(
    n_obs=50_000_000, 
    n_firms=50,        # Low dimensionality
    n_workers=500,     # Low dimensionality
    n_cities=20, 
    filename='data/data_ldfe.pq'
)

# 2. High-FE Version: 
generate_fe_data(
    n_obs=10_000_000,
    n_firms=10_000,   # High dimensionality
    n_workers=2_000, # High dimensionality
    n_cities=500,
    filename='data/data_hdfe.pq'
)


# 3. ULTRA HDF
generate_fe_data(
    n_obs=10_000_000,
    n_firms=10_000,
    n_workers=2_000,
    n_cities=500,
    n_additional_regressors=16,  # Creates x5, x6, ..., x20
    filename='data/data_uhdfe.pq'
)

# 4. "NORMAL" for in-memory 
generate_fe_data(
    n_obs=1_000_000,
    n_firms=100,
    n_workers=20,
    n_cities=10,
    filename='data/data_fe.pq'
)