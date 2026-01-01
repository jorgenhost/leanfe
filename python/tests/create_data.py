# %%
import numpy as np
import polars as pl
# %%
# Example 1: Basic HDFE Usage
# Generate sample data
np.random.seed(42)
n_obs = 10_000_000
n_firms = 100_000
n_firms_low_dim = 10

n_workers = 5000
n_workers_low_dim = 5

n_city = 250

sample_frac = 0.5

# Generate firm and worker IDs  
firm_ids = np.random.randint(0, n_firms, n_obs)
firm_ids_low_dim = np.random.randint(0, n_firms_low_dim, n_obs)
worker_ids = np.random.randint(0, n_workers, n_obs)
worker_ids_low_dim = np.random.randint(0, n_workers_low_dim, n_obs)

city_ids = np.random.randint(0, n_city, n_obs)

# Generate firm and worker fixed effects
firm_effects = np.random.normal(0, 1, n_firms)
firm_effects_low_dim = np.random.normal(0, 1, n_firms_low_dim)

worker_effects = np.random.normal(0, 1, n_workers)
worker_effects_low_dim = np.random.normal(0, 1, n_workers_low_dim)

city_effects = np.random.normal(0, 1, n_workers)

# Generate continuous variables
X1 = np.random.normal(2, 1, n_obs)  # Experience
X2 = np.random.normal(0, 1, n_obs)  # Education
X3 = np.random.randint(30, 60, n_obs) # age
X4 = X3**2                             # age²

# True coefficients
beta_X1 = 0.05  # Return to experience
beta_X2 = 0.10  # Return to education
beta_X3 = 1.5   # age
beta_X4 = -0.2  # age²
# Generate dependent variable (log wages)
log_wage = (beta_X1 * X1 + beta_X2 * X2 + beta_X3 * X3 + beta_X4 * X4 +
           firm_effects[firm_ids] + worker_effects[worker_ids] 
        #    + city_effects[city_ids]  
            +
           np.random.normal(0, 1, n_obs))

log_wage_low_dim = (beta_X1 * X1 + beta_X2 * X2 + beta_X3 * X3 + beta_X4 * X4 +
           firm_effects_low_dim[firm_ids_low_dim] + worker_effects_low_dim[worker_ids_low_dim] 
        #    + city_effects[city_ids]  
            +
           np.random.normal(0, 1, n_obs))

# Create DataFrame
data_example = pl.LazyFrame({
    'log_wage': log_wage,
    'experience': X1,
    'education': X2, 
    'age': X3,
    'age_sq': X4,
    'firm_id': firm_ids,
    'worker_id': worker_ids,
    # 'city_id': city_ids
})

data_example_low_dim = pl.LazyFrame({
    'log_wage': log_wage_low_dim,
    'experience': X1,
    'education': X2, 
    'age': X3,
    'age_sq': X4,
    'firm_id': firm_ids_low_dim,
    'worker_id': worker_ids_low_dim,
    # 'city_id': city_ids
})

data_example_low_dim.sink_parquet(f'data_low_dim.pq')
data_example_low_dim.head(int(sample_frac * n_obs)).sink_parquet(f'data_low_dim_frac{sample_frac}.pq')

data_example.sink_parquet(f'data.pq')
data_example.head(int(sample_frac * n_obs)).sink_parquet(f'data_frac{sample_frac}.pq')