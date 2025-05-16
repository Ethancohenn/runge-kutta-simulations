# test_search_theta.py

import numpy as np
from src.search_theta import brute_force_search

# Coarse scan over θ ∈ [0, 2π]
theta0, t0, thetas, times = brute_force_search(
    0.0,        # θ_min
    2*1.4125,  # θ_max
    11,        # number of samples
    T=3.0,
    h=0.001,
    tol=0.02,
    method='Dormand-Prince'
)

print(f"Coarse best: θ ≈ {theta0:.6f}, time ≈ {t0:.6f}")
