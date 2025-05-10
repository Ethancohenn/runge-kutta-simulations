# test_flight_time.py

from .flight_time import compute_flight_time
import numpy as np

# Test a single angle
theta_test = 6.0083
t = compute_flight_time(theta_test)
print(f"Time to reach target for θ={theta_test:.4f} → t={t:.4f}")

# Test multiple angles
thetas = np.array([5.5, 6.0, -0.27489])
for th in thetas:
    tm = compute_flight_time(th)
    print(f"θ={th:.2f} → t={tm:.4f}")