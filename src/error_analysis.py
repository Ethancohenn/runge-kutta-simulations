"""
 Compute numerical errors vs. time step and estimate convergence order

"""

import numpy as np
import math
from .butcher_tableaus import Adict, Bdict, Cdict
from .rk_general import rk_fixed_h

def exponential_rhs(t, y):
    """Test ODE: y' = y."""
    return y

def compute_errors(y0, T, method, pas_list):
    """
    Compute the error |y_num(T)-y_exact(T)| for the exponential problem
    using a given RK method.

    Returns
    -------
    errors : np.ndarray
        Absolute errors at each h
    """
    errors = np.zeros(len(pas_list))
    # exact solution at T
    y_exact = math.exp(T)
    A = Adict[method]
    B = Bdict[method]
    C = Cdict[method]

    for idx, h in enumerate(pas_list):
        # integrate
        yN, Y = rk_fixed_h(y0, exponential_rhs, A, B, C, T, h)
        y_num = Y[-1][0]
        errors[idx] = abs(y_exact - y_num)

    return errors

def estimate_order(hs, errors):
    """
    Estimate convergence order p by:
      1) dropping errors ≤ machine epsilon,
      2) picking the two largest h among the rest,
      3) computing p = log(e1/e2)/log(h1/h2) (we know that log(errors) ≈ p * log(hs) + C)
    """
    eps = np.finfo(float).eps
    # keep only “significant” errors
    mask = errors > eps
    hs_sig = hs[mask]
    err_sig = errors[mask]

    if len(err_sig) < 2:
        return float("nan"), float("nan")

    # pick the two largest step sizes
    idx_desc = np.argsort(-hs_sig)
    i1, i2 = idx_desc[0], idx_desc[1]

    # compute slope p and intercept c
    p = np.log(err_sig[i1] / err_sig[i2]) / np.log(hs_sig[i1] / hs_sig[i2])
    c = np.log(err_sig[i1]) - p * np.log(hs_sig[i1])
    return p, c

# Example usage
if __name__ == "__main__":
    pas_list = np.array([2**(-i-1) for i in range(1, 5)])
    for method in ["Euler-Heun", "Bogacki–Shampine", "Dormand-Prince", "7-8"]:
        errs = compute_errors(np.array([1]), 1.0, method, pas_list)
        order, _ = estimate_order(pas_list, errs)
        print(f"{method}: errors {errs}, estimated order ~= {abs(order):.2f}")
