"""
Compute time to reach a fixed target under the gravity problem.

"""

import numpy as np
from .simulation import simulate_fixed_steps
from .problems import gravite

def compute_flight_time(theta,
                        f=gravite,
                        T=3.0,
                        h=0.001,
                        target=(4.0, 1.0),
                        tol=0.02,
                        method="Dormand-Prince"):
    """
    Returns the time it takes for a particle launched at angle theta to
    reach within `tol` distance of `target`, or np.inf if not reached by T.
    """
    # initial state: (x, y, vx, vy)
    y0 = np.array([0.0, 0.0, np.cos(theta), np.sin(theta)])
    times, Y = simulate_fixed_steps(f, y0, T=T, h=h, method=method)
    # extract positions
    positions = Y[:, :2]  # shape (n_steps+1, 2)
    # compute distances to target
    dists = np.linalg.norm(positions - np.array(target), axis=1)
    # find first index within tol
    idx = np.where(dists <= tol)[0]
    if idx.size > 0:
        return times[idx[0]]
    else:
        return np.inf

def flight_time_curve(thetas, **kwargs):
    """
    Compute flight times for each theta in `thetas`.
    Returns an array of same shape as `thetas`.
    Pass any of the compute_flight_time kwargs to control T, h, tol, etc.
    """
    times = np.array([compute_flight_time(theta, **kwargs) for theta in thetas])
    return times
