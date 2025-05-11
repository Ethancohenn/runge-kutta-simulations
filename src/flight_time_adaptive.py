"""
Compute flight time to target using adaptive‐step RK.
"""

import numpy as np
from .rk_adaptive import rk_adaptive
from .problems import gravite

def compute_flight_time_adaptive(theta,
                                 f=gravite,
                                 T=3.0,
                                 h_init=0.001,
                                 tol_local=1e-3,
                                 tol_target=0.02,
                                 method="Dormand-Prince"):
    """
    Returns the time to reach within tol_target of (4,1)
    using an adaptive RK integrator to control local error.

    Returns
    -------
    t_reach : float
        Time at which the trajectory first enters the tol_target
        neighborhood, or np.inf if it never does before T.
    """
    # initial state
    y0 = np.array([0.0, 0.0, np.cos(theta), np.sin(theta)])

    from .butcher_tableaus import Adict, Bdict, Cdict
    A = Adict[method]
    B = Bdict[method]
    C = Cdict[method]

    # run adaptive integrator
    Y, H = rk_adaptive(y0, f, A, B, C, T, h_init, tol_local)

    times = np.cumsum(H)  # approximate times of each step
    positions = np.array(Y)[:, :2]  # x,y positions

    dists = np.linalg.norm(positions - np.array([4.0, 1.0]), axis=1)
    idx = np.where(dists <= tol_target)[0]
    if idx.size > 0:
        return times[idx[0]]
    else:
        return np.inf
