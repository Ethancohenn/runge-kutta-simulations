"""
Generic simulation routines for fixed-step RK integration applied to any of the problems in problems.py.
"""

import numpy as np
from .butcher_tableaus import Adict, Bdict, Cdict
from .rk_general import rk_fixed_h

def simulate_fixed_steps(f, y0, T, h, method):
    """
    Integrate dy/dt = f(t, y) from t=0 to T with fixed step size h
    using the Runge–Kutta variant 'method'.

    Returns
    -------
    times : ndarray, shape (n_steps+1,)
        Sequence of time points from 0 to T.
    Y : ndarray, shape (n_steps+1, dim_y)
        Solution at each time point.
    """

    A = Adict[method]
    B = Bdict[method]
    C = Cdict[method]

    yN, Y_list = rk_fixed_h(y0, f, A, B, C, T, h)
    times = np.linspace(0.0, T, len(Y_list))
    return times, np.vstack(Y_list)
