"""
 Runge–Kutta with adaptive time step

"""

import numpy as np
from .rk_general import somme

def updateRK_pas_adaptatif(y0, h, A, B, C, f, t):
    """
    Single adaptive RK step
    
    Returns
    -------
    y1 : ndarray
        Primary solution at t + h
    err : float
        ||y1 - z1||_2, estimate of local error
    """
    K=np.zeros((np.shape(A)[0],np.shape(y0)[0]))
    for i in range(len(K)):
        ti = t + C[i] * h
        yi = y0 + h * somme(A[i],K)
        K[i] = f(ti, yi)

    # Primary solution
    y1 = y0 + h * np.dot(B[0], K)
    # Embedded solution
    z1 = y0 + h * np.dot(B[1], K)

    err = np.linalg.norm(y1 - z1)
    return y1, err


def rk_adaptive(y0, f, A, B, C, T, h_init, tolerance, max_iter=100000):
    """
    Adaptive-step integrator

    Uses updateRK_pas_adaptatif to control local error

    Returns
    -------
    Y : list of ndarray
        States at accepted steps (starting from y0, ending at T)
    H : list of float
        Step sizes accepted at each step.
    """
    y = np.array(y0, dtype=float)
    Y = [y.copy()]
    t = 0.0
    it = 0
    h = h_init
    H = []

    while t < T and it < max_iter:
        # adjust final step to land exactly at T
        if t + h > T:
            h = T - t

        y_temp, err = updateRK_pas_adaptatif(y, h, A, B, C, f, t)

        if err < tolerance:
            # accept step
            t += h
            y = y_temp
            Y.append(y.copy())
            H.append(h)       # record accepted h
            # increase for next
            h *= 1.2
        else:
            # reject and decrease step
            h /= 2.0

        it += 1

    # Final adjustment if last t < T
    if t < T:
        h_last = T - t
        y_temp, err = updateRK_pas_adaptatif(y, h_last, A, B, C, f, t)
        if err < tolerance:
            t += h_last
            y = y_temp
            Y.append(y.copy())
            H.append(h_last)

    return Y, H