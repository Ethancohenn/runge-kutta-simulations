"""
General Runge–Kutta implementation

"""

import numpy as np
import math

def somme(A_row, K):
    """
    Compute sum_j A_row[j] * K[j] as a vector
    """
    s = len(A_row)
    out = np.zeros_like(K[0])
    for j in range(s):
        out += np.dot(A_row[j], K[j])
    return out

def updateRK(y0, h, A, B, C, f, t):
    """
    Single-step RK update (primary solution).

    Returns
    -------
    y1 : state at t + h
    """
    K = K=np.zeros((np.shape(A)[0],np.shape(y0)[0]))
    for i in range(len(K)):
        ti = t + C[i] * h
        yi = y0 + h * somme(A[i], K)
        K[i] = f(ti, yi)
    # primary solution uses B[0]
    y1 = y0 + h * np.dot(B[0], K)
    return y1

def rk_fixed_steps(y0, f, A, B, C, T, n):
    """
    Integrate dy/dt = f(t, y) over [0, T] with exactly n steps.

    Returns
    -------
    y : state at t = T
    Y  : all intermediate states including y0
    """
    y = y0
    Y = [y0]
    t = 0
    h = T / n
    for _ in range(1, n):
        y = updateRK(y, h, A, B, C, f, t)
        Y.append(y.copy())
        t += h
    return y, Y


def rk_fixed_h(y0, f, A, B, C, T, h):
    """
    Integrate dy/dt = f(t, y) from t=0 until reaching T by step size h.

    Returns
    -------
    y : final state
    Y  : all intermediate states including y0
    """
    y = y0
    Y = [y0]
    t = 0
    nb_steps = math.ceil( T / h)
    for _ in range(nb_steps):
        y = updateRK(y, h, A, B, C, f, t)
        Y.append(y.copy())
        t += h
    return y, Y