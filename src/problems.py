"""
Definitions of ODE right-hand sides for the test problems
(exponential, simple pendulum) and for the 2D gravitational particle problem.
"""

import numpy as np

def exponentielle(t, x):
    """
    Test ODE y' = y (exponential growth)
    """
    return x

def pendule_simple(t, x):
    """
    Simple pendulum
    """
    theta, omega = x
    return np.array([omega, -np.sin(theta)])

def gravite(t, x):
    """
    2D gravitational problem (six fixed unit masses)
    """
    # Fixed source positions
    X = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    Y = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])

    # current state
    a, b, vx, vy = x

    # compute vector from particle to each source
    dx = X - a
    dy = Y - b

    # distances cubed: (dx^2 + dy^2)^(3/2)
    r3 = (dx**2 + dy**2)**1.5

    # G=1
    ax = np.sum(dx / r3)
    ay = np.sum(dy / r3)

    return np.array([vx, vy, ax, ay])