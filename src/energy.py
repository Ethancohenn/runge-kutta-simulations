"""
 Mechanical energy of the 2-D gravity test case.

Conventions
-----------
• All seven masses are 1 kg
• Gravitational constant G = 1
• Fixed sources are the same six points used everywhere else
• Potential energy  V = -Σ 1/r_i   (attractive gravity)
• Kinetic  energy  K = 0.5 (vx² + vy²)
"""

import numpy as np

# Fixed sources
SOURCES = np.array([
    [2.0, -1.0],
    [2.0,  0.0],
    [2.0,  1.0],
    [3.0, -1.0],
    [3.0,  0.0],
    [3.0,  1.0]
])

def potential_energy(pos, sources=SOURCES):
    """ G = 1, masses = 1 """
    diff = sources - pos 
    r = np.linalg.norm(diff, axis=1)
    return -np.sum(1.0 / r)

def kinetic_energy(vel):
    return 0.5 * np.dot(vel, vel)

def total_energy(state):
    """
    state = [x, y, vx, vy]
    """
    pos = state[:2]
    vel = state[2:]
    return kinetic_energy(vel) + potential_energy(pos)
