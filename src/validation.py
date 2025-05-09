"""
Demonstrations of the RK2 (Euler–Heun) method on the exponential and simple pendulum problems.
"""

import numpy as np
import matplotlib.pyplot as plt

from .butcher_tableaus import Adict, Bdict, Cdict
from .rk_general import updateRK, rk_fixed_h
from .problems import exponentielle, pendule_simple

# For RK2 we use the Euler–Heun tableau
METHOD = "Euler-Heun"

def test_exponentielle(h, T):
    """
    Integrate y' = y from t=0 to T with step h using RK2.
    Returns (times, Y), where Y is the list of [y] values.
    """
    A = Adict[METHOD]
    B = Bdict[METHOD]
    C = Cdict[METHOD]

    # initial condition
    y0 = np.array([1.0])
    # integrate
    yN, Y = rk_fixed_h(y0, exponentielle, A, B, C, T, h)
    times = np.linspace(0.0, T, len(Y))
    return times, Y

def test_pendule(h, T):
    """
    Integrate simple pendulum from t=0 to T with step h using RK2.
    Returns (times, angles), where angles = [theta_i].
    """
    A = Adict[METHOD]
    B = Bdict[METHOD]
    C = Cdict[METHOD]

    y0 = np.array([np.pi/4, 0.0])  # theta=π/4, omega=0
    yN, Y = rk_fixed_h(y0, pendule_simple, A, B, C, T, h)
    times = np.linspace(0.0, T, len(Y))
    # extract only the angle component
    angles = [state[0] for state in Y]
    return times, angles

def plot_validation(h=0.01, T_exp=2.0, T_pend=10.0):

    # Exponential test
    t_exp, Y_exp = test_exponentielle(h, T_exp)
    Y_exp = [y[0] for y in Y_exp]  # unpack scalar

    # Pendulum test
    t_pend, angles = test_pendule(h, T_pend)

    # Plot
    plt.figure(figsize=(8, 6))

    plt.subplot(211)
    plt.plot(t_exp, Y_exp, label="Exponential (RK2)")
    # Uncomment to compare to exact:
    # plt.plot(t_exp, np.exp(t_exp), label="Exact exp(t)")
    plt.xlabel("Time")
    plt.ylabel("y(t)")
    plt.legend()

    plt.subplot(212)
    plt.plot(t_pend, angles, label="Simple pendulum θ(t) (RK2)")
    plt.xlabel("Time")
    plt.ylabel("Angle (rad)")
    plt.legend()

    plt.tight_layout()
    plt.savefig('figures/validation.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_validation()
