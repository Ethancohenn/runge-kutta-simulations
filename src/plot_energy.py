"""
 Compare total-energy drift for different RK methods
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from .simulation import simulate_fixed_steps
from .problems import gravite
from .energy import total_energy

def energy_curve(theta,
                 T=3.0,
                 h=0.001,
                 method="Dormand-Prince"):
    """
    Integrate trajectory and return (times, energies)
    """
    y0 = np.array([0.0, 0.0, np.cos(theta), np.sin(theta)])
    times, Y = simulate_fixed_steps(gravite, y0, T=T, h=h, method=method)
    energies = np.array([total_energy(state) for state in Y])
    return times, energies

def main(theta=6.0083,
         T=3.0,
         h=0.001,
         methods=("Euler-Heun", "Bogacki–Shampine",
                  "Dormand-Prince", "7-8"),
         save=False):
    plt.figure(figsize=(8, 5))

    for m in methods:
        t, E = energy_curve(theta, T=T, h=h, method=m)
        dE = E - E[0]
        plt.plot(t, dE, label=f"{m}")

    plt.xlabel("Time")
    plt.ylabel("ΔE(t) = E(t) – E(0)")
    plt.title(f"Energy drift (θ = {theta:.4f} rad, h = {h})")
    plt.legend()
    plt.tight_layout()

    if save:
        out = "figures/energy_drift.png"
        plt.savefig(out, dpi=300)
        print(f"Saved energy plot to {out}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, default=1.4125,
                        help="Launch angle (rad)")
    parser.add_argument("--T", type=float, default=3.0,
                        help="End time of simulation")
    parser.add_argument("--h", type=float, default=0.001,
                        help="Fixed time step")
    parser.add_argument("--methods", nargs="+",
                        default=["Euler-Heun", "Dormand-Prince"],
                        help="RK methods to plot")
    parser.add_argument("--save", action="store_true",
                        help="Save figure to figures/energy_drift.png")
    args = parser.parse_args()

    main(theta=args.theta,
         T=args.T,
         h=args.h,
         methods=args.methods,
         save=args.save)
