"""
Plot an adaptive-step trajectory for the gravity problem:
  1) Particle trajectory up to T.
  2) Corresponding step sizes h as function of x.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from .problems import gravite
from .rk_adaptive import rk_adaptive

def plot_adaptive_trajectory(theta, T, h_init, tol_local, save_fig=False):
    """
    Compute and plot:
      Top:    trajectory (x vs y) up to time T, launch angle theta
      Bottom: step sizes h vs x-coordinate
    """
    # Initial state
    y0 = np.array([0.0, 0.0, np.cos(theta), np.sin(theta)])
    
    from src.butcher_tableaus import Adict, Bdict, Cdict
    A = Adict["Dormand-Prince"]
    B = Bdict["Dormand-Prince"]
    C = Cdict["Dormand-Prince"]

    # Run adaptive integrator
    Y, H = rk_adaptive(y0, gravite, A, B, C, T, h_init, tol_local)

    # Extract X, Y coordinates
    LesX = [state[0] for state in Y]
    LesY = [state[1] for state in Y]
    
    # Fixed masses + target points
    PointsX = np.array([2, 2, 2, 3, 3, 3, 4])
    PointsY = np.array([-1, 0, 1, -1, 0, 1, 1])

    # Plot
    plt.figure(figsize=(8, 6))

    # Top panel: trajectory
    plt.subplot(2, 1, 1)
    plt.plot(LesX, LesY, '-', lw=1)
    theta_deg = np.round(theta, 5)
    plt.title(f"Trajectory up to T={T} (θ = {theta_deg})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(PointsX, PointsY, c='red', s=20)

    # Bottom panel: step sizes vs X
    plt.subplot(2, 1, 2)
    plt.plot(LesX[:-1], H, '-o', markersize=3)
    plt.xlim(0, 4)
    plt.title("Step-size h vs X")
    plt.xlabel("X")
    plt.ylabel("h")

    plt.tight_layout()

    if save_fig:
        out_path = "figures/adaptive_trajectory.png"
        plt.savefig(out_path, dpi=300)
        print(f"Saved adaptive trajectory plot to {out_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot adaptive RK trajectory and step sizes for gravity problem"
    )
    parser.add_argument('--theta', type=float, default=6.0083,
                        help='Launch angle in radians')
    parser.add_argument('--T', type=float, default=1.9,
                        help='Simulation end time')
    parser.add_argument('--h-init', type=float, default=1e-3,
                        help='Initial step size')
    parser.add_argument('--tol-local', type=float, default=1e-3,
                        help='Local error tolerance')
    parser.add_argument('--save', action='store_true',
                        help='Save the figure to figures/adaptive_trajectory.png')
    args = parser.parse_args()

    plot_adaptive_trajectory(
        args.theta,
        args.T,
        args.h_init,
        args.tol_local,
        save_fig=args.save
    )