"""
Plot a single trajectory of the 2D gravitational problem for a given launch angle.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from .problems import gravite
from .simulation import simulate_fixed_steps

# Define the six source positions again, or import if you have it elsewhere:
DEFAULT_SOURCE_POSITIONS = np.array([
    [2.0, -1.0],
    [2.0,  0.0],
    [2.0,  1.0],
    [3.0, -1.0],
    [3.0,  0.0],
    [3.0,  1.0],
])

def main(theta, T, h, method, save_fig=False):
    # initial state: (x,y,vx,vy)
    y0 = np.array([0.0, 0.0, np.cos(theta), np.sin(theta)])

    times, Y = simulate_fixed_steps(gravite, y0, T=T, h=h, method=method)

    plt.figure(figsize=(6,6))
    plt.plot(Y[:,0], Y[:,1], '-', label='Trajectory')

    sources = np.vstack((DEFAULT_SOURCE_POSITIONS, [4.0, 1.0]))
    plt.scatter(sources[:,0], sources[:,1], c='red', label='Masses & target')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Gravity trajectory, θ={theta:.4f}')
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()

    if save_fig:
        plt.savefig('figures/gravity_trajectory.png', dpi=300)
        print(f"Saved gravity trajectory plot to figures/gravity_trajectory.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the 2D gravity trajectory for a given launch angle"
    )
    parser.add_argument('--theta', type=float, default=6.0083,
                        help='Launch angle in radians')
    parser.add_argument('--T', type=float, default=2.0,
                        help='Total simulation time')
    parser.add_argument('--h', type=float, default=0.01,
                        help='Time step size')
    parser.add_argument('--method', type=str, default='Dormand-Prince',
                        choices=['Euler-Heun','Bogacki–Shampine','Dormand-Prince','7-8'],
                        help='RK method to use')
    parser.add_argument('--save', action='store_true',
                        help='Save figure to figures/gravity_trajectory.png')
    args = parser.parse_args()

    main(args.theta, args.T, args.h, args.method, save_fig=args.save)
