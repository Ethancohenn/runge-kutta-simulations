import argparse
import numpy as np
import matplotlib.pyplot as plt
import math

from .search_theta import GoldenS, _distance_only, FT_DEFAULTS
from .flight_time import min_distance_time

def main(tau=0.2, golden_tol=1e-2, save_fig=False):
    # Coarse scan: θ → min distance
    theta_vals = np.arange(0.0, 2 * np.pi, tau)
    distances = []
    for theta in theta_vals:
        d, _ = min_distance_time(theta, T=FT_DEFAULTS["T"], h=FT_DEFAULTS["h"], method=FT_DEFAULTS["method"])
        distances.append(d)
    distances = np.array(distances)

    idx_min = np.argmin(distances)
    theta_coarse = theta_vals[idx_min]

    # Golden Section around best coarse θ
    a = theta_vals[max(0, idx_min - 1)]
    b = theta_vals[min(len(theta_vals) - 1, idx_min + 1)]

    L, U, _, _, _ = GoldenS(_distance_only, a, b, tol=golden_tol)

    theta_opt = 0.5 * (L[-1] + U[-1])
    d_opt, t_opt = min_distance_time(theta_opt, T=FT_DEFAULTS["T"], h=FT_DEFAULTS["h"], method=FT_DEFAULTS["method"])

    # Plot
    plt.figure(figsize=(8, 6))

    # Top plot: min distance vs θ
    plt.subplot(2, 1, 1)
    plt.plot(theta_vals, distances, '-o', label='min distance d(θ)', markersize=4)
    plt.axvline(theta_opt, color='red', linestyle='--',
                label=f'Optimal θ ≈ {theta_opt:.4f}')
    plt.xlabel("Launch angle θ (rad)")
    plt.ylabel("Minimum distance to target (4, 1)")
    plt.title("Coarse scan of θ → min distance")
    plt.legend()

    # Bottom plot: Golden Section bracket shrinkage
    plt.subplot(2, 1, 2)
    k_vals = np.arange(len(L))
    plt.plot(k_vals, L, label='L_k (lower bound)')
    plt.plot(k_vals, U, label='U_k (upper bound)')
    plt.fill_between(k_vals, L, U, alpha=0.2)
    plt.xlabel("Golden section iteration k")
    plt.ylabel("θ interval bounds")
    plt.title("Convergence of [L_k, U_k] during Golden Section")
    plt.legend()

    plt.tight_layout()

    if save_fig:
        out_path = "figures/theta_optimization.png"
        plt.savefig(out_path, dpi=300)
        print(f"Saved figure to {out_path}")

    plt.show()

    print(f"\nOptimal θ ≈ {theta_opt:.6f} rad")
    print(f"Minimum distance ≈ {d_opt:.4e}")
    print(f"Time at min distance ≈ {t_opt:.4f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=0.2,
                        help="Angular step for the coarse scan (in radians)")
    parser.add_argument("--golden-tol", type=float, default=1e-2,
                        help="Tolerance for the golden section search")
    parser.add_argument("--save", action="store_true",
                        help="Save the figure to figures/theta_optimization.png")
    args = parser.parse_args()

    main(tau=args.tau, golden_tol=args.golden_tol, save_fig=args.save)
