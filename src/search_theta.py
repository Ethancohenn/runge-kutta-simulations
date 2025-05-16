import math
import numpy as np
from src.flight_time import compute_flight_time, min_distance_time

# Default kwargs reused in every call to compute_flight_time
FT_DEFAULTS = dict(T=3.0, h=0.001, tol=0.02, method="Dormand-Prince")


def angle_optimal(tau=0.1, t_init=3.0):
    """
    Sweep [0, 2π) with angular step `tau`
    """
    theta = 0.0
    t_opt, theta_opt = t_init, 0.0

    while theta < 2 * math.pi:
        t_curr = compute_flight_time(theta, **FT_DEFAULTS)
        if t_curr < t_opt:
            t_opt, theta_opt = t_curr, theta
        theta += tau

    return t_opt, theta_opt


def angle_optimal_refined(tau=0.1,
                          neighbourhood=10,
                          local_step=0.01,
                          t_init=3.0):
    """
    Sweep with step `tau`; around each angle giving time < 3 s,
    re‑evaluate ±`neighbourhood`·`local_step`
    """
    theta = 0.0
    t_opt, theta_opt = t_init, 0.0

    while theta < 2 * math.pi:
        t_curr = compute_flight_time(theta, **FT_DEFAULTS)
        if t_curr < 3.0:  # heuristic “interesting” region
            for i in range(-neighbourhood, neighbourhood + 1):
                d_theta = theta + i * local_step
                t_local = compute_flight_time(d_theta, **FT_DEFAULTS)
                if t_local < t_opt:
                    t_opt, theta_opt = t_local, d_theta
        theta += tau

    return t_opt, theta_opt


def GoldenS(fun, a, b, tol=1e-6, n_max=100):
    """
    Golden‑section minimum search on [a, b]
    Returns histories (L, U, x1, x2) and evaluation count
    """
    phi = (1 + math.sqrt(5)) / 2
    inv_phi = 1 / phi

    L, U, X1, X2 = [a], [b], [], []
    nfeval = 0

    x1 = b - inv_phi * (b - a)
    x2 = a + inv_phi * (b - a)
    f1, f2 = fun(x1), fun(x2)
    nfeval = 2

    while abs(b - a) > tol and nfeval < n_max:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - inv_phi * (b - a)
            f1 = fun(x1); nfeval += 1
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + inv_phi * (b - a)
            f2 = fun(x2); nfeval += 1

        L.append(a)
        U.append(b)
        X1.append(x1)
        X2.append(x2)

    return L, U, X1, X2, nfeval


# Wrapper giving only the distance
def _distance_only(theta):
    d_min, _t_min = min_distance_time(theta, T=FT_DEFAULTS["T"], h=FT_DEFAULTS["h"], method=FT_DEFAULTS["method"])
    return d_min


def angle_optimal_golden(tau=0.3, golden_tol=1e-2):
    """
    1) Sweep [0, 2π) with step `tau`, recording (θ, distance, time)
    2) Find local minima of **distance**
    3) Refine each minimum with GoldenS on distance
    4) Return angle giving minimal **time**
    """
    measurements = []
    theta = 0.0
    while theta < 2 * math.pi:
        d, t =  min_distance_time(theta, T=FT_DEFAULTS["T"], h=FT_DEFAULTS["h"], method=FT_DEFAULTS["method"])
        measurements.append((theta, d, t))
        theta += tau

    # Find local minima in distance
    candidates = []
    for i in range(1, len(measurements) - 1):
        d_prev = measurements[i-1][1]
        d_curr = measurements[i][1]
        d_next = measurements[i+1][1]
        if d_curr < d_prev and d_curr < d_next:
            θ_left = measurements[i-1][0]
            θ_right = measurements[i+1][0]

            # Golden‑section on distance
            L, U, *_ = GoldenS(_distance_only,
                               θ_left, θ_right, tol=golden_tol)
            θ_star = 0.5 * (L[-1] + U[-1])
            d_star, t_star = min_distance_time(θ_star, T=FT_DEFAULTS["T"], h=FT_DEFAULTS["h"], method=FT_DEFAULTS["method"])
            candidates.append((θ_star, (d_star, t_star)))

    # Select best time
    if not candidates:
        raise RuntimeError("No candidate angles found.")
    θ_opt, (d_opt, t_opt) = min(candidates, key=lambda x: x[1][1])
    return θ_opt, (d_opt, t_opt)



if __name__ == "__main__":
    # print("Coarse sweep:")
    # print(angle_optimal(tau=1.4125/10))

    # print("\nRefined sweep:")
    # print(angle_optimal_refined(1.4125/10))

    print("\nGolden‑section:")
    θ_best, (d_best, t_best) = angle_optimal_golden(tau=0.3)
    print(f"θ ≈ {θ_best:.6f} rad,  d_min ≈ {d_best:.4e},  t_min ≈ {t_best:.6f} s")
