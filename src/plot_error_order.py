# src/plot_error_order.py

import numpy as np
import matplotlib.pyplot as plt

from .error_analysis import compute_errors, estimate_order
from .butcher_tableaus import Adict, Bdict, Cdict

def main():
    liste_pas = np.array([2**(-i-1) for i in range(1, 5)])
    methods = [
        "Dormand-Prince",
        "Euler-Heun",
        "Bogacki–Shampine",
        "7-8"
    ]

    # compute errors for each method
    erreurs = {}
    for m in methods:
        erreurs[m] = compute_errors(np.array([1.0]), 1.0, m, liste_pas)
        p, _ = estimate_order(liste_pas, erreurs[m])
        print(f"{m}: estimated order ≈ {abs(p):.2f}")

    # four-panel plot
    plt.figure(figsize=(8, 10))

    # 1) Dormand-Prince (order 4)
    plt.subplot(411)
    plt.loglog(liste_pas, erreurs["Dormand-Prince"], label='error Dormand-Prince')
    plt.loglog(liste_pas, liste_pas**4,            label='h→h^4')
    plt.legend()

    # 2) Euler-Heun (order 2)
    plt.subplot(412)
    plt.loglog(liste_pas, erreurs["Euler-Heun"], label='error Euler-Heun')
    plt.loglog(liste_pas, liste_pas**2,           label='h→h^2')
    plt.legend()

    # 3) Bogacki–Shampine (order 3)
    plt.subplot(413)
    plt.loglog(liste_pas, erreurs["Bogacki–Shampine"], label='error Bogacki–Shampine')
    plt.loglog(liste_pas, liste_pas**3,               label='h→h^3')
    plt.legend()

    # 4) 7-8 (order 7)
    plt.subplot(414)
    plt.loglog(liste_pas, erreurs["7-8"],    label='error 7-8')
    plt.loglog(liste_pas, liste_pas**7,      label='h→h^7')
    plt.legend()

    plt.tight_layout()
    plt.savefig('figures/error_order_plot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
