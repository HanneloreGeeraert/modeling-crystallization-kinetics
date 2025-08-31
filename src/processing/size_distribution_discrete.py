import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import re
from .plotting import parse_sheetname

def evolve_spherulites(dimensions, t, alpha, G, F, Na):
    N = len(t)
    
    # Arrays for storing variables
    rho = np.zeros((N, N))       # If 2D: squared radii [time, nucleus_time], if 3D: cubed radii
    # n_kl = np.zeros((N, N))     # cumulative number with radius <= rho_kl
    dn_kl = np.zeros((N, N))    # number of spherulites in interval

    alpha_safe = np.clip(alpha, 0.0, 1.0)

    # --- main loop ---
    for j in range(N-1):   # nucleation index τ_l
        # store growth parameters for this nucleus j
        denom = max(1.0 - alpha_safe[j], 1e-12)
        a = dimensions / denom
        b = a * (dimensions-1) * F[j]

        if dimensions == 3:
            c = b * 0.5 * F[j]

        for i in range(j, N-1):   
            X_right = (1.0 - alpha_safe[i+1]) * G[i+1]
            X_left  = (1.0 - alpha_safe[i]) * G[i]
            Y_right = X_right * F[i+1]
            Y_left  = X_left * F[i]

            t_step = t[i+1] - t[i]

            X_k = 0.5 * (X_left + X_right) * t_step
            Y_k = 0.5 * (Y_left + Y_right) * t_step
            
            if dimensions == 2:
                # Update squared radius
                rho[i+1, j] = rho[i, j] + a * Y_k - b * X_k

            elif dimensions == 3:
                Z_right = Y_right * F[i+1]
                Z_left = Y_left * F[i]
                Z_k = 0.5 * (Z_left + Z_right) * t_step
                
                rho[i+1, j] = rho[i, j] + a * Z_k - b * Y_k + c * X_k

            # --- number distributions ---
            #n_kl[i+1, j] = Na[i+1] - Na[j]     # total up to radius ρ_kl
            if j <= i-1:
                dn_kl[i+1, j] = max(Na[j+1] - Na[j], 0.0)

    return rho, dn_kl

def plot_spherulites(dimensions, results, i=-1, bins=100):
    n_results = len(results)
    colors = plt.cm.viridis(np.linspace(0, 1, n_results))  # automatic colors

    sheetnames = [r['sheet'] for r in results]
    varying_keys, labels, title, mode = parse_sheetname(sheetnames)

    plt.figure(figsize=(8, 6))

    temp_legend = {}
    style_legend = [Line2D([0], [0], color='black', linestyle='-', label='Model')]

    for x, r in enumerate(results):
        color = colors[x]

        # ===== Label from parse_sheetname =====
        label_val = labels[r['sheet']]

        rho = r["rho"]
        dn_kl = r["dn_kl"]

        if dimensions == 2:
            radii = np.sqrt(rho[i, :])
        elif dimensions == 3:
            radii = np.cbrt(rho[i, :])

        counts = dn_kl[i, :]

        # histogram weighted by counts
        hist, edges = np.histogram(radii, bins=bins, weights=counts)

        # plot
        centers = 0.5 * (edges[1:] + edges[:-1]) * 1e6  # in µm
        plt.bar(
            centers, hist, width=np.diff(edges) * 1e6, 
            align="center", color=color, alpha=0.5
        )

        # Legend entry
        if label_val not in temp_legend:
            temp_legend[label_val] = Line2D([0], [0], color=color, label=f"{label_val}")

    plt.xlabel("Radius (µm)")
    plt.ylabel("Number of spherulites")

    # Always show style legend
    legend1 = plt.legend(handles=style_legend, loc="upper left", frameon=False)
    plt.gca().add_artist(legend1)

    # Only add parameter legend if there are varying parameters
    if title is not None:
        legend2 = plt.legend(handles=temp_legend.values(), title=title, loc="center right", frameon=False)
        plt.gca().add_artist(legend2)

    plt.grid(False)
    plt.tight_layout()
    plt.show()
