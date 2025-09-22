import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.processing.plotting import parse_sheetname   # keep your existing helper

def evolve_spherulites(dimensions, t, alpha, G, F, Na, bins=100, plot=False):
    """
    Streamed version: evolves spherulites without storing full (N,N) arrays.
    Returns histograms at each time step instead of full arrays.
    """
    N = len(t)

    alpha_safe = np.clip(alpha, 0.0, 1.0)

    # per nucleus properties
    rho = np.zeros(N)     # squared/cubed radii for each nucleus
    dn = np.zeros(N)      # number of spherulites per nucleus at current step

    # container for histograms
    histograms = []

    for i in range(1, N):
        t_step = t[i] - t[i-1]

        # update each nucleus j ≤ i-1
        for j in range(i):
            denom = max(1.0 - alpha_safe[j], 1e-12)
            a = dimensions / denom
            b = a * (dimensions-1) * F[j]

            if dimensions == 3:
                c = b * 0.5 * F[j]

            X_right = (1.0 - alpha_safe[i]) * G[i]
            X_left  = (1.0 - alpha_safe[i-1]) * G[i-1]
            Y_right = X_right * F[i]
            Y_left  = X_left * F[i-1]

            X_k = 0.5 * (X_left + X_right) * t_step
            Y_k = 0.5 * (Y_left + Y_right) * t_step

            if dimensions == 2:
                rho[j] = rho[j] + a * Y_k - b * X_k
            else:  # dimensions == 3
                Z_right = Y_right * F[i]
                Z_left = Y_left * F[i-1]
                Z_k = 0.5 * (Z_left + Z_right) * t_step
                rho[j] = rho[j] + a * Z_k - b * Y_k + c * X_k

            # number distributions
            if j <= i-1:
                dn[j] = max(Na[j+1] - Na[j], 0.0)

        # compute histogram for current time step
        if dimensions == 2:
            radii = np.sqrt(rho[:i])
        else:
            radii = np.cbrt(rho[:i])

        counts = dn[:i]
        hist, edges = np.histogram(radii, bins=bins, weights=counts)
        centers = 0.5 * (edges[1:] + edges[:-1]) * 1e6  # µm

        histograms.append((centers, hist))

        if plot:
            plt.clf()
            plt.bar(centers, hist, width=np.diff(edges)*1e6, align="center", alpha=0.5)
            plt.pause(0.01)

    return histograms

def plot_spherulites_from_histograms(results, time_index=-1):
    """
    Plot spherulite distributions directly from streamed histograms stored in results.
    
    Parameters
    ----------
    results : list of dicts
        Each dict must have keys: 'sheet' and 'histograms'.
    time_index : int
        Which time step to plot (default -1 = last one).
    """
    n_results = len(results)
    colors = plt.cm.viridis(np.linspace(0, 1, n_results))  # automatic colors

    # Use your existing parse_sheetname to get labels/titles
    sheetnames = [r['sheet'] for r in results]
    varying_keys, labels, title, mode = parse_sheetname(sheetnames)

    plt.figure(figsize=(8, 6))

    temp_legend = {}
    style_legend = [Line2D([0], [0], color='black', linestyle='-', label='Model')]

    for x, r in enumerate(results):
        color = colors[x]

        # ===== Label from parse_sheetname =====
        label_val = labels[r['sheet']]

        # Get the histogram at the desired time step
        centers, hist = r['histograms'][time_index]

        # plot
        plt.bar(
            centers, hist, width=(centers[1]-centers[0]),
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
