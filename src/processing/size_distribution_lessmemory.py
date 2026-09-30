import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from src.processing.plotting import parse_sheetname   # keep your existing helper

def evolve_spherulites(
    dimensions,
    t,
    alpha,
    G,
    F,
    Na,
    bins=100,
    size_range_um=None,
    size_type="diameter",
    plot=False
):
    """
    Streamed version for calculating spherulite size distributions.

    Parameters
    ----------
    dimensions : int
        2 or 3.
    t : array
        Time array.
    alpha : array
        Transformed fraction.
    G : array or scalar
        Growth rate.
    F : array
        Integrated growth variable.
    Na : array
        Activated nuclei density.
    bins : int
        Number of histogram bins.
    size_range_um : tuple or None
        Fixed histogram range in µm, e.g. (0, 100).
        Strongly recommended when comparing multiple simulations.
    size_type : str
        "radius" or "diameter".
    plot : bool
        If True, plots the distribution during evolution.

    Returns
    -------
    histograms : list of pandas DataFrames
        One size-distribution table per time step.
    """

    t = np.asarray(t, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    F = np.asarray(F, dtype=float)
    Na = np.asarray(Na, dtype=float)

    if np.isscalar(G):
        G = np.full_like(t, G, dtype=float)
    else:
        G = np.asarray(G, dtype=float)

    N = len(t)

    if not (len(alpha) == len(G) == len(F) == len(Na) == N):
        raise ValueError("t, alpha, G, F, and Na must have the same length.")

    if dimensions not in [2, 3]:
        raise ValueError("dimensions must be 2 or 3.")

    if size_type not in ["radius", "diameter"]:
        raise ValueError("size_type must be 'radius' or 'diameter'.")

    alpha_safe = np.clip(alpha, 0.0, 1.0)

    # Current squared/cubed radii for each activated population
    rho = np.zeros(N)

    # Incremental activated nuclei density
    dn = np.zeros(N)
    dn[:-1] = np.maximum(np.diff(Na), 0.0)

    histograms = []

    for i in range(1, N):
        t_step = t[i] - t[i - 1]

        if t_step <= 0:
            raise ValueError("t must be strictly increasing.")

        for j in range(i):
            denom = max(1.0 - alpha_safe[j], 1e-12)

            a = dimensions / denom
            b = a * (dimensions - 1) * F[j]

            X_right = (1.0 - alpha_safe[i]) * G[i]
            X_left = (1.0 - alpha_safe[i - 1]) * G[i - 1]

            Y_right = X_right * F[i]
            Y_left = X_left * F[i - 1]

            X_k = 0.5 * (X_left + X_right) * t_step
            Y_k = 0.5 * (Y_left + Y_right) * t_step

            if dimensions == 2:
                rho[j] += a * Y_k - b * X_k

            else:
                c = b * 0.5 * F[j]

                Z_right = Y_right * F[i]
                Z_left = Y_left * F[i - 1]
                Z_k = 0.5 * (Z_left + Z_right) * t_step

                rho[j] += a * Z_k - b * Y_k + c * X_k

        # Avoid numerical negative values
        rho_pos = np.clip(rho[:i], 0.0, None)

        if dimensions == 2:
            radius_m = np.sqrt(rho_pos)
        else:
            radius_m = np.cbrt(rho_pos)

        radius_um = radius_m * 1e6

        if size_type == "radius":
            size_um = radius_um
            size_label = "Radius"
        else:
            size_um = 2.0 * radius_um
            size_label = "Diameter"

        counts = dn[:i]

        # Use fixed size range if provided
        if size_range_um is None:
            max_size = np.nanmax(size_um)

            if max_size <= 0:
                max_size = 1e-6

            hist_range = (0.0, max_size)
        else:
            hist_range = size_range_um

        hist, edges = np.histogram(
            size_um,
            bins=bins,
            range=hist_range,
            weights=counts
        )

        centers = 0.5 * (edges[1:] + edges[:-1])
        bin_widths = np.diff(edges)

        total = np.sum(hist)

        if total > 0:
            normalized = hist / total
            cumulative = np.cumsum(normalized)
        else:
            normalized = np.zeros_like(hist)
            cumulative = np.zeros_like(hist)

        distribution = pd.DataFrame({
            f"{size_label} lower (µm)": edges[:-1],
            f"{size_label} upper (µm)": edges[1:],
            f"{size_label} center (µm)": centers,
            "Bin width (µm)": bin_widths,
            "Number density": hist,
            "Normalized number fraction": normalized,
            "Cumulative number fraction": cumulative,
            "Time (s)": t[i],
            "Alpha": alpha[i]
        })

        histograms.append(distribution)

        if plot:
            plt.clf()
            plt.bar(
                centers,
                hist,
                width=bin_widths,
                align="center",
                alpha=0.5
            )
            plt.xlabel(f"{size_label} (µm)")
            plt.ylabel("Number density of spherulites")
            plt.pause(0.01)

    return histograms


def plot_spherulites_from_histograms(results, time_index=-1, normalize=False):
    import numpy as np
    import matplotlib.pyplot as plt

    sheetnames = [r["sheet"] for r in results]
    varying_keys, labels, title, mode = parse_sheetname(sheetnames)

    valid_results = [r for r in results if "histograms" in r]
    n_results = len(valid_results)

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, n_results))

    plt.figure(figsize=(6, 4))

    xlabel = "Spherulite size (µm)"
    ylabel = "Value"

    for i, r in enumerate(valid_results):
        distribution = r["histograms"][time_index]
        label_val = labels[r["sheet"]]

        # Detect diameter or radius columns
        if "Diameter center (µm)" in distribution.columns:
            center_col = "Diameter center (µm)"
            lower_col = "Diameter lower (µm)"
            upper_col = "Diameter upper (µm)"
            xlabel = "Spherulite diameter (µm)"
        elif "Radius center (µm)" in distribution.columns:
            center_col = "Radius center (µm)"
            lower_col = "Radius lower (µm)"
            upper_col = "Radius upper (µm)"
            xlabel = "Spherulite radius (µm)"
        else:
            raise ValueError("Could not find size-center column in distribution.")

        if normalize:
            y_col = "Normalized number fraction"
            ylabel = "Normalized number fraction (-)"
        else:
            y_col = "Number density"
            ylabel = "Number density of spherulites"

        centers = distribution[center_col].to_numpy()
        heights = distribution[y_col].to_numpy()
        widths = (distribution[upper_col] - distribution[lower_col]).to_numpy()

        plt.bar(
            centers,
            heights,
            width=widths,
            align="center",
            alpha=0.35,
            color=colors[i],
            label=label_val
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title is not None:
        plt.legend(title=title, frameon=False)
    else:
        plt.legend(frameon=False)

    plt.grid(False)
    plt.tight_layout()
    plt.show()