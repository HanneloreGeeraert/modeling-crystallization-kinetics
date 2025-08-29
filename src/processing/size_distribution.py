import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import re

def mean_radius_matrix(mode, t, alpha, G, F, eps=1e-12, return_r=True):
    """
    R2[i,j] = mean squared radius of cohort nucleated at tau=t[j], observed at t[i] (i>j).
    """
    t = np.asarray(t, float)
    alpha = np.asarray(alpha, float)
    G = np.asarray(G, float)
    F = np.asarray(F, float)
    N = t.size

    R2 = np.full((N, N), np.nan, float)

    for j in range(N-1):
        denom = max(1.0 - alpha[j], eps)
        acc = 0.0
        for i in range(j+1, N):
            # trapezoid over [t[i-1], t[i]]
            I_left  = (1.0 - alpha[i-1]) * (F[i-1] - F[j]) * G[i-1]
            I_right = (1.0 - alpha[i])   * (F[i]   - F[j]) * G[i]
            acc += 0.5 * (I_left + I_right) * (t[i] - t[i-1])
            R2[i, j] = 2.0 * acc / denom

    return np.sqrt(np.maximum(R2, 0.0)) if return_r else R2

def mean_radius_vs_time(t, Na, Rbar, idx=None):
    dNa_dt = np.gradient(Na, t)
    dt = np.diff(t, prepend=t[0])
    dNa = dNa_dt * dt  # number of nuclei per interval

    N = len(t)
    r_mean = np.full(N, np.nan)

    i = idx if idx is not None else len(t) - 1

    for i in range(N):
        numer = 0.0
        denom = 0.0
        for j in range(i):
            if np.isnan(Rbar[i, j]): 
                continue
            w = max(dNa[j], 0.0)
            numer += Rbar[i, j] * w
            denom += w
        if denom > 0:
            r_mean[i] = numer / denom
    return r_mean

def spherulite_distribution(t, Na, Rbar, nbins=150, idx=None):
    dt = np.diff(t, prepend=t[0])
    dNa_dt = np.gradient(Na, t)  # nucleation density
    
    # radii for all cohorts j<i
    radii = []
    weights = []

    i = idx if idx is not None else len(t) - 1

    for j in range(i):
        r_tau = Rbar[i, j]
        if np.isnan(r_tau) or r_tau <= 0: 
            continue
        w_tau = max(dNa_dt[j], 0) * dt[j]  # number nucleated in [τ, τ+dt]
        radii.append(r_tau)
        weights.append(w_tau)
    
    radii = np.array(radii)
    weights = np.array(weights)

    hist, edges = np.histogram(radii, bins=nbins, weights=weights, density=True)
    r_centers = 0.5*(edges[:-1] + edges[1:])
    return r_centers, hist

def plot_spherulites(results, mode, nbins=150, idx=None):
    """
    Plot both mean radius vs temperature and spherulite size distribution.
    """

    n_results = len(results)
    colors = plt.cm.viridis(np.linspace(0, 1, n_results))  # automatic colors

    # Decide how to parse labels
    if mode == "noniso_SN":
        regex, title = r"Ts(\d+)", r"$T_{\mathrm{s}}$ (°C)"
    elif mode == "noniso_CR":
        regex, title = r"C(\d+)", r"Cooling rate (°C/min)"
    elif mode == "iso":
        regex, title = r"Tiso(\d+)", r"$T_{\mathrm{iso}}$ (°C)"
    else:
        raise ValueError("mode must be one of {'noniso_SN', 'noniso_CR', 'iso'}")

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
    ax_mean, ax_dist = axes

    temp_legend = {}
    style_legend = [Line2D([0], [0], color='black', linestyle='-', label='Model')]

    for i, r in enumerate(results):
        color = colors[i]

        # ===== Extract numeric label =====
        match = re.search(regex, r['sheet'])
        label_val = int(match.group(1)) if match else r['sheet']

        # ===== Mean radius vs T =====
        t_idx = idx if idx is not None else len(r['t_exp']) - 1
        R_mean = mean_radius_vs_time(r['t_exp'], r['Na'], r['Rbar'], idx=t_idx)
        ax_mean.plot(r['T_exp'] - 273.15, R_mean * 1e6, 
                     label=f"{r['sheet']} model", color=color)

        # ===== Distribution at idx =====
        r_centers, pdf = spherulite_distribution(
            r['t_exp'], r['Na'], r['Rbar'], nbins=nbins, idx=t_idx
        )
        # Compute bar widths assuming uniform spacing
        width = np.diff(r_centers)
        # For the last bar, repeat the previous width
        width = np.append(width, width[-1])

        ax_dist.bar(r_centers * 1e6, pdf, width=width * 1e6, 
                    align='center', color=color, alpha=0.6, label=f"{r['sheet']} model")

        # Legend entry
        if label_val not in temp_legend:
            temp_legend[label_val] = Line2D([0], [0], color=color, label=f"{label_val}")

    # ===== Formatting: Mean radius =====
    ax_mean.set_xlabel("Temperature (°C)")
    ax_mean.set_ylabel("Mean radius (μm)")
    ax_mean.set_xlim(10, 150)
    legend1 = ax_mean.legend(handles=style_legend, loc="lower right", frameon=False)
    legend2 = ax_mean.legend(handles=temp_legend.values(), title=title, loc="center right", frameon=False)
    ax_mean.add_artist(legend1)
    ax_mean.grid(False)

    # ===== Formatting: Distribution =====
    ax_dist.set_xlabel("Radius (μm)")
    ax_dist.set_ylabel("Probability density")
    legend1 = ax_dist.legend(handles=style_legend, loc="upper left", frameon=False)
    legend2 = ax_dist.legend(handles=temp_legend.values(), title=title, loc="center right", frameon=False)
    ax_dist.add_artist(legend1)
    ax_dist.grid(False)

    plt.tight_layout()
    plt.show()

    return fig, axes
