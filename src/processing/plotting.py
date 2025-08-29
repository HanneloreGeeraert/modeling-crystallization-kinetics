import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_results(results, mode="noniso_SN"):
    n_sheets = len(results)
    colors = plt.cm.viridis(np.linspace(0, 1, n_sheets)) 
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
    ax1, ax2 = axes

    # Legend helpers
    temp_legend = {}
    style_legend = [
        Line2D([0], [0], color='black', linestyle='-', label='Model'),
        Line2D([0], [0], color='black', linestyle='--', label='Experimental')
    ]

    # Decide how to parse labels
    if mode == "noniso_SN":
        regex, title, unit = r"Ts(\d+)", r"$T_{\mathrm{s}}$ (°C)", "°C"
    elif mode == "noniso_CR":
        regex, title, unit = r"C(\d+)", r"Cooling rate (°C/min)", "°C/min"
    elif mode == "iso":
        regex, title, unit = r"Tiso(\d+)", r"$T_{\mathrm{iso}}$ (°C)", "°C"
    else:
        raise ValueError("mode must be one of {'noniso_SN', 'noniso_CR', 'iso'}")

    for i, r in enumerate(results):
        color = colors[i]

        # Try to extract numeric value from sheet name
        match = re.search(regex, r['sheet'])
        if match:
            label_val = int(match.group(1))
        else:
            label_val = r['sheet']  # fallback: raw name

        # Heat Flow
        ax1.plot(r['T_exp']-273.15, r['HF_model'], label=f"{r['sheet']} model", color=color)
        ax1.plot(r['T_exp']-273.15, r['HF_exp'], '--', label=f"{r['sheet']} exp", color=color)

        # Crystallinity
        ax2.plot(r['T_exp']-273.15, r['alpha_model'], label=f"{r['sheet']} model", color=color)
        ax2.plot(r['T_exp']-273.15, r['alpha_exp'], '--', label=f"{r['sheet']} exp", color=color)

        if label_val not in temp_legend:
            temp_legend[label_val] = Line2D([0], [0], color=color, label=f"{label_val}")

    # Formatting Heat Flow
    ax1.set_ylabel('Heat Flow (W/g)')
    ax1.set_xlim(10, 200)
    legend1 = ax1.legend(handles=style_legend, loc="lower right", frameon=False)
    legend2 = ax1.legend(handles=temp_legend.values(), title=title, loc="center right", frameon=False)
    ax1.add_artist(legend1)
    ax1.grid(False)

    # Formatting Crystallinity
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Crystallinity (-)')
    ax2.set_xlim(10, 200)
    legend1 = ax2.legend(handles=style_legend, loc="lower right", frameon=False)
    legend2 = ax2.legend(handles=temp_legend.values(), title=title, loc="center right", frameon=False)
    ax2.add_artist(legend1)
    ax2.grid(False)

    plt.tight_layout()
    plt.show()

    return fig, axes
