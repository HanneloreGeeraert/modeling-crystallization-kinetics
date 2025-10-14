import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def parse_sheetname(sheetnames):
    def extract(sheet):
        ts = re.search(r"Ts(\d+)", sheet)
        tiso = re.search(r"Tiso(\d+)", sheet)
        c = re.search(r"C(\d+)", sheet)
        return {
            "Ts": int(ts.group(1)) if ts else None,
            "Tiso": int(tiso.group(1)) if tiso else None,
            "C": int(c.group(1)) if c else None,
        }

    parsed = {s: extract(s) for s in sheetnames}

    # Determine varying keys
    varying_keys = []
    for key in ["Ts", "Tiso", "C"]:
        values = {v[key] for v in parsed.values() if v[key] is not None}
        if len(values) > 1:
            varying_keys.append(key)

    # Determine mode
    # Priority: C → noniso_CR, Ts → noniso_SN, Tiso → iso
    mode = None
    for sheet_vals in parsed.values():
        if sheet_vals["C"] is not None:
            mode = "noniso"
            break
        elif sheet_vals["Tiso"] is not None:
            mode = "iso"
            break

    key_to_title = {
        "Ts": (r"$T_{\mathrm{s}}$ (°C)", "°C"),
        "Tiso": (r"$T_{\mathrm{iso}}$ (°C)", "°C"),
        "C": ("Cooling rate (°C/min)", "°C/min"),
    }

    # Build labels
    labels = {}
    if varying_keys:
        for sheet, vals in parsed.items():
            parts = []
            for key in ["Ts", "Tiso", "C"]:
                if key in varying_keys and vals[key] is not None:
                    _, unit = key_to_title[key]
                    parts.append(f"{vals[key]} {unit}")
            labels[sheet] = ", ".join(parts)
        title = key_to_title[varying_keys[0]][0] if len(varying_keys) == 1 else "Parameters"
    else:
        labels = {s: s for s in sheetnames}  # fallback: just sheetnames
        title = None  # means "no legend"

    return varying_keys, labels, title, mode

def plot_results(results):
    n_sheets = len(results)
    cmap = plt.get_cmap("viridis")  
    colors = cmap(np.linspace(0, 1, n_sheets))
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
    ax1, ax2 = axes

    # Legend helpers
    style_legend = [
        Line2D([0], [0], color='black', linestyle='-', label='Model'),
        Line2D([0], [0], color='black', linestyle='--', label='Experimental')
    ]

    sheetnames = [r['sheet'] for r in results]
    varying_keys, labels, title, mode = parse_sheetname(sheetnames)

    temp_legend = {}

    for i, r in enumerate(results):
        color = colors[i]
        label_val = labels[r['sheet']]  # smart label

        # Build mask
        mask = (r['weight'] == 1)

        if mode == 'iso':
            x_data_exp = r['t_exp'][mask]
            x_data_mod = r['t_exp']
        elif mode == 'noniso':
            x_data_exp = r['T_exp'][mask] - 273.15
            x_data_mod = r['T_exp'] - 273.15

        # Heat Flow
        ax1.plot(x_data_mod, r['HF_model'], label=f"{label_val} model", color=color) # pyright: ignore[reportPossiblyUnboundVariable]
        ax1.plot(x_data_exp, r['HF_exp'][mask], '--', label=f"{label_val} exp", color=color) # pyright: ignore[reportPossiblyUnboundVariable]

        # Crystallinity
        ax2.plot(x_data_mod, r['alpha_model'], label=f"{label_val} model", color=color) # pyright: ignore[reportPossiblyUnboundVariable]
        ax2.plot(x_data_exp, r['alpha_exp'][mask], '--', label=f"{label_val} exp", color=color) # pyright: ignore[reportPossiblyUnboundVariable]


        if label_val not in temp_legend:
            temp_legend[label_val] = Line2D([0], [0], color=color, label=label_val)

    # Formatting Heat Flow
    ax1.set_ylabel('Heat Flow (W/g)')
    legend1 = ax1.legend(handles=style_legend, loc="lower right", frameon=False)
    ax1.add_artist(legend1)
    if title is not None:  # only add if varying parameter found
        legend2 = ax1.legend(handles=temp_legend.values(), title=title, loc="center right", frameon=False)
        ax1.add_artist(legend2)
    ax1.grid(False)

    # Formatting Crystallinity
    if mode == 'iso':
        ax2.set_xlabel('Step time (s)')
        ax2.set_xlim(0,3600)
    if mode == 'noniso':
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_xlim(50, 150)
    ax2.set_ylabel('Crystallinity (-)')
    legend1 = ax2.legend(handles=style_legend, loc="lower right", frameon=False)
    ax2.add_artist(legend1)
    if title is not None:
        legend2 = ax2.legend(handles=temp_legend.values(), title=title, loc="center right", frameon=False)
        ax2.add_artist(legend2)
    ax2.grid(False)

    plt.tight_layout()
    plt.show()

    return fig, axes
