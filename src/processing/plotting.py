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
    colors = plt.cm.viridis(np.linspace(0, 1, n_sheets)) 
    
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
        ax1.plot(x_data_mod, r['HF_model'], label=f"{label_val} model", color=color)
        ax1.plot(x_data_exp, r['HF_exp'][mask], '--', label=f"{label_val} exp", color=color)

        # Crystallinity
        ax2.plot(x_data_mod, r['alpha_model'], label=f"{label_val} model", color=color)
        ax2.plot(x_data_exp, r['alpha_exp'][mask], '--', label=f"{label_val} exp", color=color)


        if label_val not in temp_legend:
            temp_legend[label_val] = Line2D([0], [0], color=color, label=label_val)

        # === Find 50% crystallinity point for model ===
        alpha = r['alpha_model']
        xdata = x_data_mod
        idx = np.where(np.diff(np.sign(alpha - 0.5)) != 0)[0]
        if len(idx) > 0:
            i0 = idx[0]
            x0, x1 = xdata[i0], xdata[i0+1]
            y0, y1 = alpha[i0], alpha[i0+1]
            x_half = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
            ax2.scatter(x_half, 0.5, color=color, edgecolor='black', zorder=5)
            if mode == 'iso':
                label = f"{x_half:.1f} s"
            elif mode == 'noniso':
                label = f"{x_half:.1f} °C"
            ax2.annotate(label, (x_half, 0.5), xytext=(10,0),
                         textcoords="offset points", fontsize=12, color=color)

        # === Find 50% crystallinity point for experimental data ===
        alpha_exp = r['alpha_exp'][mask]
        xdata_exp2 = x_data_exp  # already masked
        idx_exp = np.where(np.diff(np.sign(alpha_exp - 0.5)) != 0)[0]
        if len(idx_exp) > 0:
            i0 = idx_exp[0]
            x0, x1 = xdata_exp2[i0], xdata_exp2[i0+1]
            y0, y1 = alpha_exp[i0], alpha_exp[i0+1]
            x_half_exp = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
            ax2.scatter(x_half_exp, 0.5, color=color, marker='x', zorder=6)  # different marker to distinguish
            if mode == 'iso':
                label_exp = f"{x_half_exp:.1f} s"
            elif mode == 'noniso':
                label_exp = f"{x_half_exp:.1f} °C"
            ax2.annotate(label_exp, (x_half_exp, 0.5), xytext=(-60, 0),
                         textcoords="offset points", fontsize=12, color=color)
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
