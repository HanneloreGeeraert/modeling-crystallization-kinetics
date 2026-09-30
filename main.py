import numpy as np
import math
from src.models.kinetics import run_model, run_model_with_induction
from src.models.utils import get_deltaHm
from src.processing.plotting import plot_results
from src.processing.size_distribution_lessmemory import evolve_spherulites, plot_spherulites_from_histograms
from src.processing.data_loadonce import get_dataset # type: ignore
from src.models.specs import get_modelspec

import matplotlib.pyplot as plt

# Load experimental data
[all_sheets, sheet_names] = get_dataset("noniso_DII_C10", reload = True)

results = []
plot_spherulites = 0
plot_Na = 0

# Select model and parameters
model_spec = get_modelspec("DII_3D_induction")

for i, sheet_name in enumerate(sheet_names):
    
    Ts = float(sheet_name.split("Ts")[1].split("_")[0]) + 273.15
    model_spec.params.fixed_params['Ts'] = Ts

    df = all_sheets[sheet_name]
    t_exp = df['StepTime (s)'].values
    T_exp = df['Temperature (°C)'].values + 273.15
    DT_exp = df['DT (K/min)'].values / 60
    alpha_exp = df['Alpha (-)'].values
    weight = df['Weight (-)'].values
    HF_exp = df['Heat Flow Baseline Corrected (W/g)'].values
    #deltaH_m = get_deltaHm(df['Int (J/g)'].values)
    deltaH_m = model_spec.params.functions["deltaH_m"](model_spec.params)

    # Initial conditions for the model
    y0 = model_spec.make_y0(sheet_name, df)
    N0  = model_spec.params.functions["N0"](model_spec.params)

    y0[0] = N0
    if "I_d" in model_spec.state_names:
        y0[-1] = model_spec.params.adaptable_params['t_ref']

    # Run chosen model with parameter set
    if "I_d" in model_spec.state_names:
        [T_func, sol] = run_model_with_induction(model_spec.func, y0, t_exp, T_exp, DT_exp, model_spec.params)
    else:
        [T_func, sol] = run_model(model_spec.func, y0, t_exp, T_exp, DT_exp, model_spec.params)


    # Extract solution
    t_model = sol.t
    state_dict = dict(zip(model_spec.state_names, sol.y))

    alpha_model_solver = state_dict["alpha"]

    # Compute model heat flow on solver grid
    dalpha_dt_solver = np.gradient(alpha_model_solver, t_model)
    HF_model_solver = deltaH_m * dalpha_dt_solver

    # Interpolate model output back to experimental grid for plotting/comparison
    alpha_model = np.interp(t_exp, t_model, alpha_model_solver)
    HF_model = np.interp(t_exp, t_model, HF_model_solver)

    # Extract variables needed for spherulite size distribution
    F_model = state_dict["F"]
    Na_model = state_dict["Na"]

    dalpha_dt = np.gradient(state_dict["alpha"], t_exp)
    HF_model = deltaH_m * dalpha_dt
    alpha_model = state_dict["alpha"]

    print(Na_model[-1])

    if plot_Na == 1 and sheet_name == 'Ts164_C10':
        fig, ax1 = plt.subplots(figsize=(4,3)) 
        ax1.scatter(T_exp-273.15, state_dict["Na"], c='#DC2593',s=15) 
        ax1.set_ylabel(r'$N_\mathrm{a}$ (nuclei/m$^3$)', color='#DC2593') 
        ax1.tick_params(axis='y', labelcolor='#DC2593') 
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_xlim(90,150)

        ax2 = ax1.twinx() 
        ax2.scatter(T_exp-273.15, state_dict["alpha"], c='#11A3D8',s=15) 
        ax2.set_ylabel('α (-)', color='#11A3D8', rotation=270, labelpad = 15) 
        ax2.tick_params(axis='y', labelcolor='#11A3D8') 

        plt.tight_layout()
        plt.savefig("Na evolution.png", dpi=600)
        plt.show()

    result=[]

    # Store results
    result = {
        'sheet': sheet_name,
        't_exp': t_exp,
        'T_exp': T_exp,
        'HF_exp': HF_exp,
        'HF_model': HF_model,
        'alpha_exp': alpha_exp,
        'alpha_model': alpha_model,
        'weight': weight
    }

    if plot_spherulites == 1:
        T_model = T_func(t_model)
        G_model = model_spec.params.functions["G"](T_model, model_spec.params)

        result["histograms"] = evolve_spherulites(
            dimensions=3,
            t=t_model,
            alpha=alpha_model_solver,
            G=G_model,
            F=F_model,
            Na=Na_model,
            bins=100,
            size_range_um=(0, 60),
            size_type="diameter"
        )

        # Store the final spherulite size distribution
        result["final_distribution"] = result["histograms"][-1]

    results.append(result)

    print(f"Done processing sheet {sheet_name}")

print("Done processing all sheets.")

# Plot results
plot_results(results, xlim=30000, xmin=20, xmax=120)

if plot_spherulites == 1:
    plot_spherulites_from_histograms(results, time_index=-1, normalize=True)