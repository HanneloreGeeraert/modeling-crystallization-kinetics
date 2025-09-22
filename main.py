import numpy as np
from src.models.kinetics import run_model
from src.models.utils import get_deltaHm
from src.processing.plotting import plot_results
from src.processing.size_distribution_lessmemory import evolve_spherulites, plot_spherulites_from_histograms
from src.processing.data import get_dataset
from src.models.specs import get_modelspec

# Load experimental data 
[all_sheets, sheet_names] = get_dataset("non_iso_DII")

results = []

# Select model and parameters 
model_spec = get_modelspec("DII_3D_spec")

for i, sheet_name in enumerate(sheet_names):
    
    Ts = float(sheet_name.split("Ts")[1].split("_")[0]) + 273.15
    model_spec.params.fixed_params['Ts'] = Ts

    df = all_sheets[sheet_name]
    t_exp = df['StepTime_sec'].values
    T_exp = df['Temperature'].values + 273.15
    DT_exp = df['DT'].values / 60
    HF_exp = df['HF_Corrected_x_W'].values
    alpha_exp = df['alpha_x_weight'].values
    weight = df['Weight'].values
    deltaH_m = get_deltaHm(df['Int'])

    # Initial conditions for the model
    y0 = model_spec.make_y0(sheet_name, df)
    y0[0] = 9e12

    # Run chosen model with parameter set
    [T_func, sol] = run_model(model_spec.func, y0, t_exp, T_exp, DT_exp, model_spec.params)

    # Extract solution
    state_dict = dict(zip(model_spec.state_names, sol.y))
   
    dalpha_dt = np.gradient(state_dict["alpha"], t_exp)

    # Check if "alphas" exists in solution
    if "alphas" in state_dict:
        dalphas_dt = np.gradient(state_dict["alphas"], t_exp)
        HF_model = deltaH_m * (dalpha_dt + dalphas_dt)
        alpha_model = state_dict["alpha"] + state_dict["alphas"]
    else:
        HF_model = deltaH_m * dalpha_dt
        alpha_model = state_dict["alpha"]

    # Recompute G(t) on the solver grid (same interpolation you already use)
    G_model  = model_spec.params.functions["G"](T_func(t_exp), model_spec.params)

    # Mean radius (matrix over i>j)
    histograms = evolve_spherulites(model_spec.dimensions, t_exp, state_dict['alpha'], G_model, state_dict["F"], state_dict["Na"], bins=100, plot=False)

    # Store results
    results.append({
        'sheet': sheet_name,
        't_exp': t_exp,
        'T_exp': T_exp,
        'HF_exp': HF_exp,
        'HF_model': HF_model,
        'alpha_exp': alpha_exp,
        'alpha_model': alpha_model,
        'weight': weight,
        'histograms': histograms
    })

    print(f"Done processing sheet {sheet_name}")

print("Done processing all sheets.")

# Plot results
plot_results(results)
plot_spherulites_from_histograms(results, time_index=-1)