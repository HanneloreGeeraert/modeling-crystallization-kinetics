import numpy as np
from src.models.kinetics import run_model
from src.models.utils import get_deltaHm, load_sheets
from src.processing.plotting import plot_results
from src.processing.size_distribution_discrete import plot_spherulites, evolve_spherulites
from src.processing.data import get_dataset
from src.models.specs import noniso_DII_2D_spec, noniso_DII_3D_spec, noniso_DII_2D_sc_spec, noniso_DII_3D_sc_spec

# Load experimental data 
[all_sheets, sheet_names] = get_dataset("non_iso_DII")

results = []

# Select model and parameters 
model_spec = noniso_DII_3D_spec

for i, sheet_name in enumerate(sheet_names):
    
    if model_spec.mode == "noniso_SN":    
        # Extract Ts from sheet name
        Ts = float(sheet_name.split("Ts")[1].split("_")[0]) + 273.15
        model_spec.params.fixed_params['Ts'] = Ts
        if Ts == 176 + 273.15:
            model_spec.params.adaptable_params['Tsref'] = 105 + 273.15 

    df = all_sheets[sheet_name]
    t_exp = df['StepTime_sec'].values
    T_exp = df['Temperature'].values + 273.15
    DT_exp = df['DT'].values / 60
    HF_exp = df['HF_Corrected_x_W'].values
    alpha_exp = df['alpha_x_weight'].values
    deltaH_m = get_deltaHm(df['Int'])

    # Initial conditions for the model
    y0 = model_spec.make_y0(sheet_name, df)

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
    [rho, dn_kl] = evolve_spherulites(model_spec.dimensions, t_exp, state_dict['alpha'], G_model, state_dict["F"], state_dict["Na"])

    # Store results
    results.append({
        'sheet': sheet_name,
        't_exp': t_exp,
        'T_exp': T_exp,
        'HF_exp': HF_exp,
        'HF_model': HF_model,
        'alpha_exp': alpha_exp,
        'alpha_model': alpha_model,
        'rho': rho,
        'dn_kl': dn_kl
    })

print("Done processing all sheets.")

# Plot results
plot_results(results, mode = model_spec.mode)
plot_spherulites(model_spec.dimensions, results, mode = model_spec.mode, i = -1)