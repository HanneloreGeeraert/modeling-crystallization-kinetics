import numpy as np
from scipy.optimize import minimize

# === Local imports ===
from src.models.kinetics import run_model
from src.models.utils import get_deltaHm
from src.processing.plotting import plot_results
from src.processing.data_loadonce import get_dataset
from src.models.specs import get_modelspec


# Load experimental data
all_sheets, sheet_names = get_dataset("iso_Ts176", reload=True)

# Select model and parameters
model_spec = get_modelspec("DII_3D_onestep")

# Choose which parameters to optimize
param_names = ['U']
x0 = [model_spec.params.adaptable_params[name] for name in param_names]


# Define objective (error) function
def objective_function(param_values):
    """
    Compute total squared error between experimental and model HF curves,
    but only for alpha_exp in a certain range
    """
    for name, value in zip(param_names, param_values):
        model_spec.params.adaptable_params[name] = value

    total_error = 0.0

    for sheet_name in sheet_names:
        Ts = float(sheet_name.split("Ts")[1].split("_")[0]) + 273.15
        model_spec.params.fixed_params['Ts'] = Ts

        df = all_sheets[sheet_name]
        t_exp = df['StepTime (s)'].values
        T_exp = df['Temperature (°C)'].values + 273.15
        DT_exp = df['DT (K/min)'].values / 60
        HF_exp = df['Heat Flow Baseline Corrected (W/g)'].values
        alpha_exp = df['Alpha (-)'].values
        weight = df['Weight (-)'].values
        deltaH_m = get_deltaHm(df['Int (J/g)'].values)

        # Skip empty signals
        if np.max(np.abs(HF_exp)) < 1e-6:
            print(f"Skipping {sheet_name} — no measurable HF signal")
            continue

        N0  = model_spec.params.functions["N0"](model_spec.params)
        y0 = model_spec.make_y0(sheet_name, df)
        y0[0] = N0

        try:
            [T_func, sol] = run_model(model_spec.func, y0, t_exp, T_exp, DT_exp, model_spec.params)
        except Exception as e:
            print(f"Model failed on {sheet_name}: {e}")
            return np.inf

        state_dict = dict(zip(model_spec.state_names, sol.y))
        dalpha_dt = np.gradient(state_dict["alpha"], t_exp)

        if "alphas" in state_dict:
            dalphas_dt = np.gradient(state_dict["alphas"], t_exp)
            HF_model = deltaH_m * (dalpha_dt + dalphas_dt)
        else:
            HF_model = deltaH_m * dalpha_dt

        # Safety checks
        if np.any(np.isnan(HF_model)) or np.any(np.isinf(HF_model)):
            print(f"Invalid HF_model for {sheet_name}")
            return np.inf

        # Restrict comparison 
        mask = (alpha_exp >= 0.02) & (alpha_exp <= 0.4)
        if not np.any(mask):
            print(f"Skipping {sheet_name} — no alpha_exp in range")
            continue

        alpha_exp = alpha_exp[mask]
        alpha_model = state_dict["alpha"][mask]

        # Normalized MSE within that range
        eps = 1e-8
        scale = (np.max(alpha_exp) - np.min(alpha_exp)) + eps
        mse = np.mean(((alpha_exp - alpha_model) / scale) ** 2)
        total_error += mse / len(sheet_names)

    print(f"Test params {param_values} -> total error {total_error:.3e}")
    return total_error

# Run optimization
print("\n=== Starting optimization ===")
result = minimize(
    objective_function,
    x0,
    method='Nelder-Mead',     # derivative-free, robust for noisy models
    options={'maxiter': 25, 'disp': True}
)

print("\n=== Optimization complete ===")
for name, value in zip(param_names, result.x):
    print(f"  {name} = {value:.4e}")

# Update model with best-fit values
for name, value in zip(param_names, result.x):
    model_spec.params.adaptable_params[name] = value


# Final model run and plotting with optimized parameters
results = []

for sheet_name in sheet_names:
    Ts = float(sheet_name.split("Ts")[1].split("_")[0]) + 273.15
    model_spec.params.fixed_params['Ts'] = Ts

    df = all_sheets[sheet_name]
    t_exp = df['StepTime (s)'].values
    T_exp = df['Temperature (°C)'].values + 273.15
    DT_exp = df['DT (K/min)'].values / 60
    HF_exp = df['Heat Flow Baseline Corrected (W/g)'].values
    alpha_exp = df['Alpha (-)'].values
    weight = df['Weight (-)'].values
    deltaH_m = get_deltaHm(df['Int (J/g)'].values)

    N0  = model_spec.params.functions["N0"](model_spec.params)
    y0 = model_spec.make_y0(sheet_name, df)
    y0[0] = N0

    [T_func, sol] = run_model(model_spec.func, y0, t_exp, T_exp, DT_exp, model_spec.params)
    state_dict = dict(zip(model_spec.state_names, sol.y))
    dalpha_dt = np.gradient(state_dict["alpha"], t_exp)

    if "alphas" in state_dict:
        dalphas_dt = np.gradient(state_dict["alphas"], t_exp)
        HF_model = deltaH_m * (dalpha_dt + dalphas_dt)
        alpha_model = state_dict["alpha"] + state_dict["alphas"]
    else:
        HF_model = deltaH_m * dalpha_dt
        alpha_model = state_dict["alpha"]

    results.append({
        'sheet': sheet_name,
        't_exp': t_exp,
        'T_exp': T_exp,
        'HF_exp': HF_exp,
        'HF_model': HF_model,
        'alpha_exp': alpha_exp,
        'alpha_model': alpha_model,
        'weight': weight
    })

    print(f"Done processing sheet {sheet_name}")

print("Done processing all sheets — now plotting.")
plot_results(results)