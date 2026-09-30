# pyright: reportCallIssue=false

import numpy as np
from scipy.optimize import differential_evolution, minimize

# === Local imports ===
from src.models.kinetics import run_model, run_model_with_induction
from src.models.utils import get_deltaHm
from src.processing.plotting import plot_results
from src.processing.data_loadonce import get_dataset
from src.models.specs import get_modelspec

# -------------------------------------------------------------------------
# === Define parameters and bounds ===
# -------------------------------------------------------------------------

log_params = ["Kg", "G0", "C"]
lin_params = ["t_ref"]
param_names = log_params + lin_params

param_bounds = {
    "G0": (10,1000),
    "Kg": (3.5e5,4.5e5), 
    "C": (25,75),
    "t_ref": (1000,20000)}

# -------------------------------------------------------------------------
# === Helper functions ===
# -------------------------------------------------------------------------

def safe_exp(x, limit=40.0):
    """Exponentiate safely to prevent overflow."""
    return np.exp(np.clip(x, -limit, limit))


def run_model_and_compute_loss(model_spec, all_sheets, sheet_names, params):
    """Runs model and returns total normalized MSE loss."""
    total_error = 0.0
    total_weight = 0.0

    sheet_weights = {
        "Ts176_C1": 3.0,
        "Ts176_CR2": 1.0,
        "Ts176_CR5": 1.0,
        "Ts176_CR10": 1.0,
        "Ts176_CR30": 3.0,
    }

    # Assign parameters to model
    for name, value in params.items():
        model_spec.params.adaptable_params[name] = value

    for sheet_name in sheet_names:
        Ts = float(sheet_name.split("Ts")[1].split("_")[0]) + 273.15
        model_spec.params.fixed_params["Ts"] = Ts

        df = all_sheets[sheet_name]
        t_exp = df["StepTime (s)"].values
        T_exp = df["Temperature (°C)"].values + 273.15
        DT_exp = df["DT (K/min)"].values / 60
        HF_exp = df["Heat Flow Baseline Corrected (W/g)"].values
        alpha_exp = df["Alpha (-)"].values
        deltaH_m = model_spec.params.functions["deltaH_m"](model_spec.params)

        if np.max(np.abs(HF_exp)) < 1e-6:
            continue

        N0  = model_spec.params.functions["N0"](model_spec.params)
        y0 = model_spec.make_y0(sheet_name, df)
        
        y0[0] = N0
        y0[-1] = model_spec.params.adaptable_params['t_ref']

        try:
            [T_func, sol] = run_model_with_induction(model_spec.func, y0, t_exp, T_exp, DT_exp, model_spec.params)
            if not sol.success:
                print(f"Solver failed for {params}")
                print(sol.message)
                return 1e10
            if not np.all(np.isfinite(sol.y)):
                print(f"Non-finite solution for {params}")
                return 1e10
        except Exception as e:
            print("Objective failed:", e)
            raise

        state_dict = dict(zip(model_spec.state_names, sol.y))

        mask = (alpha_exp >= 0.05) & (alpha_exp <= 0.5)
        if not np.any(mask):
            continue

        alpha_exp = alpha_exp[mask]
        alpha_model = state_dict["alpha"][mask]

        eps = 1e-8
        mse = np.mean((alpha_exp - alpha_model) ** 2)

        weight = sheet_weights.get(sheet_name, 1.0)

        total_error += weight * mse
        total_weight += weight

    return total_error/total_weight


# -------------------------------------------------------------------------
# === Mixed-scale parameter handling ===
# -------------------------------------------------------------------------

# --- Construct bounds in the same order as param_names
bounds = []
for name in param_names:
    low, high = param_bounds[name]
    if name in log_params:
        bounds.append((np.log(low), np.log(high)))
    else:
        bounds.append((low, high))

def unpack_params(x):
    """Convert optimizer vector x into model parameters."""
    params = {}
    for i, name in enumerate(param_names):
        if name in log_params:
            params[name] = safe_exp(x[i])
        else:
            params[name] = x[i]
    return params


# -------------------------------------------------------------------------
# === Objective function (mixed-scale) ===
# -------------------------------------------------------------------------

# Define globals at the top of your script
best_loss = np.inf
best_x = None

def objective_function_mixed(x, model_spec, all_sheets, sheet_names):
    global best_loss, best_x

    try:
        params = unpack_params(x)
        local_spec = get_modelspec("DII_3D_induction")
        loss = run_model_and_compute_loss(local_spec, all_sheets, sheet_names, params)
        print(f"Test params {params} -> total error {loss:.3e}")

        # Handle NaN or inf
        if not np.isfinite(loss):
            return 1e10

        # Track best-so-far
        if loss < best_loss:
            best_loss = loss
            best_x = x.copy()
            print("🔥 New best:", best_loss, best_x)

        return loss

    except Exception as e:
        print("⚠️ Model crashed for params:", x)
        print("Error:", e)
        return 1e9


# -------------------------------------------------------------------------
# === Main optimization routine ===
# -------------------------------------------------------------------------

def main():
    print("Loading dataset from Excel...")
    all_sheets, sheet_names = get_dataset("noniso_Ts176", reload=True)
    model_spec = get_modelspec("DII_3D_induction")

    # ------------------ GLOBAL OPTIMIZATION ------------------
    print("\n=== Global optimization (Differential Evolution) ===")

    result_global = differential_evolution(
        func=objective_function_mixed,
        bounds=bounds,
        args=(model_spec, all_sheets, sheet_names),
        maxiter=40,
        popsize=5,
        disp=True,
        polish=False,
        seed=42)

    print("\nGlobal optimization complete.")
    print("Best parameters (raw optimizer values):", result_global.x)

    # ------------------ LOCAL REFINEMENT ------------------
    print("\n=== Local optimization (L-BFGS-B) ===")
    result_local = minimize(
        objective_function_mixed,
        result_global.x,
        args=(model_spec, all_sheets, sheet_names),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 20,   # maximum iterations
            "maxfun": 50,    # maximum function evaluations
            "ftol": 1e-6,      # tolerance for function value convergence
            "gtol": 1e-6      # tolerance for gradient norm
        },
    )
    best_params = unpack_params(result_local.x)
    print("\n=== Optimization complete ===")
    for name, value in best_params.items():
        print(f"  {name} = {value:.4e}")

    # ------------------ Final model run and plot ------------------
    # Rebuild a clean model spec so no mutated state remains
    model_spec = get_modelspec("DII_3D_induction")

    # Apply optimized parameters
    for name, value in best_params.items():
        model_spec.params.adaptable_params[name] = value

    results = []
    for sheet_name in sheet_names:
        Ts = float(sheet_name.split("Ts")[1].split("_")[0]) + 273.15
        model_spec.params.fixed_params["Ts"] = Ts

        df = all_sheets[sheet_name]
        t_exp = df["StepTime (s)"].values
        T_exp = df["Temperature (°C)"].values + 273.15
        DT_exp = df["DT (K/min)"].values / 60
        HF_exp = df["Heat Flow Baseline Corrected (W/g)"].values
        alpha_exp = df["Alpha (-)"].values
        weight = df["Weight (-)"].values
        deltaH_m = model_spec.params.functions["deltaH_m"](model_spec.params)


        N0  = model_spec.params.functions["N0"](model_spec.params)
        y0 = model_spec.make_y0(sheet_name, df)
        y0[0] = N0
        y0[-1] = model_spec.params.adaptable_params['t_ref']    

        [T_func, sol] = run_model_with_induction(model_spec.func, y0, t_exp, T_exp, DT_exp, model_spec.params)
        state_dict = dict(zip(model_spec.state_names, sol.y))
        dalpha_dt = np.gradient(state_dict["alpha"], t_exp)

        HF_model = deltaH_m * dalpha_dt
        alpha_model = state_dict["alpha"]

        results.append({
            "sheet": sheet_name,
            "t_exp": t_exp,
            "T_exp": T_exp,
            "HF_exp": HF_exp,
            "HF_model": HF_model,
            "alpha_exp": alpha_exp,
            "alpha_model": alpha_model,
            "weight": weight
        })

        print(f"Done processing sheet {sheet_name}")

    print("Done processing all sheets — now plotting.")
    plot_results(results)


# -------------------------------------------------------------------------
# === Safe multiprocessing entry point ===
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
