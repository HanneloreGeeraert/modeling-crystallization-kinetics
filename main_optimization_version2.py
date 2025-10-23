# pyright: reportCallIssue=false

import numpy as np
from scipy.optimize import differential_evolution, minimize

# === Local imports ===
from src.models.kinetics import run_model
from src.models.utils import get_deltaHm
from src.processing.plotting import plot_results
from src.processing.data_loadonce import get_dataset
from src.models.specs import get_modelspec

# -------------------------------------------------------------------------
# === Define parameters and bounds ===
# -------------------------------------------------------------------------

log_params = ["q2_0", "q2_1"]
lin_params = ["N0_164", "N0_168", "N0_172", "N0_176"]
param_names = log_params + lin_params

param_bounds = {
    "q2_0": (1e-7, 1e-3),
    "q2_1": (1e-2, 1e0),
    "N0_164": (1e13, 1e15),
    "N0_168": (6e12, 6e14),
    "N0_172": (2e12, 2e14),
    "N0_176": (1e11, 1e13)
}


# -------------------------------------------------------------------------
# === Helper functions ===
# -------------------------------------------------------------------------

def safe_exp(x, limit=40.0):
    """Exponentiate safely to prevent overflow."""
    return np.exp(np.clip(x, -limit, limit))


def run_model_and_compute_loss(model_spec, all_sheets, sheet_names, params):
    """Runs model and returns total normalized MSE loss."""
    total_error = 0.0

    # Assign parameters to model
    for name, value in params.items():
        model_spec.params.adaptable_params[name] = value

    for sheet_name in sheet_names:
        Ts = float(sheet_name.split("Ts")[1].split("_")[0]) + 273.15
        model_spec.params.fixed_params["Ts"] = Ts

        df = all_sheets[sheet_name]
        t_exp = df["StepTime_sec"].values
        T_exp = df["Temperature"].values + 273.15
        DT_exp = df["DT"].values / 60
        HF_exp = df["HF_Corrected_x_W"].values
        alpha_exp = df["alpha_x_weight"].values
        deltaH_m = get_deltaHm(df["Int"])

        if np.max(np.abs(HF_exp)) < 1e-6:
            continue

        N0  = model_spec.params.functions["N0"](model_spec.params)
        y0 = model_spec.make_y0(sheet_name, df)
        y0[0] = N0

        try:
            [T_func, sol] = run_model(model_spec.func, y0, t_exp, T_exp, DT_exp, model_spec.params)
        except Exception as e:
            print(f"[Invalid params] {params} → {e}")
            return 1e6 

        state_dict = dict(zip(model_spec.state_names, sol.y))
        dalpha_dt = np.gradient(state_dict["alpha"], t_exp)

        if "alphas" in state_dict:
            dalphas_dt = np.gradient(state_dict["alphas"], t_exp)
            HF_model = deltaH_m * (dalpha_dt + dalphas_dt)
        else:
            HF_model = deltaH_m * dalpha_dt

        if np.any(np.isnan(HF_model)) or np.any(np.isinf(HF_model)):
            return np.inf

        mask = (alpha_exp >= 0.05) & (alpha_exp <= 0.4)
        if not np.any(mask):
            continue

        alpha_exp = alpha_exp[mask]
        alpha_model = state_dict["alpha"][mask]

        eps = 1e-8
        mse = np.mean(((alpha_exp - alpha_model) / (np.max(np.abs(alpha_exp)) + eps)) ** 2)
        total_error += mse / len(sheet_names)

    return total_error


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

def objective_function_mixed(x, model_spec, all_sheets, sheet_names):
    try:
        params = unpack_params(x)
        loss = run_model_and_compute_loss(model_spec, all_sheets, sheet_names, params)
        print(f"Test params {params} -> total error {loss:.3e}")

        if not np.isfinite(loss):
            return 1e10
        return loss
    except Exception:
        return 1e10


# -------------------------------------------------------------------------
# === Main optimization routine ===
# -------------------------------------------------------------------------

def main():
    print("Loading dataset from Excel...")
    all_sheets, sheet_names = get_dataset("noniso_DII", reload=True)
    model_spec = get_modelspec("DII_3D")

    # ------------------ GLOBAL OPTIMIZATION ------------------
    print("\n=== Global optimization (Differential Evolution) ===")

    result_global = differential_evolution(
        func=objective_function_mixed,
        bounds=bounds,
        args=(model_spec, all_sheets, sheet_names),
        maxiter=20,
        popsize=8,
        disp=True,
        polish=False,
        seed=42,
    )

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
            "maxiter": 2000,   # maximum iterations
            "maxfun": 4000,    # maximum function evaluations
            "ftol": 1e-8,      # tolerance for function value convergence
            "gtol": 1e-8,      # tolerance for gradient norm
            "verbose": 1       # print convergence messages
        },
    )
    best_params = unpack_params(result_local.x)
    print("\n=== Optimization complete ===")
    for name, value in best_params.items():
        print(f"  {name} = {value:.4e}")

    # ------------------ Final model run and plot ------------------
    for name, value in best_params.items():
        model_spec.params.adaptable_params[name] = value

    results = []
    for sheet_name in sheet_names:
        Ts = float(sheet_name.split("Ts")[1].split("_")[0]) + 273.15
        model_spec.params.fixed_params["Ts"] = Ts

        df = all_sheets[sheet_name]
        t_exp = df["StepTime_sec"].values
        T_exp = df["Temperature"].values + 273.15
        DT_exp = df["DT"].values / 60
        HF_exp = df["HF_Corrected_x_W"].values
        alpha_exp = df["alpha_x_weight"].values
        weight = df["Weight"].values
        deltaH_m = get_deltaHm(df["Int"])

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
