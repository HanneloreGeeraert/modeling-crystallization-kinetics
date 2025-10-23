import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def run_model(model_func, y0, t_exp, T_exp, DT_exp, params):
    # Interpolation functions (same for any model)
    T_func = interp1d(t_exp, T_exp, kind='linear', fill_value='extrapolate') # type: ignore
    DT_func = interp1d(t_exp, DT_exp, kind='linear', fill_value='extrapolate') # type: ignore

    # Solve the system
    sol = solve_ivp(
        lambda t, y: model_func(t, y, T_func, DT_func, params),
        t_span=(t_exp[0], t_exp[-1]),
        y0=y0,
        t_eval=t_exp,
        method='Radau',
        rtol=1e-6,
        atol=1e-9
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return T_func, sol

def haudin_chenot_3D_twostep(t, y, T_func, DT_func, params):
    N, Ni, alpha, Na, Ntilde_a, F, P, Q = y

    T = T_func(t)
    DT = DT_func(t)

    q1  = params.functions["q1"](T, params)
    q2  = params.functions["q2"](T, params)
    G   = params.functions["G"](T, params)
    N0  = params.functions["N0"](params)
    dN0dT = params.functions["dN0dT"](N0, params)

    eps = 1e-10
    one_minus_alpha = np.clip(1.0 - alpha, eps, 1.0)

    # Growth and alpha
    dF_dt = G
    dNtilde_a_dt = 0.0
    dP_dt = 0.0
    dQ_dt = 0.0

    # Activation and impingement
    dNa_dt = q2 * Ni
    dNtilde_a_dt = dNa_dt / one_minus_alpha
    dP_dt = F * dNtilde_a_dt
    dQ_dt = F**2 * dNa_dt / one_minus_alpha
    dalpha_dt = 4.0 * np.pi * one_minus_alpha * G * (F**2 * Ntilde_a - 2.0 * F * P + Q)

    # Two-step activation + impingement + thermal term
    dN_dt  = -q1 * N - (N / one_minus_alpha) * dalpha_dt + (one_minus_alpha) * dN0dT * DT
    dNi_dt =  q1 * N - q2 * Ni - (Ni / one_minus_alpha) * dalpha_dt

    return [dN_dt, dNi_dt, dalpha_dt, dNa_dt, dNtilde_a_dt, dF_dt, dP_dt, dQ_dt]

def haudin_chenot_3D(t, y, T_func, DT_func, params):
    N, alpha, Na, Ntilde_a, F, P, Q = y
    
    T = T_func(t)
    DT = DT_func(t)
    
    # Call model functions from params
    q  = params.functions["q2"](T, params)
    G  = params.functions["G"](T, params)
    N0 = params.functions["N0"](params)
    dN0dT = params.functions["dN0dT"](N0, params)

    eps = 1e-10
    one_minus_alpha = max(1 - alpha, eps)
    
    # ODE system
    dNa_dt = q * N
    dNtilde_a_dt = q * N / one_minus_alpha
    dF_dt = G
    dP_dt = F * dNtilde_a_dt
    dQ_dt = F**2 * q * N / one_minus_alpha
    dalpha_dt = 4 * np.pi * one_minus_alpha * G * (F**2 * Ntilde_a - 2 * F * P + Q)
    dN_dt = - N * (q + (1 / one_minus_alpha) * dalpha_dt) + one_minus_alpha * dN0dT * DT

    return [dN_dt, dalpha_dt, dNa_dt, dNtilde_a_dt, dF_dt, dP_dt, dQ_dt]


def haudin_chenot_3D_test(t, y, T_func, DT_func, params):
    N, alpha, Na, Ntilde_a, F, P, Q = y
    
    T = T_func(t)
    DT = DT_func(t)
    
    # Call model functions from params
    N0 = params.functions["N0"](params)
    I  = params.functions["q2"](T, params) * N0
    G  = params.functions["G"](T, params)

    dN0dT = params.functions["dN0dT"](N0, params)

    eps = 1e-10
    one_minus_alpha = max(1 - alpha, eps)
    
    # ODE system
    if Na <= N0:
        dNa_dt = I
    else:
        dNa_dt = 0
    dNtilde_a_dt = dNa_dt / one_minus_alpha
    dF_dt = G
    dP_dt = F * dNtilde_a_dt
    dQ_dt = F**2 * dNa_dt / one_minus_alpha
    dalpha_dt = 4 * np.pi * one_minus_alpha * G * (F**2 * Ntilde_a - 2 * F * P + Q)
    dN_dt = max(0, - I - N *(1 / one_minus_alpha) * dalpha_dt)

    return [dN_dt, dalpha_dt, dNa_dt, dNtilde_a_dt, dF_dt, dP_dt, dQ_dt]






# Models not used
def haudin_chenot_3D_sharp(t, y, T_func, DT_func, params):
    N, alpha, Na, Ntilde_a, F, P, Q = y

    # Temperature and derivatives
    T = T_func(t)
    DT = DT_func(t)

    # Model functions
    q  = params.functions["q2"](T, params)
    G  = params.functions["G"](T, params)
    N0 = params.functions["N0"](params)
    dN0dT = params.functions["dN0dT"](N0, params)

    eps = 1e-12
    one_minus_alpha = max(1 - alpha, eps)

    # --- Kinetically limited activation with fixed smooth saturation ---
    smooth_eps = 1e-3 * N0
    arg = np.clip((N0 - Na) / smooth_eps, -60, 60)
    S = 1.0 / (1.0 + np.exp(-arg))  # S≈1 early, S→0 near saturation

    # Activation flux
    dNa_dt = q * S
    dNtilde_a_dt = dNa_dt / one_minus_alpha

    # Growth and morphology
    dF_dt = G
    dP_dt = F * dNtilde_a_dt
    dQ_dt = F**2 * dNtilde_a_dt

    # Transformation kinetics
    dalpha_dt = 4 * np.pi * one_minus_alpha * G * (F**2 * Ntilde_a - 2 * F * P + Q)

    # Potential nuclei evolution
    dN_dt = -dNa_dt + one_minus_alpha * dN0dT * DT

    return [dN_dt, dalpha_dt, dNa_dt, dNtilde_a_dt, dF_dt, dP_dt, dQ_dt]

