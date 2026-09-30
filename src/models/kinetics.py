from ast import arg

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from types import SimpleNamespace


def induction_event(t, y):
    return y[7]   # I_d

induction_event.terminal = True # type: ignore[attr-defined]
induction_event.direction = -1 # type: ignore[attr-defined]


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


def run_model_with_induction(model_func, y0, t_exp, T_exp, DT_exp, params):
    # Interpolation functions (same for any model)
    T_func = interp1d(t_exp, T_exp, kind='linear', fill_value='extrapolate') # type: ignore
    DT_func = interp1d(t_exp, DT_exp, kind='linear', fill_value='extrapolate') # type: ignore

    # First solve with induction
 # ------------------------
    # Phase 1: induction
    # ------------------------

    sol1 = solve_ivp(
        lambda t, y: model_func(
            t, y,
            T_func,
            DT_func,
            params,
            induction_finished=False
        ),
        t_span=(t_exp[0], t_exp[-1]),
        y0=y0,
        method='Radau',
        events=induction_event,
        rtol=1e-6,
        atol=1e-9
    )

    if not sol1.success:
        raise RuntimeError(
            f"Phase 1 failed: {sol1.message}"
        )

    # No induction event found
    if len(sol1.t_events[0]) == 0:

        sol = solve_ivp(
            lambda t, y: model_func(
                t, y,
                T_func,
                DT_func,
                params,
                induction_finished=False
            ),
            t_span=(t_exp[0], t_exp[-1]),
            y0=y0,
            t_eval=t_exp,
            method='Radau',
            rtol=1e-6,
            atol=1e-9
        )

        return T_func, sol

    # ------------------------
    # Event found
    # ------------------------

    t_switch = sol1.t_events[0][0]

    print(f"Induction completed at t = {t_switch:.3f} s")

    y_switch = sol1.y_events[0][0].copy()

    # force exact switching state
    y_switch[6] = 0.0

    # evaluation points after induction
    mask = t_exp >= t_switch
    t_eval2 = t_exp[mask]

    # ------------------------
    # Phase 2: crystallization
    # ------------------------

    sol2 = solve_ivp(
        lambda t, y: model_func(
            t, y,
            T_func,
            DT_func,
            params,
            induction_finished=True
        ),
        t_span=(t_switch, t_exp[-1]),
        y0=y_switch,
        t_eval=t_eval2,
        method='Radau',
        rtol=1e-6,
        atol=1e-9
    )

    if not sol2.success:
        raise RuntimeError(
            f"Phase 2 failed: {sol2.message}"
        )

    # ------------------------
    # Reconstruct full solution
    # ------------------------

    mask1 = t_exp < t_switch

    t_combined = np.concatenate([
        t_exp[mask1],
        sol2.t
    ])

    y_pre = np.empty((len(y0), np.sum(mask1)))

    for i in range(len(y0)):
        y_pre[i] = np.interp(
            t_exp[mask1],
            sol1.t,
            sol1.y[i]
        )

    y_combined = np.hstack([
        y_pre,
        sol2.y
    ])

    sol = SimpleNamespace(
        t=t_combined,
        y=y_combined,
        success=True
    )

    return T_func, sol

def haudin_chenot_3D(t, y, T_func, DT_func, params):
    N, alpha, Na, Ntilde_a, F, P, Q = y
    
    T = T_func(t)
    DT = DT_func(t)
    
    # Call model functions from params
    q  = params.functions["q_1"](T, params)
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

def haudin_chenot_3D_twostep(t, y, T_func, DT_func, params):
    N, N1, alpha, Na, Ntilde_a, F, P, Q = y
    
    T = T_func(t)
    DT = DT_func(t)
    
    # Call model functions from params
    q1  = params.functions["q_1"](T, params)
    q2  = params.functions["q_2"](T, params)
    G  = params.functions["G"](T, params)
    N0 = params.functions["N0"](params)
    dN0dT = params.functions["dN0dT"](N0, params)

    eps = 1e-10
    one_minus_alpha = max(1 - alpha, eps)
    
    # ODE system
    dNa_dt = q2 * N1
    dNtilde_a_dt = q2 * N1 / one_minus_alpha
    dF_dt = G
    dP_dt = F * dNtilde_a_dt
    dQ_dt = F**2 * dNtilde_a_dt
    dalpha_dt = 4 * np.pi * one_minus_alpha * G * (F**2 * Ntilde_a - 2 * F * P + Q)
    dN_dt = -q1 * N - N * dalpha_dt / one_minus_alpha 
    dN1_dt = q1 * N - q2 * N1  - N1 * dalpha_dt / one_minus_alpha

    return [dN_dt, dN1_dt, dalpha_dt, dNa_dt, dNtilde_a_dt, dF_dt, dP_dt, dQ_dt]

def haudin_chenot_2D(t, y, T_func, DT_func, params):
    N, alpha, Na, Ntilde_a, F, P = y
    
    T = T_func(t)
    DT = DT_func(t)
    
    # Call model functions from params
    q = params.functions["q_1"](T, params)
    G = params.functions["G"](T, params)
    N0 = params.functions["N0"](params)

    eps = 1e-10
    one_minus_alpha = np.maximum(1 - alpha, eps)
    
    # ODE system
    dNa_dt = q * N
    dNtilde_a_dt = dNa_dt / one_minus_alpha
    dF_dt = G
    dP_dt = F * dNtilde_a_dt
    dalpha_dt = 2 * np.pi * one_minus_alpha * G * (F * Ntilde_a - P)
    dN_dt = - dNa_dt - N * dalpha_dt / one_minus_alpha

    return [dN_dt, dalpha_dt, dNa_dt, dNtilde_a_dt, dF_dt, dP_dt]

def haudin_chenot_2D_induction(t, y, T_func, DT_func, params, induction_finished = False):
    N, alpha, Na, Ntilde_a, F, P, I_d = y
    
    T = T_func(t)
    DT = DT_func(t)
    
    # Call model functions from params
    q = params.functions["q_1"](T, params)
    G = params.functions["G"](T, params)
    C = params.adaptable_params['C']

    eps = 1e-10
    one_minus_alpha = max(1 - alpha, eps)
    
    if induction_finished:
        K2 = 1.0
        dId_dt = 0.0
    else:
        K2 = 0.0
        arg = np.clip(C * (423.15 / T - 1), -30, 30)
        dId_dt = -np.exp(arg)

    dNa_dt = q * N * K2
    dNtilde_a_dt = dNa_dt / one_minus_alpha
    dF_dt = G
    dP_dt = F * dNtilde_a_dt
    dalpha_dt = 2 * np.pi * one_minus_alpha * G * (F * Ntilde_a - P)
    dN_dt = - dNa_dt - N * dalpha_dt / one_minus_alpha

    return [dN_dt, dalpha_dt, dNa_dt, dNtilde_a_dt, dF_dt, dP_dt, dId_dt]

def haudin_chenot_3D_induction(t, y, T_func, DT_func, params, induction_finished = False):
    N, alpha, Na, Ntilde_a, F, P, Q, I_d = y
    
    T = T_func(t)
    DT = DT_func(t)
    
    # Call model functions from params
    q = params.functions["q_1"](T, params)
    G = params.functions["G"](T, params)
    C = params.adaptable_params['C']

    eps = 1e-10
    one_minus_alpha = max(1 - alpha, eps)
    
    if induction_finished:
        K2 = 1.0
        dId_dt = 0.0
    else:
        K2 = 0.0
        arg = np.clip(C * (423.15 / T - 1), -30, 30)
        dId_dt = -np.exp(arg)

    dNa_dt = q * N * K2
    dNtilde_a_dt = dNa_dt / one_minus_alpha
    dF_dt = G
    dP_dt = F * dNtilde_a_dt
    dQ_dt = F**2 * dNtilde_a_dt
    dalpha_dt = 4 * np.pi * one_minus_alpha * G * (F**2 * Ntilde_a - 2 * F * P + Q)
    dN_dt = - dNa_dt - N * dalpha_dt / one_minus_alpha

    return [dN_dt, dalpha_dt, dNa_dt, dNtilde_a_dt, dF_dt, dP_dt, dQ_dt, dId_dt]

def haudin_chenot_3D_sat(t, y, T_func, DT_func, params):
    # State variables
    N, alpha, Na, Ntilde_a, F, P, Q = y
    
    T = T_func(t)
    
    q  = params.functions["q_1"](T, params)
    G  = params.functions["G"](T, params)
    N0 = params.functions["N0"](params)

    eps = 1e-10
    one_minus_alpha = max(1 - alpha, eps)

    # Saturation logic
    if Na < N0:
        dNa_dt = q * N0
    else:
        dNa_dt = 0.0
        Na = N0  # clamp to avoid numerical drift

    dNtilde_a_dt = dNa_dt / one_minus_alpha

    dF_dt = G
    dP_dt = F * dNtilde_a_dt
    dQ_dt = F**2 * dNa_dt / one_minus_alpha

    dalpha_dt = (
        4 * np.pi
        * one_minus_alpha
        * G
        * (F**2 * Ntilde_a - 2 * F * P + Q)
    )
    dN_dt = 0

    return [dN_dt, dalpha_dt, dNa_dt, dNtilde_a_dt, dF_dt, dP_dt, dQ_dt]
