import numpy as np

from typing import Callable, Dict, Any, Optional

class ModelParams:
    def __init__(
        self,
        functions: Dict[str, Callable[..., float]],
        fixed_params: Optional[Dict[str, Any]] = None,
        adaptable_params: Optional[Dict[str, Any]] = None
    ):
        self.functions = functions
        self.fixed_params = fixed_params or {}
        self.adaptable_params = adaptable_params or {}

def G_LH(T, params): 
    R = params.fixed_params['R']
    U = params.fixed_params['U']  

    T_infty = params.adaptable_params['T_infty']
    Tm0 = params.adaptable_params['Tm0']
    G0 = params.adaptable_params['G0']
    Kg = params.adaptable_params['Kg']
    
    # Growth rate according to Lauritzen and Hoffman (in m/s)
    delta_T = Tm0 - T
    f = 2 * T / (Tm0 + T)
    
    exp1 = np.exp(np.clip(-U / (R * (T - T_infty)), -100, 100))
    exp2 = np.exp(np.clip(-Kg / (T * delta_T * f), -100, 100))
    
    G = G0 * exp1 * exp2

    # Set G = 0 outside allowed temperature range
    G = np.where((T < -20 + 273.15) | (T > 170 + 273.15), 0.0, G)
    
    return G

# Used in two-step model:
def q_1(T, params): 
    q1_0 = params.adaptable_params['q1_0']
    q1_1 = params.adaptable_params['q1_1']
    Ts = params.fixed_params['Ts']
    
    T0 = 150 + 273.15
    
    val = q1_0 * np.exp(-q1_1 * (T - T0))
    return np.clip(val, 1e-20, 1e20)

# Used in one-step and two-step model:
def q_2(T, params): 
    q2_0 = params.adaptable_params['q2_0']
    q2_1 = params.adaptable_params['q2_1']

    T0 = 150 + 273.15

    val = q2_0 * np.exp(-q2_1 * (T - T0))
    return np.clip(val, 1e-20, 1e20)

def N0_fixed(params):
    Ts = params.fixed_params['Ts']

    if Ts == 164+273.15:
        N0 = params.adaptable_params['N0_164']
    elif Ts == 168+273.15:
        N0 = params.adaptable_params['N0_168']
    elif Ts == 172+273.15:
        N0 = params.adaptable_params['N0_172']
    elif Ts == 176+273.15:
        N0 = params.adaptable_params['N0_176']
    else:
        N0 = 1
    
    return N0

def dN0dT_fixed(N0, params):
    return 0


# Not used:
def alpha_max_test(T, params):
    eps = 1e-10
    alpha_max = 1 - eps # Default value

    if T > (95 + 273.15):
        a = -2.089853e-4
        b = 1.533829e-1
        c = -2.714187e1
        alpha_max = poly_val = a * T**2 + b * T + c

    return 0.8 #np.clip(alpha_max, 0.6, 1-eps)  # Safe lower bound

def N0_logistic(params):
    # Extract from params
    Ts = params.fixed_params['Ts']
    Tsref = params.adaptable_params['Tsref']
    Nmax = params.adaptable_params['Nmax']
    k = params.adaptable_params['k']

    # Compute N0
    log_N0 = np.log(Nmax) / (1 + np.exp(-k * (Ts - Tsref)))
    N0 = np.exp(log_N0)

    return np.clip(N0, 0, Nmax)