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


# Shared functions for model parameters
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

def q_1(T, params): 
    q1_0 = params.adaptable_params['q1_0']
    q1_1 = params.adaptable_params['q1_1']
    T0 = params.adaptable_params['T0']
    return q1_0 * np.exp(-q1_1 * (T - T0))

def q_2(T, params): 
    q2_0 = params.adaptable_params['q2_0']
    q2_1 = params.adaptable_params['q2_1']
    T0 = params.adaptable_params['T0']
    return q2_0 * np.exp(-q2_1 * (T - T0))

def Nmax_exponential(params):
    Ts = params.fixed_params['Ts']
    Nmax_a = params.adaptable_params['Nmax_a']
    Nmax_b = params.adaptable_params['Nmax_b']

    Ts_DIIa = 166 + 273.15
    if Ts > Ts_DIIa:    
        log10Nmax = Nmax_b + Ts * Nmax_a
    else: 
        log10Nmax = Nmax_b + Ts_DIIa * Nmax_a
    return 10**log10Nmax

def Nmax_fixed():
    return 8e10

def N0_logistic(T, params):
    # Extract from params
    k_N = params.adaptable_params['k_N']
    Tsref = params.adaptable_params['Tsref']
    Nmax = Nmax_fixed()
    
    # Compute N0
    x = np.clip(k_N * (T - Tsref), -700, 700)
    return Nmax / (1 + np.exp(x))

def dN0dT_logistic(N0, params):
    k_N = params.adaptable_params['k_N']
    Nmax = Nmax_fixed()
    return -k_N * N0 * (1 - N0 / Nmax)

def N0_fixed(T, params):
    return 0

def dN0dT_fixed(N0, params):
    return 0

def alpha_max_test(T, params):
    eps = 1e-10
    alpha_max = 1 - eps # Default value

    if T > (95 + 273.15):
        a = -2.089853e-4
        b = 1.533829e-1
        c = -2.714187e1
        alpha_max = poly_val = a * T**2 + b * T + c

    return 0.8 #np.clip(alpha_max, 0.6, 1-eps)  # Safe lower bound