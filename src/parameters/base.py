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
    G = np.where((T < 20 + 273.15) | (T > 150 + 273.15), 0.0, G)
    return G

def G_fixed(T, params): 
    G0 = params.adaptable_params['G0']  
    return G0

def q1_exp(T, params): 
    q_0 = params.adaptable_params['q_10']
    q_1 = params.adaptable_params['q_11']
    
    T0 = 150 + 273.15
    
    val = q_0 * np.exp(-q_1 * (T - T0))
    return np.clip(val, 1e-20, 1e20)

def q2_exp(T, params): 
    q_0 = params.adaptable_params['q_20']
    q_1 = params.adaptable_params['q_21']
    
    T0 = 150 + 273.15
    
    val = q_0 * np.exp(-q_1 * (T - T0))
    return np.clip(val, 1e-20, 1e20)

def q_CNT(T, params): 
    A = params.adaptable_params['A']
    K = params.adaptable_params['K']
    Tm0 = params.adaptable_params['Tm0']

    val = A * np.exp(- K / (T * (Tm0 - T)**2))
    return np.clip(val, 1e-20, 1e20)

def N0_fixed(params):
    Ts = int(params.fixed_params['Ts']-273.15)
    N0 = params.adaptable_params['N0_' + str(Ts)]
    return N0

def dN0dT_fixed(N0, params):
    return 0

def deltaH_m_fixed(params):
    Ts = int(params.fixed_params['Ts']-273.15)
    deltaH_m = params.adaptable_params['deltaH_m_' + str(Ts)]
    return deltaH_m