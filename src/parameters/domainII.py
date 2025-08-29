import numpy as np
from .base import ModelParams
from .base import G_LH, q_HC, N0_logistic, dN0dT_logistic, alpha_max_test

# Fixed parameters (never optimized)
fixed_params = dict(
    R=8.314, 
    U=6270,
    Ts=None # This will be set per data sheet
)

# Adaptable parameters (can be optimized)
adaptable_DII_2D = dict(
    T_infty=267.85-30,
    Tm0=454.65,
    G0=28.05607404,
    Kg=327350.6189,
    T0=150+273.15, 
    q0=1,
    q1=0,
    Tsref=105+273.15,
    k_N=0.8,
    Nmax_a = -0.165920,
    Nmax_b = 82.4701
)

DII_2D = ModelParams(
    functions={
        "q": q_HC,
        "G": G_LH,
        "N0": N0_logistic,
        "dN0dT": dN0dT_logistic
    },
    fixed_params=fixed_params.copy(),
    adaptable_params=adaptable_DII_2D
)

adaptable_DII_3D = dict(
    T_infty=267.85-30,
    Tm0=454.65,
    G0=28.05607404,
    Kg=327350.6189,
    T0=150+273.15, 
    q0=1 / (2*100e-6) ,  # Adjusted for 3D model through sample thickness
    q1=0,
    Tsref=110+273.15,
    k_N=0.5,
    Nmax_a = -0.165920,
    Nmax_b = 82.4701 - np.log10(2*100e-6)  # Adjusted for 3D model through sample thickness
)

DII_3D = ModelParams(
    functions={
        "q": q_HC,
        "G": G_LH,
        "N0": N0_logistic,
        "dN0dT": dN0dT_logistic
    },
    fixed_params=fixed_params.copy(),
    adaptable_params=adaptable_DII_3D
)

# Adaptable parameters (can be optimized)
adaptable_DII_2D_sc = dict(
    T_infty=267.85-30,
    Tm0=454.65,
    G0=28.05607404,
    Kg=327350.6189,
    T0=150+273.15, 
    q0=1,
    q1=0,
    Tsref=105+273.15,
    k_N=0.8,
    Nmax_a = -0.165920,
    Nmax_b = 82.4701,
    k_s = 0.0005
)

DII_2D_sc = ModelParams(
    functions={
        "q": q_HC,
        "G": G_LH,
        "N0": N0_logistic,
        "dN0dT": dN0dT_logistic,
        "alpha_max": alpha_max_test
    },
    fixed_params=fixed_params.copy(),
    adaptable_params=adaptable_DII_2D_sc
)

adaptable_DII_3D_sc = dict(
    T_infty=267.85-30,
    Tm0=454.65,
    G0=28.05607404,
    Kg=327350.6189,
    T0=150+273.15, 
    q0=1 / 100e-6 ,  # Adjusted for 3D model through sample thickness
    q1=0,
    Tsref=107+273.15,
    k_N=0.5,
    Nmax_a = -0.165920,
    Nmax_b = 82.4701 - np.log10(100e-6),  # Adjusted for 3D model through sample thickness
    k_s = 0.5
)

DII_3D_sc = ModelParams(
    functions={
        "q": q_HC,
        "G": G_LH,
        "N0": N0_logistic,
        "dN0dT": dN0dT_logistic,
        "alpha_max": alpha_max_test
    },
    fixed_params=fixed_params.copy(),
    adaptable_params=adaptable_DII_3D_sc
)