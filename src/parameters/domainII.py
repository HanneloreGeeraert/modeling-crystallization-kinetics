import numpy as np
from .base import ModelParams
from .base import G_LH, q_1, q_2, N0_logistic, dN0dT_logistic, alpha_max_test, N0_fixed, dN0dT_fixed

# Parameters that will be used:
# Fixed parameters (never optimized)
fixed_params = dict(
    R=8.314, 
    U=6270,
    Ts=None, # This will be set per data sheet
)

adaptable_DII_3D = dict(
    T_infty=267.85-30,
    Tm0=454.65,
    G0=28.05607404,
    Kg=327350.6189,
    q1_0= 1.40163003e-05,
    q1_1=8.28730989e-03,
    q2_0=4.04595759e-02,
    q2_1=1.25382425e-01,
    T0=150+273.15,
    N0=6.86950184e+12
)

DII_3D = ModelParams(
    functions={
        "G": G_LH,
        "q1": q_1,
        "q2": q_2,
        "N0": N0_fixed,
        "dN0dT": dN0dT_fixed
    },
    fixed_params=fixed_params.copy(),
    adaptable_params=adaptable_DII_3D
)