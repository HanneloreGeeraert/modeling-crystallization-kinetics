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
    q1_0=4e-6,
    q1_1=1.42e-1,
    q2_0=4.66e-4,
    q2_1=9.20e-2,
    T0=150+273.15,
    N0=5e+10
)

DII_3D = ModelParams(
    functions={
        "G": G_LH, # pyright: ignore[reportArgumentType]
        "q1": q_1,
        "q2": q_2,
        "N0": N0_fixed,
        "dN0dT": dN0dT_fixed
    },
    fixed_params=fixed_params.copy(),
    adaptable_params=adaptable_DII_3D
)