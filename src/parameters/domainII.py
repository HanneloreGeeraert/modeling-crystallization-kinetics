import numpy as np
from .base import ModelParams
from .base import G_LH, q_1, q_2, N0_fixed, N0_logistic, dN0dT_fixed

# Parameters that will be used:
# Fixed parameters (never optimized)
fixed_params = dict(
    R=8.314, 
    U=6270,
    Ts=None, # This will be set per data sheet
)

adaptable_DII_3D = dict(
    T_infty=237.85,
    Tm0=454.65,
    G0=104.605607404,
    Kg=369500,
    #q10_0=0.747e-7,
    #q1_1=0.29593,
    q2_0=8.58537e-4,
    q2_1=0.08102793,
    #N0_164=202635789141506.56,
    #N0_168=83572747116894.94,
    N0_172=1.0228e13,
    N0_176=1.0302e12,
    #Tsref=456,
    #Nmax=3e+14,
    #k= -2.1e-01
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