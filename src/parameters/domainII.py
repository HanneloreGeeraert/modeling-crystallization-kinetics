import numpy as np
from .base import G_fixed, ModelParams
from .base import G_LH, q1_exp, q2_exp, q_CNT, N0_fixed, dN0dT_fixed, deltaH_m_fixed

# Fixed parameters (never optimized)
fixed_params = dict(
    R=8.314, 
    U=6270,
    Ts=None, # This will be set per data sheet
)

adaptable_DII_3D = dict(
    T_infty=245.15,
    Tm0=455.95,
    G0= 232.31,  
    Kg = 4.1097e5, 
    q_10=0.000206,
    q_11=0.155,
    N0_164= 1.5e15,
    N0_168= 6.5e14,
    N0_172=2e14,
    N0_176=1.15e13,
    N0_180= 8.5e10,
    t_ref = 9444.8,
    C = 57.491,
    deltaH_m_164= -37,
    deltaH_m_168= -39,
    deltaH_m_172= -41,
    deltaH_m_176= -42,
    deltaH_m_180= -44,
)

DII_3D = ModelParams(
    functions={
        "G": G_LH, # pyright: ignore[reportArgumentType]
        "q_1": q1_exp,
        "q_2": q2_exp,
        "N0": N0_fixed,
        "dN0dT": dN0dT_fixed,
        "deltaH_m": deltaH_m_fixed
    },
    fixed_params=fixed_params.copy(),
    adaptable_params=adaptable_DII_3D
)

adaptable_DII_2D = dict(
    T_infty=245.15,
    Tm0=454.65,
    G0=72,
    Kg=3.3e5,
    q_10=0.0000206399,
    q_11=0.1547,
    #A=0.987098626,
    #K=6492733.383,
    N0_164=3e13,
    N0_168=2e13,
    N0_172=8e12,
    N0_176=2e7,
    N0_180=3e6,
    C =-55,
    t_ref = 1.6e4
)

DII_2D = ModelParams(
    functions={
        "G": G_LH, # pyright: ignore[reportArgumentType]
        "q": q1_exp,
        "N0": N0_fixed,
        "dN0dT": dN0dT_fixed
    },
    fixed_params=fixed_params.copy(),
    adaptable_params=adaptable_DII_2D
)

adaptable_DII_2D_direct = dict(
    G0= 2.2713e-8, #0.027333e-6,
    q_10 = 1.4078e-3,#0.0046
    q_11=0,
    #A=0.987098626,
    #K=6492733.383, 
    N0_176= 4.2411e7,
    C = 0,
    t_ref = 1200 #280
)

DII_2D_fixed = ModelParams(
    functions={
        "G": G_fixed, # pyright: ignore[reportArgumentType]
        "q": q1_exp,
        "N0": N0_fixed,
        "dN0dT": dN0dT_fixed
    },
    fixed_params=fixed_params.copy(),
    adaptable_params=adaptable_DII_2D_direct
)

adaptable_DII_3D_direct = dict(
    G0= 0.08e-6,
    q_10 = 0.000206399,
    q_11= 0.1547,
    N0_164= 4e13,
    N0_168= 9.5e12,
    N0_172= 5e12,
    N0_176=2.5e11,
    N0_180= 4e9
)

DII_3D_fixed = ModelParams(
    functions={
        "G": G_fixed, # pyright: ignore[reportArgumentType]
        "q": q1_exp,
        "N0": N0_fixed,
        "dN0dT": dN0dT_fixed
    },
    fixed_params=fixed_params.copy(),
    adaptable_params=adaptable_DII_3D_direct
)