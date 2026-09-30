from .kinetics import haudin_chenot_2D, haudin_chenot_3D, haudin_chenot_3D_sat, haudin_chenot_2D_induction, haudin_chenot_3D_induction, haudin_chenot_3D_twostep
from ..parameters.domainII import DII_2D, DII_3D, DII_2D_fixed, DII_3D_fixed

class ModelSpec:
    def __init__(self, dimensions, func, params, n_states, state_names):
        self.dimensions = dimensions
        self.func = func
        self.params = params
        self.n_states = n_states
        self.state_names = state_names

    def make_y0(self, sheet_name, df):
        return [0.0] * self.n_states
    
def get_modelspec(name: str) -> ModelSpec:
    """Retrieve a ready-to-use ModelSpec by variable name."""
    try:
        return globals()[name]
    except KeyError:
        available = [k for k, v in globals().items() if isinstance(v, ModelSpec)]
        raise ValueError(f"Model '{name}' not found. Available: {available}")
    
# Define the ready-to-use models
DII_3D_onestep = ModelSpec(
    dimensions=3,
    func=haudin_chenot_3D,
    params=DII_3D,
    n_states=7,
    state_names=["N", "alpha", "Na", "Ntilde_a", "F", "P", "Q"]
)

DII_3D_twostep = ModelSpec(
    dimensions=3,
    func=haudin_chenot_3D_twostep,
    params=DII_3D,
    n_states=8,
    state_names=["N", "N1", "alpha", "Na", "Ntilde_a", "F", "P", "Q"]
)

DII_2D_onestep = ModelSpec(
    dimensions=2,
    func=haudin_chenot_2D,
    params=DII_2D,
    n_states=6,
    state_names=["N", "alpha", "Na", "Ntilde_a", "F", "P"]
)

DII_2D_induction = ModelSpec(
    dimensions=2,
    func=haudin_chenot_2D_induction,
    params=DII_2D,
    n_states=7,
    state_names=["N", "alpha", "Na", "Ntilde_a", "F", "P", "I_d"]
)

DII_2D_direct = ModelSpec(
    dimensions=2,
    func=haudin_chenot_2D_induction,
    params=DII_2D_fixed,
    n_states=7,
    state_names=["N", "alpha", "Na", "Ntilde_a", "F", "P", "I_d"]
)

DII_3D_direct = ModelSpec(
    dimensions=3,
    func=haudin_chenot_3D,
    params=DII_3D_fixed,
    n_states=7,
    state_names=["N", "alpha", "Na", "Ntilde_a", "F", "P", "Q"]
)

DII_3D_induction = ModelSpec(
    dimensions=3,
    func= haudin_chenot_3D_induction,
    params=DII_3D,
    n_states=8,
    state_names=["N", "alpha", "Na", "Ntilde_a", "F", "P", "Q", "I_d"]
)

DII_3D_sat = ModelSpec(
    dimensions=3,
    func=haudin_chenot_3D_sat,
    params=DII_3D,
    n_states=7,
    state_names=["N", "alpha", "Na", "Ntilde_a", "F", "P", "Q"]
)