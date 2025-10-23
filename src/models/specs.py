from .kinetics import haudin_chenot_3D, haudin_chenot_3D_twostep, haudin_chenot_3D_test
from ..parameters.domainII import DII_3D

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
DII_3D_twostep = ModelSpec(
    dimensions=3,
    func=haudin_chenot_3D_twostep,
    params=DII_3D,
    n_states=8,
    state_names=["N", "Ni", "alpha", "Na", "Ntilde_a", "F", "P", "Q"]
)

DII_3D_onestep = ModelSpec(
    dimensions=3,
    func=haudin_chenot_3D,
    params=DII_3D,
    n_states=7,
    state_names=["N", "alpha", "Na", "Ntilde_a", "F", "P", "Q"]
)

DII_3D_test = ModelSpec(
    dimensions=3,
    func=haudin_chenot_3D_test,
    params=DII_3D,
    n_states=7,
    state_names=["N", "alpha", "Na", "Ntilde_a", "F", "P", "Q"]
)
