import pandas as pd
import os
import pickle

# In-memory cache (lives as long as the Python process)
_dataset_cache = {}

# Base folder for datasets
BASE_FOLDER = r"C:\Users\Hannelore\OneDrive - Vrije Universiteit Brussel\Bestanden PhD\Modeling\Haudin-Chenot"

# Dictionary of available datasets
DATASETS = {
    "noniso_DII": {
        "file": "Self-nucleation P(3HB-co 5% 4HB) non-iso C10 (rep 1).xlsx",
        "sheets": ["Ts176_C10"]  #["Ts164_C10", "Ts168_C10", "Ts172_C10", "Ts176_C10"]
    },
    "iso_DII": {
        "file": "Self-nucleation P(3HB-co-5% 4HB) 176x3 iso with cooling.xlsx",
        "sheets": ["Ts176_Tiso100", "Ts176_Tiso105", "Ts176_Tiso110", "Ts176_Tiso120"]
    },
    "noniso_CR": {
        "file": "Self-nucleation P(3HB-co 5% 4HB) 176x3 non-iso.xlsx",
        "sheets": ["Ts176_C1", "Ts176_C2", "Ts176_C5", "Ts176_C10", "Ts176_C20", "Ts176_C30", "Ts176_C100"]
    }
}

_dataset_cache = {}

def load_sheets(file_path, sheet_names):
    """Load and filter the requested sheets from an Excel file."""
    cols_needed = [
        'StepTime_sec', 'Temperature', 'Weight', 'HF_Corrected',
        'HF_Corrected_x_W', 'DT', 'alpha_x_weight', 'Int', 'Weight'
    ]

    all_sheets_raw = pd.read_excel(file_path, sheet_name=sheet_names, skiprows=[1], usecols=cols_needed)

    all_sheets = {}
    for sheet, df in all_sheets_raw.items():
        df_filtered = df[(df['Temperature'] >= 30) & (df['Temperature'] <= 150)].copy()
        all_sheets[sheet] = df_filtered

    return all_sheets

def get_dataset(name, reload=False):
    """Return (all_sheets, sheet_names) with memory + on-disk caching."""
    if not reload and name in _dataset_cache:
        print(f"Using in-memory cache for {name}")
        return _dataset_cache[name]

    cache_file = f"cache_{name}.pkl"

    # Try on-disk cache if available
    if not reload and os.path.exists(cache_file):
        print(f"Loading {name} from pickle cache...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        _dataset_cache[name] = data
        return data

    # Otherwise read from Excel
    print(f"Loading dataset '{name}' from Excel...")
    dataset_info = DATASETS[name]
    file_path = os.path.join(BASE_FOLDER, dataset_info["file"])
    sheet_names = dataset_info["sheets"]
    all_sheets = load_sheets(file_path, sheet_names)

    # Save to both caches
    data = (all_sheets, sheet_names)
    _dataset_cache[name] = data
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

    return data

