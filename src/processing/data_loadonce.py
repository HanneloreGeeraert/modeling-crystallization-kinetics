import pandas as pd
import os
import pickle

# In-memory cache (lives as long as the Python process)
_dataset_cache = {}

# Base folder for datasets
BASE_FOLDER = r"C:\Users\Hannelore\OneDrive - Vrije Universiteit Brussel\Bestanden PhD\Studies\Modeling\01_Data\Final modeling data Haudin-Chenot"

# Dictionary of available datasets
DATASETS = {
    "iso_DII_Tiso120_withcooling": {
        "file": "Isothermal data - processed - full range - final - with cooling.xlsx",
        "sheets": ["Ts164_Tiso120", "Ts168_Tiso120", "Ts172_Tiso120", "Ts176_Tiso120",  "Ts180_Tiso120"]
    },
    "iso_DII_Tiso130_withcooling": {
        "file": "Isothermal data - processed - full range - final - with cooling.xlsx",
        "sheets": ["Ts164_Tiso130", "Ts168_Tiso130", "Ts172_Tiso130", "Ts176_Tiso130"]
    },
    "iso_Ts180_withcooling": {
        "file": "Isothermal data - processed - full range - final - with cooling.xlsx",
        "sheets": ["Ts180_Tiso120"]
    },
    "iso_Ts176_withcooling": {
        "file": "Isothermal data - processed - full range - final - with cooling.xlsx",
        "sheets": ["Ts176_Tiso100", "Ts176_Tiso105", "Ts176_Tiso110", "Ts176_Tiso115", "Ts176_Tiso120", "Ts176_Tiso125", "Ts176_Tiso130"]
    },
    "iso_Ts172_withcooling": {
        "file": "Isothermal data - processed - full range - final - with cooling.xlsx",
        "sheets": ["Ts172_Tiso120", "Ts172_Tiso130"]
    },
    "iso_Ts168_withcooling": {
        "file": "Isothermal data - processed - full range - final - with cooling.xlsx",
        "sheets": ["Ts168_Tiso120", "Ts168_Tiso130"]
    },
    "iso_Ts164_withcooling": {
        "file": "Isothermal data - processed - full range - final - with cooling.xlsx",
        "sheets": ["Ts164_Tiso120", "Ts164_Tiso130"]
    },
    "noniso_DII_C1": {
        "file": "Non-isothermal data - processed.xlsx",
        "sheets": ["Ts164_C1", "Ts168_C1", "Ts172_C1", "Ts176_C1",  "Ts180_C1"] 
    },
    "noniso_DII_C5": {
        "file": "Non-isothermal data - processed.xlsx",
        "sheets": ["Ts164_C5", "Ts168_C5", "Ts172_C5", "Ts176_C5",  "Ts180_C5"]
    },
    "noniso_DII_C10": {
        "file": "Non-isothermal data - processed.xlsx",
        "sheets": ["Ts164_C10", "Ts168_C10", "Ts172_C10", "Ts176_C10"]
    },
     "noniso_DII_C30": {
        "file": "Non-isothermal data - processed.xlsx", 
        "sheets": ["Ts164_C30", "Ts168_C30", "Ts172_C30", "Ts176_C30"]
    },
    "noniso_Ts180": {
        "file": "Non-isothermal data - processed.xlsx",
        "sheets": ["Ts180_C1", "Ts180_C5", "Ts180_C10", "Ts180_C30"]
    },
    "noniso_Ts176": {
        "file": "Non-isothermal data - processed.xlsx",
        "sheets": ["Ts176_C1", "Ts176_C2", "Ts176_C5", "Ts176_C10", "Ts176_C30"]
    },
    "noniso_Ts168": {
        "file": "Non-isothermal data - processed.xlsx",
        "sheets": ["Ts168_C1", "Ts168_C5", "Ts168_C10", "Ts168_C30"]
    },
    "noniso_Ts164": {
        "file": "Non-isothermal data - processed.xlsx",
        "sheets": ["Ts164_C1", "Ts164_C5", "Ts164_C10", "Ts164_C30"]
    },
    "noniso_Ts172": {
        "file": "Non-isothermal data - processed.xlsx",
        "sheets": ["Ts172_C1", "Ts172_C5", "Ts172_C10", "Ts172_C30"]
    }
}

_dataset_cache = {}

def load_sheets(file_path, sheet_names):
    """Load and filter the requested sheets from an Excel file."""
    cols_needed = [
        'StepTime (s)', 'Temperature (°C)', 'Weight (-)', 'Heat Flow Baseline Corrected (W/g)',
        'DT (K/min)', 'Alpha (-)', 'Int (J/g)'
    ]

    all_sheets_raw = pd.read_excel(file_path, sheet_name=sheet_names, skiprows=[1], usecols=cols_needed)

    all_sheets = {}
    for sheet, df in all_sheets_raw.items():
        df_filtered = df[(df['Temperature (°C)'] >= 20) & (df['Temperature (°C)'] <= 150)].copy()
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

