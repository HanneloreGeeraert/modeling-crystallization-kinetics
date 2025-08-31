import os
import pandas as pd

import pandas as pd

def load_sheets(file_path, sheet_names):
    cols_needed = [
        'StepTime_sec', 'Temperature', 'Weight', 'HF_Corrected',
        'HF_Corrected_x_W', 'DT', 'alpha', 'alpha_x_weight', 'Int'
    ]

    # Read all sheets
    all_sheets_raw = pd.read_excel(file_path, sheet_name=sheet_names, skiprows=[1], usecols=cols_needed)
    
    # Filter temperature for each sheet
    all_sheets = {}
    for sheet, df in all_sheets_raw.items():
        df_filtered = df[(df['Temperature'] >= 30) & (df['Temperature'] <= 150)].copy()
        all_sheets[sheet] = df_filtered

    return all_sheets

# Base folder for datasets
BASE_FOLDER = r"C:\Users\hgeeraer\OneDrive - Vrije Universiteit Brussel\Bestanden PhD\Modeling\Haudin-Chenot"

# Dictionary of available datasets
DATASETS = {
    "non_iso_DII": {
        "file": "Self-nucleation P(3HB-co 5% 4HB) non-iso C10 (rep 1).xlsx",
        "sheets": ["Ts164_C10", "Ts168_C10", "Ts172_C10", "Ts176_C10"]
    },
    "iso_DII": {
        "file": "Self-nucleation P(3HB-co-5% 4HB) 176x3 - with cooling.xlsx",
        "sheets": ["Ts176_Tiso105"]#, "Ts176_Tiso105", "Ts176_Tiso110", "Ts176_Tiso115", "Ts176_Tiso120"]
    }
}

def get_dataset(name: str):
    if name not in DATASETS:
        raise ValueError(f"Dataset '{name}' not found. Available: {list(DATASETS.keys())}")

    dataset_info = DATASETS[name]
    file_path = os.path.join(BASE_FOLDER, dataset_info["file"])
    sheet_names = dataset_info["sheets"]

    sheets_dict = load_sheets(file_path, sheet_names)
    return sheets_dict, sheet_names