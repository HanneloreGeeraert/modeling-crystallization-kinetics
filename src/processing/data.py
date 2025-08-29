import os
from src.models.utils import load_sheets

# Base folder for datasets
BASE_FOLDER = r"C:\Users\hgeeraer\OneDrive - Vrije Universiteit Brussel\Bestanden PhD\Modeling\Haudin-Chenot"

# Dictionary of available datasets
DATASETS = {
    "non_iso_DII": {
        "file": "Self-nucleation P(3HB-co 5% 4HB) non-iso C10 (rep 1).xlsx",
        "sheets": ["Ts164_C10", "Ts168_C10", "Ts172_C10", "Ts176_C10"]
    },
    "iso_DII": {
        "file": "Self-nucleation P(3HB-co 5% 4HB) iso.xlsx",
        "sheets": ["T176_Tiso100", "Ts176_Tiso105", "Ts176_Tiso110", "Ts176_Tiso115", "Ts176_Tiso120"]
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