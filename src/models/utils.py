import pandas as pd

# Heat of fusion normalization factor
def get_deltaHm(Int): 
    return -58

def load_sheets(file_path, sheet_names):
    cols_needed = ['StepTime_sec', 'Temperature', 'Weight', 'HF_Corrected_x_W', 'DT', 'alpha_x_weight', 'Int']

    all_sheets = pd.read_excel(file_path, sheet_name=sheet_names, skiprows=[1], usecols=cols_needed)
    
    # Filter for weight column
    for sheet in all_sheets:
        df = all_sheets[sheet]
        if 'Weight' in df.columns:
            df = df[df['Weight'] == 1].reset_index(drop=True)
        all_sheets[sheet] = df

    return all_sheets