import pandas as pd 
from pathlib import Path

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw"

def load_raw_data(file_name='diabetic_data.csv'):
    """
    Loading the raw diabetes dataset.
    
    Returns 
    ------------
    Raw dataset with the column names and values
    """
    file_path = RAW_DATA_PATH / file_name
    df = pd.read_csv(file_path)
    
    return df 

