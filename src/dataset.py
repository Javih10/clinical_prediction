import pandas as pd 

from data.raw.downloaded_data import load_raw_data
from src.cleaning import (drop_unwanted_columns, replacing_missing_values, readmission_label, correcting_col_types)
from src.features import (adding_all_features, drop_col_for_train)

def finalizing_dataset():
    df = load_raw_data()
    df = drop_unwanted_columns(df)
    df = replacing_missing_values(df)
    df = readmission_label(df)
    df = correcting_col_types(df)
    df = adding_all_features(df)
    df = drop_col_for_train(df)
    return df