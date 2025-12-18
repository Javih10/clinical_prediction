import pandas as pd 
import numpy as np 

unwanted_columns = UNWANTED_COLS = [
    'encounter_id',
    'patient_nbr',
    'weight',
    'payer_code',
    'medical_specialty',
    'examide',
    'citoglipton'
]

def drop_unwanted_columns(df):
    """ 
    Dropping columns that are not meaningful or are identifiers
    
    Args:
        df (dataframe): with similar columns
    """
    df = df.copy()
    return(df.drop(columns=unwanted_columns, errors='ignore'))

def replacing_missing_values(df):
    """
    Replaces '?' placeholder with NaN values

    Args:
        df (dataframe): dataframe that contains values such as '?'
    """
    
    df = df.copy()
    df.replace('?',np.nan, inplace=True)
    return df