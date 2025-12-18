import pandas as pd 
import numpy as np 

unwanted_columns = UNWANTED_COLS = [
    'encounter_id',
    'patient_nbr',
    'weight',
    'payer_code',
    'medical_specialty'
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

def readmission_label(df):
    """Creates readmission label for 30 days 

    Args:
        df (dataframe):dataframe containing the col readmission

    Returns:
        dataframe: return a dataframe with a new column 
    """
    df = df.copy()
    df['readmitted_30'] = df['readmitted'].apply(
        lambda x: 1 if x == '<30' else 0
    )
    return df 

def correcting_col_types(df):
    df = df.copy()
    df['age'] = df['age'].astype('category')
    df['race'] = df['race'].astype('category')
    df['gender'] = df['gender'].astype('category')
    
    return df 