import pandas as pd 
import numpy as np 
import re

medications = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone']
nums = (
    '250.4', '250.5', '250.6', '250.7', '250.8', '250.9'
)

def cleaning_up_cols(df):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    df = df.copy()
    
    df['num_inpatient'] = df['number_inpatient']
    df['num_emergency'] = df['number_emergency']
    df['num_outpatient'] = df['number_outpatient']
    
    return df 

def total_visits_col(df):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    df = df.copy()
    
    df['total_visits'] = df['num_inpatient'] + df['num_emergency'] + df['num_outpatient']
    
    return df 


def medication_changes(df):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df.copy()
    
    df['num_medication_change'] = df[medications].isin(['Up','Down']).sum(axis=1)
    
    return df 

def usage_of_insulin(df):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    df = df.copy()
    
    df['insulin_used'] = (df['insulin'] != 'No').astype(int)
    
    return df


def has_complication(code):
    if pd.isna(code):
        return 0
    return int(any(str(code).startswith(prefix) for prefix in nums))


def add_has_diabetes_complications(df):
    df = df.copy()
    df['has_diabetes_complications'] = (
        df[['diag_1', 'diag_2', 'diag_3']]
        .apply(lambda row: max(has_complication(c) for c in row), axis=1)
    )
    return df

def adding_all_features(df):
    
    df = cleaning_up_cols(df)
    df = total_visits_col(df)
    df = medication_changes(df)
    df = usage_of_insulin(df)
    df = add_has_diabetes_complications(df)
    
    return df

def readmission_target(df):
    df = df.copy()
    df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
    
    return df

def drop_col_for_train(df):
    df = df.copy()
    return df.drop(columns = ['readmitted', 'encounter_id', 'patient_nbr'])