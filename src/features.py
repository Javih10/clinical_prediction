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

def renaming_cols(df):
    """Renaming some columns to make them similar to others 

    Args:
        df (dataframe): the data being worked with in the form of a df

    Returns:
        dataframe: a dataframe with some columns being renamed
    """
    
    df = df.copy()
    
    df['num_inpatient'] = df['number_inpatient']
    df['num_emergency'] = df['number_emergency']
    df['num_outpatient'] = df['number_outpatient']
    
    return df 

def total_visits_col(df):
    """Creates a new column with total number of visits per patient

    Args:
        df (dataframe): dataframe with the columns 'num_inpatient','num_emergency','num_outpatient'

    Returns:
        dataframe: returns a dataframe with a new column total_visits
    """
    
    df = df.copy()
    
    df['total_visits'] = df['num_inpatient'] + df['num_emergency'] + df['num_outpatient']
    
    return df 


def medication_changes(df):
    """Returns whether the medication has changed and how many times it has 

    Args:
        df (dataframe): dataframe with columns that contains medications and whether used on the patient

    Returns:
        dataframe: a dataframe with a new column describing num_medication_change
    """
    df = df.copy()
    
    df['num_medication_change'] = df[medications].isin(['Up','Down']).sum(axis=1)
    
    return df 

def usage_of_insulin(df):
    """Returns whether insulin was used on the patient with 1 for yes and 0 for no 

    Args:
        df (dataframe): dataframe with a column named 'insulin'

    Returns:
        dataframe: a dataframe with a new column insulin_used
    """
    
    df = df.copy()
    
    df['insulin_used'] = (df['insulin'] != 'No').astype(int)
    
    return df


def has_complication(code):
    """Returns whether a patient has complications with any diagnoses 

    """
    if pd.isna(code):
        return 0
    return int(any(str(code).startswith(prefix) for prefix in nums))


def add_has_diabetes_complications(df):
    """Whether a patient has diabtes complications or not adding three different diagnoses together

    Args:
        df (dataframe): A dataframe with variables called diag_1, diag_2, and diag_3

    Returns:
        dataframe: return a column with either 1 (has diabetes effect) or 0 (does not have diabetes effect)
    """
    df = df.copy()
    df['has_diabetes_complications'] = (
        df[['diag_1', 'diag_2', 'diag_3']]
        .apply(lambda row: max(has_complication(c) for c in row), axis=1)
    )
    return df

def adding_all_features(df):
    """Adding most of the features together into just one function

    Args:
        df (dataframe): dataframe

    Returns:
        dataframe: a dataframe with functions applied for getting new features
    """
    
    df = renaming_cols(df)
    df = total_visits_col(df)
    df = medication_changes(df)
    df = usage_of_insulin(df)
    df = add_has_diabetes_complications(df)
    
    return df

def drop_col_for_train(df):
    
    df = df.copy()
    return df.drop(columns = ['encounter_id', 'patient_nbr'])