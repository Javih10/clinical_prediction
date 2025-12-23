import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def column_type(X):
    """

    Args:
        X (_type_): _description_
    """
    
    num_cols = X.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()
    
    cat_cols = X.select_dtypes(
        include=['object','category']
    ).columns.tolist()
    
    return cat_cols, num_cols


def building_pipeline(cat_cols, num_cols):
    num_trans = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    cat_trans = Pipeline(steps=[
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            dtype=np.float64
        ))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_trans, num_cols),
            ('cat', cat_trans, cat_cols)
        ]
    )
    
    return preprocessor