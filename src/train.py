import pandas as pd 
import numpy as np 
from src.dataset import finalizing_dataset
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from src.preprocessing import building_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, precision_score, recall_score)

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from src.preprocessing import building_pipeline

# General features for all models being trained
def training_model():
    df = finalizing_dataset()
    target = 'readmitted_30'
    y = df[target]
    X = df.drop(columns=[target, 'readmitted'])
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify= y, 
    random_state = 42)
    
    return X_train, X_test, y_train, y_test

def dumm_model(x,y):
    """_summary_

    Args:
        x (column): categorical column 
        y (column): numerical column 
    """
    dum_pipeline = Pipeline(steps=[
    ('preprocessor', building_pipeline(x, y)),
    ('model', DummyClassifier(strategy='most_frequent'))]
    )
    
    return dum_pipeline

def log_pipeline(x,y):
    """_summary_

    Args:
        x (column): categorical column
        y (column): numerical column 
    """
    
    LR_pipeline = Pipeline(steps=[
    ('preprocessor', building_pipeline(
        x, y)),
    ('model', LogisticRegression(
        max_iter = 100, 
        class_weight= 'balanced',
        solver='liblinear'
        ))])
    return LR_pipeline

def RF_pipe(x,y):
    """_summary_

    Args:
        x (column): categorical column 
        y (column): numerical column 

    Returns:
        _type_: _description_
    """
    
    RF_pipeline = Pipeline(steps=[
    ('preprocessor', building_pipeline(
        x, y)),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
    ))])
    
    return RF_pipeline

def GB_pipe(x,y):
    """_summary_

    Args:
        x (column): categorical column 
        y (column): numerical column 

    Returns:
        _type_: _description_
    """
    GB_pipeline = Pipeline(steps=[
    ('preprocessor', building_pipeline(
        x, y)),
    ('model', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05, 
        max_depth = 3, 
        random_state = 42
    ))])
    
    return GB_pipeline

def fit_and_pred(pipeline, x, y, z):
    """_summary_

    Args:
        pipeline (_type_): model which trying to get the information from 
        x (_type_): X_training from the split 
        y (_type_): y_training from split 
        z (_type_): X_test from the split 
    """
    pipeline.fit(x,y)
    ypred = pipeline.predict(z)
    yprob_pred = pipeline.predict_proba(z)[:,1]
    
    return ypred, yprob_pred
    



def stat_metrics(x, y, z):
    """_summary_

    Args:
        x (_type_): y_test from split 
        y (_type_): y predictor
        z (_type_): y probability

    Returns:
        _type_: _description_
    """
    roc = roc_auc_score(x,z)
    precision = precision_score(x,y)
    recall = recall_score(x,y)
    
    return roc, precision, recall
    
    

