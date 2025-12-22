# Hospital Readmission Risk Prediction 

## Overview 
Predicting 3-day hospital readmission using clinical data, framed an imbalanced binary classification problem 

## Data 
- Kaggle: https://www.kaggle.com/datasets/brandao/diabetes
- Name: Diabetes 130 US Hospitals for years 1998-2008
- Readmission Rate = 11%

## Methods
- Preprocessing the columns with ColumnTransformer.
- Trained Logistic Regression, Random, Forest, Gradient Boosting models. Compared them to one another to see which gave the best performance.
- Recalled-target threshold optimixation
- Evaluation was done with PR curves and ROC AUC
 
## Results
- ROC AUC = 0.76
- Recall = 50%
- - Precision = 20%
 
## Intrepretability 
- Feature importance and SHAP analysis was done.
- Model drivers aligned with clinical expectations.

## Repository Strucrure 

`` 
├── data/              # Raw datasets
├── notebooks/         # Analysis notebooks 
├── src/               # .py files 
├── README.md          # Project overview
```

