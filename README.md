# Hospital Readmission Risk Prediction 

## Overview 
Predicting 30-day hospital readmission using clinical data, framed an imbalanced binary classification problem. This project demonstarets an end-to-end ML pipeline: data preprocessing, feature engineer, model training, and evaluating a large clinical dataset. 

## Project Motivation 
Hospital readmissions can be costly but most often are highky preventable. Being able to predict which patients are likely to be readmitted within 30 days can not only help the patient but also the healthcare provider:
- Able to allocate resources more efficiently
- Improving patient care
- Reducing the healthcare cost for patients

## Repository Strucrure 

```
Clinical_prediction/
├── data/             
    ├──raw/            # Raw dataset 
├── notebooks/         # Exploratory Analysis & Visualizations 
├── src/               # Modular Scripts
    ├── cleaning.py
    ├── dataset.py
    ├── features.py
    ├── preprocessing.py
    ├── train.py
    ├── evaluate.py

├── README.md          # Project overview

```

## Technology Used 
Language: Python
Libraries: Pandas, Numpy, scikit-learn, matplotlib, seaborn 
ML Models: Logistic Regression, Random Forest, Gradient Boosting
Evaluation: AUC_ROC, Precision, Recall, F1-score

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
| Model | ROC-AUC | Recall | Precision |
|----------|----------|----------|-----------|
| Logistic Regression | 0.64 | 0.54 | 0.16 |
|----------|----------|----------|-----------|
|Random Forest | 0.66 | 0.55 | 0.17 |
|----------|----------|----------|-----------|
|Gradient Boosting | 0.76 | 0.5 | 0.2 |
 
## Intrepretability 
- Feature importance and SHAP analysis was done.
- Model drivers aligned with clinical expectations.

## How to Run 
1. Clone the repository
   git clone https://github.com/Javih10/clinical_prediction.git

2. Install dependicies:
   pip install -r requirements.txt

3. Run preprocessing and model training scripts:

4. Explore notebooks for visualizations and analysis


