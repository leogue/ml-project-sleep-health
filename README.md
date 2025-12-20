# Sleep & Health project's information

## Team

- Léo Guérin - CCC2 - leo.guerin@edu.devinci.fr
- Léa Montaron - CCC2 - lea.montaron@edu.devinci.fr
- Marie-Lou Jodet - CCC2 - marie-lou.jodet@edu.devinci.fr

## Project Overview 

This project aims to predict sleep disorders (Insomnia and Sleep Apnea) using simple lifestyle indicators and physiological biomarkers. 
The main objective is to maximize Recall, ensuring that no pathological cases are missed — a critical requirement in medical screening.
Our methodology includes data preprocessing (encoding, normalization, feature engineering), exploratory analysis, and the evaluation of 
several supervised models (Logistic Regression, SVM, Random Forest, XGBoost) using Stratified Cross-Validation.

We followed a three-phase experimental strategy:
- Baseline: Train models on all available variables.
- Noise reduction: Remove non-informative features such as occupation.
- Minimalist biomarkers: Use only 5 key variables (Age, BMI Category, Sleep Duration, Systolic BP, Diastolic BP) to build a lightweight predictive model.

The best performance is achieved with an optimized XGBoost model, reaching:
- Recall (Macro): 0.97
- 0 False Negatives on the validation set

## Source Code Description (src/)

- *pre_process.py* : Data preprocessing, data cleaning and feature engineering. This file generate processed_dataset.csv and validation_dataset.csv.
- *analysis.py* : Performs exploratory data analysis and generates visualization figures (PCA, UMAP, Correlation) saved in report/figures.
- *model_selection.py* : Compares multiple machine learning models (XGBoost, SVM, RF, LR) to select the best one based on Recall.
- *hyperparameter_tuning.py* : Optimizes the hyperparameters of the selected XGBoost model using Grid Search.
- *final_validation.py* : Trains the final optimized model and evaluates performance on the separate validation dataset.
- *ensemble_methods.py* : Implements ensemble methods (Voting, Weighted Voting) to combine model predictions.


