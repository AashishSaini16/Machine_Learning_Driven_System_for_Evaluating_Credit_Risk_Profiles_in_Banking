# Machine Learning Driven System for Evaluating Credit Risk Profiles in Banking

This repository contains a comprehensive machine learning pipeline for assessing credit risk in banking using the Credit Risk Dataset from Kaggle. The system preprocesses data, performs exploratory data analysis (EDA), trains an XGBoost model for binary classification (loan approved/rejected), and provides an interactive widget-based interface for real-time predictions. The project is implemented in a Jupyter Notebook and includes statistical tests, model evaluation, and visualizations.

## Overview
Credit risk evaluation is crucial for banks to minimize defaults and make informed lending decisions. This project builds a predictive model using XGBoost to classify loan applications as "Approved" or "Rejected" based on features like age, income, employment length, loan amount, interest rate, and credit history. It handles imbalanced data with SMOTE, performs hyperparameter tuning, and compares with baseline models (Logistic Regression, Random Forest). An interactive UI allows users to input customer details and get predictions with confidence scores.

The notebook is self-contained and can be run on Google Colab or locally.

## Features
- **Data Preprocessing**: Handling missing values (imputation), outlier removal, categorical encoding (LabelEncoder), and feature scaling (StandardScaler).
- **Exploratory Data Analysis (EDA)**: Summary statistics, correlation heatmaps, distribution plots, statistical tests (Chi-square, T-test, VIF for multicollinearity).
- **Model Training**: XGBoost classifier with cross-validation, grid search for hyperparameters, and SMOTE for class imbalance.
- **Model Evaluation**: Metrics including accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and cross-validation scores.
- **Model Comparison**: Benchmarks against Logistic Regression and Random Forest.
- **Interactive Prediction**: Widget-based form for user input and real-time predictions.
- **Visualizations**: Heatmaps, count plots, ROC curves, and feature importance plots.

## Dataset
The dataset used is the [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) from Kaggle, containing 32,581 entries with 12 features:
- **Key Features**: `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`, `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_status` (target), `loan_percent_income`, `cb_person_default_on_file`, `cb_person_cred_hist_length`.
- **Target**: `loan_status` (0: Approved, 1: Rejected/Defaulted).
- **Imbalance**: Approximately 78% approved vs. 22% rejected.
- **Download**: Automated via Kaggle API in the notebook (requires `kaggle.json` credentials).

Sample Data:
| person_age | person_income | person_home_ownership | person_emp_length | loan_intent | loan_grade | loan_amnt | loan_int_rate |
|------------|---------------|-----------------------|-------------------|-------------|------------|-----------|---------------|
| 22         | 59000         | RENT                  | 123.0             | PERSONAL    | D          | 35000     | 16.02         |
| 21         | 9600          | OWN                   | 5.0               | EDUCATION   | B          | 1000      | 11.14         |

## Technologies Used
- **Python Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Imbalanced-learn (SMOTE), Statsmodels, IPyWidgets.
- **Environment**: Jupyter Notebook (compatible with Google Colab).
- **Hardware Acceleration**: Optional GPU support via Colab (T4 GPU).

## Methodology
1. **Data Loading & Preprocessing**: Impute missing values (mean strategy for `person_emp_length` and `loan_int_rate`), remove outliers (`person_age` > 100), encode categoricals, scale numerics.

![Correlation Heatmap](https://github.com/AashishSaini16/Machine_Learning_Driven_System_for_Evaluating_Credit_Risk_Profiles_in_Banking/blob/main/Correlation_Matrix.PNG)

2. **EDA**: Correlation analysis, distribution checks, statistical significance tests (Chi-square for categoricals, T-test for numerics, VIF for multicollinearity).

![Feature Importance](https://github.com/AashishSaini16/Machine_Learning_Driven_System_for_Evaluating_Credit_Risk_Profiles_in_Banking/blob/main/Feature_Importance.PNG)

3. **Handling Imbalance**: Apply SMOTE to oversample the minority class (defaults).
4. **Modeling**:
   - Primary: XGBoost with parameters tuned via GridSearchCV (e.g., `max_depth`, `learning_rate`).
   - Baselines: Logistic Regression, Random Forest.
   - Evaluation: 5-fold cross-validation, ROC-AUC.

![ROC Curve](https://github.com/AashishSaini16/Machine_Learning_Driven_System_for_Evaluating_Credit_Risk_Profiles_in_Banking/blob/main/ROC.PNG)

5. **Explainability**: Feature importance plots and SHAP values.
6. **Deployment**: Interactive UI using IPyWidgets for user-friendly predictions.

## Interactive Demo
The notebook includes a widget-based form ("Banker's Credit Approval Checker") for entering customer details (e.g., age, income, loan amount). It predicts approval/rejection with confidence and handles invalid inputs.

![Banker's Credit Approval Checker Demo](https://github.com/AashishSaini16/Machine_Learning_Driven_System_for_Evaluating_Credit_Risk_Profiles_in_Banking/blob/main/Output.PNG)
