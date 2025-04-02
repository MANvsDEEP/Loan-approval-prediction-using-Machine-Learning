## Loan-approval-prediction-using-Machine-Learning

Overview

This project aims to predict loan approvals using machine learning models such as LightGBM, XGBoost, and CatBoost. The dataset consists of financial and personal details of applicants, which are used to determine whether a loan should be approved or not.

Dataset

The project utilizes the following datasets:

train.csv: Contains labeled data with applicant details and loan approval status.

test.csv: Contains applicant details without labels, used for predictions.

credit_risk_dataset.csv: Additional dataset for credit risk analysis.

sample_submission.csv: Sample format for submission.

Project Structure

├── code.ipynb                # Jupyter Notebook with model training and evaluation
├── train.csv                 # Training dataset
├── test.csv                  # Test dataset
├── credit_risk_dataset.csv   # Additional dataset
├── sample_submission.csv     # Submission format
├── README.md                 # Project documentation

Installation

Prerequisites

Ensure you have the following installed:

Python 3.8+

Jupyter Notebook

Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, lightgbm, xgboost, catboost, lime

Install Dependencies

Run the following command to install the required libraries:

pip install -r requirements.txt

Model Training

The model is trained using Stratified K-Fold Cross Validation to ensure balanced training. It uses the following machine learning models:

LightGBM (LGBMClassifier): A gradient boosting framework that uses tree-based learning algorithms.

XGBoost (XGBClassifier): A scalable, efficient, and flexible implementation of gradient boosting.

CatBoost (CatBoostClassifier): A high-performance gradient boosting on decision trees for categorical features.

Training the Model

To train the model, open code.ipynb and execute all cells. The training steps include:

Data Preprocessing: Handling missing values, encoding categorical features, and feature scaling.

Feature Selection: Identifying important features using model-based techniques.

Model Training: Training each classifier using Stratified K-Fold cross-validation.

Hyperparameter Tuning: Optimizing model parameters for better accuracy.

Evaluation Metrics

The model performance is evaluated using:

ROC-AUC Score: Measures the ability of the model to distinguish between loan approvals and denials.

Confusion Matrix: Shows true positives, false positives, true negatives, and false negatives.

Feature Importance: Determines which features are most influential in decision-making.
