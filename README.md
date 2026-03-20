# Customer Churn Prediction

## Overview
This project predicts customer churn using machine learning techniques.
It includes data preprocessing, exploratory data analysis, model building, and performance evaluation.

## Problem Statement
Customer churn affects business growth and profitability.
The aim of this project is to predict whether a customer will leave the company based on historical customer data.

## Dataset
The dataset contains customer information such as demographics, services subscribed, account details, and churn status.

## Tools and Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Workflow
1. Data collection
2. Data cleaning
3. Exploratory Data Analysis
4. Feature engineering
5. Model building
6. Model evaluation
7. Insights and conclusion

## Exploratory Data Analysis
The following analysis was performed:
- Churn count
- Contract type vs churn
- Monthly charges distribution
- Tenure vs churn
- Internet service vs churn

## Models Used
- Logistic Regression
- Decision Tree
- Random Forest

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Results
The best-performing model achieved good accuracy in predicting customer churn.
Important factors affecting churn included tenure, contract type, and monthly charges.

## Key Insights
- Customers with short tenure are more likely to churn.
- Month-to-month contract customers show higher churn.
- Customers with higher monthly charges may have higher churn probability.

## Folder Structure
customer-churn-project/
│── data/
│── notebooks/
│── src/
│── outputs/
│── README.md

## How to Run
1. Clone the repository
2. Install dependencies:
   pip install pandas numpy matplotlib seaborn scikit-learn
3. Run the script:
   python churn_prediction.py

## Future Improvements
- Hyperparameter tuning
- Streamlit deployment
- Power BI dashboard integration

## Author
Durgesh Desale
