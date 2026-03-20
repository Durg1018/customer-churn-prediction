# Customer Churn Prediction using Machine Learning

## Project Overview
This project predicts whether a telecom customer is likely to churn or not using machine learning techniques.  
The main objective is to analyze customer behavior, identify important churn patterns, and build predictive models that can help businesses improve customer retention.

---

## Dataset
The project uses the **Telco Customer Churn Dataset** containing customer demographic details, account information, services subscribed, billing details, and churn status.

### Dataset Size
- **Rows:** 7043
- **Columns:** 21

### Target Column
- **Churn**
  - Yes = Customer left
  - No = Customer stayed

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code

---

## Project Steps

### 1. Data Loading
The dataset was loaded from a CSV file using Pandas.

### 2. Data Cleaning
- Removed `customerID` because it is not useful for prediction
- Converted `TotalCharges` to numeric format
- Handled missing values in `TotalCharges`
- Converted the target column `Churn` into numeric values:
  - Yes → 1
  - No → 0

### 3. Exploratory Data Analysis
Several visualizations were created to understand churn behavior:
- Churn Count
- Contract Type vs Churn
- Internet Service vs Churn
- Monthly Charges Distribution
- Tenure vs Churn

### 4. Data Preprocessing
- Converted categorical variables into numeric using one-hot encoding
- Split the dataset into training and testing sets
- Applied feature scaling using StandardScaler

### 5. Model Building
The following machine learning models were trained:
- Logistic Regression
- Decision Tree
- Random Forest

### 6. Model Evaluation
Models were evaluated using:
- Accuracy
- Classification Report
- Confusion Matrix

---

## Model Results

- **Logistic Regression:** 80.70%
- **Decision Tree:** 74.10%
- **Random Forest:** 78.57%

### Best Model
**Logistic Regression** performed the best with an accuracy of **80.70%**.

---

## Confusion Matrix of Best Model
```text
[[923 112]
 [190 184]]