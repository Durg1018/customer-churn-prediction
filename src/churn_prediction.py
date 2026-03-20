import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Create output folders
os.makedirs("outputs/charts", exist_ok=True)

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/churn.csv")

print("Dataset loaded successfully!\n")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

# -----------------------------
# 2. Data cleaning
# -----------------------------
print("\nChecking blank spaces in TotalCharges...")
print((df["TotalCharges"] == " ").sum())

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nMissing values after converting TotalCharges:")
print(df.isnull().sum())

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Convert target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Fill missing TotalCharges with median
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

print("\nUpdated data types:")
print(df.dtypes)

print("\nFirst 5 rows after cleaning:")
print(df.head())

# -----------------------------
# 3. EDA
# -----------------------------
sns.set_style("whitegrid")

plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df)
plt.title("Churn Count")
plt.savefig("outputs/charts/churn_count.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Contract Type vs Churn")
plt.xticks(rotation=10)
plt.savefig("outputs/charts/contract_vs_churn.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Internet Service vs Churn")
plt.savefig("outputs/charts/internet_vs_churn.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["MonthlyCharges"], bins=30, kde=True)
plt.title("Monthly Charges Distribution")
plt.savefig("outputs/charts/monthly_charges_distribution.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="Churn", y="tenure", data=df)
plt.title("Tenure vs Churn")
plt.savefig("outputs/charts/tenure_vs_churn.png")
plt.show()

print("\nCharts saved in outputs/charts/")

# -----------------------------
# 4. Prepare data for ML
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Convert categorical columns into numeric
X = pd.get_dummies(X, drop_first=True)

print("\nEncoded feature columns:")
print(X.columns.tolist())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 5. Train models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name}")
    print("Accuracy:", round(acc, 4))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    results.append((name, acc))

# -----------------------------
# 6. Save results
# -----------------------------
best_model = max(results, key=lambda x: x[1])

with open("outputs/model_results.txt", "w") as f:
    f.write("Model Performance Results\n")
    f.write("=========================\n\n")
    for model_name, acc in results:
        f.write(f"{model_name}: Accuracy = {acc:.4f}\n")
    f.write(f"\nBest Model: {best_model[0]} with Accuracy = {best_model[1]:.4f}\n")

print("\nBest model:", best_model[0], "with accuracy:", round(best_model[1], 4))
print("Results saved in outputs/model_results.txt")
print("\nProject completed successfully.")