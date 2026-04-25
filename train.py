import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load dataset (place CSV in same folder)
df = pd.read_csv("credit_fraud.csv")

print("Columns:", df.columns)

# Data Cleaning
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.fillna(df.mean(), inplace=True)

print("Null values:\n", df.isnull().sum())

# Target column
target_col = "is_fraud"

X = df.drop(target_col, axis=1)
y = df[target_col]

# Train-Test Split (FIXED missing bracket)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Close any active MLflow run
if mlflow.active_run():
    mlflow.end_run()

# Set experiment
mlflow.set_experiment("Fraud_Detection")

# Train models
for n in [50, 100]:
    with mlflow.start_run(run_name=f"RF_{n}_trees"):

        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)

        print(f"\nRun: RF with {n} trees")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print("-" * 40)

        # Log to MLflow
        mlflow.log_param("n_estimators", n)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        mlflow.sklearn.log_model(model, "model")
