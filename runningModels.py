import sys
import os
import time
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from streamlit_app.project_utils import *
from streamlit_app.data_cleaning_and_preprocessing import *

# === Redirect stdout to both console and file ===
class Logger(object):
    def __init__(self, logfile="run_log.txt"):
        self.terminal = sys.stdout
        self.log = open(logfile, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
sys.stdout = Logger(os.path.join(log_dir, "model_training_log.txt"))

# === Script starts ===
total_start_time = time.time()

# Load dataset
print("Loading financial dataset...")
financial_df = pd.read_csv('D:\\jaheem\\COMP3610_Project-main\\financial_data.csv') 
print("Loaded financial dataset successfully.")
print("First few rows of raw data:")
print(financial_df.head())

# Preprocessing
print("\nStarting feature engineering and preprocessing...")
preprocessing_start = time.time()
financial_df_preprocessed = feature_engineering_financial(financial_df, False)
print(f"Preprocessing complete. Time taken: {time.time() - preprocessing_start:.2f} seconds.")
print("First few rows of preprocessed data:")
print(financial_df_preprocessed.head())

# Prepare features and target
print("\nPreparing feature matrix X and target vector y...")
X = financial_df_preprocessed.drop(columns=['Date', 'Ticker', 'Close'])  # Features
y = financial_df_preprocessed['Close']  # Target
print(f"Feature columns: {X.columns.tolist()}")
print(f"Number of samples: {len(X)}")

# Split the dataset
print("\nSplitting dataset into training and test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# Scale the features
print("\nScaling features using StandardScaler...")
scaling_start = time.time()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Feature scaling complete. Time taken: {time.time() - scaling_start:.2f} seconds.")

# Train optimized Random Forest model
print("\nTraining optimized Random Forest model (n_estimators=50, max_depth=12, min_samples_leaf=5)...")
rf_start = time.time()
rf = RandomForestRegressor(n_estimators=50, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
print(f"Optimized Random Forest training complete. Time taken: {time.time() - rf_start:.2f} seconds.")

# Train Linear Regression model
print("\nTraining Linear Regression model...")
lr_start = time.time()
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print(f"Linear Regression training complete. Time taken: {time.time() - lr_start:.2f} seconds.")

# Evaluate both models
print("\nEvaluating models...")
eval_start = time.time()
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Evaluation complete. Time taken: {time.time() - eval_start:.2f} seconds.")

print(f"\n📊 Random Forest - MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}")
print(f"📊 Linear Regression - MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}")

# Save models and scaler (with compression)
print("\nSaving models and scaler to 'models/' directory with compression...")
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(rf, os.path.join(model_dir, 'random_forest_model.pkl'), compress=('zlib', 3))
joblib.dump(lr, os.path.join(model_dir, 'linear_regression_model.pkl'), compress=('zlib', 3))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'), compress=('zlib', 3))
print("✅ Models and scaler saved successfully in the 'models/' folder with compression.")

# Plot feature importances for Random Forest using Matplotlib
feature_importances = rf.feature_importances_
features = X.columns.tolist()

print("\nPlotting feature importances...")
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color='royalblue')
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig('random_forest_feature_importances.png')
plt.close()

# Plot Actual vs Predicted values for both models using Matplotlib
print("\nPlotting Actual vs Predicted values...")
sample_size = int(0.2 * len(y_test))
sampled_indices = np.random.choice(len(y_test), size=sample_size, replace=False)

plt.figure(figsize=(10, 6))
plt.scatter(y_test.iloc[sampled_indices], y_pred_rf[sampled_indices], color='red', marker='o', label='Random Forest', s=10, alpha=0.5)
plt.scatter(y_test.iloc[sampled_indices], y_pred_lr[sampled_indices], color='blue', marker='X', label='Linear Regression', s=10, alpha=0.5)
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend(title="Models")
plt.tight_layout()
plt.savefig('actual_vs_predicted_values.png')
plt.close()

print("✅ Figures saved as 'random_forest_feature_importances.png' and 'actual_vs_predicted_values.png'.")

# Print total runtime
print(f"\n✅ Total script runtime: {time.time() - total_start_time:.2f} seconds.")
