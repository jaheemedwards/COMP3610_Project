from streamlit_app.project_utils import *

# Directories for saving
models_dir = "models"
log_dir = "logs"

# Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Logging setup
log_file = os.path.join(log_dir, "training_log_updown.txt")
def log(message):
    # Open the log file with UTF-8 encoding to support all characters (including emojis)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{message}\n")
    print(message)


log("🚀 Starting training pipeline...")

# Track the start time
start_time = time.time()

# Load data
log("📂 Loading data files...")
news_df = pd.read_csv("D:/jaheem/COMP3610_Project-main/streamlit_app/financial_and_news_merged.csv", parse_dates=["Date"])
full_prices_df = pd.read_csv("D:/jaheem/COMP3610_Project-main/financial_data.csv", parse_dates=["Date"])
log(f"Loaded {len(news_df)} news records 📈 and {len(full_prices_df)} price records 💵.")

# Sort both DataFrames
news_df = news_df.sort_values(["Ticker", "Date"])
full_prices_df = full_prices_df.sort_values(["Ticker", "Date"])

# Get next day's close price
log("📊 Calculating next day's closing price...")
full_prices_df["NextClose"] = full_prices_df.groupby("Ticker")["Close"].shift(-1)

# Merge
log("🔗 Merging news with next close prices...")
merged = pd.merge(
    news_df,
    full_prices_df[["Date", "Ticker", "NextClose"]],
    on=["Date", "Ticker"],
    how="left"
)
initial_len = len(merged)
merged.dropna(subset=["NextClose"], inplace=True)
log(f"Merged data has {len(merged)} records 🔥 (dropped {initial_len - len(merged)} rows without NextClose).")

# Create label
log("📝 Creating binary labels (1 = price up/stay, 0 = price down)...")
merged["Target"] = (merged["NextClose"] >= merged["Close"]).astype(int)
log(f"Class distribution:\n{merged['Target'].value_counts().to_dict()}")

# Prepare the feature set (Including Ticker)
log("🔠 Vectorizing news titles using TF-IDF and encoding Ticker...")

# Preprocessing pipeline: OneHotEncoder for Ticker + TF-IDF for Title
preprocessor = ColumnTransformer(
    transformers=[
        ('ticker', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), ['Ticker']),
        ('title', TfidfVectorizer(max_features=1000, stop_words="english"), 'Title')
    ]
)

# Split data
X = merged[['Title', 'Ticker']]
y = merged['Target']
log(f"Data shape: {X.shape}, Labels shape: {y.shape}")

# Train-test split
log("📚 Splitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
log(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Create a pipeline with preprocessing and XGBoost
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric="logloss"))
])

# Train the model
log("🤖 Training XGBoost classifier...")
pipeline.fit(X_train, y_train)
log("✅ Model training complete.")

# Evaluate
log("📈 Evaluating model...")
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

log("📊 Classification Report:\n" + report)
log("🔴 Confusion Matrix:\n" + str(matrix))

# Save confusion matrix as chart
log("📉 Saving confusion matrix chart...")

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")

# Plot confusion matrix with Seaborn heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=["Down", "Up"], yticklabels=["Down", "Up"], cbar=False)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix", fontsize=14)
plt.tight_layout()
confusion_matrix_chart_path = "confusion_matrix_updown.png"
plt.savefig(confusion_matrix_chart_path)
plt.close()
log(f"📸 Confusion matrix saved as {confusion_matrix_chart_path}")

# Save model and vectorizer
log("💾 Saving model and vectorizer to disk...")
model_path = os.path.join(models_dir, "xgb_updown_model.joblib")
joblib.dump(pipeline, model_path)  # Save the whole pipeline including preprocessor
log(f"💾 Saved {model_path}.")

# Log total runtime
end_time = time.time()
elapsed_time = end_time - start_time
log(f"⏱️ Model training completed in {elapsed_time:.2f} seconds.")

log("🎉 Training pipeline complete.\n")
