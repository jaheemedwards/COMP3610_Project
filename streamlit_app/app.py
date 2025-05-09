# ==========================
# 📁 Imports
# ==========================
from project_utils import *
from data_visualization import *
from data_aquisition import *
from data_cleaning_and_preprocessing import *
from finalDatasetScript import *
from PIL import Image
import streamlit as st

# ==========================
# ⚙️ Page Configuration
# ==========================
st.set_page_config(page_title="📊 Stock Analysis App", layout="wide")


# ==========================
# 🏠 Home Page
# ==========================
def page_home():
    st.title("📊 Home Page")
    st.write("Welcome to the Stock Analysis App!")

    st.markdown("""
    This application allows you to:
    - 📈 *Analyze financial data* of S&P 500 companies.
    - 📰 *Review financial news and sentiment analysis*.
    - 🕵‍♂ *Explore stock performance*, volume trends, and price movements.

    Use the tabs at the top to navigate between:
    - *Financial Data*: Select a stock to view detailed visualizations or general market trends.
    - *News & Sentiment*: Review recent news headlines and their associated sentiment.
    - *Model Evaluation*: Compare ML models and explore their methodology.
    - *Prediction Dashboard*: See predictions of stock price movements using ML models.
    """)

    st.success("Get started by selecting a tab above!")

    st.image(
        "https://media.giphy.com/media/3oKIPtjElfqwMOTbH2/giphy.gif",
        caption="Let's analyze some stocks!",
        use_container_width=True
    )


# ==========================
# 🧠 Model Evaluation Page
# ==========================
def page_model_and_methodology():
    st.title("📊 Model Evaluation Dashboard")
    st.markdown("Performance comparison of ML models for predicting stock price movement using headlines and tickers.")

    # Dataset Overview
    st.header("📂 Dataset Overview")
    st.markdown("""
    - **News Records**: 45,043  
    - **Price Records**: 1,836,992  
    - **Merged Dataset**: 43,643 usable records  
    - **Dropped**: 1,400 rows with missing next-day close prices  
    - **Vectorized using**: TF-IDF for headlines + One-Hot Encoding for tickers  
    """)

    st.divider()

    # Binary Classifier Evaluation
    st.header("🔴 Binary Classifier (Up/Down)")

    st.subheader("📊 Class Distribution")
    st.markdown("""
    - **Class 1 (Up or Stay)**: 23,273  
    - **Class 0 (Down)**: 20,370  
    """)

    st.subheader("🧪 Train/Test Split")
    st.markdown("""
    - Train: 34,914  
    - Test: 8,729  
    - Accuracy: **57%**
    """)

    st.subheader("📋 Classification Report")
    st.code("""
              precision    recall  f1-score   support

           0       0.58      0.30      0.39      4074
           1       0.57      0.81      0.67      4655

    accuracy                           0.57      8729
   macro avg       0.58      0.56      0.53      8729
weighted avg       0.58      0.57      0.54      8729
    """, language="text")

    st.subheader("📉 Confusion Matrix")
    try:
        st.image("streamlit_app/confusion_matrix_updown.jpg")
    except FileNotFoundError:
        st.warning("Confusion matrix image not found.")

    st.markdown("""
    **Insights**:
    - Model performs better on 'Up/Stay' predictions.
    - Struggles with 'Down' class — may indicate data imbalance or overlapping features.
    """)

    st.divider()

    # Regression Model Evaluation
    st.title("📈 Machine Learning Models Overview")

    st.markdown("Overview of ML models used to predict closing prices based on financial data.")

    st.header("📊 Dataset Information")
    st.markdown("""
    - **Source**: `financial_data.csv`
    - **Total Samples**: 1,836,992  
    - **Key Features**: `High`, `Low`, `Open`, `Volume`, `Price_Change`, `Volatility`
    """)

    st.header("🛠️ Data Preprocessing")
    st.markdown("""
    - Added `Price_Change` and `Volatility`
    - Standardized features using `StandardScaler`
    """)

    st.subheader("Sample of Preprocessed Data")
    st.dataframe({
        "Date": ["2010-01-04"] * 5,
        "Ticker": ["A", "AAPL", "ABT", "ACGL", "ACN"],
        "Close": [19.97, 6.44, 18.58, 7.60, 31.65],
        "High": [20.18, 6.46, 18.61, 7.63, 31.75],
        "Low": [19.87, 6.39, 18.40, 7.58, 31.22],
        "Open": [20.03, 6.42, 18.49, 7.59, 31.24],
        "Volume": [3815561, 493729600, 10829095, 4813200, 3650100],
        "Price_Change": [-0.68, -0.68, 1.88, -0.59, 3.16],
        "Volatility": [0.91] * 5
    })

    st.header("🔍 Feature Matrix & Target")
    st.markdown("""
    - **Target**: `Close`
    - **Dropped from X**: `Date`, `Ticker`, `Close`
    - **Shape**: X → (1,836,992, 6), y → closing prices
    """)

    st.header("🧪 Train/Test Split")
    st.markdown("""
    - **Ratio**: 80/20  
    - **Train**: 1,469,593  
    - **Test**: 367,399
    """)

    st.header("⚙️ Models Trained")

    # Random Forest
    st.subheader("1. Random Forest Regressor")
    st.markdown("""
    - **Params**: `n_estimators=50`, `max_depth=12`, `min_samples_leaf=5`  
    - **Training Time**: 48.97s  
    - **MSE**: 4.80  
    - **R²**: 1.00
    """)

    # Linear Regression
    st.subheader("2. Linear Regression")
    st.markdown("""
    - **Training Time**: 0.15s  
    - **MSE**: 3.34  
    - **R²**: 1.00
    """)

    st.info("💡 Both models achieve perfect R² — consider validating further for data leakage or overfitting.")

    st.header("💾 Model Saving")
    st.markdown("""
    Saved with `joblib` (zlib compression):

    - `models/random_forest_model.pkl`  
    - `models/linear_regression_model.pkl`  
    - `models/scaler.pkl`
    """)

    st.header("📊 Visualizations")

    st.subheader("📌 Feature Importances (Random Forest)")
    image1 = Image.open("streamlit_app/random_forest_feature_importances.jpg")
    st.image(image1, caption="Feature Importances")

    st.subheader("📌 Actual vs Predicted Values")
    image2 = Image.open("streamlit_app/actual_vs_predicted_values.jpg")
    st.image(image2, caption="Actual vs Predicted Closing Prices")

    st.header("⏱️ Total Script Runtime")
    st.markdown("**⏳ 53.07 seconds**")


# ==========================
# 💹 Financial Data Page
# ==========================
def page_financial(df, tickers, start_date, end_date):
    st.title("📈 Financial Data")

    st.write("""
    Analyze financial metrics of selected stocks or get a general overview of market performance.
    """)

    options = ["-- General Overview --"] + sorted(tickers)
    selected_ticker = st.selectbox("Select a stock", options)

    if selected_ticker == "-- General Overview --":
        st.subheader(f"Data from {start_date} to {end_date}")
        st.write(df.describe())

        st.subheader("📊 General Stock Trends")
        st.plotly_chart(show_gainers_chart(df))
        st.plotly_chart(show_volume_chart(df))
        st.plotly_chart(show_price_range_chart(df))
    else:
        st.subheader(f"📌 Stock Data for '{selected_ticker}'")
        st.plotly_chart(plot_stock_with_moving_averages_plotly(df, selected_ticker))
        st.plotly_chart(plot_candlestick_chart(df, selected_ticker))


# ==========================
# 🗞️ News & Sentiment Page
# ==========================
# Function to display news and sentiment analysis page
def page_news(df, tickers):
    # Set the page title and introductory text
    st.title("📰 News & Sentiment")
    st.write("Explore news headlines and sentiment analysis.")

    # Allow user to select a ticker from a dropdown menu
    selected_ticker = st.selectbox("Select a Ticker", options=sorted(df['Ticker'].unique()))

    # Filter the dataframe to only include rows for the selected ticker
    filtered_df = df[df['Ticker'] == selected_ticker]

    # Display a subheader and show recent news headlines for the selected ticker
    st.subheader("Recent News Headlines")
    st.dataframe(
        filtered_df[['Date', 'Title', 'Sentiment', 'Sentiment_Label']]
        .sort_values('Date', ascending=False)  # Sort by most recent
        .reset_index(drop=True),               # Reset index for clean display
        use_container_width=True               # Use full width of container
    )

    # Subheader for visual insights
    st.subheader("Visual Analysis")

    # Plot 1: Bar chart showing count of each sentiment label (Positive, Neutral, Negative)
    sentiment_bar_fig = plot_sentiment_label_bar_chart(df, selected_ticker)
    st.plotly_chart(sentiment_bar_fig)

    # Plot 2: Line chart showing sentiment trends over time
    sentiment_time_fig = plot_sentiment_over_time(df, selected_ticker)
    st.plotly_chart(sentiment_time_fig)

    # Plot 3: Average sentiment score by day of the week
    by_day_fig = plot_sentiment_by_day(df, selected_ticker)
    st.plotly_chart(by_day_fig)

    # Plot 4: Bar chart of top keywords from positive sentiment headlines
    top_keywords_fig = plot_top_positive_keywords(df, selected_ticker)
    st.plotly_chart(top_keywords_fig)

    # Plot 5: Line chart showing monthly average sentiment scores
    monthly_avg_fig = plot_monthly_avg_sentiment(df, selected_ticker)
    st.plotly_chart(monthly_avg_fig)

# ==========================
# 🔮 Prediction Page
# ==========================
# Function to predict stock movement direction based on news title and provide confidence insights
def make_prediction_with_insights(news_title, selected_ticker, xgb_updown_model):
    # Step 1: Create a DataFrame with the news title and ticker to feed into the model
    data = pd.DataFrame({'Title': [news_title], 'Ticker': [selected_ticker]})
    
    # Step 2: Use the model to predict stock movement (1 = Up, 0 = Down) and get probabilities
    prediction = xgb_updown_model.predict(data)[0]
    prediction_prob = xgb_updown_model.predict_proba(data)[0]
    
    # Step 3: Convert prediction result to readable text and calculate confidence percentage
    prediction_text = "Stock will go *Up*" if prediction == 1 else "Stock will go *Down*"
    confidence = max(prediction_prob) * 100  # Use the higher probability as the confidence
    
    # Return the prediction message, confidence, and full probability distribution
    return prediction_text, confidence, prediction_prob

# Function to display predictions based on financial data and news sentiment
def page_predictions(news_financial_df, financial_df, tickers, rf_model, lr_model, xgb_updown_model, scaler_rf_lr):
    # Page title and description
    st.title("📈 Financial and News Data")
    st.write("Financial and News Headlines Sentiment Analysis with Predictions.")

    # Dropdown to select a stock ticker
    selected_ticker = st.selectbox(
        "Select a Ticker",
        sorted(tickers),
        key="financial_news_ticker_selectbox"
    )

    # Section: Predict stock movement from news headline
    st.header("Stock Movement Prediction from News")

    # Text input for entering a news article title
    news_title = st.text_input("Enter News Article Title")

    # If both a news title and ticker are provided, run prediction
    if news_title and selected_ticker:
        # Call prediction function to get direction and confidence
        prediction_text, confidence, prediction_prob = make_prediction_with_insights(news_title, selected_ticker, xgb_updown_model)
        
        # Display the predicted direction and confidence
        st.write(f"📊 *XGBoost UpDown Prediction:* {prediction_text}")
        st.write(f"🔮 *Confidence Level:* {confidence:.2f}%")

        # Visualize the prediction probabilities
        st.write("📈 *Prediction Probability Distribution*")
        fig = plot_prediction_confidence(prediction_prob)
        st.plotly_chart(fig)
        
        # Provide contextual insight based on predicted direction
        if prediction_text == "Stock will go *Up*":
            st.write("🧐 Insight: This news article indicates a positive sentiment towards the stock, suggesting it may rise. This could be due to favorable financial news or a market trend.")
        else:
            st.write("🧐 Insight: The news article suggests a negative sentiment towards the stock, which could be linked to poor financial reports or market downturns.")

    # Section: Predict future closing price of selected stock
    st.header("📉 Predict Closing Price")

    # Filter financial data for the selected ticker only
    filtered_df = financial_df[financial_df['Ticker'] == selected_ticker]

    # Apply preprocessing and feature engineering
    financial_preprocessed_df = feature_engineering_financial(filtered_df, False)
    X_rf_lr = financial_preprocessed_df.drop(columns=['Date', 'Ticker', 'Close'])  # Remove non-feature columns
    
    # Scale features before prediction
    X_to_predict_scaled = scaler_rf_lr.transform(X_rf_lr)

    # Make predictions using both Random Forest and Linear Regression models
    y_pred_rf = rf_model.predict(X_to_predict_scaled)
    y_pred_lr = lr_model.predict(X_to_predict_scaled)

    # Copy the processed data for display and visualization
    final_df = financial_preprocessed_df.copy()

    # Add model predictions to the dataframe
    final_df['RF_Prediction'] = y_pred_rf
    final_df['LR_Prediction'] = y_pred_lr

    # Plot actual vs predicted prices
    st.subheader(f"Actual Closing Price vs Predictions for {selected_ticker}")
    prediction_chart = create_prediction_plot(final_df, selected_ticker)
    st.plotly_chart(prediction_chart)

    # Display most recent predicted prices from both models
    st.subheader("Predicted Close Prices (most recent day)")
    st.write(f"📘 *Random Forest Prediction:* ${y_pred_rf[-1]:.2f}")
    st.write(f"📙 *Linear Regression Prediction:* ${y_pred_lr[-1]:.2f}")

    # Optional: Sentiment Analysis and Price Movement
    # Uncomment the sections below if needed, making sure they reference the correct dataframes (news_financial_df, financial_df)
    
    # Sentiment Analysis Section (e.g., analyze_sentiment function should be defined elsewhere)
    # st.header("🧠 Sentiment Analysis")
    # news_financial_df = analyze_sentiment(news_financial_df)
    # sentiment_counts = news_financial_df[news_financial_df['Ticker'] == selected_ticker]['Sentiment_Label'].value_counts().reset_index()
    # sentiment_counts.columns = ['Sentiment_Label', 'Count']
    # fig1 = px.bar(sentiment_counts, x='Sentiment_Label', y='Count', color='Sentiment_Label',
    #               title=f'Sentiment Distribution for {selected_ticker}', text='Count')
    # fig1.update_traces(textposition='outside')
    # fig1.update_layout(showlegend=False)
    # st.plotly_chart(fig1)

    # Price Movement Section (e.g., analyze_price_movement function should be defined elsewhere)
    # st.header("📈 Price Movement Around News Events")
    # price_df = analyze_price_movement(news_financial_df)
    # st.dataframe(price_df[price_df['Ticker'] == selected_ticker].sort_values('Date', ascending=False))

    # Stock Indicators Section (e.g., calculate_stock_indicators function should be defined elsewhere)
    # st.header("📊 Stock Performance Indicators")
    # indicators_df = calculate_stock_indicators(financial_df)
    # st.dataframe(indicators_df[indicators_df['Ticker'] == selected_ticker].sort_values('Date', ascending=False))

    # News Event Visualization Section (e.g., visualize_news_on_price function should be defined elsewhere)
    # st.header("🗞 News on Price Chart")
    # fig5 = visualize_news_on_price(news_financial_df, selected_ticker)
    # st.plotly_chart(fig5)



# Cache the function so that data loading doesn't re-run unnecessarily
@st.cache_data
def load_data(start_date=None, end_date=None):
    # 1. Load S&P 500 tickers from external source or API
    sp500_tickers = get_sp500_tickers()

    # 2. Load pre-cleaned financial data from yfinance or a CSV file
    yahoo_df = fetch_yahoo_data(sp500_tickers, start_date, end_date)
    financial_cleaned_data = clean_financial_data(yahoo_df)
    # If the function above is not downloading the data due to too many
    # requests being made to yfinance then comment out the previous 2 
    # lines (yahoo_df..., financial_cleaned_data...) and uncomment the 
    # line below (financial_cleaned_data...) to run a previously saved
    # CSV file.
    
    #financial_cleaned_data = pd.read_csv('C:/Users/18684/OneDrive/Desktop/comp3610_finalProject/COMP3610_Project/data/financial_data.csv')

    # 3. Load preprocessed news data with parsed dates
    preprocessed_news = pd.read_csv("streamlit_app/preprocessed_news.csv", parse_dates=["Date"])

    # 4. Load merged financial and news dataset with parsed dates
    financial_and_news_data = pd.read_csv("streamlit_app/financial_and_news_merged.csv", parse_dates=["Date"])

    # 5. Return all loaded datasets
    return sp500_tickers, financial_cleaned_data, preprocessed_news, financial_and_news_data


# Cache the function so that models are only loaded once per session, improving performance
@st.cache_resource
def load_models():
    # Define paths to the pre-trained model and scaler files
    model_dir = "streamlit_app/models"
    rf_model_path = os.path.join(model_dir, 'random_forest_model.pkl')
    lr_model_path = os.path.join(model_dir, 'linear_regression_model.pkl')
    xgb_updown_model_path = os.path.join(model_dir, 'xgb_updown_model.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')

    # Load each model and scaler using joblib
    rf_model = joblib.load(rf_model_path)
    lr_model = joblib.load(lr_model_path)
    xgb_updown_model = joblib.load(xgb_updown_model_path)
    scaler = joblib.load(scaler_path)

    # Return the models and scaler to be used in the app
    return rf_model, lr_model, xgb_updown_model, scaler


# === Set Date Range for historical financial data ===
start_date = '2010-01-01'  # Start from Jan 1, 2010
end_date = datetime.now().strftime("%Y-%m-%d")  # Set end date to today's date

# Load data and models with a loading spinner for better user experience
with st.spinner("Loading data and models..."):
    sp500_tickers, financial_df, news_df, financial_and_news_df = load_data(start_date, end_date)  # Load datasets
    rf_model, lr_model, xgb_updown_model, scaler = load_models()  # Load ML models and scaler
st.success("Data and models loaded successfully!")  # Display success message

# Create tab layout for different sections of the app
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Home",                # Welcome and overview
    "Financial Data",      # Display historical stock data
    "News & Sentiment",    # Show news headlines and sentiment analysis
    "Models & Methodology",# Explain ML models used
    "Predictions"          # Provide prediction tools and visualizations
])

# Render each page within its corresponding tab
with tab1:
    page_home()  # Load Home page content
with tab2:
    page_financial(financial_df, sp500_tickers, start_date, end_date)  # Load Financial Data page
with tab3:
    page_news(news_df, sp500_tickers)  # Load News & Sentiment page
with tab4:
    page_model_and_methodology()  # Load Models & Methodology page
with tab5:
    page_predictions(financial_and_news_df, financial_df, sp500_tickers, rf_model, lr_model, xgb_updown_model, scaler)  # Load Predictions page
