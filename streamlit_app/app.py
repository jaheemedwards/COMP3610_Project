from project_utils import *
import streamlit as st
from data_visualization import *
from data_aquisition import *
from data_cleaning_and_preprocessing import *
from finalDatasetScript import *

def page_home():
    st.title("📊 Home Page")
    st.write("Welcome to the Stock Analysis App!")

    st.markdown("""
    This application allows you to:
    - 📈 *Analyze financial data* of S&P 500 companies.
    - 📰 *Review financial news and sentiment analysis*.
    - 🕵‍♂ *Explore stock performance*, volume trends, and price movements.

    Use the tabs at the top to navigate between:
    - *Financial Data*: Select a stock to view detailed visualizations or see general market trends.
    - *News & Sentiment*: Read recent news headlines and view the associated sentiment score.
    """)

    st.success("Get started by selecting a tab above!")

    st.image(
        "https://media.giphy.com/media/3oKIPtjElfqwMOTbH2/giphy.gif",
        caption="Let's analyze some stocks!",
        use_container_width=True  # Updated parameter
    )

def page_model_and_methodology():
    pass

def page_financial(df, tickers, start_date, end_date):
    st.title("📈 Financial Data")

    # General discussion about the data
    st.write("""
        This page provides an analysis of financial data for various stocks. 
        You can select a stock from the dropdown below to explore detailed charts 
        including moving averages and candlestick patterns, or view general stock 
        trends like gainers, trading volume, and price ranges.
    """)

    # Insert a default option at the top of the dropdown
    options = ["-- General Overview --"] + sorted(tickers)
    selected_ticker = st.selectbox("Select a stock", options)

    if selected_ticker == "-- General Overview --":

        st.subheader("Data from " + start_date + " to " + end_date)
        st.write(df.describe())
        # Show the general charts
        st.subheader("📊 General Stock Trends")

        gainers_fig = show_gainers_chart(df)
        st.plotly_chart(gainers_fig)

        volume_fig = show_volume_chart(df)
        st.plotly_chart(volume_fig)

        price_range_fig = show_price_range_chart(df)
        st.plotly_chart(price_range_fig)

    else:
        # Show stock-specific charts
        st.subheader(f"📌 Stock Data for '{selected_ticker}'")

        moving_avg_fig = plot_stock_with_moving_averages_plotly(df, selected_ticker)
        st.plotly_chart(moving_avg_fig)

        candlestick_fig = plot_candlestick_chart(df, selected_ticker)
        st.plotly_chart(candlestick_fig)

        #Show the stock with the model prediction on it 

def page_news(df, tickers):
    st.title("📰 News & Sentiment")
    st.write("Explore news headlines and sentiment analysis.")

    # Ticker selection
    selected_ticker = st.selectbox("Select a Ticker", options=sorted(df['Ticker'].unique()))

    # Filter by selected ticker
    filtered_df = df[df['Ticker'] == selected_ticker]

    # Recent headlines
    st.subheader("Recent News Headlines")
    st.dataframe(
        filtered_df[['Date', 'Title', 'Sentiment', 'Sentiment_Label']]
        .sort_values('Date', ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )

    # Visual Analysis
    st.subheader("Visual Analysis")

    # Sentiment distribution
    sentiment_bar_fig = plot_sentiment_label_bar_chart(df, selected_ticker)
    st.plotly_chart(sentiment_bar_fig)

    sentiment_time_fig = plot_sentiment_over_time(df, selected_ticker)
    st.plotly_chart(sentiment_time_fig)

    by_day_fig = plot_sentiment_by_day(df, selected_ticker)
    st.plotly_chart(by_day_fig)

    top_keywords_fig = plot_top_positive_keywords(df, selected_ticker)
    st.plotly_chart(top_keywords_fig)

    monthly_avg_fig = plot_monthly_avg_sentiment(df, selected_ticker)
    st.plotly_chart(monthly_avg_fig)

def make_prediction_with_insights(news_title, selected_ticker, xgb_updown_model):
    # Step 1: Prepare the input data
    data = pd.DataFrame({'Title': [news_title], 'Ticker': [selected_ticker]})
    
    # Step 2: Get prediction and prediction probabilities from the model
    prediction = xgb_updown_model.predict(data)[0]
    prediction_prob = xgb_updown_model.predict_proba(data)[0]
    
    # Step 3: Determine the prediction result and confidence level
    prediction_text = "Stock will go *Up*" if prediction == 1 else "Stock will go *Down*"
    confidence = max(prediction_prob) * 100  # Confidence as a percentage
    
    return prediction_text, confidence, prediction_prob


def page_predictions(news_financial_df, financial_df, tickers, rf_model, lr_model, xgb_updown_model, scaler_rf_lr):
    st.title("📈 Financial and News Data")
    st.write("Financial and News Headlines Sentiment Analysis with Predictions.")

    # Select a ticker from the dropdown list
    selected_ticker = st.selectbox(
        "Select a Ticker",
        sorted(tickers),
        key="financial_news_ticker_selectbox"
    )

    st.header("Stock Movement Prediction from News")

    # Input for news article title
    news_title = st.text_input("Enter News Article Title")

    # When the user inputs both the news article and selects a stock, make the prediction
    if news_title and selected_ticker:
        # Get prediction and confidence level
        prediction_text, confidence, prediction_prob = make_prediction_with_insights(news_title, selected_ticker, xgb_updown_model)
        
        # Display prediction and confidence level
        st.write(f"📊 *XGBoost UpDown Prediction:* {prediction_text}")
        st.write(f"🔮 *Confidence Level:* {confidence:.2f}%")

        # Plot the prediction confidence
        st.write("📈 *Prediction Probability Distribution*")
        fig = plot_prediction_confidence(prediction_prob)
        st.plotly_chart(fig)
        
        # Add insight based on the prediction
        if prediction_text == "Stock will go *Up*":
            st.write("🧐 Insight: This news article indicates a positive sentiment towards the stock, suggesting it may rise. This could be due to favorable financial news or a market trend.")
        else:
            st.write("🧐 Insight: The news article suggests a negative sentiment towards the stock, which could be linked to poor financial reports or market downturns.")

    # Predict Closing Price Section
    st.header("📉 Predict Closing Price")

    filtered_df = financial_df[financial_df['Ticker'] == selected_ticker]

    # Apply feature engineering only on this subset
    financial_preprocessed_df = feature_engineering_financial(filtered_df, False)
    X_rf_lr = financial_preprocessed_df.drop(columns=['Date', 'Ticker', 'Close'])  # Features
    
    X_to_predict_scaled = scaler_rf_lr.transform(X_rf_lr)
    y_pred_rf = rf_model.predict(X_to_predict_scaled)
    y_pred_lr = lr_model.predict(X_to_predict_scaled)

    final_df = financial_preprocessed_df.copy()

    # Add predictions to the latest data
    final_df['RF_Prediction'] = y_pred_rf
    final_df['LR_Prediction'] = y_pred_lr

    # Plot Actual Closing Price, RF Prediction, and LR Prediction
    st.subheader(f"Actual Closing Price vs Predictions for {selected_ticker}")
    prediction_chart = create_prediction_plot(final_df, selected_ticker)
    st.plotly_chart(prediction_chart)

    # Display predictions for the most recent day
    st.subheader("Predicted Close Prices (most recent day)")
    st.write(f"📘 *Random Forest Prediction:* ${y_pred_rf[-1]:.2f}")
    st.write(f"📙 *Linear Regression Prediction:* ${y_pred_lr[-1]:.2f}")



    # Optional: Sentiment Analysis and Price Movement (currently commented out)
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



@st.cache_data
def load_data():

    # 1. Load S&P 500 tickers
    sp500_tickers = get_sp500_tickers()

    # 2. Fetch financial data and clean it
    #yahoo_df = fetch_yahoo_data(sp500_tickers, start_date, end_date)
    #financial_cleaned_data = clean_financial_data(yahoo_df)
    financial_cleaned_data = pd.read_csv('C:/Users/18684/OneDrive/Desktop/COMP3610_Project/data/financial_data.csv')

    # 3. Load pre-cleaned news data and preprocess it
    preprocessed_news = pd.read_csv("streamlit_app\preprocessed_news.csv", parse_dates=["Date"])

    financial_and_news_data = pd.read_csv("streamlit_app/financial_and_news_merged.csv", parse_dates=["Date"])

    return sp500_tickers, financial_cleaned_data, preprocessed_news, financial_and_news_data

# Cache the loading of models to optimize performance
@st.cache_resource
def load_models():
    # Define paths to your models
    model_dir = "streamlit_app/models"
    rf_model_path = os.path.join(model_dir, 'random_forest_model.pkl')
    lr_model_path = os.path.join(model_dir, 'linear_regression_model.pkl')
    xgb_updown_model_path = os.path.join(model_dir, 'xgb_updown_model.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')

    # Load models and scaler
    rf_model = joblib.load(rf_model_path)
    lr_model = joblib.load(lr_model_path)
    xgb_updown_model = joblib.load(xgb_updown_model_path)
    scaler = joblib.load(scaler_path)

    # Return models and scaler
    return rf_model, lr_model, xgb_updown_model, scaler

# === Set Date Range ===
start_date = '2010-01-01'
end_date = datetime.now().strftime("%Y-%m-%d")


with st.spinner("Loading data and models..."):
    sp500_tickers, financial_df, news_df, financial_and_news_df = load_data()
    rf_model, lr_model, xgb_updown_model, scaler = load_models()
st.success("Data and models loaded successfully!")



#Page selector
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Financial Data", "News & Sentiment", "Predictions"])

with tab1:
    page_home()
with tab2:
    page_financial(financial_df, sp500_tickers, start_date, end_date)
with tab3:
    page_news(news_df, sp500_tickers)
with tab4:
    page_predictions(financial_and_news_df, financial_df, sp500_tickers, rf_model, lr_model, xgb_updown_model, scaler)