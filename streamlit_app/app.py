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
    - 📈 **Analyze financial data** of S&P 500 companies.
    - 📰 **Review financial news and sentiment analysis**.
    - 🕵️‍♂️ **Explore stock performance**, volume trends, and price movements.

    Use the tabs at the top to navigate between:
    - **Financial Data**: Select a stock to view detailed visualizations or see general market trends.
    - **News & Sentiment**: Read recent news headlines and view the associated sentiment score.
    """)

    st.success("Get started by selecting a tab above!")

    st.image(
        "https://media.giphy.com/media/3oKIPtjElfqwMOTbH2/giphy.gif",
        caption="Let's analyze some stocks!",
        use_container_width=True  # Updated parameter
    )

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


def page_financial_and_news(df, tickers):
    st.title("📈 Financial and News Data")
    st.write("Financial and News Headlines Sentiment Analysis.")

    # Ensure 'Date' is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # ✅ Fix: Add a unique key to prevent duplicate element ID error
    selected_ticker = st.selectbox(
        "Select a Ticker",
        sorted(tickers),
        key="financial_news_ticker_selectbox"
    )

    # 1. Sentiment Analysis
    st.header("🧠 Sentiment Analysis")
    df = analyze_sentiment(df)
    sentiment_counts = df[df['Ticker'] == selected_ticker]['Sentiment_Label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment_Label', 'Count']
    fig1 = px.bar(sentiment_counts, x='Sentiment_Label', y='Count', color='Sentiment_Label',
                  title=f'Sentiment Distribution for {selected_ticker}', text='Count')
    fig1.update_traces(textposition='outside')
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1)

    # 2. Price Movement Analysis
    st.header("📈 Price Movement Around News Events")
    price_df = analyze_price_movement(df)
    st.dataframe(price_df[price_df['Ticker'] == selected_ticker].sort_values('Date', ascending=False))

    # 3. Stock Indicators
    st.header("📊 Stock Performance Indicators")
    indicators_df = calculate_stock_indicators(df)
    st.dataframe(indicators_df[indicators_df['Ticker'] == selected_ticker].sort_values('Date', ascending=False))

    # 4. News Event Visualization
    st.header("🗞️ News on Price Chart")
    fig5 = visualize_news_on_price(df, selected_ticker)
    st.plotly_chart(fig5)

@st.cache_data
def load_data():

    # 1. Load S&P 500 tickers
    sp500_tickers = get_sp500_tickers()

    # 2. Fetch financial data and clean it
    yahoo_df = fetch_yahoo_data(sp500_tickers, start_date, end_date)
    financial_cleaned_data = clean_financial_data(yahoo_df)

    # 3. Load pre-cleaned news data and preprocess it
    preprocessed_news = pd.read_csv("preprocessed_news.csv", parse_dates=["Date"])

    financial_and_news_data = pd.read_csv("financial_and_news_merged.csv", parse_dates=["Date"])

    return sp500_tickers, financial_cleaned_data, preprocessed_news, financial_and_news_data

# === Set Date Range ===
start_date = '2010-01-01'
end_date = datetime.now().strftime("%Y-%m-%d")

# Load cached data
sp500_tickers, financial_df, news_df, financial_and_news_df = load_data()

#Page selector
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Financial Data", "News & Sentiment", "Financial and News Headlines"])

with tab1:
    page_home()
with tab2:
    page_financial(financial_df, sp500_tickers, start_date, end_date)
with tab3:
    page_news(news_df, sp500_tickers)
with tab4:
    page_financial_and_news(financial_and_news_df, sp500_tickers)
