#from src.utils import *
import streamlit as st
# from src.data_visualization import *
#from ..data_visualization import *
from src.data_aquisition import *
from src.data_cleaning_and_preprocessing import *

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def page_home():
    st.title("📊 Home Page")
    st.write("Welcome to the Stock Analysis App!")

def page_financial(df, tickers):
    st.title("📈 Financial Data")

    # General discussion about the data
    st.write("""
        This page provides an analysis of financial data for various stocks. 
        You can select a stock from the dropdown below to explore detailed charts 
        including moving averages and candlestick patterns, or view general stock 
        trends like gainers, trading volume, and price ranges.
    """)

    # Dropdown to select a stock from a list (e.g., SP500 tickers)
    selected_ticker = st.selectbox("Select a stock", tickers)

    # If a stock is selected, show the stock-specific plots
    if selected_ticker:
        # Remove the general charts and display stock-specific charts
        st.subheader(f"Stock Data for '{selected_ticker}'")

        # Moving averages and candlestick chart
        #plot_stock_with_moving_averages_plotly(df, selected_ticker)
        #plot_candlestick_chart(df, selected_ticker)
    else:
        # Show the general charts if no stock is selected
        st.subheader("General Stock Trends")

        # Display gainers chart, volume chart, and price range chart
        #show_gainers_chart(df)
        #show_volume_chart(df)
        #show_price_range_chart(df)

def page_news():
    st.title("📰 News & Sentiment")
    st.write("News headlines and sentiment analysis.")

# === Set Date Range ===
start_date = '2010-01-01'
end_date = datetime.now().strftime("%Y-%m-%d")

# === Fetch Financial Data ===
sp500_tickers = get_sp500_tickers()
yahoo_df = fetch_yahoo_data(sp500_tickers, start_date, end_date)
financial_cleaned_data = clean_financial_data(yahoo_df)
financial_df = financial_cleaned_data.copy()

# Page selector
page = st.sidebar.radio("Go to", ("Home", "Financial Data", "News & Sentiment"))

if page == "Home":
    page_home()
elif page == "Financial Data":
    page_financial(financial_df, sp500_tickers)
elif page == "News & Sentiment":
    page_news()
