from project_utils import *
import streamlit as st
from data_visualization import *
from data_aquisition import *
from data_cleaning_and_preprocessing import *

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

def page_financial(df, tickers):
    st.title("📈 Financial Data")

    # General discussion about the data
    st.write("""
        This page provides an analysis of financial data for various stocks. 
        You can select a stock from the dropdown below to explore detailed charts 
        including moving averages and candlestick patterns, or view general stock 
        trends like gainers, trading volume, and price ranges.
    """)

    # Insert a default option at the top of the dropdown
    options = ["-- General Overview --"] + tickers
    selected_ticker = st.selectbox("Select a stock", options)

    if selected_ticker == "-- General Overview --":
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
    st.write("News headlines and sentiment analysis.")

def page_financial_and_news(df, tickers):
    st.title("📈 Financial and News Data")
    st.write("Financial and News headlines sentiment analysis.")

@st.cache_data
def load_data():
    sp500_tickers = get_sp500_tickers()
    yahoo_df = fetch_yahoo_data(sp500_tickers, start_date, end_date)
    financial_cleaned_data = clean_financial_data(yahoo_df)
    #news_data = pd.read_csv("C:\Users\18684\OneDrive\Desktop\COMP3610_Project\data\news_data.csv")
    #financial_and_news_data = pd.read_csv("C:\Users\18684\OneDrive\Desktop\COMP3610_Project\data\financial_and_news_merged.csv")
    return sp500_tickers, financial_cleaned_data#, news_data, financial_and_news_data

# === Set Date Range ===
start_date = '2010-01-01'
end_date = datetime.now().strftime("%Y-%m-%d")

# Load cached data
sp500_tickers, financial_df = load_data()#, news_df, financial_and_news_df 

#Page selector
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Financial Data", "News & Sentiment", "Financial and News Headlines"])

with tab1:
    page_home()

with tab2:
    page_financial(financial_df, sp500_tickers)

with tab3:
    page_news()

with tab4:
    page_financial_and_news()
