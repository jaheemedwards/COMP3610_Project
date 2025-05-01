# === Imports ===
from project_utils import *
from data_aquisition import *
from data_cleaning_and_preprocessing import *

# === Set Date Range ===
start_date = '2010-01-01'
end_date = datetime.now().strftime("%Y-%m-%d")

# === Fetch Financial Data ===
sp500_tickers = get_sp500_tickers()
yahoo_df = fetch_yahoo_data(sp500_tickers, start_date, end_date)
financial_cleaned_data = clean_financial_data(yahoo_df)
financial_df = financial_cleaned_data.copy()

# === Fetch and Clean News Data ===
stock_tickers = sp500_tickers
finviz_urls = generate_finviz_urls(stock_tickers)
news_df = fetch_and_parse_finviz_news(stock_tickers, finviz_urls, delay=1)
news_df_cleaned = clean_news_data(news_df)

# === Save Financial Data ===
financial_df.to_csv('data/financial_data.csv', index=False)

# === Save News Data ===
news_df_cleaned.to_csv('data/news_data.csv', index=False)

# === Merge Financial and News Data ===
financial_and_news_df = merge_datasets(financial_df, news_df_cleaned)

# Full dataset with all rows
full_df = financial_and_news_df.copy()

# Filtered dataset with only rows that have news (for sentiment analysis)
sentiment_df = financial_and_news_df.dropna(subset=['Title']).reset_index(drop=True)
sentiment_df.to_csv('data/financial_and_news_merged.csv', index=False)

# === Preprocess Data for Modeling ===
financial_df_preprocessed = feature_engineering_financial(financial_df)
news_df_preprocessed = preprocess_news_data(news_df_cleaned)
