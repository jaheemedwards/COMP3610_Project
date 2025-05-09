from project_utils import *

# -----------------------------------------------
# 1. Sentiment Analysis on News Headlines
# -----------------------------------------------
def analyze_sentiment(df):
    def get_sentiment(text):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'

    df['Sentiment_Label'] = df['Title'].apply(get_sentiment)
    return df

# -----------------------------------------------
# 2. Price Movement Analysis Based on News
# -----------------------------------------------
def analyze_price_movement(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Price Change'] = df.groupby('Ticker')['Close'].pct_change() * 100
    df['Price Before News'] = df.groupby('Ticker')['Close'].shift(1)
    df['Price After News'] = df.groupby('Ticker')['Close'].shift(-1)
    return df[['Date', 'Ticker', 'Close', 'Price Change', 'Price Before News', 'Price After News']]

# -----------------------------------------------
# 3. Stock Performance Indicators
# -----------------------------------------------
def calculate_stock_indicators(df):
    df['Volatility'] = df.groupby('Ticker')['Close'].rolling(window=30).std().reset_index(0, drop=True)
    df['Daily Returns'] = df.groupby('Ticker')['Close'].pct_change() * 100
    df['Volume Change'] = df.groupby('Ticker')['Volume'].pct_change() * 100
    return df[['Date', 'Ticker', 'Close', 'Volatility', 'Daily Returns', 'Volume Change']]

# -----------------------------------------------
# 4. Visualization of News Events on Stock Chart
# -----------------------------------------------
def visualize_news_on_price(df, ticker):
    filtered_df = df[df['Ticker'] == ticker].copy()
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

    fig = px.line(filtered_df, x='Date', y='Close', title=f'Stock Price for {ticker}')

    for _, row in filtered_df.iterrows():
        fig.add_annotation(x=row['Date'], y=row['Close'],
                           text=row['Title'], showarrow=True,
                           arrowhead=2, ax=-40, ay=-40,
                           font=dict(size=8))
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Close Price')
    return fig
