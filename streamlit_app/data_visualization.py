from project_utils import *

def plot_stock_with_moving_averages(df, ticker):
    """
    Plots the Close price with 100-day and 200-day moving averages for a given stock in dark mode.

    Parameters:
    - df_tidy (pd.DataFrame): Tidy DataFrame with columns 'Date', 'Ticker', 'Close'.
    - ticker (str): Ticker symbol of the stock to plot.
    """
    # Set dark mode style
    plt.style.use('dark_background')

    # Filter for the selected stock
    stock_df = df[df['Ticker'] == ticker].copy()

    # Ensure 'Date' is datetime and sorted
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df = stock_df.sort_values('Date')

    # Calculate moving averages
    stock_df['MA100'] = stock_df['Close'].rolling(window=100).mean()
    stock_df['MA200'] = stock_df['Close'].rolling(window=200).mean()

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(stock_df['Date'], stock_df['Close'], label='Close Price', color='cyan')
    plt.plot(stock_df['Date'], stock_df['MA100'], label='100-Day MA', color='orange')
    plt.plot(stock_df['Date'], stock_df['MA200'], label='200-Day MA', color='red')

    plt.title(f'{ticker} Close Price with 100-Day & 200-Day Moving Averages', color='white')
    plt.xlabel('Date', color='white')
    plt.ylabel('Price', color='white')
    plt.legend()
    plt.grid(True, color='gray', linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_stock_with_moving_averages_plotly(df, ticker):
    """
    Plots the Close price with 100-day and 200-day moving averages for a given stock using Plotly (dark mode, with slider).

    Parameters:
    - df_tidy (pd.DataFrame): Tidy DataFrame with columns 'Date', 'Ticker', 'Close'.
    - ticker (str): Ticker symbol of the stock to plot.
    """
    # Filter for the selected stock
    stock_df = df[df['Ticker'] == ticker].copy()

    # Ensure 'Date' is datetime and sorted
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df = stock_df.sort_values('Date')

    # Calculate moving averages
    stock_df['MA100'] = stock_df['Close'].rolling(window=100).mean()
    stock_df['MA200'] = stock_df['Close'].rolling(window=200).mean()

    # Plotting
    fig = go.Figure()

    # Close price line
    fig.add_trace(go.Scatter(
        x=stock_df['Date'],
        y=stock_df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='cyan')
    ))

    # 100-day MA line
    fig.add_trace(go.Scatter(
        x=stock_df['Date'],
        y=stock_df['MA100'],
        mode='lines',
        name='100-Day MA',
        line=dict(color='orange')
    ))

    # 200-day MA line
    fig.add_trace(go.Scatter(
        x=stock_df['Date'],
        y=stock_df['MA200'],
        mode='lines',
        name='200-Day MA',
        line=dict(color='red')
    ))

    # Layout
    fig.update_layout(
        title=f'{ticker} Close Price with 100-Day & 200-Day Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        xaxis_rangeslider_visible=True,  # This shows the slider
        autosize=True,
        width=1000,
        height=600
    )

    #fig.show()
    return fig

def plot_candlestick_chart(df, ticker):
    """
    Plots a candlestick chart for a given stock.

    Parameters:
    - df_tidy (pd.DataFrame): Tidy DataFrame with columns 'Date', 'Ticker', 'Open', 'High', 'Low', 'Close'.
    - ticker (str): Ticker symbol of the stock to plot.
    """
    # Filter for the selected stock
    stock_df = df[df['Ticker'] == ticker].copy()

    # Ensure 'Date' is datetime and sorted
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df = stock_df.sort_values('Date')

    # Plotting
    fig = go.Figure(data=[go.Candlestick(
        x=stock_df['Date'],
        open=stock_df['Open'],
        high=stock_df['High'],
        low=stock_df['Low'],
        close=stock_df['Close']
    )])

    fig.update_layout(
        title=f'{ticker} Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=True,
        template='plotly_dark',
        autosize=True,
        width=1000,
        height=600
    )

    #fig.show()
    return fig

def prepare_stock_data(df, date=None):
    """
    Prepares stock data for visualization by calculating daily percentage changes.

    Args:
        df (pd.DataFrame): Input DataFrame with stock data
        date (str): Date to filter for (YYYY-MM-DD format)

    Returns:
        pd.DataFrame: Prepared DataFrame with percentage changes
        str: Formatted date string
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Pct_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100

    if date is None:
        date = df['Date'].max()
    else:
        date = pd.to_datetime(date)

    daily_data = df[df['Date'] == date].sort_values('Pct_Change', ascending=False)
    return daily_data, date.strftime('%Y-%m-%d')

def show_gainers_chart(df, date=None):
    """
    Displays a bar chart of top gaining stocks with a green/orange color scheme.

    Args:
        df (pd.DataFrame): Input stock data
        date (str): Date to analyze (YYYY-MM-DD format)
    """
    daily_data, date_str = prepare_stock_data(df, date)
    top_gainers = daily_data.nlargest(15, 'Pct_Change')

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top_gainers['Ticker'],
            y=top_gainers['Pct_Change'],
            text=top_gainers['Pct_Change'].round(2).astype(str) + '%',
            textposition='auto',
            marker=dict(
                color=top_gainers['Pct_Change'],
                colorscale='tealrose',
                colorbar=dict(title='% Change')
            ),
            hovertemplate='<b>%{x}</b><br>Change: %{y:.2f}%<extra></extra>'
        )
    )

    fig.update_layout(
        title=f"<b>Top Gainers - {date_str}</b>",
        plot_bgcolor='#121212',
        paper_bgcolor='#121212',
        font=dict(color='white'),
        xaxis=dict(
            showline=True,
            linecolor='#333333',
            gridcolor='#333333'
        ),
        yaxis=dict(
            title="<b>Percentage Change</b>",
            showline=True,
            linecolor='#333333',
            gridcolor='#333333'
        ),
        hoverlabel=dict(
            bgcolor='#222222',
            font_size=14
        ),
        height=500
    )

    #fig.show()
    return fig

def show_volume_chart(df, date=None):
    """
    Displays a bar chart of most active stocks by volume with a purple/blue color scheme.

    Args:
        df (pd.DataFrame): Input stock data
        date (str): Date to analyze (YYYY-MM-DD format)
    """
    daily_data, _ = prepare_stock_data(df, date)
    most_active = daily_data.nlargest(15, 'Volume')

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=most_active['Ticker'],
            y=most_active['Volume']/1e6,
            text=most_active['Volume'].apply(lambda x: f"{x/1e6:.1f}M"),
            textposition='auto',
            marker=dict(
                color=most_active['Volume']/1e6,
                colorscale='purpor',
                colorbar=dict(title='Volume (M)')
            ),
            hovertemplate='<b>%{x}</b><br>Volume: %{y:.1f}M<extra></extra>'
        )
    )

    fig.update_layout(
        title="<b>Most Active Stocks</b>",
        plot_bgcolor='#121212',
        paper_bgcolor='#121212',
        font=dict(color='white'),
        xaxis=dict(
            showline=True,
            linecolor='#333333',
            gridcolor='#333333'
        ),
        yaxis=dict(
            title="<b>Volume (Millions)</b>",
            showline=True,
            linecolor='#333333',
            gridcolor='#333333'
        ),
        hoverlabel=dict(
            bgcolor='#222222',
            font_size=14
        ),
        height=500
    )

    #fig.show()
    return fig

def show_price_range_chart(df, date=None):
    """
    Displays a price range visualization with a red/blue color scheme.

    Args:
        df (pd.DataFrame): Input stock data
        date (str): Date to analyze (YYYY-MM-DD format)
    """
    daily_data, _ = prepare_stock_data(df, date)
    top_performers = daily_data.nlargest(8, 'Pct_Change')

    fig = go.Figure()

    for _, row in top_performers.iterrows():
        # Price range line (low to high)
        fig.add_trace(
            go.Scatter(
                x=[row['Ticker'], row['Ticker']],
                y=[row['Low'], row['High']],
                mode='lines',
                line=dict(width=3, color='#666666'),
                showlegend=False,
                hovertemplate=f"<b>{row['Ticker']}</b><br>High: {row['High']:.2f}<br>Low: {row['Low']:.2f}<extra></extra>"
            )
        )

        # Open price marker
        fig.add_trace(
            go.Scatter(
                x=[row['Ticker']],
                y=[row['Open']],
                mode='markers',
                marker=dict(
                    symbol='line-ns-open',
                    size=20,
                    line=dict(width=3, color='#1f77b4')  # Blue
                ),
                name='Open',
                showlegend=False,
                hovertemplate=f"<b>{row['Ticker']}</b><br>Open: {row['Open']:.2f}<extra></extra>"
            )
        )

        # Close price marker
        fig.add_trace(
            go.Scatter(
                x=[row['Ticker']],
                y=[row['Close']],
                mode='markers',
                marker=dict(
                    symbol='line-ns-open',
                    size=20,
                    line=dict(width=3, color='#ff7f0e')  # Orange
                ),
                name='Close',
                showlegend=False,
                hovertemplate=f"<b>{row['Ticker']}</b><br>Close: {row['Close']:.2f}<extra></extra>"
            )
        )

    fig.update_layout(
        title="<b>Price Ranges (Top Performers)</b>",
        plot_bgcolor='#121212',
        paper_bgcolor='#121212',
        font=dict(color='white'),
        xaxis=dict(
            showline=True,
            linecolor='#333333',
            gridcolor='#333333'
        ),
        yaxis=dict(
            title="<b>Price</b>",
            showline=True,
            linecolor='#333333',
            gridcolor='#333333'
        ),
        hoverlabel=dict(
            bgcolor='#222222',
            font_size=14
        ),
        height=500,
        showlegend=False
    )

    #fig.show()
    return fig

def plot_sentiment_label_bar_chart(df, ticker):
    filtered_df = df[df['Ticker'] == ticker]
    sentiment_counts = filtered_df['Sentiment_Label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment_Label', 'Count']

    fig = px.bar(sentiment_counts, x='Sentiment_Label', y='Count',
                 color='Sentiment_Label',
                 title=f'Sentiment Label Count for {ticker}',
                 text='Count')
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, xaxis_title='Sentiment Label', yaxis_title='Count')
    #fig.show()
    return fig

def plot_sentiment_over_time(df, ticker):
    filtered = df[df['Ticker'] == ticker]
    fig = px.line(filtered, x='Date', y='Sentiment', color='Sentiment_Label',
                  title=f'Sentiment Over Time for {ticker}')
    #fig.show()
    return fig

def plot_sentiment_by_day(df, ticker):
    filtered = df[df['Ticker'] == ticker]
    fig = px.box(filtered, x='Day_of_Week', y='Sentiment', color='Sentiment_Label',
                 title=f'Sentiment Distribution by Day of Week for {ticker}')
    #fig.show()
    return fig

def plot_top_positive_keywords(df, ticker, top_n=10):
    filtered = df[(df['Ticker'] == ticker) & (df['Sentiment_Label'] == 'Positive')]
    
    # Adjust the column range if your keyword columns are different
    keyword_cols = df.columns[4:-5]
    
    keyword_means = filtered[keyword_cols].mean().sort_values(ascending=False).head(top_n)
    
    fig = px.bar(x=keyword_means.index, y=keyword_means.values,
                 title=f'Top {top_n} Keywords in Positive Articles for {ticker}',
                 labels={'x': 'Keyword', 'y': 'Average Score'})
    #fig.show()
    return fig

def plot_monthly_avg_sentiment(df, ticker):
    filtered = df[df['Ticker'] == ticker].copy()
    filtered['Month_Year'] = filtered['Date'].astype(str).str[:7]
    monthly_avg = filtered.groupby('Month_Year')['Sentiment'].mean().reset_index()
    
    fig = px.line(monthly_avg, x='Month_Year', y='Sentiment',
                  title=f'Monthly Average Sentiment for {ticker}')
    #fig.show()
    return fig

# Example usage (uncomment and update with your data and desired ticker)
# df = pd.read_csv("your_data.csv")
# plot_sentiment_over_time(df, 'AAPL')
# plot_sentiment_by_day(df, 'AAPL')
# plot_top_positive_keywords(df, 'AAPL')
# plot_sentiment_label_count(df, 'AAPL')
# plot_monthly_avg_sentiment(df, 'AAPL')
