# Import all necessary utilities from the project utilities module
from project_utils import *

def clean_financial_data(yahoo_df):
    """
    Clean and preprocess the financial dataset.

    Parameters:
    - yahoo_df (pd.DataFrame): Raw financial data with multi-level columns (Date, Ticker, Price attributes).

    Returns:
    - pd.DataFrame: Cleaned financial data with relevant columns and no missing 'Close' values.
    """
    # Ensure the index is named 'Date'
    yahoo_df.index.name = 'Date'

    # Stack both levels ('Ticker' and price attributes), flattening the dataframe
    df_stacked = yahoo_df.stack(level=[0, 1], future_stack=True).reset_index()

    # Rename columns to match the new flattened structure
    df_stacked.columns = ['Date', 'Ticker', 'Attribute', 'Value']

    # Pivot the data so that each row corresponds to a (Date, Ticker) and each column is an attribute
    df_tidy = df_stacked.pivot(index=['Date', 'Ticker'], columns='Attribute', values='Value').reset_index()

    # Remove the name of the columns index
    df_tidy.columns.name = None

    # Drop rows where 'Close' value is missing (essential for financial analysis)
    df_tidy = df_tidy.dropna(subset=['Close'])

    # Drop the 'Adj Close' column if it exists (often redundant)
    if 'Adj Close' in df_tidy.columns:
        df_tidy = df_tidy.drop(columns=['Adj Close'])

    # Convert 'Date' column to datetime format
    df_tidy['Date'] = pd.to_datetime(df_tidy['Date'], errors='coerce')

    return df_tidy

def feature_engineering_financial(financial_df, include_advanced_features=True):
    """
    Perform feature engineering on financial data.

    Parameters:
    - financial_df (pd.DataFrame): Cleaned financial dataset.
    - include_advanced_features (bool): Whether to include SMA, EMA, RSI, and Bollinger Bands.

    Returns:
    - pd.DataFrame: Financial dataset with new features added.
    """
    # Daily percentage change in closing price
    financial_df['Price_Change'] = financial_df['Close'].pct_change()

    # Rolling volatility based on the standard deviation of daily returns
    financial_df['Volatility'] = financial_df['Price_Change'].rolling(window=20).std()

    if include_advanced_features:
        # Simple and Exponential Moving Averages
        financial_df['SMA_50'] = financial_df['Close'].rolling(window=50).mean()
        financial_df['SMA_200'] = financial_df['Close'].rolling(window=200).mean()
        financial_df['EMA_50'] = financial_df['Close'].ewm(span=50, adjust=False).mean()
        financial_df['EMA_200'] = financial_df['Close'].ewm(span=200, adjust=False).mean()

        # Relative Strength Index (RSI)
        rsi = RSIIndicator(financial_df['Close'], window=14)
        financial_df['RSI'] = rsi.rsi()

        # Bollinger Bands
        bollinger = BollingerBands(financial_df['Close'], window=20, window_dev=2)
        financial_df['Bollinger_Upper'] = bollinger.bollinger_hband()
        financial_df['Bollinger_Lower'] = bollinger.bollinger_lband()
        financial_df['Bollinger_Mid'] = bollinger.bollinger_mavg()

    # Forward fill and then backward fill any missing values
    financial_df = financial_df.ffill().bfill()

    return financial_df

def clean_news_data(news_df):
    """
    Clean the news dataset by formatting dates, removing invalid entries, and cleaning text.

    Parameters:
    - news_df (pd.DataFrame): Raw news data containing 'Ticker', 'Date', and 'Title'.

    Returns:
    - pd.DataFrame: Cleaned news dataset.
    """
    # Convert 'Date' to datetime format
    news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce')

    # Drop rows with missing values in critical columns
    news_df = news_df.dropna(subset=['Date', 'Ticker', 'Title'])

    # Remove non-ASCII characters
    news_df['Title'] = news_df['Title'].str.replace(r'[^\x00-\x7F]+', '', regex=True)

    # Remove punctuation and special characters
    news_df['Title'] = news_df['Title'].str.replace(r'[^\w\s]', '', regex=True)

    # Convert to lowercase and strip whitespaces
    news_df['Title'] = news_df['Title'].str.lower().str.strip()

    # Ensure 'Ticker' column is string
    news_df['Ticker'] = news_df['Ticker'].astype(str)

    # Reset index after cleaning
    news_df = news_df.reset_index(drop=True)

    return news_df

def preprocess_text(text):
    """
    Clean and tokenize a string of text for NLP.

    Parameters:
    - text (str): Raw text to process.

    Returns:
    - str: Cleaned and tokenized text.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def get_sentiment(text):
    """
    Compute sentiment polarity of text using TextBlob.

    Parameters:
    - text (str): Input text.

    Returns:
    - float: Sentiment polarity score (-1 to 1).
    """
    return TextBlob(text).sentiment.polarity

def preprocess_news_data(news_df):
    """
    Perform feature engineering on cleaned news data.

    Parameters:
    - news_df (pd.DataFrame): Cleaned news data with 'Date', 'Ticker', and 'Title'.

    Returns:
    - pd.DataFrame: News data enriched with sentiment, TF-IDF, and date-based features.
    """
    # Clean and tokenize text
    news_df['Cleaned_Title'] = news_df['Title'].apply(preprocess_text)

    # Generate TF-IDF features (limit to top 100 terms)
    vectorizer = TfidfVectorizer(max_features=100)
    X_tfidf = vectorizer.fit_transform(news_df['Cleaned_Title'])
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

    # Combine original data with TF-IDF features
    news_df = pd.concat([news_df, tfidf_df], axis=1)

    # Compute sentiment polarity and label
    news_df['Sentiment'] = news_df['Cleaned_Title'].apply(get_sentiment)
    news_df['Sentiment_Label'] = news_df['Sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

    # Extract time-based features
    news_df['Day_of_Week'] = news_df['Date'].dt.dayofweek
    news_df['Month'] = news_df['Date'].dt.month
    news_df['Year'] = news_df['Date'].dt.year

    # Perform basic topic modeling (optional)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X_tfidf)

    # Display top 10 words from each topic
    top_words = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words.append([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]])
    print("Top Topics:", top_words)

    news_df = news_df.reset_index(drop=True)
    return news_df

def merge_datasets(financial_data, news_data):
    """
    Merge financial and news data on 'Date' and 'Ticker'.

    Parameters:
    - financial_data (pd.DataFrame): Financial data with price and indicator features.
    - news_data (pd.DataFrame): News data with sentiment and text features.

    Returns:
    - pd.DataFrame: Combined dataset.
    """
    merged_data = pd.merge(financial_data, news_data, how='left', on=['Date', 'Ticker'])
    return merged_data
