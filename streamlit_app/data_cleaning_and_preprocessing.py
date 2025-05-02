from project_utils import *

def clean_financial_data(yahoo_df):
    """
    Clean and preprocess the financial dataset.

    Parameters:
    - yahoo_df (pd.DataFrame): Raw financial data with multi-level columns (Date, Ticker, Price attributes).

    Returns:
    - pd.DataFrame: Cleaned financial data with relevant columns and no missing 'Close' values.
    """
    # 1. Ensure index is named
    yahoo_df.index.name = 'Date'

    # 2. Stack both levels: ['Ticker', 'Price']
    df_stacked = yahoo_df.stack(level=[0, 1], future_stack=True).reset_index()

    # 3. Rename columns
    df_stacked.columns = ['Date', 'Ticker', 'Attribute', 'Value']

    # 4. Pivot: rows = (Date, Ticker), columns = Attribute
    df_tidy = df_stacked.pivot(index=['Date', 'Ticker'], columns='Attribute', values='Value').reset_index()

    # 5. Remove column index name
    df_tidy.columns.name = None

    # 6. Drop rows where 'Close' is missing
    df_tidy = df_tidy.dropna(subset=['Close'])

    # 7. Drop the 'Adj Close' column if it exists
    if 'Adj Close' in df_tidy.columns:
        df_tidy = df_tidy.drop(columns=['Adj Close'])

    # Also make sure df 'Date' is datetime
    df_tidy['Date'] = pd.to_datetime(df_tidy['Date'], errors='coerce')

    return df_tidy

def feature_engineering_financial(financial_df):
    # Calculate Moving Averages (SMA and EMA)
    financial_df['SMA_50'] = financial_df['Close'].rolling(window=50).mean()
    financial_df['SMA_200'] = financial_df['Close'].rolling(window=200).mean()
    financial_df['EMA_50'] = financial_df['Close'].ewm(span=50, adjust=False).mean()
    financial_df['EMA_200'] = financial_df['Close'].ewm(span=200, adjust=False).mean()

    # Calculate Daily Percentage Change (Price Change)
    financial_df['Price_Change'] = financial_df['Close'].pct_change()

    # Calculate Volatility (Rolling Standard Deviation of Daily Percentage Change)
    financial_df['Volatility'] = financial_df['Price_Change'].rolling(window=20).std()

    # Calculate RSI (Relative Strength Index)
    rsi = RSIIndicator(financial_df['Close'], window=14)
    financial_df['RSI'] = rsi.rsi()

    # Calculate Bollinger Bands
    bollinger = BollingerBands(financial_df['Close'], window=20, window_dev=2)
    financial_df['Bollinger_Upper'] = bollinger.bollinger_hband()
    financial_df['Bollinger_Lower'] = bollinger.bollinger_lband()
    financial_df['Bollinger_Mid'] = bollinger.bollinger_mavg()

    # Forward fill missing values
    df = financial_df.ffill()

    # Apply backward fill for any remaining NaNs
    df = df.bfill()

    return df

def clean_news_data(news_df):
    """
    Cleans the news dataset by handling missing values, formatting the date,
    and cleaning the text in the 'Title' column.

    Parameters:
    - news_df (pd.DataFrame): The news data containing 'Ticker', 'Date', and 'Title'.

    Returns:
    - pd.DataFrame: Cleaned news dataset.
    """
    # 1. Convert 'Date' column to datetime format
    news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce')

    # 2. Drop rows where 'Date', 'Ticker', or 'Title' are missing
    news_df = news_df.dropna(subset=['Date', 'Ticker', 'Title'])

    # 3. Remove any non-ASCII characters from the 'Title'
    news_df['Title'] = news_df['Title'].str.replace(r'[^\x00-\x7F]+', '', regex=True)

    # 4. Remove quotes or special characters from the 'Title'
    news_df['Title'] = news_df['Title'].str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation

    # 5. Standardize text: convert all titles to lowercase for consistency
    news_df['Title'] = news_df['Title'].str.lower()

    # 6. Remove leading/trailing spaces
    news_df['Title'] = news_df['Title'].str.strip()

    # 7. Ensure the 'Ticker' column is a string (to avoid errors when merging with other data)
    news_df['Ticker'] = news_df['Ticker'].astype(str)

    # 8. Reset the index after cleaning
    news_df = news_df.reset_index(drop=True)

    return news_df

# Preprocess text: Tokenize, remove stopwords, and clean text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and digits
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Function to perform sentiment analysis
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def preprocess_news_data(news_df):
    """
    Performs feature engineering on the cleaned news dataset.
    Adds sentiment analysis, TF-IDF features, and date-based features.

    Parameters:
    - news_df (pd.DataFrame): The cleaned news data containing 'Ticker', 'Date', and 'Title'.

    Returns:
    - pd.DataFrame: News dataset with additional features.
    """
    # Step 1: Preprocess Titles for feature engineering (clean and tokenize text)
    news_df['Cleaned_Title'] = news_df['Title'].apply(preprocess_text)

    # Step 2: TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=100)  # Limit to top 100 features for simplicity
    X_tfidf = vectorizer.fit_transform(news_df['Cleaned_Title'])
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    news_df = pd.concat([news_df, tfidf_df], axis=1)

    # Step 3: Sentiment Analysis
    news_df['Sentiment'] = news_df['Cleaned_Title'].apply(get_sentiment)
    news_df['Sentiment_Label'] = news_df['Sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

    # Step 4: Extract Date Features (Day of Week, Month, Year)
    news_df['Day_of_Week'] = news_df['Date'].dt.dayofweek
    news_df['Month'] = news_df['Date'].dt.month
    news_df['Year'] = news_df['Date'].dt.year

    # Step 5: Topic Modeling (Optional)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X_tfidf)
    top_words = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words.append([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]])

    print("Top Topics:", top_words)  # You can print the top words for topics here

    # Reset the index after feature engineering
    news_df = news_df.reset_index(drop=True)

    return news_df

def merge_datasets(financial_data, news_data):
    """
    Merges financial data with news titles based on the 'Date' and 'Ticker' columns.

    Parameters:
    financial_data (DataFrame): The DataFrame containing the financial data (Date, Ticker, Close, etc.).
    news_data (DataFrame): The DataFrame containing the news data (Date, Ticker, Title).

    Returns:
    DataFrame: A new DataFrame with the merged data.
    """
    # Merge the two datasets on 'Date' and 'Ticker'
    merged_data = pd.merge(financial_data, news_data, how='left', on=['Date', 'Ticker'])

    return merged_data