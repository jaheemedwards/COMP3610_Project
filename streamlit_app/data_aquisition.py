from project_utils import *

def get_sp500_tickers():
    """
    Fetch the list of S&P 500 tickers from Wikipedia.
    
    Returns:
        list: A list of cleaned S&P 500 ticker symbols.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    tables = pd.read_html(response.text)

    # The first table contains the S&P 500 tickers
    sp500_table = tables[0]
    tickers = sp500_table['Symbol'].tolist()

    # Replace special characters (e.g., BRK.B → BRK-B) for compatibility
    tickers = [ticker.replace(".", "-") for ticker in tickers]

    print(f"Fetched {len(tickers)} S&P 500 tickers.")
    return tickers


def fetch_yahoo_data(tickers, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance for multiple tickers.
    
    Parameters:
        tickers (list): List of stock symbols.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        
    Returns:
        DataFrame: Historical financial data grouped by ticker.
    """
    print("Fetching Yahoo Finance data...")

    # Download data using yfinance
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

    print("Yahoo Finance data fetched successfully.")
    return data


def generate_finviz_urls(tickers):
    """
    Generate Finviz URLs for a list of tickers.

    Parameters:
        tickers (list): List of stock ticker symbols.
        
    Returns:
        list: URLs for each ticker on Finviz.
    """
    base_url = "https://finviz.com/quote.ashx?t={}&p=d"
    urls = [base_url.format(ticker) for ticker in tickers]
    return urls


def fetch_and_parse_finviz_news(stock_tickers, finviz_urls, delay=1):
    """
    Fetch and parse news headlines from Finviz for a list of stock tickers.
    
    Parameters:
        stock_tickers (list): Stock ticker strings.
        finviz_urls (list): Corresponding Finviz URLs.
        delay (int): Delay between requests (in seconds).
        
    Returns:
        DataFrame: Parsed news headlines with ticker and date.
    """
    news_tables = {}

    for ticker, url in zip(stock_tickers, finviz_urls):
        print(f"Fetching news for {ticker}...")

        try:
            # Set user-agent to avoid blocks
            req = Request(
                url=url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                                  '(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
                }
            )
            response = urlopen(req)
            html = BeautifulSoup(response, 'html.parser')

            # Locate news table
            news_table = html.find(id='news-table')
            if news_table:
                news_tables[ticker] = news_table
            else:
                print(f"No news table found for {ticker}.")

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

        # Delay to avoid overwhelming the server
        time.sleep(delay)

    # Parse the fetched HTML tables
    parsed_data = []
    most_recent_date = None

    for ticker, news_table in news_tables.items():
        for row in news_table.find_all('tr'):
            link = row.find('a')
            if not link:
                continue

            title = link.get_text().strip()
            date_data = row.find_all('td')

            if len(date_data) > 1:
                # Separate date and time if available
                date_time = date_data[0].text.strip().split(' ')
                if len(date_time) == 1:
                    # Only time is available; use last known date
                    time_of_news = date_time[0]
                    date_of_news = most_recent_date
                else:
                    date_of_news = date_time[0]
                    time_of_news = date_time[1]

                # Convert 'Today' to actual date
                if date_of_news == 'Today':
                    most_recent_date = datetime.today().strftime('%Y-%m-%d')
                    date_of_news = most_recent_date
                else:
                    try:
                        date_of_news = datetime.strptime(date_of_news, '%b-%d-%y').strftime('%Y-%m-%d')
                        most_recent_date = date_of_news
                    except ValueError:
                        # Skip if date parsing fails
                        pass
            else:
                # No date info found
                date_of_news = most_recent_date
                time_of_news = None

            parsed_data.append([ticker, date_of_news, title])

    # Remove entries with missing titles
    parsed_data = [row for row in parsed_data if row[2] is not None]

    # Create a DataFrame with the results
    news_df = pd.DataFrame(parsed_data, columns=['Ticker', 'Date', 'Title'])

    return news_df
