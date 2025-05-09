import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
#import pandas_datareader.data as web
import matplotlib.pyplot as plt
from textblob import TextBlob  # For sentiment analysis
import plotly.express as px  # For interactive charts
import plotly.graph_objects as go  # For more cutomization (candlestick plot)

#from tqdm import tqdm  # For progress bar
import ssl
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import re
import nltk
import time
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from datetime import datetime
import time

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import sys
import os
import time
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

nltk.download('stopwords')
nltk.download('punkt')