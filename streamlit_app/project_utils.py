# Core libraries
import os
import sys
import time
import re
import ssl
import joblib
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Date and time
from datetime import datetime

# Web scraping
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

# NLP & sentiment analysis
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Machine learning models and tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Financial data & technical indicators
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Streamlit for web interface
import streamlit as st

# NLTK resource downloads
nltk.download('stopwords')
nltk.download('punkt')
