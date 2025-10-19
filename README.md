# 📈 Stock Market Prediction using Machine Learning

## 📖 Overview

This project is a **web-based application** that predicts stock market performance using machine learning models such as **Linear Regression**, **Random Forest**, and **XGBoost**.
It integrates **historical** and **real-time stock data** from S&P 500 companies, visualizes key metrics, and performs **sentiment analysis on financial news** to enhance prediction accuracy.

Built with **Streamlit**, the app provides an interactive dashboard where users can:
✅ Explore stock market trends
✅ Analyze individual stock performance
✅ View predictions supported by sentiment analysis

---

## 🎯 Key Features

* 📊 Real-time financial data with **Yahoo Finance (yFinance)**
* 🤖 Predictive modelling using **ML algorithms (LR, RF, XGBoost)**
* 💬 Sentiment analysis on financial news articles (TextBlob / NLTK)
* 🌐 Fully interactive **Streamlit dashboard**
* 🧱 Clean, modular, and open-source architecture

---

## 🧰 Technologies Used

| Category            | Tools/Libraries                                         |
| ------------------- | ------------------------------------------------------- |
| **Frontend**        | Streamlit                                               |
| **Data Source**     | Yahoo Finance (yFinance)                                |
| **ML Models**       | scikit-learn, Random Forest, Linear Regression, XGBoost |
| **Data Processing** | Pandas, NumPy                                           |
| **Visualization**   | Matplotlib, Seaborn                                     |
| **NLP / Sentiment** | NLTK, TextBlob                                          |

---

## 🚀 Getting Started

### ✅ Prerequisites

* Python **3.8+** installed
* `pip` package manager
* (Optional but recommended) **pyenv** for Python version control

### 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/jaheemedwards/COMP3610_Project.git
cd COMP3610_Project

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app/app.py
```

---

## ⚙️ Running the App Locally (Easy Method)

A helper script called `run_app.zsh` is included to automate setup.
This script will:

1. **Create a virtual environment (`venv`) with Python 3.11**
2. **Activate the environment**
3. **Upgrade pip**
4. **Install required packages from `requirements.txt`**
5. **Start the app at `streamlit_app/app.py`**

### ▶ How to Use

```bash
chmod +x run_app.zsh
./run_app.zsh
```

This will automatically create the environment (if missing), install dependencies, and launch the project in your browser.

> **Note:** The script assumes Python **3.11** is installed on your system.
> If you're using `pyenv`, it will attempt to install Python 3.11 automatically if missing.

---

## 🎥 Demo

Watch the app in action:
👉 **YouTube Demo Video:** [https://www.youtube.com/watch?v=60bXAOetOQs&ab_channel=Eddie](https://www.youtube.com/watch?v=60bXAOetOQs&ab_channel=Eddie)

---
