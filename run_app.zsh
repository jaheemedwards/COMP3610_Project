#!/bin/zsh

# ---------------------------
# run_app.zsh
# ---------------------------

# Exit on error
set -e

# Python version to use
PYTHON_VERSION=3.11

# Virtual environment folder
VENV_DIR=venv

echo "Creating Python $PYTHON_VERSION virtual environment..."

# Check if pyenv is available
if command -v pyenv >/dev/null 2>&1; then
    PYTHON_BIN=$(pyenv which python$PYTHON_VERSION || echo "")
    if [[ -z "$PYTHON_BIN" ]]; then
        echo "Python $PYTHON_VERSION not found in pyenv. Installing..."
        pyenv install $PYTHON_VERSION
    fi
    PYTHON_BIN=$(pyenv which python$PYTHON_VERSION)
else
    PYTHON_BIN=$(which python3 || which python)
fi

# Create virtual environment if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
    $PYTHON_BIN -m venv $VENV_DIR
    echo "Virtual environment created at $VENV_DIR"
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Run the Streamlit app
echo "Starting Streamlit app..."
streamlit run streamlit_app/app.py
