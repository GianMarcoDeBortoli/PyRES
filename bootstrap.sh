#!/bin/bash

ENV_NAME="pyres-env"
USE_CONDA=true

# If an argument is given and is "no-conda", force venv
if [ "$1" == "no-conda" ]; then
    USE_CONDA=false
fi

# Detect OS
OS_TYPE="$(uname)"

if $USE_CONDA && command -v conda &> /dev/null
then
    echo "Conda detected."
    echo "Creating conda environment..."
    conda env create -f environment.yml --name $ENV_NAME
    exit 0
fi

# Otherwise, fallback to python venv
echo "Proceeding with python -m venv..."

python3 -m venv $ENV_NAME
source $ENV_NAME/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# OS-specific libsndfile installation
if [ "$OS_TYPE" == "Darwin" ]; then
    ...
    echo "macOS detected."
    if ! brew list libsndfile &> /dev/null; then
        echo "Installing libsndfile with brew..."
        brew install libsndfile
    else
        echo "libsndfile already installed via brew."
    fi
    # Add DYLD_LIBRARY_PATH setup into the activate script
    echo "Configuring environment to find libsndfile..."
    echo "export DYLD_LIBRARY_PATH=\$(brew --prefix libsndfile)/lib:\$DYLD_LIBRARY_PATH" >> $ENV_NAME/bin/activate
elif [ "$OS_TYPE" == "Linux" ]; then
    echo "Linux detected."
    echo "Installing libsndfile with apt..."
    sudo apt-get update
    sudo apt-get install -y libsndfile1
else
    echo "Unsupported OS: $OS_TYPE. Please install libsndfile manually."
fi

echo "Done. To activate the environment, run: source $ENV_NAME/bin/activate"