# PyRES
### A Python library for Reverberation Enhancement System development and simulation.

---

PyRES is designed to interface with the related open-access dataset:

De Bortoli, G. M. (2025). DataRES: Dataset for research on Reverberation Enhancement Systems (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15165524

---

## Installation

Follow the instructions below to install **PyRES** and to set up a working environment.

1. Clone the repository
```shell
git clone https://github.com/GianMarcoDeBortoli/PyRES.git
cd pyres
```

2. Install Python:

Make sure you have Python version >=3.10 installed on your system.

3. Set up the environment
- Automatic setup (recommended):
  - On **MacOS/Linux** (bash):
    ```shell
    bash bootstrap.sh
    ```
  - on **Windows** (cmd):
    ```shell
    call bootstrap.bat
    ```
- Manual Setup:
  - If you are using **Conda**:
    ```shell
    conda env create -f environment.yml --name pyres-env
    ```
  - If you are using **Pip**:
    - On **MacOS** (bash):
      ```shell
      brew install libsndfile
      python -m venv pyres-env
      source pyres-env/bin/activate
      echo "export DYLD_LIBRARY_PATH=$(brew --prefix libsndfile)/lib:$DYLD_LIBRARY_PATH" >> pyres-env/bin/activate
      python -m pip install --upgrade pip
      pip install -r requirements.txt
      ```
    - On **Linux** (bash):
      ```shell
      sudo apt-get update && sudo apt-get install -y libsndfile1
      python -m venv pyres-env
      source pyres-env/bin/activate
      python -m pip install --upgrade pip
      pip install -r requirements.txt
      ```
    - On **Windows** (cmd):
      ```shell
      python -m venv pyres-env
      .\pyres-env\Scripts\activate.bat
      python -m pip install --upgrade pip
      pip install -r requirements.txt
      ```

---

## Tutorial

Please refer to the example files for a tutorial on how to use this library.
