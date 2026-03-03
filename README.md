# PyRES
### Python library for Reverberation Enhancement System development and simulation.

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

Please refer to the .examples/ folder for a series of tutorial files.

1. `VrRoom` class

   The `VrRoom` class represents the DSP architecture in a reverberation enhancement system.
   
   **PyRES** contains several architecture implementations.
   
   This tutorial shows the instantiation of a DSP and the most important attributes.

3. `PhRoom` class

   The `PhRoom` class represents the physical space in which the reverberation enhancement system is located.
   
   The physical space hosts the stage sources, the audience receivers, and the system transducers.
   
   **PyRES**, `PhRoom` has two subclasses:
   - `PhRoom_wgn`, which simulates the room impulse responses through exponentially-decaying white-Gaussian-noise sequences
   - `PhRoom_dataset`, which loads the room impulse responses from the accompanying dataset[1]
   
   These tutorials show the use and the most important attributes of the two.
  

4. `RES` class

   In **PyRES** the reverberation enhancement system is implemented by the RES class.
   
   The `RES` class receives an instance of `VrRoom` and `PhRoom` each, and controls the interaction between them.
   
   This tutorial shows the use and the most important attributes and methods of the `RES` class.
  
6. Training of a DSP

   **PyRES** relies on **FLAMO**[2] as backend for the signal processing.
   
   Thus, the DSP architectures are defined as chains of differential processing modules which can be trained through a machine-learning-like pipeline.
   
   These tutorials show the training of DSP architecture repeating previous work of the author[3][4].

---

## References

[1] De Bortoli, G., Prawda, K., Coleman, P., and Schlecht, S. J. "DataRES: Dataset for research on Reverberation Enhancement Systems" (2.0.0) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.15737243](https://doi.org/10.5281/zenodo.15737243)

[2] Dal Santo G., De Bortoli, G., Prawda, K., Schlecht, S. J., and Välimäki, V. "FLAMO: An Open-Source Library for Frequency-Domain Differentiable Audio Processing" Proceedings of the International Conference on Acoustics, Speech, and Signal Processing, pp.1--5, 2025

[3] De Bortoli, G., Dal Santo, G., Prawda, K., Lokki, T., Välimäki, V., and Schlecht, S. J. "Differentiable Active Acoustics: Optimizing Stability via Gradient Descent" Proceedings of the International Conference on Digital Audio Effects, pp. 254-261, 2024.

[4] De Bortoli, G., Prawda, K., and Schlecht, S. J. "Active Acoustics with a Phase Cancelling Modal Reverberator" Journal of the Audio Engineering Society, Vol. 72, No. 10, pp. 705-715, 2024.
