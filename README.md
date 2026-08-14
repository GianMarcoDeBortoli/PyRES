# pyRES
### Python library for reverberation enhancement system development and simulation.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/GianMarcoDeBortoli/pyRES.git
cd pyRES
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install pyRES in editable mode:

```bash
pip install -e .
```

---

## Features

pyRES provides tools for the simulation, analysis, and optimization of reverberation enhancement systems (RES), including

- DSP architectures
- physical room models, using simulations and interfacing with DataRES[1]
- open-loop and closed-loop system simulation
- optimization using FLAMO

---

## Dependencies

pyRES builds upon

- FLAMO for differentiable audio processing

---

## Quickstart

- `PhRoom` class

  It represents the physical space in which the reverberation enhancement system is located.
   
  The physical space hosts the stage sources, the audience receivers, and the system transducers.
   
  `PhRoom` has multiple subclasses:
  - `PhRoom_wgn`: models room impulse responses with exponentially-decaying white-Gaussian-noise sequences
  - `PhRoom_dataset`: loads room impulse response measurements from the accompanying dataset[1]

- `VrRoom` class

  It represents the DSP architecture in a reverberation enhancement system.
  
- `RES` class

  It implements the reverberation enhancement system as the combination of a physical room and a virtual room.
   
  The `RES` class receives an instance of `VrRoom` and `PhRoom` each, and controls the interaction between them.
  
- Training of a DSP

  **pyRES** relies on **FLAMO**[2] as backend for the signal processing.
   
  Thus, the DSP architectures are defined as chains of differential processing modules which can be trained through a machine-learning-like pipeline.

Please refer to the .examples/ folder for a series of tutorial files.

---

## References

[1] De Bortoli, G., Prawda, K., Coleman, P., and Schlecht, S. J. "DataRES: Dataset for research on Reverberation Enhancement Systems" (2.0.0) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.15737243](https://doi.org/10.5281/zenodo.15737243)

[2] Dal Santo G., De Bortoli, G., Prawda, K., Schlecht, S. J., and Välimäki, V. "FLAMO: An Open-Source Library for Frequency-Domain Differentiable Audio Processing" Proceedings of the International Conference on Acoustics, Speech, and Signal Processing, pp.1--5, 2025