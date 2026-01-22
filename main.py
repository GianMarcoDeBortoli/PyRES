# ==================================================================
# ============================ IMPORTS =============================
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
# PyTorch
import torch
# PyRES
from PyRES.physical_room import PhRoom_dataset
from PyRES.plots import plot_DRR, plot_coupling
from flamo.functional import mag2db, db2mag
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Time-frequency parameters
    samplerate = 48000              # Sampling frequency
    nfft = samplerate * 5           # FFT size
    alias_decay_db = 0              # Anti-time-aliasing decay in dB

    # Dataset directory (adjust path as needed)
    dataset_directory = './DataRES/'
    room_name = 'GLivelab-Helsinki'

    # Load physical room dataset
    phroom = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )

    # Plot Coupling
    phroom.plot_coupling()
    # Plot Direct-to-Reverberant Ratio (DRR)
    phroom.plot_DRR()