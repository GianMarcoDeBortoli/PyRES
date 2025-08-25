# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# PyTorch
import torch
# PyRES
from PyRES.physical_room import PhRoom_dataset


###########################################################################################
# In this example we implement a physical room using the PhRoom_dataset class.
# This class requires the following parameters:
# - fs: sampling frequency [Hz]
# - nfft: FFT size
# - alias_decay_db: anti-time-aliasing decay [Hz]
# - dataset_directory: path to the dataset
# - room_name: path to the room impulse responses
# The PhRoom_dataset class implements a physical room by interfacing with dataRES.
# Reference:
#   De Bortoli, G. M. (2025).
#   DataRES: Dataset for research on Reverberation Enhancement Systems (1.0.0) [Data set]. Zenodo.
#   https://doi.org/10.5281/zenodo.15165524
###########################################################################################

torch.manual_seed(12345)

if __name__ == '__main__':

    # Time-frequency
    samplerate = 48000                 # Sampling frequency
    nfft = samplerate*3                # FFT size
    alias_decay_db = 0                 # Anti-time-aliasing decay in dB

    # Physical room
    room_dataset = './dataRES'      # Path to the dataset
    room = 'Otala'                  # Path to the room impulse responses
    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=room_dataset,
        room_name=room
    )

    physical_room.plot_setup()
    physical_room.plot_coupling()
    physical_room.plot_DRR()