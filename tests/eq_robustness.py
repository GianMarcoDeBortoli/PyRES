# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# PyTorch
import torch
# PyRES
from PyRES.res import RES
from PyRES.physical_room import PhRoom_dataset
from PyRES.virtual_room import random_FIRs
from PyRES.plots import plot_evs, plot_spectrograms

###########################################################################################

torch.manual_seed(130297)

if __name__ == '__main__':

    # -------------------- Initialize RES ---------------------
    # Time-frequency
    samplerate = 48000                 # Sampling frequency in Hz
    nfft = samplerate*3                # FFT size
    alias_decay_db = 0                 # Anti-time-aliasing decay in dB

    # Physical room
    room_dataset = './dataRES'         # Path to the dataset
    room = 'Otala'                     # Path to the room impulse responses
    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=room_dataset,
        room_name=room
    )
    _, n_mcs, n_lds, _ = physical_room.get_ems_rcs_number()

    # Virtual room
    fir_order = 2**8                   # FIR filter order
    virtual_room = random_FIRs(
        n_M=n_mcs,
        n_L=n_lds,
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        FIR_order=fir_order,
        requires_grad=True
    )


    # Reverberation Enhancement System
    res_1 = RES(
        physical_room = physical_room,
        virtual_room = virtual_room
    )
    # Load previous equalization FIRs
    state = torch.load('./model_states/2025-04-26_22.49.13.pt')
    res_1.set_v_ML_state(state)
    
    # --------------------- Performance ----------------------
    evs_opt = res_1.open_loop_eigenvalues().squeeze(0)
    ir_opt = res_1.system_simulation().squeeze(0)

    # ------------------- New room state ---------------------
    # Physical room
    room_dataset = './dataRES'         # Path to the dataset
    room = 'Otala_C1_L1'                  # Path to the room impulse responses
    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=room_dataset,
        room_name=room
    )
    _, n_mcs, n_lds, _ = physical_room.get_ems_rcs_number()

    # Virtual room
    fir_order = 2**8                   # FIR filter order
    virtual_room = random_FIRs(
        n_M=n_mcs,
        n_L=n_lds,
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        FIR_order=fir_order,
        requires_grad=True
    )


    # Reverberation Enhancement System
    res_2 = RES(
        physical_room = physical_room,
        virtual_room = virtual_room
    )
    # Load previous equalization FIRs
    state = torch.load('./model_states/2025-04-26_22.49.13.pt')
    res_2.set_v_ML_state(state)

    # --------------------- Performance ----------------------
    evs_non_opt = res_2.open_loop_eigenvalues().squeeze(0)
    ir_non_opt = res_2.system_simulation().squeeze(0)
    
    # ------------------------ Plots -------------------------
    plot_evs(evs_opt, evs_non_opt, samplerate, nfft, 20, 8000)
    plot_spectrograms(ir_non_opt[:,0], ir_opt[:,0], samplerate, nfft=2**8, noverlap=2**7)

    exit()
