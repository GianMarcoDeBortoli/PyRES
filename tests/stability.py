# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
# PyTorch
import torch
# flamo
from flamo.functional import mag2db
# PyRES
from PyRES.res import RES
from PyRES.physical_room import PhRoom_dataset
from PyRES.virtual_room import *
from PyRES.plots import plot_evs, plot_spectrograms

###########################################################################################

torch.manual_seed(130297)

if __name__ == '__main__':

    # -------------------- Initialize RES ---------------------
    # Time-frequency
    samplerate = 48000                 # Sampling frequency in Hz
    nfft = samplerate                # FFT size
    alias_decay_db = 0                 # Anti-time-aliasing decay in dB

    # Physical room
    room_dataset = './dataRES'         # Path to the dataset
    room = 'Otala'              # Path to the room impulse responses
    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=room_dataset,
        room_name=room
    )
    _, n_mcs, n_lds, _ = physical_room.get_ems_rcs_number()

    rirs = physical_room.get_lds_to_mcs().param.detach().clone()
    rtfs = torch.fft.rfft(rirs, n=nfft, dim=0)
    h_SA = physical_room.get_stg_to_aud().param.detach().clone()
    H_SA = torch.fft.rfft(h_SA, n=nfft, dim=0)

    # Virtual room
    virtual_room = unitary_parallel_connections(
        n_M=n_mcs,
        n_L=n_lds,
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
    )

    # Reverberation Enhancement System
    res = RES(
        physical_room = physical_room,
        virtual_room = virtual_room
    )
    
    # --------------------- Performance ----------------------
    evs = res.open_loop_eigenvalues().squeeze(0)
    ir = res.system_simulation().squeeze(0)

    # ------------------ Curves comparison -------------------
    full_TF = torch.fft.rfft(ir, n=nfft, dim=0)
    mag_full_TF = torch.abs(full_TF)
    max_mag_full_TF = torch.max(mag_full_TF, dim=1).values
    mean_mag_full_TF = torch.mean(mag_full_TF, dim=1)
    mag_rtfs = torch.abs(rtfs)
    max_mag_rtfs = torch.amax(mag_rtfs, dim=(1,2))
    mean_mag_rtfs = torch.mean(mag_rtfs, dim=(1,2))
    real_evs = torch.real(evs)
    max_real_evs = torch.max(real_evs, dim=0).values
    mean_real_evs = torch.mean(real_evs, dim=0)
    mag_evs = torch.abs(evs)
    max_mag_evs = torch.max(mag_evs, dim=0).values
    mean_mag_evs = torch.mean(mag_evs, dim=0)
    
    f_axis = torch.linspace(0, samplerate/2, nfft//2+1)

    plt.figure()
    plt.plot(f_axis, mag2db(mag_full_TF[:,0]), color="blue", label='Full system TFs - max magnitude values')
    # plt.plot(f_axis, mag2db(mean_mag_full_TF), color="blue", linestyle='dashed', label='Full system TF - mean values')
    plt.plot(f_axis, mag2db(H_SA[:,0]), color="blue", linestyle='dashed', label='STG to AUD TF - max magnitude values')
    plt.plot(f_axis, mag2db(max_mag_rtfs), color="purple", label='Feedback loop TFs - max magnitude values')
    # plt.plot(f_axis, mag2db(mean_mag_rtfs), color="pink", linestyle='dashed', label='Feedback loop TF - mean values')
    plt.plot(f_axis, mag2db(real_evs), color="red", label='Eigenvalues - max real-part values')
    # plt.plot(f_axis, mag2db(mean_real_evs), color="red", linestyle='dashed', label='Eigenvalue real part - mean values')
    plt.plot(f_axis, mag2db(mag_evs), color="green", label='Eigenvalues - max magnitude values')
    # plt.plot(f_axis, mag2db(mean_mag_evs), color="green", linestyle='dashed', label='Eigenvalue magnitude - mean values')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    # plt.xscale('log')
    plt.legend()
    plt.show(block=True)

    exit()
