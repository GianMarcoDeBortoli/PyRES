# ==================================================================
# ============================ IMPORTS =============================
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import numpy as np
import torch
import matplotlib.pyplot as plt

# PyRES
from PyRES.physical_room import PhRoom_dataset


###########################################################################################
# Test script for visualizing the effect of the new PhRoom methods:
#   - only_direct_path
#   - only_early_reflections
#   - only_late_reverb
#
# For each method, we:
#   1. Load the Otala room with PhRoom_dataset
#   2. Take one LM impulse response (loudspeaker -> microphone)
#   3. Plot it before and after applying the method
###########################################################################################


def plot_before_after(ax_before, ax_after, t, ir_before, ir_after, title):
    """Helper to plot a single IR before and after processing."""
    ax_before.plot(t, ir_before)
    ax_before.set_title(f"{title} - before")
    ax_before.set_xlabel("Time [s]")
    ax_before.set_ylabel("Amplitude")

    ax_after.plot(t, ir_after)
    ax_after.set_title(f"{title} - after")
    ax_after.set_xlabel("Time [s]")
    ax_after.set_ylabel("Amplitude")


if __name__ == '__main__':

    # Time-frequency parameters
    samplerate = 48000              # Sampling frequency
    nfft = samplerate * 3           # FFT size
    alias_decay_db = 0              # Anti-time-aliasing decay in dB

    # Dataset directory and room name
    # NOTE: Adapt this path if your DataRES directory is located elsewhere.
    dataset_directory = './DataRES/'
    room_name = 'Otala'

    print("=" * 80)
    print("Visualizing PhRoom functions with Otala room (LM impulse responses)")
    print("=" * 80)

    # Figure with 3 rows (one per function) and 2 columns (before/after)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle("Effect of PhRoom time-windowing functions on one h_LM IR", fontsize=14)

    # Common settings: we always look at LM, first mic / first loudspeaker
    mic_idx = 1
    lds_idx = 0

    # ------------------------------------------------------------------
    # Test 1: only_direct_path
    # ------------------------------------------------------------------
    print("\nLoading PhRoom_dataset for only_direct_path...")
    ph_dp = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )

    rirs_dp_before = ph_dp.get_rirs()['LM'][:, mic_idx, lds_idx].clone().detach()
    t_dp = np.arange(rirs_dp_before.shape[0]) / samplerate

    print("Applying only_direct_path...")
    ph_dp.only_direct_path(window_duration_ms=1.0)
    rirs_dp_after = ph_dp.get_rirs()['LM'][:, mic_idx, lds_idx].clone().detach()

    plot_before_after(
        axes[0, 0],
        axes[0, 1],
        t_dp,
        rirs_dp_before.numpy(),
        rirs_dp_after.numpy(),
        "only_direct_path (LM[0,0])"
    )

    # ------------------------------------------------------------------
    # Test 2: only_early_reflections
    # ------------------------------------------------------------------
    print("\nLoading PhRoom_dataset for only_early_reflections...")
    ph_er = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )

    rirs_er_before = ph_er.get_rirs()['LM'][:, mic_idx, lds_idx].clone().detach()
    t_er = np.arange(rirs_er_before.shape[0]) / samplerate

    print("Applying only_early_reflections...")
    ph_er.only_early_reflections(window_duration_ms=25.0, suppress_direct_path_ms=1.0)
    rirs_er_after = ph_er.get_rirs()['LM'][:, mic_idx, lds_idx].clone().detach()

    plot_before_after(
        axes[1, 0],
        axes[1, 1],
        t_er,
        rirs_er_before.numpy(),
        rirs_er_after.numpy(),
        "only_early_reflections (LM[0,0])"
    )

    # ------------------------------------------------------------------
    # Test 3: only_late_reverb
    # ------------------------------------------------------------------
    print("\nLoading PhRoom_dataset for only_late_reverb...")
    ph_lr = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )

    rirs_lr_before = ph_lr.get_rirs()['LM'][:, mic_idx, lds_idx].clone().detach()
    t_lr = np.arange(rirs_lr_before.shape[0]) / samplerate

    print("Applying only_late_reverb...")
    ph_lr.only_late_reverb(transition_duration_ms=25.0)
    rirs_lr_after = ph_lr.get_rirs()['LM'][:, mic_idx, lds_idx].clone().detach()

    plot_before_after(
        axes[2, 0],
        axes[2, 1],
        t_lr,
        rirs_lr_before.numpy(),
        rirs_lr_after.numpy(),
        "only_late_reverb (LM[0,0])"
    )

    # Layout and show
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print("\nDone. Close the plot window to end the script.")


