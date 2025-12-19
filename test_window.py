# ==================================================================
# ============================ IMPORTS =============================
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import numpy as np
import torch
import matplotlib.pyplot as plt

###########################################################################################
# Visualize the Hann windows used in the PhRoom windowing functions:
# - only_direct_path: fade-out window (1 -> 0)
# - only_early_reflections: fade-in (0 -> 1) and fade-out (1 -> 0)
# - only_late_reverb: fade-in window (0 -> 1)
###########################################################################################


if __name__ == '__main__':

    # Parameters matching typical usage
    samplerate = 48000              # Sampling frequency
    window_duration_ms = 5.0        # For only_direct_path
    early_window_ms = 50.0          # For only_early_reflections
    transition_ms = 50.0            # For only_late_reverb

    # Convert to samples
    window_samples = int(window_duration_ms * samplerate / 1000.0)
    early_window_samples = int(early_window_ms * samplerate / 1000.0)
    transition_samples = int(transition_ms * samplerate / 1000.0)

    # ==================================================================
    # 1. only_direct_path: fade-out window (1 -> 0)
    # ==================================================================
    # Half Hann window going from 1 -> 0
    hann_full_dp = torch.hann_window(2 * window_samples, periodic=False)
    fade_out_dp = hann_full_dp[window_samples:]  # second half: 1 down to 0
    
    # Create full window (1s before, then fade-out)
    # For visualization, show a window starting at some point
    vis_length = window_samples * 3
    window_dp = torch.ones(vis_length)
    window_dp[-window_samples:] = fade_out_dp

    # ==================================================================
    # 2. only_early_reflections: fade-in (0 -> 1) and fade-out (1 -> 0)
    # ==================================================================
    # Fade-in and fade-out lengths
    fade_in_len = min(int(0.05 * early_window_samples), int(3.0 * samplerate / 1000.0))
    fade_out_len = min(int(0.1 * early_window_samples), int(5.0 * samplerate / 1000.0))
    fade_in_len = max(fade_in_len, 1)
    fade_out_len = max(fade_out_len, 1)

    # Fade-in: 0 -> 1
    hann_in = torch.hann_window(2 * fade_in_len, periodic=False)
    fade_in_er = hann_in[:fade_in_len]  # first half: 0 -> 1

    # Fade-out: 1 -> 0
    hann_out = torch.hann_window(2 * fade_out_len, periodic=False)
    fade_out_er = hann_out[fade_out_len:]  # second half: 1 -> 0

    # Create full window
    window_er = torch.zeros(early_window_samples)
    window_er[:fade_in_len] = fade_in_er
    window_er[fade_in_len:-fade_out_len] = 1.0  # flat region
    window_er[-fade_out_len:] = fade_out_er

    # ==================================================================
    # 3. only_late_reverb: fade-in window (0 -> 1)
    # ==================================================================
    # Fade-in: 0 -> 1
    hann_full_lr = torch.hann_window(2 * transition_samples, periodic=False)
    fade_in_lr = hann_full_lr[:transition_samples]  # first half: 0 -> 1
    
    # Create full window (zeros before, then fade-in, then 1s)
    vis_length_lr = transition_samples * 3
    window_lr = torch.zeros(vis_length_lr)
    window_lr[transition_samples:transition_samples*2] = fade_in_lr
    window_lr[transition_samples*2:] = 1.0

    # ==================================================================
    # Time domain plots
    # ==================================================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Hann Windows Used in PhRoom Functions', fontsize=16, fontweight='bold')

    # Time axes
    t_dp = np.arange(len(window_dp)) / samplerate * 1000  # ms
    t_er = np.arange(len(window_er)) / samplerate * 1000  # ms
    t_lr = np.arange(len(window_lr)) / samplerate * 1000  # ms

    # Plot 1: only_direct_path - time domain
    axes[0, 0].plot(t_dp, window_dp.numpy(), 'b-', linewidth=2)
    axes[0, 0].set_title('only_direct_path: Fade-out Window (1 → 0)')
    axes[0, 0].set_xlabel('Time [ms]')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.1, 1.1)
    axes[0, 0].axvline(t_dp[-window_samples], color='r', linestyle='--', alpha=0.5, label='Fade start')
    axes[0, 0].legend()

    # Plot 2: only_direct_path - frequency domain
    window_dp_padded = torch.zeros(2**14)  # Pad for better frequency resolution
    window_dp_padded[:len(window_dp)] = window_dp
    fft_dp = torch.fft.rfft(window_dp_padded)
    freqs_dp = np.fft.rfftfreq(len(window_dp_padded), 1/samplerate)
    magnitude_dp = torch.abs(fft_dp)
    
    axes[0, 1].semilogx(freqs_dp[1:], 20 * np.log10(magnitude_dp[1:].numpy() + 1e-10), 'b-', linewidth=2)
    axes[0, 1].set_title('only_direct_path: Frequency Response')
    axes[0, 1].set_xlabel('Frequency [Hz]')
    axes[0, 1].set_ylabel('Magnitude [dB]')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(20, samplerate / 2)

    # Plot 3: only_early_reflections - time domain
    axes[1, 0].plot(t_er, window_er.numpy(), 'g-', linewidth=2)
    axes[1, 0].set_title('only_early_reflections: Fade-in + Flat + Fade-out')
    axes[1, 0].set_xlabel('Time [ms]')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-0.1, 1.1)
    axes[1, 0].axvline(t_er[fade_in_len], color='r', linestyle='--', alpha=0.5, label='Fade-in end')
    axes[1, 0].axvline(t_er[-fade_out_len], color='orange', linestyle='--', alpha=0.5, label='Fade-out start')
    axes[1, 0].legend()

    # Plot 4: only_early_reflections - frequency domain
    window_er_padded = torch.zeros(2**14)
    window_er_padded[:len(window_er)] = window_er
    fft_er = torch.fft.rfft(window_er_padded)
    freqs_er = np.fft.rfftfreq(len(window_er_padded), 1/samplerate)
    magnitude_er = torch.abs(fft_er)
    
    axes[1, 1].semilogx(freqs_er[1:], 20 * np.log10(magnitude_er[1:].numpy() + 1e-10), 'g-', linewidth=2)
    axes[1, 1].set_title('only_early_reflections: Frequency Response')
    axes[1, 1].set_xlabel('Frequency [Hz]')
    axes[1, 1].set_ylabel('Magnitude [dB]')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(20, samplerate / 2)

    # Plot 5: only_late_reverb - time domain
    axes[2, 0].plot(t_lr, window_lr.numpy(), 'r-', linewidth=2)
    axes[2, 0].set_title('only_late_reverb: Fade-in Window (0 → 1)')
    axes[2, 0].set_xlabel('Time [ms]')
    axes[2, 0].set_ylabel('Amplitude')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_ylim(-0.1, 1.1)
    axes[2, 0].axvline(t_lr[transition_samples], color='r', linestyle='--', alpha=0.5, label='Fade start')
    axes[2, 0].axvline(t_lr[transition_samples*2], color='orange', linestyle='--', alpha=0.5, label='Fade end')
    axes[2, 0].legend()

    # Plot 6: only_late_reverb - frequency domain
    window_lr_padded = torch.zeros(2**14)
    window_lr_padded[:len(window_lr)] = window_lr
    fft_lr = torch.fft.rfft(window_lr_padded)
    freqs_lr = np.fft.rfftfreq(len(window_lr_padded), 1/samplerate)
    magnitude_lr = torch.abs(fft_lr)
    
    axes[2, 1].semilogx(freqs_lr[1:], 20 * np.log10(magnitude_lr[1:].numpy() + 1e-10), 'r-', linewidth=2)
    axes[2, 1].set_title('only_late_reverb: Frequency Response')
    axes[2, 1].set_xlabel('Frequency [Hz]')
    axes[2, 1].set_ylabel('Magnitude [dB]')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_xlim(20, samplerate / 2)

    plt.tight_layout()
    plt.show(block=True)

    print("Window visualization complete!")
    print(f"\nWindow parameters:")
    print(f"  only_direct_path: fade-out duration = {window_duration_ms} ms ({window_samples} samples)")
    print(f"  only_early_reflections: window = {early_window_ms} ms ({early_window_samples} samples)")
    print(f"    - fade-in: {fade_in_len} samples ({fade_in_len/samplerate*1000:.2f} ms)")
    print(f"    - fade-out: {fade_out_len} samples ({fade_out_len/samplerate*1000:.2f} ms)")
    print(f"  only_late_reverb: fade-in duration = {transition_ms} ms ({transition_samples} samples)")

