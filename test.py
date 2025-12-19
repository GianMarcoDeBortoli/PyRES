# ==================================================================
# ============================ IMPORTS =============================
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
# PyTorch
import torch
# PyRES
from PyRES.physical_room import PhRoom_dataset
from PyRES.virtual_room import unitary_mixing_matrix
from PyRES.res import RES
from PyRES.plots import plot_distributions
from flamo.functional import mag2db
import matplotlib.pyplot as plt


###########################################################################################
# Test script for the new PhRoom functions:
# - only_direct_path
# - only_early_reflections
# - only_late_reverb
# - suppress_feedback
#
# For each function, we create a RES with PhRoom_dataset (Otala) and unitary_mixing_matrix,
# apply the function, and compute the open-loop eigenvalues.
###########################################################################################


if __name__ == '__main__':

    # Time-frequency parameters
    samplerate = 48000              # Sampling frequency
    nfft = samplerate * 5           # FFT size
    alias_decay_db = 0              # Anti-time-aliasing decay in dB

    # Dataset directory (adjust path as needed)
    dataset_directory = './DataRES/'
    room_name = 'Otala'

    print("=" * 80)
    print("Testing PhRoom functions with Otala room")
    print("=" * 80)

    # Load the physical room
    print(f"\nLoading PhRoom_dataset: {room_name}")
    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )

    print(f"Transducer numbers:")
    print(f"  Stage emitters: {physical_room.transducer_number['stg']}")
    print(f"  System microphones: {physical_room.transducer_number['mcs']}")
    print(f"  System loudspeakers: {physical_room.transducer_number['lds']}")
    print(f"  Audience receivers: {physical_room.transducer_number['aud']}")

    # Get number of microphones and loudspeakers for virtual room
    n_M = physical_room.transducer_number['mcs']
    n_L = physical_room.transducer_number['lds']

    # Test 1: only_direct_path
    print("\n" + "=" * 80)
    print("Test 1: only_direct_path")
    print("=" * 80)
    
    # Create a fresh copy of the physical room for this test
    physical_room_dp = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )
    
    # Create virtual room with unitary mixing matrix
    virtual_room_dp = unitary_mixing_matrix(
        n_M=n_M,
        n_L=n_L,
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db
    )
    
    # Create RES
    res_dp = RES(
        physical_room=physical_room_dp,
        virtual_room=virtual_room_dp
    )
    
    # Apply only_direct_path
    print("Applying only_direct_path...")
    physical_room_dp.only_direct_path(window_duration_ms=1.0)
    
    # Compute open-loop eigenvalues
    evs_dp = res_dp.open_loop_eigenvalues()
    print(f"Open-loop eigenvalues shape: {evs_dp.shape}")
    print(f"Max eigenvalue magnitude: {mag2db(torch.max(torch.abs(evs_dp)))} dB")
    print(f"Max eigenvalue real part: {mag2db(torch.max(torch.real(evs_dp)))} dB")

    # Test 2: only_early_reflections
    print("\n" + "=" * 80)
    print("Test 2: only_early_reflections")
    print("=" * 80)
    
    # Create a fresh copy of the physical room for this test
    physical_room_er = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )
    
    # Create virtual room with unitary mixing matrix
    virtual_room_er = unitary_mixing_matrix(
        n_M=n_M,
        n_L=n_L,
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db
    )
    
    # Create RES
    res_er = RES(
        physical_room=physical_room_er,
        virtual_room=virtual_room_er
    )
    
    # Apply only_early_reflections
    print("Applying only_early_reflections...")
    physical_room_er.only_early_reflections(window_duration_ms=25.0, suppress_direct_path_ms=1.0)
    
    # Compute open-loop eigenvalues
    evs_er = res_er.open_loop_eigenvalues()
    print(f"Open-loop eigenvalues shape: {evs_er.shape}")
    print(f"Max eigenvalue magnitude: {mag2db(torch.max(torch.abs(evs_er)))} dB")
    print(f"Max eigenvalue real part: {mag2db(torch.max(torch.real(evs_er)))} dB")

    # Test 3: only_late_reverb
    print("\n" + "=" * 80)
    print("Test 3: only_late_reverb")
    print("=" * 80)
    
    # Create a fresh copy of the physical room for this test
    physical_room_lr = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )
    
    # Create virtual room with unitary mixing matrix
    virtual_room_lr = unitary_mixing_matrix(
        n_M=n_M,
        n_L=n_L,
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db
    )
    
    # Create RES
    res_lr = RES(
        physical_room=physical_room_lr,
        virtual_room=virtual_room_lr
    )
    
    # Apply only_late_reverb
    print("Applying only_late_reverb...")
    physical_room_lr.only_late_reverb(transition_duration_ms=25.0)
    
    # Compute open-loop eigenvalues
    evs_lr = res_lr.open_loop_eigenvalues()
    print(f"Open-loop eigenvalues shape: {evs_lr.shape}")
    print(f"Max eigenvalue magnitude: {mag2db(torch.max(torch.abs(evs_lr)))} dB")
    print(f"Max eigenvalue real part: {mag2db(torch.max(torch.real(evs_lr)))} dB")

    # Test 4: suppress_feedback
    print("\n" + "=" * 80)
    print("Test 4: suppress_feedback")
    print("=" * 80)
    
    # Create a fresh copy of the physical room for this test
    physical_room_sf = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )
    
    # Create virtual room with unitary mixing matrix
    virtual_room_sf = unitary_mixing_matrix(
        n_M=n_M,
        n_L=n_L,
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db
    )
    
    # Create RES
    res_sf = RES(
        physical_room=physical_room_sf,
        virtual_room=virtual_room_sf
    )
    
    # Apply suppress_feedback with gain=0.1 (half amplitude)
    print("Applying suppress_feedback with gain=0.1...")
    physical_room_sf.suppress_feedback(gain=0.1)
    
    # Compute open-loop eigenvalues
    evs_sf = res_sf.open_loop_eigenvalues()
    print(f"Open-loop eigenvalues shape: {evs_sf.shape}")
    print(f"Max eigenvalue magnitude: {mag2db(torch.max(torch.abs(evs_sf)))} dB")
    print(f"Max eigenvalue real part: {mag2db(torch.max(torch.real(evs_sf)))} dB")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("All tests completed successfully!")
    print(f"  Test 1 (only_direct_path): Max |ev| = {mag2db(torch.max(torch.abs(evs_dp))):.2f} dB")
    print(f"  Test 2 (only_early_reflections): Max |ev| = {mag2db(torch.max(torch.abs(evs_er))):.2f} dB")
    print(f"  Test 3 (only_late_reverb): Max |ev| = {mag2db(torch.max(torch.abs(evs_lr))):.2f} dB")
    print(f"  Test 4 (suppress_feedback): Max |ev| = {mag2db(torch.max(torch.abs(evs_sf))):.2f} dB")


    # Plot maximum real part eigenvalue per frequency bin
    print("\n" + "=" * 80)
    print("Plotting maximum real part eigenvalue vs frequency")
    print("=" * 80)
    
    # Get real parts of eigenvalues (shape: [nfft, n_M] or [nfft, n_M, n_M])
    real_evs_dp_freq = torch.real(evs_dp)
    real_evs_er_freq = torch.real(evs_er)
    real_evs_lr_freq = torch.real(evs_lr)
    
    # Find maximum real part per frequency bin
    # If shape is [nfft, n_M, n_M], take max over last two dims
    # If shape is [nfft, n_M], take max over last dim
    if len(real_evs_dp_freq.shape) == 3:
        max_real_dp = torch.max(real_evs_dp_freq.view(real_evs_dp_freq.shape[0], -1), dim=1)[0]
        max_real_er = torch.max(real_evs_er_freq.view(real_evs_er_freq.shape[0], -1), dim=1)[0]
        max_real_lr = torch.max(real_evs_lr_freq.view(real_evs_lr_freq.shape[0], -1), dim=1)[0]
    else:
        max_real_dp = torch.max(real_evs_dp_freq, dim=-1)[0]
        max_real_er = torch.max(real_evs_er_freq, dim=-1)[0]
        max_real_lr = torch.max(real_evs_lr_freq, dim=-1)[0]
    
    # Create frequency axis (for rfft, we have nfft//2+1 frequency bins)
    n_freq_bins = max_real_dp.shape[0]
    freqs = torch.linspace(0, samplerate / 2, n_freq_bins)
    
    # Convert to dB for plotting
    max_real_dp_db = mag2db(max_real_dp + 1e-10)
    max_real_er_db = mag2db(max_real_er + 1e-10)
    max_real_lr_db = mag2db(max_real_lr + 1e-10)
    
    # Plot
    plt.rcParams.update({'font.family':'serif', 'font.size':14, 'font.weight':'heavy', 'text.usetex':True})
    plt.figure(figsize=(10, 6))
    plt.plot(freqs.numpy(), max_real_dp_db.numpy(), label='only_direct_path', linewidth=1)
    plt.plot(freqs.numpy(), max_real_er_db.numpy(), label='only_early_reflections', linewidth=1)
    plt.plot(freqs.numpy(), max_real_lr_db.numpy(), label='only_late_reverb', linewidth=1)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Max real part eigenvalue [dB]')
    plt.title('Maximum real part eigenvalue per frequency bin')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.xlim(20, samplerate / 2)
    plt.tight_layout()
    plt.show(block=True)


    # Plot eigenvalue real part distributions
    print("\n" + "=" * 80)
    print("Plotting eigenvalue real part distributions")
    print("=" * 80)
    
    # Extract real parts of eigenvalues
    real_evs_dp = torch.real(evs_dp).flatten()
    real_evs_er = torch.real(evs_er).flatten()
    real_evs_lr = torch.real(evs_lr).flatten()
    
    # Stack distributions for plotting
    # plot_distributions expects shape [n_samples, n_distributions]
    # We need to pad to same length or use the minimum length
    min_len = min(len(real_evs_dp), len(real_evs_er), len(real_evs_lr))
    real_evs_dp_trimmed = real_evs_dp[:min_len]
    real_evs_er_trimmed = real_evs_er[:min_len]
    real_evs_lr_trimmed = real_evs_lr[:min_len]
    
    # Convert to dB scale for plotting
    # Note: For real parts that can be negative, we convert magnitude to dB
    # and preserve sign. However, this creates symmetry around 0 dB by construction.
    # If you want to see actual asymmetry, plot in linear scale instead.
    real_evs_dp_db = mag2db(real_evs_dp_trimmed + 1e-10)
    real_evs_er_db = mag2db(real_evs_er_trimmed + 1e-10)
    real_evs_lr_db = mag2db(real_evs_lr_trimmed + 1e-10)
    
    distributions = torch.stack((
        real_evs_dp_db,
        real_evs_er_db,
        real_evs_lr_db
    ), dim=1)
    
    print("Plotting eigenvalue real part distributions (in dB)...")
    print("Note: The symmetry around 0 dB is an artifact of the conversion.")
    print("      Positive real parts map to +dB, negative to -dB with same magnitude.")
    plot_distributions(
        distributions=distributions,
        n_bins=nfft//100,
        labels=['only_direct_path', 'only_early_reflections', 'only_late_reverb']
    )

    exit(0)

