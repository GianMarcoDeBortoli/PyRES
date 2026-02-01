# ==================================================================
# ============================ IMPORTS =============================
import numpy as np
from external.DecayFitNet.python.toolbox.DecayFitNetToolbox import DecayFitNetToolbox
from external.DecayFitNet.python.toolbox.utils import calc_mse
from external.DecayFitNet.python.toolbox.core import discard_last_n_percent, decay_model, PreprocessRIR
import pyfar as pf
import pyrato as pr
import torch
# PyRES
from PyRES.utils import expand_to_dimension, find_direct_path

import matplotlib.pyplot as plt

# ==================================================================
# ===================== OCTAVE BANDS ANALYSIS ======================

def _octave_bands_filter(signal: torch.Tensor, fs: int, octave_bands: int | str='broadband', frequency_range: tuple=(125.0, 8000.0)) -> pf.Signal:
    f"""
    Applies octave band filtering to the input signal.

        **Args**:
            - signal (torch.Tensor): Input signal.
            - fs (int): Sampling frequency [Hz].
            - octave_bands (int | str): Number of octave bands or 'broadband'. Defaults to 'broadband'.
            - frequency_range (tuple): Frequency range for the octave bands. Defaults to (63.0, 16000.0).

        **Returns**:
            - pf.TimeData: Filtered signal.
    """
    signal = expand_to_dimension(signal, 3)
    assert signal.shape[1] == 1 and signal.shape[2] == 1, "The input signal must be a single-channel signal."

    signal = pf.Signal(data=signal.squeeze().numpy(), sampling_rate=fs, domain='time')

    if octave_bands != 'broadband':
        assert isinstance(octave_bands, int) and octave_bands in [1,3], "The number of octave bands must be 1 or 3 integer value."
        filterBank = pf.dsp.filter.fractional_octave_bands(signal=None, num_fractions=octave_bands, sampling_rate=fs, frequency_range=frequency_range, order=14)
        signal = pf.Signal(data=filterBank.process(signal=signal, reset=True).time.squeeze(), sampling_rate=fs, domain='time')

    return signal

def _energy_decay_curve(rir: torch.Tensor, fs: int, octave_bands: int | str='broadband', frequency_range: tuple=(125.0, 8000.0)) -> pf.TimeData:
    f"""
    Computes the energy decay curve of a room impulse response.

        **Args**:
            - rir (torch.Tensor): Room impulse response.
            - fs (int): Sampling frequency [Hz].
            - octave_bands (int | str): Number of octave bands or 'broadband'. Defaults to 'broadband'.
            - frequency_range (tuple): Frequency range for the octave bands. Defaults to (63.0, 16000.0).

        **Returns**:
            - pf.TimeData: Energy decay curve.
    """

    rir = _octave_bands_filter(signal=rir, fs=fs, octave_bands=octave_bands, frequency_range=frequency_range)
    edc = pr.energy_decay_curve_chu(data=rir, time_shift=False, plot=True)
    plt.show(block=True)

    return edc

def reverberation_time(rirs: torch.Tensor, fs: int, decay_interval: str='T30', octave_bands: int | str='broadband', frequency_range: tuple=(125.0, 8000.0)) -> torch.Tensor:
    f"""
    Computes the reverberation time of a room impulse response matrix.

        **Args**:
            - rirs (torch.Tensor): Room impulse responses.
            - fs (int): Sampling frequency [Hz].
            - decay_interval (str): Decay interval. Defaults to 'T30'.

        **Returns**:
            - torch.Tensor: Reverberation time [s].
    """

    rirs = expand_to_dimension(rirs, 3)
    if octave_bands != 'broadband':
        center_frequencies, _ = pf.dsp.filter.fractional_octave_frequencies(num_fractions=octave_bands, frequency_range=frequency_range)
    else:
        center_frequencies = np.array([1.0])
    rt = np.zeros([len(center_frequencies), rirs.shape[1], rirs.shape[2]])

    for i in range(rirs.shape[1]):
        for j in range(rirs.shape[2]):
            rir = rirs[:,i,j]
            edc = _energy_decay_curve(rir=rir, fs=fs, octave_bands=octave_bands, frequency_range=frequency_range)
            rt[:,i,j] = pr.reverberation_time_linear_regression(energy_decay_curve=edc, T=decay_interval)

    rt = torch.tensor(rt)
    nan_idxs = torch.isnan(rt).nonzero()

    if len(nan_idxs) > 0:
        for idx in nan_idxs:
            rt_mean = torch.mean(rt[idx[0],:,:][~torch.isnan(rt[idx[0],:,:])])
            rt[idx[0], idx[1], idx[2]] = rt_mean

    return rt

def reverberation_time_multislope(rirs: torch.Tensor, fs: int, n_slopes: int=2, decay_interval: str='T30', octave_bands: int | str='broadband', frequency_range: tuple=(125.0, 8000.0)) -> torch.Tensor:
    f"""
    Computes the reverberation time of a room impulse response matrix.

        **Args**:
            - rirs (torch.Tensor): Room impulse responses.
            - fs (int): Sampling frequency [Hz].
            - decay_interval (str): Decay interval. Defaults to 'T30'.

        **Returns**:
            - torch.Tensor: Reverberation time [s].
    """

    rirs = expand_to_dimension(rirs, 3)

    if octave_bands != 'broadband':
        center_frequencies, _ = pf.dsp.filter.fractional_octave_frequencies(num_fractions=octave_bands, frequency_range=frequency_range)
        n_bands = len(center_frequencies)
    else:
        center_frequencies = [500, 1000]
        n_bands = 2
    rt = np.zeros([n_bands, rirs.shape[1], rirs.shape[2], n_slopes])

    for i in range(rirs.shape[1]):
        for j in range(rirs.shape[2]):
            rir = rirs[:,i,j]
            # Init Preprocessing
            rir_preprocessing = PreprocessRIR(sample_rate=fs, filter_frequencies=center_frequencies)
            # Schroeder integration, analyse_full_rir: if RIR onset should be detected, set this to False
            true_edc, __ = rir_preprocessing.schroeder(rir, analyse_full_rir=True)
            time_axis = (torch.linspace(0, true_edc.shape[2] - 1, true_edc.shape[2]) / fs)
            # Permute into [n_bands, n_batches, n_samples]
            true_edc = true_edc.permute(1, 0, 2)
            # Prepare the model
            decayfitnet = DecayFitNetToolbox(n_slopes=n_slopes, sample_rate=fs, filter_frequencies=center_frequencies)
            # Process
            estimated_parameters_decayfitnet, _ = decayfitnet.estimate_parameters(rir, analyse_full_rir=True)

            # ----------
            # Get fitted EDC from estimated parameters
            # fitted_edc_decayfitnet = decay_model(torch.from_numpy(estimated_parameters_decayfitnet[0]),
            #                                     torch.from_numpy(estimated_parameters_decayfitnet[1]),
            #                                     torch.from_numpy(estimated_parameters_decayfitnet[2]),
            #                                     time_axis=time_axis,
            #                                     compensate_uli=True,
            #                                     backend='torch')
            # # Discard last 5% for MSE evaluation
            # true_edc = discard_last_n_percent(true_edc, 5)
            # fitted_edc_decayfitnet = discard_last_n_percent(fitted_edc_decayfitnet, 5)

            # # Plot
            # time_axis_excl5 = time_axis[0:round(0.95 * len(time_axis))]  # discard last 5 percent of plot time axis
            # colors = ['b', 'g', 'r', 'c', 'm', 'y']
            # for band_idx in range(true_edc.shape[0]):
            #     plt.plot(time_axis_excl5, 10 * torch.log10(true_edc[band_idx, 0, :].squeeze()),
            #             colors[band_idx], label='Measured EDC, {} Hz'.format(center_frequencies[band_idx]))
            #     plt.plot(time_axis_excl5, 10 * torch.log10(fitted_edc_decayfitnet[band_idx, 0, :].squeeze()),
            #             colors[band_idx] + '--', label='DecayFitNet fit, {} Hz'.format(center_frequencies[band_idx]))

            # plt.xlabel('time [s]')
            # plt.ylabel('energy [dB]')
            # plt.subplots_adjust(right=0.6)
            # plt.legend(loc='upper right', bbox_to_anchor=(1.8, 1))
            # plt.title('DecayFitNet')
            
            # ----------------

            # Store estimated values
            rt[:,i,j,:] = estimated_parameters_decayfitnet[0]
            print(rt[:,i,j,:])
            # plt.show(block=True)

    rt = torch.tensor(rt)
    nan_idxs = torch.isnan(rt).nonzero()
    print(nan_idxs)

    if len(nan_idxs) > 0:
        for idx in nan_idxs:
            rt_mean = torch.mean(rt[idx[0],:,:,idx[3]][~torch.isnan(rt[idx[0],:,:,idx[3]])])
            rt[idx[0], idx[1], idx[2], idx[3]] = rt_mean # TODO: slice(idx)?

    return rt

def energy_coupling(rirs: torch.Tensor, fs: int, octave_bands: int | str='broadband', frequency_range: tuple=(125.0, 8000.0)) -> torch.Tensor:
    f"""
    Computes the energy coupling of a room impulse response matrix.

        **Args**:
            - rirs (torch.Tensor): Room impulse responses.
            - fs (int): Sampling frequency [Hz].
            - octave_bands (int | str): Number of octave bands or 'broadband'. Defaults to 'broadband'.
            - frequency_range (tuple): Frequency range for the octave bands. Defaults to (63.0, 16000.0).

        **Returns**:
            - torch.Tensor: Energy coupling.
    """

    rirs = expand_to_dimension(rirs, 3)
    if octave_bands != 'broadband':
        center_frequencies, _ = pf.dsp.filter.fractional_octave_frequencies(num_fractions=octave_bands, frequency_range=frequency_range)
        n_bands = len(center_frequencies)
    else:
        n_bands = 1
    ec = np.zeros([n_bands, rirs.shape[1], rirs.shape[2]])

    for i in range(rirs.shape[1]):
        for j in range(rirs.shape[2]):
            rir = rirs[:,i,j]
            rir_temp = pf.Signal(data=rir, sampling_rate=fs, domain='time')

            direct_path = find_direct_path(impulse_response=rir, fs=fs)
            t_03 = 0.003 # 3 ms before direct sound
            index_start = np.max([0, direct_path - fs*t_03]).astype(int).item()
            index_end = index_start + (pr.intersection_time_lundeby(data=rir_temp, time_shift=True)[0] * fs).astype(int).item()

            rir = _octave_bands_filter(rir, fs=fs, octave_bands=octave_bands, frequency_range=frequency_range)
            rir_windowed = pf.dsp.time_window(signal=rir, interval=(index_start, index_end), window='boxcar', unit='samples', crop='window')
            energy = np.sum(np.square(rir_windowed.time), axis=1)

            ec[:,i,j] = 10 * np.log10(energy + 1e-10)

    return torch.tensor(ec)

def clarity_index(rirs: torch.Tensor, fs: int, octave_bands: int | str='broadband', frequency_range: tuple=(125.0, 8000.0)) -> torch.Tensor:
    f"""
    Computes the clarity index of a room impulse response matrix.

        **Args**:
            - rirs (torch.Tensor): Room impulse responses.
            - fs (int): Sampling frequency [Hz].
            - t (float): Time interval [s]. Defaults to 0.05s.
            - octave_bands (int | str): Number of octave bands or 'broadband'. Defaults to 'broadband'.
            - frequency_range (tuple): Frequency range for the octave bands. Defaults to (63.0, 16000.0).

        **Returns**:
            - torch.Tensor: Clarity index [dB].
    """
    rirs = expand_to_dimension(rirs, 3)

    if octave_bands != 'broadband':
        center_frequencies, _ = pf.dsp.filter.fractional_octave_frequencies(num_fractions=octave_bands, frequency_range=frequency_range)
        n_bands = len(center_frequencies)
    else:
        n_bands = 1
    ci = np.zeros([n_bands, rirs.shape[1], rirs.shape[2]])

    for i in range(rirs.shape[1]):
        for j in range(rirs.shape[2]):
            rir = rirs[:,i,j]
            rir_temp = pf.Signal(data=rir, sampling_rate=fs, domain='time')

            direct_path = find_direct_path(impulse_response=rir, fs=fs)
            t_03 = 0.003 # 3 ms before direct sound
            t_80 = 0.080 # 5 ms after direct sound
            index_start = np.max([0, direct_path - fs*t_03]).astype(int).item()
            index_mid = index_start + (fs*np.array([t_80])).astype(int).item()
            index_end = index_start + (pr.intersection_time_lundeby(data=rir_temp, time_shift=True)[0] * fs).astype(int).item()

            rir = _octave_bands_filter(rir, fs=fs, octave_bands=octave_bands, frequency_range=frequency_range)
            rir_early = pf.dsp.time_window(signal=rir, interval=(index_start, index_mid), window='boxcar', unit='samples', crop='window')
            rir_late = pf.dsp.time_window(signal=rir, interval=(index_mid, index_end), window='boxcar', unit='samples', crop='window')

            energy_early = np.sum(np.square(rir_early.time), axis=1)
            energy_late = np.sum(np.square(rir_late.time), axis=1)

            ci[:,i,j] = 10 * np.log10((energy_early / energy_late) + 1e-10)

    return torch.tensor(ci)

def definition(rirs: torch.Tensor, fs: int, octave_bands: int | str='broadband', frequency_range: tuple=(125.0, 8000.0)) -> torch.Tensor:
    f"""
    Computes the clarity index of a room impulse response matrix.

        **Args**:
            - rirs (torch.Tensor): Room impulse responses.
            - fs (int): Sampling frequency [Hz].
            - t (float): Time interval [s]. Defaults to 0.05s.
            - octave_bands (int | str): Number of octave bands or 'broadband'. Defaults to 'broadband'.
            - frequency_range (tuple): Frequency range for the octave bands. Defaults to (63.0, 16000.0).

        **Returns**:
            - torch.Tensor: Clarity index [dB].
    """
    rirs = expand_to_dimension(rirs, 3)

    if octave_bands != 'broadband':
        center_frequencies, _ = pf.dsp.filter.fractional_octave_frequencies(num_fractions=octave_bands, frequency_range=frequency_range)
        n_bands = len(center_frequencies)
    else:
        n_bands = 1
    ci = np.zeros([n_bands, rirs.shape[1], rirs.shape[2]])

    for i in range(rirs.shape[1]):
        for j in range(rirs.shape[2]):
            rir = rirs[:,i,j]
            rir_temp = pf.Signal(data=rir, sampling_rate=fs, domain='time')

            direct_path = find_direct_path(impulse_response=rir, fs=fs)
            t_03 = 0.003 # 3 ms before direct sound
            t_50 = 0.050 # 5 ms after direct sound
            index_start = np.max([0, direct_path - fs*t_03]).astype(int).item()
            index_mid = index_start + (fs*np.array([t_50])).astype(int).item()
            index_end = index_start + (pr.intersection_time_lundeby(data=rir_temp, time_shift=True)[0] * fs).astype(int).item()

            rir = _octave_bands_filter(rir, fs=fs, octave_bands=octave_bands, frequency_range=frequency_range)
            rir_early = pf.dsp.time_window(signal=rir, interval=(index_start, index_mid), window='boxcar', unit='samples', crop='window')
            rir_late = pf.dsp.time_window(signal=rir, interval=(index_mid, index_end), window='boxcar', unit='samples', crop='window')

            energy_early = np.sum(np.square(rir_early.time), axis=1)
            energy_late = np.sum(np.square(rir_late.time), axis=1)

            ci[:,i,j] = 10 * np.log10((energy_early / energy_late) + 1e-10)

    return torch.tensor(ci)

def direct_to_reverb_ratio(rirs: torch.Tensor, fs: int, octave_bands: int | str='broadband', frequency_range: tuple=(125.0, 8000.0)) -> torch.Tensor:
    f"""
    Computes the direct-to-reverberant ratio of a room impulse response matrix.

        **Args**:
            - rirs (torch.Tensor): Room impulse responses.
            - fs (int): Sampling frequency [Hz].
            - decay_interval (str): Decay interval. Defaults to 'T30'.

        **Returns**:
            - torch.Tensor: Direct-to-reverberant ratio.
    """

    rirs = expand_to_dimension(rirs, 3)

    if octave_bands != 'broadband':
        center_frequencies, _ = pf.dsp.filter.fractional_octave_frequencies(num_fractions=octave_bands, frequency_range=frequency_range)
        n_bands = len(center_frequencies)
    else:
        n_bands = 1
    drr = np.zeros([n_bands, rirs.shape[1], rirs.shape[2]])

    for i in range(rirs.shape[1]):
        for j in range(rirs.shape[2]):
            rir = rirs[:,i,j]
            rir_temp = pf.Signal(data=rir, sampling_rate=fs, domain='time')

            direct_path = find_direct_path(impulse_response=rir, fs=fs)
            t_03 = 0.003 # 3 ms before direct sound
            t_05 = 0.005 # 5 ms after direct sound
            index_start = np.max([0, direct_path - fs*t_03]).astype(int).item()
            index_mid = index_start + (fs*np.array([t_05])).astype(int).item()
            index_end = index_start + (pr.intersection_time_lundeby(data=rir_temp, time_shift=True)[0] * fs).astype(int).item()

            rir = _octave_bands_filter(rir, fs=fs, octave_bands=octave_bands, frequency_range=frequency_range)
            rir_direct = pf.dsp.time_window(signal=rir, interval=(index_start, index_mid), window='boxcar', unit='samples', crop='window')
            rir_reverb = pf.dsp.time_window(signal=rir, interval=(index_mid, index_end), window='boxcar', unit='samples', crop='window')

            energy_early = np.sum(np.square(rir_direct.time), axis=1)
            energy_late = np.sum(np.square(rir_reverb.time), axis=1)

            drr[:,i,j] = 10 * np.log10((energy_early / energy_late) + 1e-10)

    return torch.tensor(drr)