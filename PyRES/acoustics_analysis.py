# ==================================================================
# ============================ IMPORTS =============================
import numpy as np
from decayfitnet.toolbox.DecayFitNetToolbox import DecayFitNetToolbox
from decayfitnet.toolbox.core import PreprocessRIR
import pyfar as pf
import pyrato as pr
import torch
# PyRES
from PyRES.utils import expand_to_dimension, find_direct_path

# ==================================================================
# ===================== OCTAVE BANDS ANALYSIS ======================

def _octave_band_filter(signal: torch.Tensor, fs: int, octave_bands: int | str='broadband', frequency_range: tuple=(125.0, 8000.0)) -> pf.Signal:
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

def octave_band_filter(signal: torch.Tensor, fs: int, octave_bands: int, frequency_range: tuple=(125.0, 8000.0)) -> pf.Signal:
    f"""
    Applies octave band filtering to the input signal.

        **Args**:
            - signal (torch.Tensor): Input signal.
            - fs (int): Sampling frequency [Hz].
            - octave_bands (int | str): Number of octave bands.
            - frequency_range (tuple): Frequency range for the octave bands. Defaults to (125.0, 8000.0).

        **Returns**:
            - pf.TimeData: Filtered signal.
    """
    rirs = expand_to_dimension(rirs, 3)
    assert len(rirs.shape) == 3, "The input signal must be at most 3-dimensional."

    assert isinstance(octave_bands, int) and octave_bands in [1,3], "The number of octave bands must be 1 or 3 integer value."
    filterBank = pf.dsp.filter.fractional_octave_bands(signal=None, num_fractions=octave_bands, sampling_rate=fs, frequency_range=frequency_range, order=14)

    rirs_filtered = torch.zeros_like(rirs).unsqueeze(-1).repeat(1, 1, 1, filterBank.num_bands)

    for i in range(rirs.shape[1]):
        for j in range(rirs.shape[2]):
            rir = rirs[:,i,j,0]
            signal = pf.Signal(data=rir.squeeze().numpy(), sampling_rate=fs, domain='time')
        signal = pf.Signal(data=filterBank.process(signal=signal, reset=True).time.squeeze(), sampling_rate=fs, domain='time')

        rirs_filtered[:,i,j,:] = torch.tensor(signal.time)

    return rirs_filtered

def signal_to_noise_ratio(rirs: torch.Tensor, fs: int) -> torch.Tensor:
    f"""
    Computes the noise energy of a room impulse response matrix.
    Reference:
        Prawda, K., Schlecht, S. J., & Välimäki, V. (2022). Robust selection of clean swept-sine measurements in non-stationary noise.
        The Journal of the Acoustical Society of America, 151(3), 2117-2126.

        **Args**:
            - rirs (torch.Tensor): Room impulse responses.
            - fs (int): Sampling frequency.

        **Returns**:
            - torch.Tensor: Noise energy.
    """

    rirs = expand_to_dimension(rirs, 3)
    snr = np.zeros([1, rirs.shape[1], rirs.shape[2]])

    for i in range(rirs.shape[1]):
        for j in range(rirs.shape[2]):
            rir = rirs[:,i,j]

            noise_start = int(0.9 * rirs.shape[0])
            noise_end = rirs.shape[0]
            chi = 1.4826

            rir_temp = pf.Signal(data=rir, sampling_rate=fs, domain='time')

            direct_path = find_direct_path(impulse_response=rir, fs=fs)
            t_03 = 0.003 # 3 ms before direct sound
            signal_start = np.max([0, direct_path - fs*t_03]).astype(int).item()
            signal_end = direct_path + (pr.intersection_time_lundeby(data=rir_temp, time_shift=True)[0] * fs).astype(int).item()

            rir_windowed = pf.dsp.time_window(signal=rir_temp, interval=(signal_start, signal_end), window='boxcar', unit='samples', crop='window')
            signal_energy = np.sum(np.square(rir_windowed.time), axis=1)
            noise_energy = (signal_end-signal_start) * (chi**2) * torch.median(torch.square(rir[noise_start:noise_end]))

            snr[0,i,j] = 10*np.log10(signal_energy/noise_energy + 1e-10)


    return torch.tensor(snr)

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

    rirs = expand_to_dimension(rirs, 4)
    if octave_bands != 'broadband':
        center_frequencies, _ = pf.dsp.filter.fractional_octave_frequencies(num_fractions=octave_bands, frequency_range=frequency_range)
        n_bands = len(center_frequencies)
    else:
        n_bands = 1
    ec = np.zeros([n_bands, rirs.shape[1], rirs.shape[2]])

    for i in range(rirs.shape[1]):
        for j in range(rirs.shape[2]):
            rir = rirs[:,i,j,0]
            rir_temp = pf.Signal(data=rir, sampling_rate=fs, domain='time')

            direct_path = find_direct_path(impulse_response=rir, fs=fs)
            t_03 = 0.003 # 3 ms before direct sound
            index_start = np.max([0, direct_path - fs*t_03]).astype(int).item()
            index_end = direct_path + (pr.intersection_time_lundeby(data=rir_temp, time_shift=True)[0] * fs).astype(int).item()

            rir = _octave_band_filter(rir, fs=fs, octave_bands=octave_bands, frequency_range=frequency_range)
            rir_windowed = pf.dsp.time_window(signal=rir, interval=(index_start, index_end), window='boxcar', unit='samples', crop='window')
            energy = np.sum(np.square(rir_windowed.time), axis=1)

            ec[:,i,j] = 10 * np.log10(energy + 1e-10)

    return torch.tensor(ec)

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

    rir = _octave_band_filter(signal=rir, fs=fs, octave_bands=octave_bands, frequency_range=frequency_range)
    edc = pr.energy_decay_curve_chu(data=rir, time_shift=False, plot=False)

    return edc

def energy_decay_curve(rirs: torch.Tensor) -> torch.Tensor:
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

    rirs = expand_to_dimension(rirs, 3)
    edc = torch.flip(torch.cumsum(torch.flip(torch.square(rirs), dims=(0,)), dim=0), dims=(0,))

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
            # Store estimated values
            rt[:,i,j,:] = estimated_parameters_decayfitnet[0]

    rt = torch.tensor(rt)
    nan_idxs = torch.isnan(rt).nonzero()

    if len(nan_idxs) > 0:
        for idx in nan_idxs:
            rt_mean = torch.mean(rt[idx[0],:,:,idx[3]][~torch.isnan(rt[idx[0],:,:,idx[3]])])
            rt[idx[0], idx[1], idx[2], idx[3]] = rt_mean # TODO: slice(idx)?

    return rt

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
            index_mid = direct_path + (np.array([fs*t_05])).astype(int).item()
            index_end = direct_path + (pr.intersection_time_lundeby(data=rir_temp, time_shift=True)[0] * fs).astype(int).item()

            rir = _octave_band_filter(rir, fs=fs, octave_bands=octave_bands, frequency_range=frequency_range)
            rir_direct = pf.dsp.time_window(signal=rir, interval=(index_start, index_mid), window='boxcar', unit='samples', crop='window')
            rir_reverb = pf.dsp.time_window(signal=rir, interval=(index_mid, index_end), window='boxcar', unit='samples', crop='window')

            energy_early = np.sum(np.square(rir_direct.time), axis=1)
            energy_late = np.sum(np.square(rir_reverb.time), axis=1)

            drr[:,i,j] = 10 * np.log10((energy_early / energy_late) + 1e-10)

    return torch.tensor(drr)

def definition(rirs: torch.Tensor, fs: int, octave_bands: int | str='broadband', frequency_range: tuple=(125.0, 8000.0)) -> torch.Tensor:
    f"""
    Computes the clarity index of a room impulse response matrix.

        **Args**:
            - rirs (torch.Tensor): Room impulse responses.
            - fs (int): Sampling frequency [Hz].
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
    d50 = np.zeros([n_bands, rirs.shape[1], rirs.shape[2]])

    for i in range(rirs.shape[1]):
        for j in range(rirs.shape[2]):
            rir = rirs[:,i,j]
            rir_temp = pf.Signal(data=rir, sampling_rate=fs, domain='time')

            direct_path = find_direct_path(impulse_response=rir, fs=fs)
            t_03 = 0.003 #  3 ms before direct sound
            t_50 = 0.050 # 50 ms after direct sound
            index_start = np.max([0, direct_path - fs*t_03]).astype(int).item()
            index_mid = direct_path + (fs*np.array([t_50])).astype(int).item()
            index_end = direct_path + (pr.intersection_time_lundeby(data=rir_temp, time_shift=True)[0] * fs).astype(int).item()

            rir = _octave_band_filter(rir, fs=fs, octave_bands=octave_bands, frequency_range=frequency_range)
            rir_early = pf.dsp.time_window(signal=rir, interval=(index_start, index_mid), window='boxcar', unit='samples', crop='window')
            rir_late = pf.dsp.time_window(signal=rir, interval=(index_mid, index_end), window='boxcar', unit='samples', crop='window')

            energy_early = np.sum(np.square(rir_early.time), axis=1)
            energy_late = np.sum(np.square(rir_late.time), axis=1)

            d50[:,i,j] = 10 * np.log10((energy_early / energy_late) + 1e-10)

    return torch.tensor(d50)

def clarity_index(rirs: torch.Tensor, fs: int, octave_bands: int | str='broadband', frequency_range: tuple=(125.0, 8000.0)) -> torch.Tensor:
    f"""
    Computes the clarity index of a room impulse response matrix.

        **Args**:
            - rirs (torch.Tensor): Room impulse responses.
            - fs (int): Sampling frequency [Hz].
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
    c80 = np.zeros([n_bands, rirs.shape[1], rirs.shape[2]])

    for i in range(rirs.shape[1]):
        for j in range(rirs.shape[2]):
            rir = rirs[:,i,j]
            rir_temp = pf.Signal(data=rir, sampling_rate=fs, domain='time')

            direct_path = find_direct_path(impulse_response=rir, fs=fs)
            t_03 = 0.003 #  3 ms before direct sound
            t_80 = 0.080 # 80 ms after direct sound
            index_start = np.max([0, direct_path - fs*t_03]).astype(int).item()
            index_mid = direct_path + (fs*np.array([t_80])).astype(int).item()
            index_end = direct_path + (pr.intersection_time_lundeby(data=rir_temp, time_shift=True)[0] * fs).astype(int).item()

            rir = _octave_band_filter(rir, fs=fs, octave_bands=octave_bands, frequency_range=frequency_range)
            rir_early = pf.dsp.time_window(signal=rir, interval=(index_start, index_mid), window='boxcar', unit='samples', crop='window')
            rir_late = pf.dsp.time_window(signal=rir, interval=(index_mid, index_end), window='boxcar', unit='samples', crop='window')

            energy_early = np.sum(np.square(rir_early.time), axis=1)
            energy_late = np.sum(np.square(rir_late.time), axis=1)

            c80[:,i,j] = 10 * np.log10((energy_early / energy_late) + 1e-10)

    return torch.tensor(c80)

def lateral_energy_fraction(rirs: torch.Tensor, fs: int) -> torch.Tensor:
    f"""
    Computes the lateral energy fraction of a room impulse response matrix.

        **Args**:
            - rirs (torch.Tensor): Room impulse responses.
            - fs (int): Sampling frequency [Hz].

        **Returns**:
            - torch.Tensor: Direct-to-reverberant ratio.
    """

    rirs = expand_to_dimension(rirs, 4)

    octave_bands = 1
    frequency_range = [125.0, 1000.0]
    lef = np.zeros([1, rirs.shape[1], rirs.shape[2]])

    # Enconde A-format to B-format
    conversion_matrix = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                                      [1.0,-1.0,-1.0, 1.0],
                                      [1.0, 1.0,-1.0,-1.0],
                                      [1.0,-1.0, 1.0,-1.0]])
    conversion_matrix = 1/(2*torch.sqrt(torch.tensor([4*torch.pi]))) * torch.matmul( conversion_matrix, torch.diag(torch.sqrt(torch.tensor([1,3,3,3]))) )

    rirs = torch.matmul(rirs, conversion_matrix)

    for i in range(rirs.shape[1]):
        for j in range(rirs.shape[2]):
            # rir_temp = rirs[:,i,j,:].squeeze()
            # rir_temp = torch.matmul(rir_temp, conversion_matrix)
            rir_omni = rirs[:,i,j,0]
            rir_fig8 = rirs[:,i,j,1]

            direct_path = find_direct_path(impulse_response=rir_omni, fs=fs)
            t_05 = 0.005 # 5 ms after direct sound
            t_80 = 0.080
            index_start = direct_path + np.max([0, -(fs*np.array([0.003])).astype(int).item()])
            index_mid = direct_path + (fs*np.array([t_05])).astype(int).item()
            index_end = direct_path + (fs*np.array([t_80])).astype(int).item()

            rir_omni = _octave_band_filter(rir_omni, fs=fs, octave_bands=octave_bands, frequency_range=frequency_range)
            rir_fig8 = _octave_band_filter(rir_fig8, fs=fs, octave_bands=octave_bands, frequency_range=frequency_range)
            # rir_omni = pf.Signal(data=rir_omni.squeeze().numpy(), sampling_rate=fs, domain='time')
            # rir_fig8 = pf.Signal(data=rir_fig8.squeeze().numpy(), sampling_rate=fs, domain='time')
            rir_omni = pf.dsp.time_window(signal=rir_omni, interval=(index_start, index_end), window='boxcar', unit='samples', crop='window')
            rir_fig8 = pf.dsp.time_window(signal=rir_fig8, interval=(index_mid, index_end), window='boxcar', unit='samples', crop='window')

            energy_omni = np.sum(np.square(rir_omni.time), axis=1)
            energy_fig8 = np.sum(np.square(rir_fig8.time), axis=1)

            lef[0,i,j] = np.mean(energy_fig8 / energy_omni)

    return torch.tensor(lef)

def pseudo_intensity_vector(rirs: torch.Tensor) -> torch.Tensor:
    f"""
    Computes the lateral energy fraction of a room impulse response matrix.

        **Args**:
            - rirs (torch.Tensor): Room impulse responses.
            - fs (int): Sampling frequency [Hz].

        **Returns**:
            - torch.Tensor: Direct-to-reverberant ratio.
    """

    rirs = expand_to_dimension(rirs, 4)

    # Enconde A-format to B-format
    conversion_matrix = torch.tensor([[1, 1, 1, 1],
                                      [1,-1,-1, 1],
                                      [1, 1,-1,-1],
                                      [1,-1, 1,-1]])
    conversion_matrix = 1/(2*torch.sqrt(torch.tensor([4*torch.pi]))) * torch.matmul( conversion_matrix, torch.diag(torch.tensor([1,3,3,3])) )

    rirs = torch.matmul(rirs, conversion_matrix)

    pressure_signal = rirs[:,:,:,0]
    particle_velocity_x = rirs[:,:,:,3]
    particle_velocity_y = rirs[:,:,:,1]
    particle_velocity_z = rirs[:,:,:,2]

    pseudo_intensity_vector = pressure_signal.unsqueeze(-1) * torch.stack((particle_velocity_x, particle_velocity_y, particle_velocity_z), dim=-1)

    return pseudo_intensity_vector, pressure_signal, particle_velocity_x, particle_velocity_y, particle_velocity_z

def sdm_doa(pseudo_intensity_vector: torch.Tensor) -> torch.Tensor:
    f"""
    Computes the direction of arrival (DOA) from the pseudo-intensity vector.

        **Args**:
            - pi_v (torch.Tensor): Pseudo-intensity vector.

        **Returns**:
            - torch.Tensor: DOA in spherical coordinates (azimuth, elevation, radius).
    """
    pseudo_intensity_vector = pseudo_intensity_vector / (torch.norm(pseudo_intensity_vector, dim=-1, keepdim=True) + 1e-10)

    x = pseudo_intensity_vector[:,:,:,0]
    y = pseudo_intensity_vector[:,:,:,1]
    z = pseudo_intensity_vector[:,:,:,2]

    azimuth = torch.atan2(y, x)
    radius = torch.sqrt(x**2 + y**2 + z**2)
    elevation = torch.asin(z / (radius + 1e-10))

    doa = torch.stack((azimuth, elevation, radius), dim=-1)

    return doa