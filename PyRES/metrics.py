# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
import pyfar as pf
import pyrato as pr
# PyTorch
import torch
# FLAMO
from flamo.functional import find_onset
# PyRES
from PyRES.utils import expand_to_dimension

from torch.nn.functional import max_pool1d
#Scipy
from scipy.signal import find_peaks


# ==================================================================

def reverb_time(rir: torch.Tensor, fs: int, decay_interval: str='T30') -> torch.Tensor:
    f"""
    Computes the reverberation time of a room impulse response.

        **Args**:
            - rir (torch.Tensor): Room impulse response.
            - fs (int): Sampling frequency [Hz].
            - decay_interval (str): Decay interval. Defaults to 'T30'.

        **Returns**:
            - torch.Tensor: Reverberation time [s].
    """

    rir = rir.squeeze().numpy()
    pf_rir = pf.Signal(data=rir, sampling_rate=fs, domain='time')
    edc = pr.energy_decay_curve_chu(data=pf_rir, time_shift=False)
    rt = pr.reverberation_time_energy_decay_curve(energy_decay_curve=edc, T=decay_interval)

    return torch.tensor(rt.item())

def energy_coupling(rir: torch.Tensor, fs: int, decay_interval: str='T30') -> torch.Tensor:
    f"""
    Computes the energy coupling of an impulse response.

        **Args**:
            - rir (torch.Tensor): Room impulse response.
            - fs (int): Sampling frequency [Hz].
            - decay_interval (str): Decay interval. Defaults to 'T30'.

        **Returns**:
            - torch.Tensor: Energy coupling.
    """

    rir = expand_to_dimension(rir, 3)
    
    # ec = torch.zeros(rir.shape[1:])
    # for i in range(rir.shape[1]):
    #     for j in range(rir.shape[2]):
    #         r = rir[:,i,j]
    #         index1 = find_onset(r)
    #         rt = reverb_time(r, fs=fs, decay_interval=decay_interval)
    #         index2 = (index1 + fs*rt).long()
    #         r_cut = r[index1:index2]
    #         ec[i,j] = torch.sum(torch.square(r_cut))

    prev_rt = reverb_time(rir[:,0,0], fs=fs, decay_interval='T20')
    
    ec = torch.zeros(rir.shape[1:])
    for i in range(rir.shape[1]):
        for j in range(rir.shape[2]):
            r = rir[:,i,j]
            index1 = find_direct_path(r, fs=fs)
            rt = reverb_time(r, fs=fs, decay_interval=decay_interval)
            if (torch.isnan(rt) and decay_interval == 'T30') or rt > 1.5*prev_rt:
                rt = reverb_time(r, fs=fs, decay_interval='T20')
            if torch.isnan(rt):
                print(f"Warning: Could not compute reverberation time for mic {i+1}, speaker {j+1}. Using previous rir value.")
                rt = prev_rt
            index2 = (index1 + fs*rt).long()
            r_cut = r[index1:index2]
            ec[i,j] = torch.sum(torch.square(r_cut))
            prev_rt = rt

    return ec

def find_direct_path(rir: torch.Tensor, fs: int) -> int:
    f"""
    Detects the direct path onset in a room impulse response.

        **Parameters**:
            - rir (torch.Tensor): Room impulse response (1D tensor).
            - fs (int): Sampling rate (Hz)

        **Returns**:
            - direct_index (int): Sample index of estimated direct path
    """

    rir = rir.clone().detach()
    rir_abs = rir.abs()

    # Envelope approximation using max filter (peak envelope)
    kernel_size = 10
    pad = kernel_size // 2
    env = max_pool1d(rir_abs.view(1, 1, -1), kernel_size=kernel_size, stride=1, padding=pad)[0, 0]

    env_threshold = 0.5 * torch.max(env).item()
    peaks_env, properties_env = find_peaks(env.numpy(), height=env_threshold)
    
    if len(peaks_env) == 0:
        raise RuntimeError("No peaks found in the envelope.")

    env_peak_loc = peaks_env[0]
    env_peak_width = int(0.8 * properties_env["widths"][0]) if "widths" in properties_env else 20
    start = max(0, env_peak_loc - env_peak_width)
    end = min(len(rir), env_peak_loc + env_peak_width)
    env_peak_interval = torch.arange(start, end)

    # Now find actual peak within the envelope region
    rir_segment = rir_abs[env_peak_interval]
    rir_threshold = 0.5 * torch.max(rir_segment).item()
    peaks_h, _ = find_peaks(rir_segment.numpy(), height=rir_threshold)

    if len(peaks_h) == 0:
        raise RuntimeError("No peaks found in the impulse response segment.")

    delay = int(env_peak_interval[0].item() + peaks_h[0])

    return delay


def direct_to_reverb_ratio(rir: torch.Tensor, fs: int, decay_interval: str='T30') -> torch.Tensor:
    f"""
    Computes the direct-to-reverberant ratio of an impulse response.

        **Args**:
            - rir (torch.Tensor): Room impulse response.
            - fs (int): Sampling frequency [Hz].
            - decay_interval (str): Decay interval. Defaults to 'T30'.

        **Returns**:
            - torch.Tensor: Direct-to-reverberant ratio.
    """

    rir = expand_to_dimension(rir, 3)
    prev_rt = reverb_time(rir[:,0,0], fs=fs, decay_interval='T20')

    drr = torch.zeros(rir.shape[1:])
    for i in range(rir.shape[1]):
        for j in range(rir.shape[2]):
            r = rir[:,i,j]
            index1 = find_direct_path(r, fs=fs)
            index2 = (index1 + fs*torch.tensor([0.005])).long()
            rt = reverb_time(r, fs=fs, decay_interval=decay_interval)
            if (torch.isnan(rt) and decay_interval == 'T30') or rt > 1.5*prev_rt:
                rt = reverb_time(r, fs=fs, decay_interval='T20')
            if torch.isnan(rt):
                print(f"Warning: Could not compute reverberation time for mic {i+1}, speaker {j+1}. Using default value of 0.5 seconds.")
                rt = prev_rt
            index3 = (index1 + fs*rt).long()
            direct = torch.sum(torch.square(r[index1:index2]))
            reverb = torch.sum(torch.square(r[index2:index3]))
            drr[i,j] = direct/reverb
            prev_rt = rt

    return drr

# def peak_to_mean_ratio(array: torch.Tensor, dim: tuple[int]=None) -> torch.Tensor:
#     f"""
#     Computes the peak to mean ratio of an array along a given dimension.

#         **Args**:
#             - array (torch.Tensor): Input array.
#             - dim (int, tuple): Dimension(s) along which to compute the peak to mean ratio.

#         **Returns**:
#             - torch.Tensor: Peak to mean ratio.
#     """

#     if dim is None:
#         dim = tuple(range(1, array.ndim))
#     else:
#         if isinstance(dim, int):
#             dim = (dim,)
#         elif not isinstance(dim, tuple):
#             raise ValueError("dim must be an int or a tuple of ints")

#     max_vals = torch.amax(input=array, dim=dim, keepdim=True)
#     mean_vals = torch.mean(input=array, dim=dim, keepdim=True)

#     return max_vals / mean_vals


# def coloration_coefficient(ir: torch.Tensor, interval: tuple[int,int]=None) -> torch.Tensor:
#     f"""
#     Computes the coloration coefficient of an array.

#         **Args**:
#             ir (torch.Tensor): impulse response.

#         **Returns**:
#             torch.Tensor: Coloration coefficient as the standard deviation with respect to a mean of 1.
#     """
#     if interval is not None:
#         array = array[interval[0]:interval[1]]

#     mean_value = torch.mean(input=array, dim=0, keepdim=True)
#     array_norm = array / mean_value

#     return torch.std(input=array_norm, dim=0, keepdim=True)


# if __name__ == '__main__':

#     import matplotlib.pyplot as plt
#     array = torch.randn(100, 10, 10)

#     ptmr = peak_to_mean_ratio(array)

#     cc = coloration_coefficient(array)

#     cc1 = coloration_coefficient(array[:,0,0].squeeze())

#     a = 1