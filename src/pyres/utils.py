# ==================================================================
# ============================ IMPORTS =============================
# PyTorch
import torch
from torch.nn.functional import max_pool1d
#Scipy
from scipy.signal import find_peaks


# ==================================================================

def next_power_of_2(x: float) -> int: 
    r"""
    Returns the next power of 2 of the input number.

        **Args**:
            - x (float): Input number.

        **Returns**:
            - int: Next power of 2.
    """
    return 1 if x == 0 else 2**(x - 1).bit_length()

def expand_to_dimension(array: torch.Tensor, dim: int) -> torch.Tensor:
    r"""
    Expands the input array to a given dimension.

        **Args**:
            - array (torch.Tensor): Input array.
            - dim (int): Dimension to expand to.

        **Returns**:
            - torch.Tensor: Expanded array.
    """
    while len(array.shape) < dim:
        array = array.unsqueeze(-1)
    return array


import matplotlib.pyplot as plt

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


def limit_frequency_points(array: torch.Tensor, fs: int, nfft: int, f_interval: tuple[float, float]=None, f_subset: torch.Tensor=None) -> torch.Tensor:
    f"""
    Reduces the input array to a given frequency interval or to a given frequency subset.

        **Args**:
            - array (torch.Tensor): Input array.
            - fs (int): Sampling frequency [Hz].
            - nfft (int): FFT size.
            - f_interval (tuple[float, float], optional): Frequency interval [Hz]. Defaults to None.
            - f_subset (torch.Tensor, optional): Frequency points [Hz]. Defaults to None.

        **Returns**:
            - torch.Tensor: reduced array.
    """
    
    if f_interval is not None:
        freqs = torch.linspace(0, fs/2, nfft//2+1)
        index_1 = torch.argmin(torch.abs(freqs - torch.tensor(f_interval[0])))
        index_2 = torch.argmin(torch.abs(freqs - torch.tensor(f_interval[1])))
        subset = torch.arange(index_1, index_2+1)
    elif f_subset is not None:
        freqs = torch.linspace(0, fs/2, nfft//2+1)
        subset = torch.argmin(torch.abs(freqs - f_subset.unsqueeze(0)), dim=1)
    else:
        subset = torch.arange(0, array.shape[-1])
    
    return torch.take_along_dim(array, subset, 0)
    