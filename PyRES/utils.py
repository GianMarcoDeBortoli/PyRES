# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
import numpy as np
# PyTorch
import torch


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
    

def system_equalization_curve(
        evs: torch.Tensor,
        fs: int,
        nfft: int,
        f_c: float=None
    ) -> torch.Tensor:
        f"""
        Computes the system equalization curve.
        If a crossover frequency is provided, the curve is divided into two parts:
        a flat response below the crossover frequency and a moving average of the mean value above the crossover frequency.
        If no crossover frequency is provided, the curve is a horizontal line at the mean value.
        The mean is computed first from the absolute values of the eigenvalues across the channels, and then across frequencies.
        
            **Args**:
                - evs (torch.Tensor): Open-loop eigenvalues [nfft, n_M].
                - fs (int): Sampling frequency [Hz].
                - nfft (int): FFT size.
                - f_c (float, optional): Crossover frequency [Hz]. Defaults to None.
            
            **Returns**:
                - torch.Tensor: The system equalization curve.
        """
        
        # frequency samples
        freqs = torch.linspace(0, fs/2, nfft//2+1)

        # Compute RTFs
        mean_evs = torch.mean(torch.abs(evs), dim=(1))

        if f_c is not None:
            # Divide target between left and right of crossover frequency
            index_crossover = torch.argmin(torch.abs(freqs - f_c))
            left_interval = torch.arange(0, index_crossover+1)
            right_interval = torch.arange(index_crossover, mean_evs.shape[0])

            # Left target: horizontal line at mean RTFs value
            scaling_factor = torch.mean(mean_evs[left_interval])
            target_left = scaling_factor * torch.ones(index_crossover,)

            # Right target: moving average of RTFs values
            smooth_window_length = right_interval.shape[0]//6
            # TODO: Find a way to apply this convolution with torch functions
            smooth_evs = torch.tensor(np.convolve(mean_evs[right_interval], np.ones(smooth_window_length)/smooth_window_length, mode='valid'))
            pre = torch.ones(smooth_window_length//2,) * smooth_evs[0]
            post = torch.ones(smooth_window_length//2,) * smooth_evs[-1]
            target_right = torch.cat((pre, smooth_evs, post), dim=0)

            # Create continuity between left and right
            target_right = target_right * (target_left[-1] / target_right[0])
            
            # Concatenate left and right targets
            target = torch.cat((target_left, target_right), dim=0)
        else:
            # Horizontal line at mean RTFs value
            scaling_factor = torch.mean(mean_evs)
            target = scaling_factor * torch.ones(mean_evs.shape[0],)
        
        return target