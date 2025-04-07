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
    
    ec = torch.zeros(rir.shape[1:])
    for i in range(rir.shape[1]):
        for j in range(rir.shape[2]):
            r = rir[:,i,j]
            index1 = find_onset(r)
            rt = reverb_time(r, fs=fs, decay_interval=decay_interval)
            index2 = (index1 + fs*rt).long()
            r_cut = r[index1:index2]
            ec[i,j] = torch.sum(torch.square(r_cut))

    return ec

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

    drr = torch.zeros(rir.shape[1:])
    for i in range(rir.shape[1]):
        for j in range(rir.shape[2]):
            r = rir[:,i,j]
            index1 = find_onset(r)
            index2 = (index1 + fs*torch.tensor([0.005])).long()
            rt = reverb_time(r, fs=fs, decay_interval=decay_interval)
            index3 = (index1 + fs*rt).long()
            direct = torch.sum(torch.square(r[index1:index2]))
            reverb = torch.sum(torch.square(r[index2:index3]))
            drr[i,j] = direct/reverb

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