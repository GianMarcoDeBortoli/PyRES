# ==================================================================
# ============================ IMPORTS =============================
from collections import OrderedDict
import numpy as np
import pyfar as pf
import pyrato as pr
# PyTorch
import torch
import torch.nn.functional as F
# FLAMO
from flamo.functional import db2mag, get_eigenvalues, skew_matrix
# PyRES
from pyres.utils import expand_to_dimension, find_direct_path


# ==================================================================
# ========================= PHYSICAL ROOM ==========================

def simulate_setup(
        room_dims: torch.FloatTensor,
        mcs_n: int,
        lds_n: int
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    r"""
    Simulates a RES setup with the given room dimensions and number of loudspeakers and microphones.

        **Args**:
            - room_dims (torch.Tensor[float,float,float]): The dimensions of the room [m] - length, width, height.
            - mcs_n (int): The number of system microphones.
            - lds_n (int): The number of system loudspeakers.

        **Returns**:
            - torch.Tensor: The stage emitter position [m].
            - torch.Tensor: The system emitter positions [m].
            - torch.Tensor: The system receiver positions [m].
            - torch.Tensor: The audience receiver position [m].
    """
    assert len(room_dims.shape) == 1, "The room dimensions must be a 1D tensor."
    assert len(room_dims) == 3, "The room dimensions must have 3 elements."
    assert torch.all(room_dims > 0), "The room dimensions must be greater than 0."

    assert isinstance(lds_n, int), "The number of loudspeakers must be an integer."
    assert isinstance(mcs_n, int), "The number of microphones must be an integer."
    assert lds_n > 0, "The number of loudspeakers must be greater than 0."
    assert mcs_n > 0, "The number of microphones must be greater than 0."


    room_dims = torch.cat([torch.sort(room_dims[0:2], descending=True).values, room_dims[2].unsqueeze(0)], dim=0)
    surface_n = 5 # shoebox: 4 walls + 1 ceiling

    # Assign an index to each transducer
    lds_idx = torch.arange(0, lds_n)
    mcs_idx = torch.arange(0, mcs_n)
    # Assign each transducer to a surface
    lds_surface_idx = torch.fmod(lds_idx, surface_n)
    mcs_surface_idx = torch.fmod(mcs_idx, surface_n)

    # Allocate the loudspeaker and microphone positions
    lds_pos = torch.zeros((lds_n, 3))
    lds_count = 0
    mcs_pos = torch.zeros((mcs_n, 3))
    mcs_count = 0

    # For each surface
    for i in range(surface_n):
        # Count the number of loudspeaker and microphones in the surface
        lds_n_on_surface = torch.sum(lds_surface_idx == i)
        mcs_n_on_surface = torch.sum(mcs_surface_idx == i)
        # Case based on the surface index
        match i:
            case 0:
                dim_1_idx = 0
                dim_1 = room_dims[dim_1_idx]
                dim_2_idx = 2
                dim_2 = room_dims[dim_2_idx]
                const_idx = 1
                const = 0
            case 1:
                dim_1_idx = 0
                dim_1 = room_dims[dim_1_idx]
                dim_2_idx = 2
                dim_2 = room_dims[dim_2_idx]
                const_idx = 1
                const = room_dims[const_idx]
            case 2:
                dim_1_idx = 1
                dim_1 = room_dims[dim_1_idx]
                dim_2_idx = 2
                dim_2 = room_dims[dim_2_idx]
                const_idx = 0
                const = 0
            case 3:
                dim_1_idx = 1
                dim_1 = room_dims[dim_1_idx]
                dim_2_idx = 2
                dim_2 = room_dims[dim_2_idx]
                const_idx = 0
                const = room_dims[const_idx]
            case 4:
                dim_1_idx = 1
                dim_1 = room_dims[dim_1_idx]
                dim_2_idx = 0
                dim_2 = room_dims[dim_1_idx]
                const_idx = 2
                const = room_dims[const_idx]
            case _:
                raise ValueError("Invalid surface index.")
        # Get the positions of loudspeakers and microphones on the current surface
        lds_pos_on_surface, mcs_pos_on_surface = positions_on_surface(dim_1, dim_2, lds_n_on_surface, mcs_n_on_surface)
        # Assign the positions of the loudspeakers to the global positions
        lds_pos[lds_count:lds_count+lds_n_on_surface, dim_1_idx] = lds_pos_on_surface[:, 0]
        lds_pos[lds_count:lds_count+lds_n_on_surface, dim_2_idx] = lds_pos_on_surface[:, 1]
        lds_pos[lds_count:lds_count+lds_n_on_surface, const_idx] = const
        # Assign the positions of the microphones to the global positions
        mcs_pos[mcs_count:mcs_count+mcs_n_on_surface, dim_1_idx] = mcs_pos_on_surface[:, 0]
        mcs_pos[mcs_count:mcs_count+mcs_n_on_surface, dim_2_idx] = mcs_pos_on_surface[:, 1]
        mcs_pos[mcs_count:mcs_count+mcs_n_on_surface, const_idx] = const
        # Update the counts
        lds_count += lds_n_on_surface
        mcs_count += mcs_n_on_surface

    stg_pos = torch.FloatTensor([[room_dims[0] / 4, room_dims[1] / 2, room_dims[2]/2]])
    aud_pos = torch.FloatTensor([[room_dims[0] * 3 / 4, room_dims[1] / 2, room_dims[2]/2]])

    # Add some noise to the positions
    stg_pos += torch.normal(0, 0.01, size=stg_pos.shape)
    mcs_pos += torch.normal(0, 0.01, size=mcs_pos.shape)
    lds_pos += torch.normal(0, 0.01, size=lds_pos.shape)
    aud_pos += torch.normal(0, 0.01, size=aud_pos.shape)

    positions = OrderedDict()
    positions['stg'] = stg_pos
    positions['mcs'] = mcs_pos
    positions['lds'] = lds_pos
    positions['aud'] = aud_pos

    return positions

def positions_on_surface(dim_1, dim_2, lds_n, mcs_n):
    """
    Distribute microphones and loudspeakers across a 2D rectangle surface,
    prioritizing central positions and preventing clustering using farthest-point sampling.

    Args:
        dim_1 (float): First dimension (e.g., width)
        dim_2 (float): Second dimension (e.g., height)
        lds_n (int): Number of loudspeakers
        mcs_n (int): Number of microphones

    Returns:
        lds_pos: (lds_n, 2) torch.FloatTensor
        mcs_pos: (mcs_n, 2) torch.FloatTensor
    """
    # Check if the number of microphones and loudspeakers is zero
    total_n = mcs_n + lds_n
    if total_n == 0:
        return torch.empty((0, 2)), torch.empty((0, 2))

    # Compute grid resolution
    cols = int(torch.ceil(torch.sqrt(torch.tensor(total_n).float())).item())
    rows = int(torch.ceil(total_n / cols))
    while rows * cols < total_n:
        cols += 1
        rows = int(torch.ceil(total_n / cols))

    # Generate 2D grid points over area
    dx = 0.5 * dim_1 / cols
    x_vals = torch.linspace(dx, dim_1 - dx, cols)
    dy = 0.5 * dim_2 / rows
    y_vals = torch.linspace(dy, dim_2 - dy, rows)
    xx, yy = torch.meshgrid(x_vals, y_vals, indexing='ij')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Prioritize center position
    center = torch.tensor([dim_1 / 2, dim_2 / 2])
    dists = torch.norm(grid_points - center, dim=1)
    sorted_indices = torch.argsort(dists)
    grid_points = grid_points[sorted_indices[:total_n]]

    # Farthest point sampling to select microphone positions
    mcs_pos = []
    available = grid_points.clone()

    # Start with center-most point
    if mcs_n == 0:
        mcs_pos = torch.empty((0, 2))
        lds_pos = grid_points[:lds_n]
        return lds_pos, mcs_pos
    else:
        mcs_pos.append(available[0])
        mcs_idxs = {0}

    while (len(mcs_pos) < mcs_n).item():
        remaining_indices = [i for i in range(total_n) if i not in mcs_idxs]
        remaining_points = available[remaining_indices]
        current_mics = torch.stack(mcs_pos)
        dists = torch.cdist(remaining_points.unsqueeze(0), current_mics.unsqueeze(0)).squeeze(0)
        min_dists = dists.min(dim=1).values
        next_idx_in_remaining = torch.argmax(min_dists).item()
        next_idx = remaining_indices[next_idx_in_remaining]
        mcs_pos.append(available[next_idx])
        mcs_idxs.add(next_idx)

    mcs_pos = torch.stack(mcs_pos)

    # Assign remaining points to loudspeakers
    all_idxs = set(range(total_n))
    lds_idxs = list(all_idxs - mcs_idxs)
    lds_pos = available[lds_idxs]

    return lds_pos, mcs_pos

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
    edc = pr.edc.energy_decay_curve_chu(data=pf_rir, time_shift=False)
    rt = pr.parameters.reverberation_time_linear_regression(energy_decay_curve=edc, T=decay_interval)

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


# ==================================================================
# ========================== VIRTUAL ROOM ==========================

def one_pole_filter(mag_DC: float, mag_NY: float) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Returns the coefficients of a one-pole absorption filter.

        **Args**:
            - mag_DC (float): The magnitude value of the filter at 0 Hz (linear scale).
            - mag_NY (float): The magnitude value of the filter at Nyquist frequency (linear scale).

        **Returns**:
            - b (torch.Tensor): The numerator coefficients of the filter transfer function.
            - a (torch.Tensor): The denominator coefficients of the filter transfer function.
    """

    b = torch.zeros(2, *(mag_DC.shape))
    a = torch.zeros(2, *(mag_DC.shape))

    r = mag_DC / mag_NY

    a1 = (1 - r) / (1 + r)
    b0 = (1 - a1) * mag_NY

    b[0,:] = b0
    a[0,:] = 1
    a[1,:] = a1

    return b, a

def resonance_filter(
        fs: int, resonance:torch.Tensor, gain:torch.Tensor, phase:torch.Tensor, t60:torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Returns the transfer function coefficients of a complex first order resonance filter.

        **Args**:
            - fs (int, optional): The sampling frequency of the signal [Hz].
            - resonance (torch.Tensor): The resonance frequency of the mode [Hz].
            - gain (torch.Tensor): The magnitude peak values [dB].
            - phase (torch.Tensor): The phase of the resonance [rad].
            - t60 (float): The reverberation time of the resonance [s].

        **Returns**:
            - b (torch.Tensor): The numerator coefficients of the filter transfer function.
            - a (torch.Tensor): The denominator coefficients of the filter transfer function.
    """

    assert resonance.shape == gain.shape == phase.shape == t60.shape, "The input tensors must have the same shape."

    if len(resonance.shape) > 1:
        b = torch.zeros(2, *resonance.shape[1:])
        a = torch.zeros(3, *resonance.shape[1:])
    else:
        b = torch.zeros(2,1)
        a = torch.zeros(3,1)

    # Normalized resonance
    res_norm = 2*torch.pi*resonance/fs

    # Pole radius
    radius = db2mag(-60 / (t60 * fs))
    # Pole
    pole = radius * torch.exp(torch.complex(torch.zeros_like(res_norm), res_norm))

    # Residue gain
    g = (1 - radius) * gain
    # Residue
    residue = g * torch.exp(torch.complex(torch.zeros_like(phase), phase))
    
    # Rational transfer function coefficients
    b[0] = 2*residue.real
    b[1] = -2*torch.real(residue*torch.conj(pole))
    a[0] = 1
    a[1] = -2*pole.real
    a[2] = torch.abs(pole)**2

    return b, a

def modal_reverb(
        fs: int, nfft: int, resonances: torch.Tensor, gains: torch.Tensor, phases: torch.Tensor, t60: torch.Tensor, alias_decay_db: float
    ) -> torch.Tensor:
    r"""
    Returns the transfer function of the modal reverb.

        **Args**:
            - fs (int): The sampling frequency of the signal [Hz]].
            - nfft (int): FFT size.
            - resonances (torch.Tensor): The resonance frequencies of the modes [Hz].
            - gains (torch.Tensor): The magnitude peak values [dB].
            - phases (torch.Tensor): The phase of the resonances [rad].
            - t60 (float): The reverberation time of the modal reverb [s].
            - alias_decay_db (float): The anti-time-aliasing decay [dB].

        **Returns**:
            - torch.Tensor: The transfer function of the modal reverb.
    """

    assert resonances.shape == gains.shape == phases.shape == t60.shape, "The input tensors must have the same shape."

    b, a = resonance_filter(fs=fs, resonance=resonances, gain=gains, phase=phases, t60=t60)

    gamma = 10 ** (-torch.abs(torch.tensor(alias_decay_db)) / nfft / 20) # TODO: This should go inside the class
    # n -> time samples / i,j -> input-ouput / k -> number
    b_aa = torch.einsum('n, n... -> n...', (gamma ** torch.arange(0, 2, 1)), b)
    a_aa = torch.einsum('n, n... -> n...', (gamma ** torch.arange(0, 3, 1)), a)

    B = torch.fft.rfft(b_aa, nfft, dim=0)
    A = torch.fft.rfft(a_aa, nfft, dim=0)
    
    return torch.div(B, A).sum(dim=-1)


# ==================================================================
# ========================= OPTIMIZATION ===========================

def system_equalization_curve(
    feedback: torch.Tensor,
    samplerate: float,
    nfft: int,
    mc_reps: int = 1000,
    smoothing_fraction: float = 1 / 6,
) -> tuple[torch.Tensor, ...]:
    """
    Compute a system equalization target curve.

    The function:
        1. Generates random Stiefel-manifold realizations.
        2. Computes the maximum absolute eigenvalue of the open loop
           for each realization and frequency.
        3. Averages the curves over the Monte Carlo realizations.
        4. Smooths the mean eigenvalue curve using a fractional-octave
           moving average.
        5. Computes and smooths the maximum singular-value curve of
           the feedback matrix using the same fractional-octave window.

    Args:
        feedback:
            Feedback transfer matrices, shape [nfft, n_M, n_L].

        samplerate:
            Sampling rate [Hz].

        mc_reps:
            Number of Monte Carlo Stiefel-manifold realizations.

        smoothing_fraction:
            Width of the smoothing window in octaves.
            E.g. 1/3 = one-third-octave smoothing,
                 1/6 = one-sixth-octave smoothing.

    Returns:
        target:
            Fractional-octave-smoothed mean maximum-eigenvalue curve.

        mean_evs_curve:
            Mean maximum-eigenvalue curve before smoothing.

        evs:
            Maximum-eigenvalue curve from the last Monte Carlo realization.
    """

    n_M = feedback.shape[1]
    n_L = feedback.shape[2]

    max_n = max(n_M, n_L)

    # ------------------------------------------------------------------
    # Monte Carlo evaluation
    # ------------------------------------------------------------------

    mean_evs_curve = torch.zeros(
        nfft,
        dtype=feedback.real.dtype,
        device=feedback.device,
    )

    max_evs_curve = torch.zeros_like(mean_evs_curve)

    for _ in range(mc_reps):

        # Random Stiefel-manifold realization
        M = torch.randn(
            max_n,
            max_n,
            dtype=feedback.real.dtype,
            device=feedback.device,
        )

        U = torch.matrix_exp(skew_matrix(M))[:n_L, :n_M]
        U = torch.complex(U, torch.zeros_like(U))

        # Open loop
        L = torch.matmul(feedback, U)

        # Maximum absolute eigenvalue at every frequency
        evs = torch.abs(torch.linalg.eigvals(L)).amax(dim=1)

        # Monte Carlo accumulation
        mean_evs_curve += evs

        # Maximum across Monte Carlo realizations
        max_evs_curve = torch.maximum(
            max_evs_curve,
            evs,
        )

    # Monte Carlo mean
    mean_evs_curve /= mc_reps

    # ------------------------------------------------------------------
    # Fractional-octave smoothing
    # ------------------------------------------------------------------

    target = fractional_octave_smooth(
        mean_evs_curve,
        samplerate=samplerate,
        fraction_octave=smoothing_fraction,
    )

    return (
        target,
        mean_evs_curve,
        evs,
    )


def fractional_octave_smooth(
    curve: torch.Tensor,
    samplerate: float,
    fraction_octave: float,
) -> torch.Tensor:
    """
    Smooth a frequency-domain curve using a centered
    fractional-octave moving average.

    The window around frequency f is

        [f / 2^(B/2), f * 2^(B/2)]

    where B = fraction_octave.

    The result has exactly the same length as the input curve.
    """

    nfft = curve.shape[0]

    # FFT frequency bins
    freqs = torch.linspace(
        0.0,
        samplerate / 2,
        nfft,
        dtype=curve.dtype,
        device=curve.device,
    )

    target = torch.empty_like(curve)

    # DC cannot be handled logarithmically.
    # Use the first few bins up to the first meaningful
    # fractional-octave window.
    target[0] = curve[0]

    # Frequency ratio corresponding to half the octave bandwidth
    ratio = 2.0 ** (fraction_octave / 2.0)

    # Cumulative sum for efficient moving averages
    cumsum = torch.cat(
        (
            torch.zeros(
                1,
                dtype=curve.dtype,
                device=curve.device,
            ),
            torch.cumsum(curve, dim=0),
        )
    )

    for i in range(1, nfft):

        f = freqs[i]

        f_low = f / ratio
        f_high = f * ratio

        # Convert frequency limits to FFT-bin indices
        i_low = torch.searchsorted(freqs, f_low)
        i_high = torch.searchsorted(
            freqs,
            f_high,
            right=True,
        )

        i_low = max(int(i_low), 0)
        i_high = min(int(i_high), nfft)

        # Average over the fractional-octave band
        target[i] = (
            cumsum[i_high] - cumsum[i_low]
        ) / (i_high - i_low)

    return target