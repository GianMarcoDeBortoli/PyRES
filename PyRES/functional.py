# ==================================================================
# ============================ IMPORTS =============================
# PyTorch
import torch
# FLAMO
from flamo import dsp
from flamo.functional import db2mag
from flamo.auxiliary.reverb import rt2slope


# ==================================================================

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


class FDN_one_pole_absorption(dsp.parallelFilter):
    r"""
    Parallel absorption filters for the FDN reverberator.
    """
    def __init__(
        self,
        channels: int=1,
        fs: int = 48000,
        nfft: int = 2**11,
        t60_DC: float = 1.0,
        t60_NY: float = 1.0,
        alias_decay_db: float = 0.0
    ):
        r"""
        Initialize the FDN absorption filters.

            **Args**:
                - channels (int, optional): The number of channels. Defaults to 1.
                - fs (int, optional): The sampling frequency of the signal [Hz]. Defaults to 48000.
                - nfft (int, optional): FFT size. Defaults to 2**11.
                - t60_DC (float, optional): The reverberation time of the FDN at 0 Hz [s]. Defaults to 1.0.
                - t60_NY (float, optional): The reverberation time of the FDN at Nyquist frequency [s]. Defaults to 1.0.
                - alias_decay_db (float, optional): The anti-time-aliasing decay [dB]. Defaults to 0.0.
        """
        super().__init__(size=(1, channels), nfft=nfft, requires_grad=False, alias_decay_db=alias_decay_db)

        self.fs = torch.tensor([fs])
        self.t60_DC = torch.tensor([t60_DC]).repeat(channels)
        self.t60_NY = torch.tensor([t60_NY]).repeat(channels)

    def get_freq_response(self):
        r"""
        Get the frequency response of the absorption filters.
        Reference: flamo.dsp.parallelFilter.get_freq_response()
        """
        self.freq_response = lambda param: self.compute_freq_response(param.squeeze())

    def compute_freq_response(self, delays: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the frequency response of the absorption filters.
        Reference: flamo.dsp.parallelFilter.compute_freq_response()
        """

        absorp_DC = self.rt2absorption(self.t60_DC, self.fs, delays)
        absorp_NY = self.rt2absorption(self.t60_NY, self.fs, delays)

        b, a = one_pole_filter(absorp_DC, absorp_NY)

        b_aa = torch.einsum('p, p... -> p...', (self.gamma ** torch.arange(0, 2, 1)), b)
        a_aa = torch.einsum('p, p... -> p...', (self.gamma ** torch.arange(0, 2, 1)), a)

        B = torch.fft.rfft(b_aa, self.nfft, dim=0)
        A = torch.fft.rfft(a_aa, self.nfft, dim=0)

        return torch.div(B, A)
    
    def rt2absorption(self, rt60: torch.Tensor, fs: int, delay_len: torch.Tensor) -> torch.Tensor:
        r"""
        Convert time in seconds of 60 dB decay to energy decay slope relative to the delay line length.

            **Args**:
                - rt60 (torch.Tensor): The reverberation time [s].
                - fs (int): The sampling frequency of the signal [Hz].
                - delays_len (torch.Tensor): The lengths of the delay lines [samples].

            **Returns**:
                - torch.Tensor: The energy decay slope relative to the delay line length.
        """
        return db2mag(delay_len * rt2slope(rt60, fs))
