# ==================================================================
# ============================ IMPORTS =============================
# PyTorch
import torch
# FLAMO
from flamo import dsp, system
from flamo.functional import db2mag, skew_matrix
from flamo.auxiliary.reverb import rt2slope
# PyRES
from pyres.functional import modal_reverb, one_pole_filter


# ==================================================================
# =========================== BASE CLASS ===========================

class VrRoom(object):
    r"""
    Base class for virtual-room implementations.
    """
    def __init__(
        self,
        n_M: int,
        n_L: int,
        fs: int,
        nfft: int,
        alias_decay_db: float=0.0
    ):
        r"""
        Initializes the virtual room.

            **Args**:
                - n_M (int): Number of system microphones.
                - n_L (int): Number of system loudspeakers.
                - fs (int): Sampling frequency [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].

            **Attributes**:
                - n_M (int): Number of system microphones.
                - n_L (int): Number of system loudspeakers.
                - fs (int): Sampling frequency [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - v_ML (flamo.system.Series): Virtual room DSP.
                - G (flamo.dsp.parallelGain): System gain (master gain of the audio setup).
        """

        object.__init__(self)
        
        self.n_M = n_M
        self.n_L = n_L
        self.fs = fs
        self.nfft = nfft
        self.alias_decay_db = alias_decay_db

        self.v_ML: system.Series

    def get_v_ML(self) -> system.Series:
        r"""
        Returns the virtual room DSP.

            **Returns**:
                - flamo.system.Series: Virtual room DSP.
        """
        return self.v_ML

    def coupling(self, inputs: int, outputs: int, connections: str = 'mixing', requires_grad: bool=False) -> dsp.Gain:
        r"""
        Initializes the coupling matrix.
        It connects the virtual room to the physical room in case of n_L ~= n_M.
        One should be used in input, between physical room and virtual room, and one should be used in output, between virtual room and physical room

            **Args**:
                - inputs (int): Number of input channels.
                - outputs (int): Number of output channels.
                - connections (str): Type of connections. Default is 'mixing', which means that the coupling matrix is a mixing matrix.
                                     The other option is 'parallel', which means that the coupling matrix is an identity matrix.

            **Returns**:
                - flamo.dsp.Gain: Coupling matrix.
        """

        module = dsp.Gain(
            size = (outputs, inputs),
            nfft = self.nfft,
            alias_decay_db = self.alias_decay_db,
            requires_grad=requires_grad
        )
        if connections == 'mixing':
            # Mixing matrix
            max_n = torch.max(torch.tensor([outputs, inputs]))
            M = torch.randn(max_n, max_n)
            O = torch.matrix_exp(skew_matrix(M))[:outputs, :inputs]
            module.assign_value(O)
        elif connections == 'parallel':
            # Identity matrix
            module.assign_value(torch.eye(outputs, inputs))
        else:
            raise ValueError(f"Unknown connections type: {connections}. Use 'mixing' or 'parallel'.")

        return module


# ==================================================================
# ============================ MATRICES ============================

class unitary_parallel_connections(VrRoom):
    r"""
    Unitary parallel connections for systems with independent channels.
    It can only be paired to a physical room with n_L = n_M.
    """
    def __init__(
        self,
        n_M: int = 1,
        n_L: int = 1,
        fs: int = 48000,
        nfft: int = 2**11,
        alias_decay_db: float = 0.0
    ):
        r"""
        Initializes the unitary parallel connections.

            **Args**:
                - n_M (int): Number of system microphones.
                - n_L (int): Number of system loudspeakers.
                - fs (int): Sampling frequency [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].

            **Attributes**:
                [- VrRoom attributes]
        """
        
        assert n_M == n_L, "The number of system microphones and the number of system loudspeakers must be equal for unitary_independent_connections"

        VrRoom.__init__(
            self,
            n_M=n_M,
            n_L=n_L,
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )

        self.v_ML = system.Series(
            self.coupling(
                inputs = self.n_M,
                outputs = self.n_L,
                connections = 'parallel'
            )
        )


class unitary_mixing_matrix(VrRoom):
    r"""
    Unitary mixing matrix for systems with non-independent channels.
    """
    def __init__(
        self,
        n_M: int = 1,
        n_L: int = 1,
        fs: int = 48000,
        nfft: int = 2**11,
        alias_decay_db: float = 0.0,
        requires_grad: bool = False
    ):
        r"""
        Initializes the unitary mixing matrix.

            **Args**:
                - n_M (int): Number of system microphones.
                - n_L (int): Number of system loudspeakers.
                - fs (int): Sampling frequency [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].

            **Attributes**:
                [- VrRoom attributes]
        """

        VrRoom.__init__(
            self,
            n_M=n_M,
            n_L=n_L,
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )

        self.v_ML = system.Series(
            self.coupling(
                inputs = self.n_M,
                outputs = self.n_L,
                connections = 'mixing',
                requires_grad=requires_grad
            )
        )


# ==================================================================
# ==================== FINITE IMPULSE RESPONSE =====================

class random_FIRs(VrRoom):
    r"""
    Random FIR filter of given order. Learnable coefficients.
    Reference:
        De Bortoli, G., Dal Santo, G., Prawda, K., Lokki, T., Välimäki, V., and Schlecht, S. J.
        "Differentiable Active Acoustics: Optimizing Stability via Gradient Descent"
        Proceedings of the International Conference on Digital Audio Effects, pp. 254-261, 2024.
    """
    def __init__(
        self,
        n_M: int=1,
        n_L: int=1,
        fs: int=48000,
        nfft: int=2**11,
        alias_decay_db: float=0.0,
        FIR_order: int=100,
        requires_grad: bool=False
    ) -> None:
        r"""
        Initializes the random FIR filter.

            **Args**:
                - n_M (int): Number of system microphones.
                - n_L (int): Number of system loudspeakers.
                - fs (int): Sampling frequency [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - FIR_order (int): FIR filter order.
                - requires_grad (bool): Whether the filter is learnable.

            **Attributes**:
                [- VrRoom attributes]
                - FIR_order (int): FIR filter order.
        """

        VrRoom.__init__(
            self,
            n_M=n_M,
            n_L=n_L,
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )
        self.FIR_order = FIR_order
        
        self.v_ML = system.Series(
            dsp.Filter(
                size=(self.FIR_order, self.n_L, self.n_M),
                nfft=self.nfft,
                requires_grad=requires_grad,
                alias_decay_db=self.alias_decay_db
            )
        )


class phase_cancellation(VrRoom):
    r"""
    Phase cancelling modal reverb. Learnable phases.
    Reference:
        De Bortoli, G., Prawda, K., and Schlecht, S. J.
        "Active Acoustics with a Phase Cancelling Modal Reverberator"
        Journal of the Audio Engineering Society, Vol. 72, No. 10, pp. 705-715, 2024.
    """
    def __init__(
        self,
        n_M: int=1,
        n_L: int=1,
        fs: int = 48000,
        nfft: int = 2**11,
        alias_decay_db: float=0.0,
        n_modes: int=10,
        low_f_lim: float=0,
        high_f_lim: float=500,
        t60: float=1.0,
        requires_grad: bool=False
    ):
        r"""
        Initializes the phase canceling modal reverb.

            **Args**:
                - n_M (int): Number of system microphones.
                - n_L (int): Number of system loudspeakers.
                - fs (int): Sampling frequency [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - n_modes (int): Number of modes in the modal reverb.
                - low_f_lim (float): Lowest mode frequency [Hz].
                - high_f_lim (float): Highest mode frequency [Hz].
                - t60 (float): Reverberation time of the modal reverb [s].
                - requires_grad (bool): Whether the filter is learnable.
            
            **Attributes**:
                [- VrRoom attributes]
        """

        VrRoom.__init__(
            self,
            n_M=n_M,
            n_L=n_L,
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )

        self.v_ML = system.Series(
            phase_cancelling_modal_reverb(
                size=(n_L, n_M),
                fs=fs,
                nfft=nfft,
                alias_decay_db=alias_decay_db,
                n_modes=n_modes,
                low_f_lim=low_f_lim,
                high_f_lim=high_f_lim,
                t60=t60,
                requires_grad=requires_grad
            )
        )


# ==================================================================
# ======================= RECURSIVE FILTERS ========================

class FDN(VrRoom):
    r"""
    Feedback delay network.
    """
    def __init__(
        self,
        n_M: int=1,
        n_L: int=1,
        fs: int=48000,
        nfft: int=2**11,
        alias_decay_db: float=0.0,
        order: int=4,
        t60_DC: float=1.0,
        t60_NY: float=1.0
    ):
        r"""
        Initializes the feedback delay network.
        
            **Args**:
                - n_M (int): Number of system microphones.
                - n_L (int): Number of system loudspeakers.
                - fs (int): Sampling frequency [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - order (int): Order of the feedback delay network.
                - t60_DC (float): Reverberation time at 0 Hz [s].
                - t60_NY (float): Reverberation time at Nyquist frequency [s].

            **Attributes**:
                [- VrRoom attributes]
                - order (int): Order of the feedback delay network.
                - t60_DC (float): Reverberation time at 0 Hz [s].
                - t60_NY (float): Reverberation time at Nyquist frequency [s].
        """

        VrRoom.__init__(
            self,
            n_M=n_M,
            n_L=n_L,
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )

        input_gains = self.coupling(
            inputs = self.n_M,
            outputs = order,
            connections = 'mixing',
            requires_grad=False
        )

        self.order = order
        self.t60_DC = t60_DC
        self.t60_NY = t60_NY
        recursion = self.__recursion()

        output_gains = self.coupling(
            inputs = order,
            outputs = self.n_L,
            connections = 'mixing',
            requires_grad=False
        )

        self.v_ML = system.Series(
            input_gains,
            recursion,
            output_gains
        )
    
    def __recursion(self) -> system.Recursion:
        r"""
        Initializes the recursive part of the feedback delay network.

            **Returns**:
                - system.Recursion: Recursive part of the feedback delay network.
        """

        delays = dsp.parallelDelay(
            size = (self.order,),
            max_len = 2000,
            nfft = self.nfft,
            isint = True,
            requires_grad = False,
            alias_decay_db = self.alias_decay_db,
        )
        delay_lengths = torch.randint(700, 3000, (self.order,)).float()
        delays.assign_value(delays.sample2s(delay_lengths))

        feedback_matrix = dsp.Matrix(
            size = (self.order, self.order),
            nfft = self.nfft,
            matrix_type = "orthogonal",
            requires_grad = False,
            alias_decay_db = self.alias_decay_db,
        )
        M = feedback_matrix.param.clone().detach()
        O = torch.matrix_exp(skew_matrix(M))
        feedback_matrix.assign_value(O)

        attenuation = FDN_one_pole_absorption(
            channels = self.order,
            fs = self.fs,
            nfft = self.nfft,
            t60_DC = self.t60_DC,
            t60_NY = self.t60_NY,
            alias_decay_db = self.alias_decay_db,
        )
        attenuation.assign_value(delay_lengths.view(1,-1))

        recursion = system.Recursion(
            fF = system.Series(delays, attenuation),
            fB = feedback_matrix,
        )

        return recursion


class unitary_reverberator(VrRoom):
    r"""
    Unitary reverberator.
    Reference:
        Poletti, M.
        "A unitary reverberator for reduced colouration in assisted reverberation systems."
        INTER-NOISE and NOISE-CON Congress and Conference Proceedings. Vol. 1995. No. 5. Institute of Noise Control Engineering, 1995.
    """
    def __init__(
        self,
        n_M: int=1,
        n_L: int=1,
        fs: int=48000,
        nfft: int=2**11,
        alias_decay_db: float=0.0,
        order: int=4,
        t60: float=1.0
    ):
        r"""
        Initializes the unitary reverberator.

            **Args**:
                - n_M (int): Number of system microphones.
                - n_L (int): Number of system loudspeakers.
                - fs (int): Sampling frequency [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - order (int): Size of the reverberator structure.
                - t60 (float): Reverberation time [s].

            **Attributes**:
                [- VrRoom attributes]
                - t60 (float): Reverberation time [s].
        """

        VrRoom.__init__(
            self,
            n_M=n_M,
            n_L=n_L,
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )

        self.order = order
        self.t60 = t60

        coupling_in, recursion, feedforward, coupling_out  = self.__components()
        
        self.v_ML = system.Series(
            coupling_in,
            recursion,
            feedforward,
            coupling_out
        )

    def __components(self) -> tuple[dsp.Gain, system.Recursion, dsp.Filter, dsp.Gain]:
        r"""
        Initializes the components of the unitary reverberator.

            **Returns**:
                - coupling_in (dsp.Gain): Input coupling matrix.
                - recursion (system.Recursion): Recursive part of the reverberator.
                - feedforward (dsp.Filter): Feedforward part of the reverberator.
                - coupling_out (dsp.Gain): Output coupling matrix.
        """
        D, delay_lengths = self.__delays(channels=self.order)
        C = self.coupling(inputs=self.order, outputs=self.order, connections='mixing', requires_grad=False)

        gamma = db2mag(delay_lengths.mean() * rt2slope(self.t60, self.fs))
        G = self.__gains(channels=self.order, g=gamma)

        recursion = self.__recursion(channels=self.order, delays=D, mixing_matrix=C, gains=G)
        feedforward = self.__feedforward(channels=self.order, delay_lines=delay_lengths, mixing_matrix=C, gamma=gamma)

        coupling_in = self.coupling(inputs=self.n_M, outputs=self.order, connections='parallel', requires_grad=False)
        coupling_out = self.coupling(inputs=self.order, outputs=self.n_L, connections='parallel', requires_grad=False)

        return coupling_in, recursion, feedforward, coupling_out
    
    def __delays(self, channels: int) -> tuple[dsp.parallelDelay, torch.Tensor]:
        r"""
        Initialize the delays module of the unitary reverberator.

            **Args**:
                - channels (int): Number of channels.

            **Returns**:
                - dsp.parallelDelay: Delay processing module.
                - torch.Tensor: Delay lengths.
        """

        module = dsp.parallelDelay(
            size = (channels,),
            max_len = 2000,
            nfft = self.nfft,
            isint = True,
            requires_grad = False,
            alias_decay_db = self.alias_decay_db,
        )
        delay_lengths = torch.randint(700,3000,(channels,)).float()
        module.assign_value(module.sample2s(delay_lengths))

        return module, delay_lengths
    
    def __gains(self, channels: int, g: torch.Tensor) -> dsp.parallelGain:
        r"""
        Initializes the gains module of the unitary reverberator.

            **Args**:
                - channels (int): Number of channels.
                - g (torch.Tensor): Gain value.

            **Returns**:
                - dsp.Gain: Gains module.
        """

        module = dsp.parallelGain(
            size = (channels,),
            nfft = self.nfft,
            requires_grad = False,
            alias_decay_db = self.alias_decay_db,
        )
        module.assign_value(g * torch.ones(channels,))

        return module
    
    def __recursion(self, channels:int, delays: dsp.parallelDelay, mixing_matrix: dsp.Matrix, gains: dsp.parallelGain) -> system.Recursion:
        r"""
        Initializes the recursion module of the unitary reverberator.

            **Args**:
                - channels (int): Number of channels.
                - D (dsp.parallelDelay): Delay processing module.
                - C (dsp.Matrix): Mixing-matrix processing module.
                - G (dsp.Gain): Gain processing module

            **Returns**:
                - system.Recursion: Recursive part of the reverberator.
        """

        identity = dsp.parallelGain(
            size = (channels,),
            nfft = self.nfft,
            requires_grad = False,
            alias_decay_db = self.alias_decay_db,
        )
        identity.assign_value(torch.ones(channels,))

        recursion = system.Recursion(
            fF = identity,
            fB = system.Series(delays, mixing_matrix, gains)
        )

        return recursion
    
    def __feedforward(self, channels: int, delay_lines: torch.Tensor, mixing_matrix: dsp.Matrix, gamma: torch.Tensor) -> dsp.Filter:
        r"""
        Initializes the feedforward module of the unitary reverberator.

            **Args**:
                - channels (int): Number of channels.
                - delay_lines (torch.Tensor): Delay lengths [samples].
                - mixing_matrix (dsp.Matrix): Mixing-matrix processing module.
                - gamma (torch.Tensor): Gain value.

            **Returns**:
                - dsp.Filter: Feedforward part of the reverberator.
        """

        order = torch.max(delay_lines).int()

        feedforward = dsp.Filter(
            size=(order, channels, channels),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db,
        )

        first_tap = gamma * torch.eye(channels)
        second_tap = mixing_matrix.param.clone().detach()
        second_tap_idxs = delay_lines.unsqueeze(0).expand(channels, -1).long()
        new_params = torch.zeros_like(feedforward.param)
        for i in range(channels):
            for j in range(channels):
                new_params[0, i, j] = first_tap[i, j]
                new_params[second_tap_idxs[i, j]-1, i, j] = second_tap[i, j]

        feedforward.assign_value(new_params)

        return feedforward
    

# ==================================================================
# ======================= AUXILIARY CLASSES ========================

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
    

class phase_cancelling_modal_reverb(dsp.DSP):
    r"""
    Phase cancelling modal reverb. Learnable phases.
    Reference:
        De Bortoli, G., Prawda, K., and Schlecht, S. J.
        Active Acoustics with a Phase Cancelling Modal Reverberator
        Journal of the Audio Engineering Society, 2024.
    """
    def __init__(
        self,
        size: tuple[int, int] = (1, 1),
        fs: int = 48000,
        nfft: int = 2**11,
        alias_decay_db: float=0.0,
        n_modes: int=10,
        low_f_lim: float=0,
        high_f_lim: float=500,
        t60: float=1.0,
        requires_grad: bool=False
    ):
        r"""
        Initializes the phase canceling modal reverb.

            **Args**:
                - n_M (int): Number of system microphones.
                - n_L (int): Number of system loudspeakers.
                - fs (int): Sampling frequency [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - n_modes (int): Number of modes in the modal reverb.
                - low_f_lim (float): Lowest mode frequency [Hz].
                - high_f_lim (float): Highest mode frequency [Hz].
                - t60 (float): Reverberation time of the modal reverb [s].
                - requires_grad (bool): Whether the filter is learnable.
        """
        dsp.DSP.__init__(
            self,
            size=(1, *size, n_modes),
            nfft=nfft,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db
        )

        self.fs = fs

        self.n_modes = n_modes
        self.resonances = torch.linspace(low_f_lim, high_f_lim, n_modes).view(
            -1, *(1,)*(len(self.param.shape[:-1]))).permute(
                [1,2,3,0]).expand(
                    *self.param.shape)
        self.gains = torch.ones_like(self.param)
        self.t60 = t60 * torch.ones_like(self.param)

        self.initialize_class()

    def forward(self, x, ext_param=None):
        r"""
        Applies the Filter module to the input tensor x.
        Reference: FLAMO.dsp.Filter.forward()
        """
        self.check_input_shape(x)
        if ext_param is None:
            return self.freq_convolve(x, self.param)
        else:
            with torch.no_grad():
                self.assign_value(ext_param)
            return self.freq_convolve(x, ext_param)
        
    def check_input_shape(self, x):
        r"""
        Checks if the dimensions of the input tensor x are compatible with the module.
        Reference: FLAMO.dsp.Filter.check_input_shape()
        """
        if (int(self.nfft / 2 + 1), self.input_channels) != (x.shape[1], x.shape[2]):
            raise ValueError(
                f"parameter shape = {self.freq_response.shape} not compatible with input signal of shape = ({x.shape})."
            )

    def check_param_shape(self):
        r"""
        Checks if the shape of the filter parameters is valid.
        Reference: FLAMO.dsp.Filter.check_param_shape()
        """
        assert (
            len(self.size) == 4
        ), "Filter must be 3D, for 2D (parallel) filters use ParallelFilter module."

    
    def init_param(self):
        r"""
        Initializes the filter parameters.
        Reference: FLAMO.dsp.Filter.init_param()
        """
        torch.nn.init.uniform_(self.param, a=0, b=2*torch.pi)

    def get_freq_response(self):
        r"""
        Computes the frequency response of the filter.
        Reference: FLAMO.dsp.Filter.get_freq_response()
        """
        self.freq_response = lambda param: modal_reverb(fs=self.fs, nfft=self.nfft, resonances=self.resonances, gains=self.gains, phases=param, t60=self.t60, alias_decay_db=self.alias_decay_db)

    def get_freq_convolve(self):
        r"""
        Computes the frequency convolution function.
        Reference: FLAMO.dsp.Filter.get_freq_convolve()
        """
        self.freq_convolve = lambda x, param: torch.einsum(
            "fmn,bfn...->bfm...", self.freq_response(param), x
        )

    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        Reference: FLAMO.dsp.Filter.get_io()
        """
        self.input_channels = self.size[-2]
        self.output_channels = self.size[-3]

    def initialize_class(self):
        r"""
        Initializes the class.
        Reference: FLAMO.dsp.Filter.initialize_class()
        """
        self.init_param()
        self.get_gamma()
        self.check_param_shape()
        self.get_io()
        self.get_freq_response()
        self.get_freq_convolve()