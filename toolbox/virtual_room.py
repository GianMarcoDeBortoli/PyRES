# ==================================================================
# ============================ IMPORTS =============================
# Torch
import torch
# Flamo
from flamo import dsp, system
from flamo.functional import db2mag
from flamo.auxiliary.reverb import rt2slope
# PyRES
from functional import modal_reverb, FDN_absorption


# ==================================================================
# ============================ MATRICES ============================

class unitary_parallel_connections(dsp.parallelGain):
    r"""
    Unitary parallel connections for systems with independent channels.
    """
    def __init__(
        self,
        n_M: int = 1,
        n_L: int = 1,
        nfft: int = 2**11,
        alias_decay_db: float = 0.0,
    ):
        r"""
        Initializes the unitary parallel connections.

            **Args**:
                - n_M (int): Number of system microphones.
                - n_L (int): Number of system loudspeakers.
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay in dB.
        """
        
        assert n_M == n_L, "The number of system microphones and the number of system loudspeakers must be equal for unitary_independent_connections"

        self.n_M = n_M
        self.n_L = n_L

        super().__init__(
            size = (self.n_M,),
            nfft = nfft,
            requires_grad = False,
            alias_decay_db = alias_decay_db,
        )
        self.assign_value(torch.ones(self.n_M,))

class unitary_mixing_matrix(system.Series):
    r"""
    Unitary mixing matrix for systems with non-independent channels.
    """
    def __init__(
        self,
        n_M: int = 1,
        n_L: int = 1,
        nfft: int = 2**11,
        alias_decay_db: float = 0.0,
    ):
        r"""
        Initializes the unitary mixing matrix.

            **Args**:
                - n_M (int): Number of system microphones.
                - n_L (int): Number of system loudspeakers.
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay in dB.
        """
        
        self.n_M = n_M
        self.n_L = n_L
        
        coupling_1, mixing_matrix, coupling_2 = self.__components(
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )

        super().__init__(coupling_1, mixing_matrix, coupling_2)

    def __components(self, nfft: int, alias_decay_db: float) -> tuple[dsp.Gain, dsp.Matrix, dsp.Gain]:
        r"""
        Initializes the components of the unitary mixing matrix.
        The coupling matrices allow for appropriate input-output channels for systems with different numbers of microphones and loudspeakers.
        
            **Args**:
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay in dB.

            **Returns**:
                - coupling_1 (dsp.Gain): Coupling matrix 1.
                - mixing_matrix (dsp.Matrix): Mixing matrix.
                - coupling_2 (dsp.Gain): Coupling matrix 2.
        """

        max_n = torch.max(torch.tensor([self.n_M, self.n_L]))

        coupling_1 = self.__coupling(
            inputs = self.n_M,
            outputs = max_n
        )

        unitary = dsp.Matrix(
            size = (max_n, max_n),
            nfft = nfft,
            matrix_type = "orthogonal",
            requires_grad = False,
            alias_decay_db = alias_decay_db,
        )

        coupling_2 = self.__coupling(
            inputs = max_n,
            outputs = self.n_L
        )

        return coupling_1, unitary, coupling_2
    
    def __coupling(self, inputs: int, outputs: int) -> dsp.Gain:
        r"""
        Initializes the coupling matrix.

            **Args**:
                - inputs (int): Number of input channels.
                - outputs (int): Number of output channels.

            **Returns**:
                dsp.Gain: Coupling matrix.
        """

        module = dsp.Gain(
            size = (outputs, inputs),
            nfft = self.nfft,
            requires_grad = False,
            alias_decay_db = self.alias_decay_db,
        )
        module.assign_value(torch.eye(outputs, inputs))

        return module


# ==================================================================
# ==================== FINITE IMPULSE RESPONSE =====================

class random_FIRs(dsp.Filter):
    r"""
    Random FIR filter with a given order. Learnable FIR coefficients.
    """
    def __init__(
        self,
        n_M: int=1,
        n_L: int=1,
        nfft: int=2**11,
        FIR_order: int=100,
        alias_decay_db: float=0.0,
        requires_grad: bool=False
    ) -> None:
        r"""
        Initializes the random FIR filter.

            **Args**:
                - n_M (int): Number of system microphones.
                - n_L (int): Number of system loudspeakers.
                - nfft (int): FFT size.
                - FIR_order (int): FIR filter order.
                - alias_decay_db (float): Anti-time-aliasing decay in dB.
                - requires_grad (bool): Whether the filter is learnable.
        """

        self.n_M = n_M
        self.n_L = n_L
        self.FIR_order = FIR_order
        
        super().__init__(
            size=(self.FIR_order, self.n_L, self.n_M),
            nfft=nfft,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db
        )


class phase_canceling_modal_reverb(dsp.DSP):
    r"""
    Phase canceling modal reverb. Learnable phases.
    """
    def __init__(
        self,
        n_M: int=1,
        n_L: int=1,
        fs: int = 48000,
        nfft: int = 2**11,
        n_modes: int=10,
        low_f_lim: float=0,
        high_f_lim: float=500,
        t60: float=1.0,
        requires_grad: bool=False,
        alias_decay_db: float=0.0,
    ):
        r"""
        Initializes the phase canceling modal reverb.

            **Args**:
                - n_M (int): Number of system microphones.
                - n_L (int): Number of system loudspeakers.
                - fs (int): Sampling frequency.
                - nfft (int): FFT size.
                - n_modes (int): Number of modes in the modal reverb.
                - low_f_lim (float): Lowest mode frequency.
                - high_f_lim (float): Highest mode frequency.
                - t60 (float): Reverberation time of the modal reverb.
                - requires_grad (bool): Whether the filter is learnable.
                - alias_decay_db (float): Anti-time-aliasing decay in dB.
        """

        super().__init__(
            size=(1, n_L, n_M, n_modes,),
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

            **Arguments**:
                - **x** (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.
                - **ext_param** (torch.Tensor, optional): Parameter values received from external modules (hyper conditioning). Default: None.

            **Returns**:
                torch.Tensor: Output tensor of shape :math:`(B, M, N_{out}, ...)`.
        """
        self.check_input_shape(x)
        if ext_param is None:
            return self.freq_convolve(x, self.param)
        else:
            # log the parameters that are being passed
            with torch.no_grad():
                self.assign_value(ext_param)
            return self.freq_convolve(x, ext_param)
        
    def check_input_shape(self, x):
        r"""
        Checks if the dimensions of the input tensor x are compatible with the module.

            **Arguments**:
                **x** (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.
        """
        if (int(self.nfft / 2 + 1), self.input_channels) != (x.shape[1], x.shape[2]):
            raise ValueError(
                f"parameter shape = {self.freq_response.shape} not compatible with input signal of shape = ({x.shape})."
            )

    def check_param_shape(self):
        r"""
        Checks if the shape of the filter parameters is valid.
        """
        assert (
            len(self.size) == 4
        ), "Filter must be 3D, for 2D (parallel) filters use ParallelFilter module."

    
    def init_param(self):
        torch.nn.init.uniform_(self.param, a=0, b=2*torch.pi)

    def get_freq_response(self):

        self.freq_response = lambda param: modal_reverb(fs=self.fs, nfft=self.nfft, resonances=self.resonances, gains=self.gains, phases=param, t60=self.t60, alias_decay_db=self.alias_decay_db)

    def get_freq_convolve(self):
        r"""
        Computes the frequency convolution function.

        The frequency convolution is computed using the :func:`torch.einsum` function.

            **Arguments**:
                **x** (torch.Tensor): Input tensor.

            **Returns**:
                torch.Tensor: Output tensor after frequency convolution.
        """
        self.freq_convolve = lambda x, param: torch.einsum(
            "fmn,bfn...->bfm...", self.freq_response(param), x
        )

    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = self.size[-2]
        self.output_channels = self.size[-3]

    def initialize_class(self):
        r"""
        Initializes the class.
        """
        self.init_param()
        self.get_gamma()
        self.check_param_shape()
        self.get_io()
        self.get_freq_response()
        self.get_freq_convolve()


class wgn_reverb(dsp.DSP):
    # TODO: After modifying wgn_reverb in FLAMO, this class has RT and mean and/or std as parameters. Non-learnable...
    # TODO: wgn_reverb is used in get_freq_response()
    def __init__(
        self,
        n_M: int=1,
        n_L: int=1,
        nfft: int=2**11,
        RT: float=1.0,
        mean: float=0.0,
        std: float=1.0,
        requires_grad: bool=False
    ):
        pass

# ==================================================================
# =================== INFINITE IMPULSE RESPONSE ====================

class FDN(system.Series):
    r"""
    Feedback delay network.
    """
    def __init__(
        self,
        n_M: int=1,
        n_L: int=1,
        fs: int=48000,
        nfft: int=2**11,
        t60_DC: float=1.0,
        t60_NY: float=1.0,
        alias_decay_db: float=0.0,
    ):
        r"""
        Initializes the feedback delay network.
        
            **Args**:
                - n_M (int): Number of system microphones.
                - n_L (int): Number of system loudspeakers.
                - fs (int): Sampling frequency.
                - nfft (int): FFT size.
                - t60_DC (float): Reverberation time at 0 Hz.
                - t60_NY (float): Reverberation time at Nyquist frequency.
                - alias_decay_db (float): Anti-time-aliasing decay in dB.
                - requires_grad (bool): Whether the filter is learnable.
        """
        
        self.n_M = n_M
        self.n_L = n_L
        self.fs = fs
        self.nfft = nfft
        self.t60_DC = t60_DC
        self.t60_NY = t60_NY
        self.alias_decay_db = alias_decay_db

        max_n = torch.max(torch.tensor([n_M, n_L]))

        input_gains = self.__gains(
            inputs = self.n_M,
            outputs = max_n,
        )

        recursion = self.__recursion(
            channels = max_n,
        )

        output_gains = self.__gains(
            inputs = max_n,
            outputs = self.n_L,

        )

        super().__init__(input_gains, recursion, output_gains)

    def __gains(self, inputs: int, outputs: int) -> dsp.Gain:
        r"""
        Initializes a unitary parallel gain module.

            **Args**:
                - inputs (int): Number of input channels.
                - outputs (int): Number of output channels.
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay in dB.

            **Returns**:
                dsp.Gain: Unitary parallel gain module.
        """

        module = dsp.Gain(
            size = (outputs, inputs),
            nfft = self.nfft,
            requires_grad = False,
            alias_decay_db = self.alias_decay_db,
        )
        module.assign_value(torch.eye(outputs, inputs))

        return module
    
    def __recursion(self, channels: int) -> system.Recursion:
        r"""
        Initializes the recursive part of the feedback delay network.

            **Args**:
                - channels (int): Number of channels.
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay in dB.

            **Returns**:
                system.Recursion: Recursive part of the feedback delay network.
        """

        delays = dsp.parallelDelay(
            size = (channels,),
            max_len = 2000,
            nfft = self.nfft,
            isint = True,
            requires_grad = False,
            alias_decay_db = self.alias_decay_db,
        )
        delay_lengths = delays.s2sample(delays.param)

        feedback_matrix = dsp.Matrix(
            size = (channels, channels),
            nfft = self.nfft,
            matrix_type = "orthogonal",
            requires_grad = False,
            alias_decay_db = self.alias_decay_db,
        )

        attenuation = FDN_absorption(
            channels = channels,
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


class unitary_reverberator(system.Series):
    r"""
    Unitary reverberator.
    Reference:
        Poletti, Mark. "A unitary reverberator for reduced colouration in assisted reverberation systems."
        INTER-NOISE and NOISE-CON Congress and Conference Proceedings. Vol. 1995. No. 5. Institute of Noise Control Engineering, 1995.
    """
    def __init__(
        self,
        n_M: int=1,
        n_L: int=1,
        fs: int=48000,
        nfft: int=2**11,
        t60: float=1.0,
        alias_decay_db: float=0.0,
    ):
        r"""
        Initializes the unitary reverberator.
        """
        
        self.n_M = n_M
        self.n_L = n_L
        self.fs = fs
        self.nfft = nfft
        self.t60 = t60
        self.alias_decay_db = alias_decay_db

        coupling_in, recursion, feedforward, coupling_out  = self.__components()
        
        super().__init__(coupling_in, recursion, feedforward, coupling_out)

    def __components(self) -> tuple[dsp.Gain, system.Recursion, dsp.Filter, dsp.Gain]:
        r"""
        Initializes the components of the unitary reverberator.

            **Returns**:
                - coupling_in (dsp.Gain): Input coupling matrix.
                - recursion (system.Recursion): Recursive part of the reverberator.
                - feedforward (dsp.Filter): Feedforward part of the reverberator.
                - coupling_out (dsp.Gain): Output coupling matrix.
        """

        max_n = torch.max(torch.tensor([self.n_M, self.n_L]))

        D, delay_lengths = self.__delays(channels=max_n)
        C = self.__mixing_matrix(channels=max_n)

        gamma = db2mag(delay_lengths.mean() * rt2slope(self.t60, self.fs))
        G = self.__gains(channels=max_n, g=gamma)

        recursion = self.__recursion(channels=max_n, delays=D, mixing_matrix=C, gains=G)
        feedforward = self.__feedforward(channels=max_n, delay_lines=delay_lengths, mixing_matrix=C, gamma=gamma)

        coupling_in = self.__coupling(self, inputs=max_n, outputs=self.n_L)
        coupling_out = self.__coupling(self, inputs=self.n_M, outputs=max_n)

        return coupling_in, recursion, feedforward, coupling_out
    
    def __delays(self, channels: int, nfft: int, alias_decay_db: int) -> tuple[dsp.parallelDelay, torch.Tensor]:
        r"""
        Initialize the delays module of the unitary reverberator.

            **Args**:
                - channels (int): Number of channels.
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay in dB.

            **Returns**:
                - dsp.parallelDelay: Delays module.
                - torch.Tensor: Delay lengths.
        """

        module = dsp.parallelDelay(
            size = (channels,),
            max_len = 2000,
            nfft = nfft,
            isint = True,
            requires_grad = False,
            alias_decay_db = alias_decay_db,
        )

        delay_lengths = module.s2sample(module.param.clone().detach())

        return module, delay_lengths
    
    def __mixing_matrix(self, channels: int) -> dsp.Matrix:
        r"""
        Initializes the mixing matrix module of the unitary reverberator.

            **Args**:
                - channels (int): Number of channels.

            **Returns**:
                dsp.Matrix: Mixing matrix module.
        """

        module = dsp.Matrix(
            size = (channels, channels),
            nfft = self.nfft,
            matrix_type = "orthogonal",
            requires_grad = False,
            alias_decay_db = self.alias_decay_db,
        )

        return module
    
    def __gains(self, channels: int, g: torch.Tensor) -> dsp.parallelGain:
        r"""
        Initializes the gains module of the unitary reverberator.

            **Args**:
                - channels (int): Number of channels.
                - g (torch.Tensor): Gain value.

            **Returns**:
                dsp.Gain: Gains module.
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
                - D (dsp.parallelDelay): Delays module.
                - C (dsp.Matrix): Mixing matrix module.
                - G (dsp.Gain): Gains

            **Returns**:
                system.Recursion: Recursive part of the reverberator.
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
                - order (int): Order of the FIR filter.
                - delays (dsp.parallelDelay): Delays module.
                - mixing_matrix (dsp.Matrix): Mixing matrix module.
                - gamma (torch.Tensor): Gain value.

            **Returns**:
                dsp.Filter: Feedforward part of the reverberator.
        """

        order = delay_lines.max().int(), 

        feedforward = dsp.Filter(
            size = (order, channels, channels),
            nfft = self.nfft,
            requires_grad = False,
            alias_decay_db = self.alias_decay_db,
        )

        first_tap = gamma * torch.eye(channels)
        second_tap = mixing_matrix.param.clone().detach()
        second_tap_idxs = delay_lines.unsqueeze(1).expand(-1, channels).long()
        new_params = torch.zeros_like(feedforward.param)
        new_params[0, :, :] = first_tap
        new_params[second_tap_idxs - 1, :, :] = second_tap

        feedforward.assign_value(new_params)

        return feedforward
    
    def __coupling(self, inputs: int, outputs: int) -> dsp.Gain:
        r"""
        Initializes the coupling matrix.

            **Args**:
                - inputs (int): Number of input channels.
                - outputs (int): Number of output channels.

            **Returns**:
                dsp.Gain: Coupling matrix
        """

        module = dsp.Gain(
            size = (outputs, inputs),
            nfft = self.nfft,
            requires_grad = False,
            alias_decay_db = self.alias_decay_db,
        )
        module.assign_value(torch.eye(outputs, inputs))

        return module
