# ==================================================================
# ============================ IMPORTS =============================
from collections import OrderedDict
# PyTorch
import torch
import torch.nn as nn
# FLAMO
from flamo import dsp
from flamo.functional import mag2db, WGN_reverb
# PyRES
from PyRES.dataset_api import (
    get_hl_info,
    get_ll_info,
    get_rirs,
    normalize_rirs,
    get_transducer_number,
    get_transducer_positions
)
from PyRES.functional import energy_coupling, direct_to_reverb_ratio
from PyRES.functional import simulate_setup
from PyRES.utils import find_direct_path
from PyRES.plots import (
    plot_room_setup,
    plot_coupling,
    plot_DRR,
    plot_distributions
)


# ==================================================================
# =========================== BASE CLASS ===========================

class PhRoom(object):
    r"""
    Base class for physical-room implementations.
    """
    def __init__(
            self,
            fs: int,
            nfft: int,
            alias_decay_db: float
        ) -> None:
        r"""
        Initializes the PhRoom object.

            **Args**:
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].

            **Attributes**:
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - transducer_number (OrderedDict): Number of transducers in the room.
                - transducer_indices (OrderedDict): Indices of the requested stage emitters, system receivers, system emitters and audience receivers.
                - transducer_positions (OrderedDict): Positions of the requested stage emitters, system receivers, system emitters and audience receivers.
                - rir_length (int): Length of the room impulse responses in samples.
                - h_SA (nn.Module): Room impulse responses bewteen stage emitters and audience receivers.
                - h_SM (nn.Module): Room impulse responses bewteen stage emitters and system receivers.
                - h_LA (nn.Module): Room impulse responses bewteen system emitters and audience receivers.
                - h_LM (nn.Module): Room impulse responses bewteen system emitters and system receivers.
        """
        object.__init__(self)

        self.fs = fs
        self.nfft = nfft
        self.alias_decay_db = alias_decay_db

        self.transducer_number = OrderedDict(
            {'stg': int, 'mcs': int, 'lds': int, 'aud': int}
        )

        self.transducer_indices = OrderedDict(
            {'stg': list[int], 'mcs': list[int], 'lds': list[int], 'aud': list[int]}
        )

        self.transducer_positions = OrderedDict(
            {'stg': list[list[int]], 'mcs': list[list[int]], 'lds': list[list[int]], 'aud': list[list[int]]}
        )

        self.rir_length: int

        self.h_SA: nn.Module
        self.h_SM: nn.Module
        self.h_LA: nn.Module
        self.h_LM: nn.Module

        self.energy_coupling = OrderedDict(
            {'SA': torch.Tensor, 'SM': torch.Tensor, 'LM': torch.Tensor, 'LA': torch.Tensor}
        )

        self.direct_to_reverb_ratio = OrderedDict(
            {'SA': torch.Tensor, 'SM': torch.Tensor, 'LM': torch.Tensor, 'LA': torch.Tensor}
        )

    def get_ems_rcs_number(self) -> OrderedDict:
        r"""
        Returns the number of emitters and receivers.

            **Returns**:
                - OrderedDict: Number of emitters and receivers.
        """
        return self.transducer_number
    
    def get_stg_to_aud(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between stage emitters and audience receivers.

            **Returns**:
                - torch.Tensor: Stage-to-Audience RIRs. shape = (samples, n_A, n_S).
        """
        return self.h_SA

    def get_stg_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between stage emitters and system receivers.

            **Returns**:
                - torch.Tensor: Stage-to-Microphones RIRs. shape = (samples, n_M, n_S).
        """
        return self.h_SM
    
    def get_lds_to_aud(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between system emitters and audience receivers.

            **Returns**:
                - torch.Tensor: Loudspeakers-to-Audience RIRs. shape = (samples n_A, n_L).
        """
        return self.h_LA

    def get_lds_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between system emitters and system receivers.

            **Returns**:
                - torch.Tensor: Loudspeakers-to-Microphones RIRs. shape = (samples, n_M, n_L).
        """
        return self.h_LM
    
    def get_rirs(self) -> OrderedDict:
        r"""
        Returns a copy of all system room impulse responses.

            **Returns**:
                - OrderedDict: System RIRs.
        """
        RIRs = OrderedDict()
        RIRs.update({'SM': self.get_stg_to_mcs().param.clone().detach()})
        RIRs.update({'SA': self.get_stg_to_aud().param.clone().detach()})
        RIRs.update({'LM': self.get_lds_to_mcs().param.clone().detach()})
        RIRs.update({'LA': self.get_lds_to_aud().param.clone().detach()})
        return RIRs
    
    def compute_energy_coupling(self) -> OrderedDict:
        r"""
        Computes the energy coupling of the room impulse responses.

            **Returns**:
                - OrderedDict: Energy coupling of the system RIRs.
        """
        rirs = self.get_rirs()
        ec_SA = energy_coupling(rirs["SA"], fs=self.fs)
        ec_SM = energy_coupling(rirs["SM"], fs=self.fs)
        ec_LM = energy_coupling(rirs["LM"], fs=self.fs)
        ec_LA = energy_coupling(rirs["LA"], fs=self.fs)

        ec = OrderedDict()
        ec.update({'SA': ec_SA})
        ec.update({'SM': ec_SM})
        ec.update({'LM': ec_LM})
        ec.update({'LA': ec_LA})

        return ec
    
    def compute_direct_to_reverb_ratio(self) -> OrderedDict:
        r"""
        Computes the direct-to-reverberant ratio (DRR) of the room impulse responses.

            **Returns**:
                - OrderedDict: Direct-to-reverberant ratio of the system RIRs.
        """
        rirs = self.get_rirs()
        drr_SA = direct_to_reverb_ratio(rirs["SA"], fs=self.fs)
        drr_SM = direct_to_reverb_ratio(rirs["SM"], fs=self.fs)
        drr_LM = direct_to_reverb_ratio(rirs["LM"], fs=self.fs)
        drr_LA = direct_to_reverb_ratio(rirs["LA"], fs=self.fs)

        drr = OrderedDict()
        drr.update({'SA': drr_SA})
        drr.update({'SM': drr_SM})
        drr.update({'LM': drr_LM})
        drr.update({'LA': drr_LA})

        return drr
    
    def create_modules(self,
            rirs_SA: torch.Tensor,
            rirs_SM: torch.Tensor,
            rirs_LA: torch.Tensor,
            rirs_LM: torch.Tensor,
            rir_length: int
        ) -> tuple[dsp.Filter, dsp.Filter, dsp.Filter, dsp.Filter]:
        r"""
        Creates the processing modules for the room-impulse-response blocks.

            **Args**:
                - rirs_SA (torch.Tensor): Room impulse responses between stage emitters and audience receivers.
                - rirs_SM (torch.Tensor): Room impulse responses between stage emitters and system receivers.
                - rirs_LA (torch.Tensor): Room impulse responses between system emitters and audience receivers.
                - rirs_LM (torch.Tensor): Room impulse responses between system emitters and system receivers.
                - rir_length (int): Length of the room impulse responses in samples.

            **Returns**:
                - dsp.Filter: Room impulse responses bewteen stage emitters and audience receivers.
                - dsp.Filter: Room impulse responses bewteen stage emitters and system receivers.
                - dsp.Filter: Room impulse responses bewteen system emitters and audience receivers.
                - dsp.Filter: Room impulse responses bewteen system emitters and system receivers.
        """
        # Get number of transducers
        n_S = self.transducer_number['stg']
        n_M = self.transducer_number['mcs']
        n_L = self.transducer_number['lds']
        n_A = self.transducer_number['aud']

        # Stage to Audience
        h_SA = dsp.Filter(
            size=(rir_length, n_A, n_S),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        h_SA.assign_value(rirs_SA)

        # Stage to Microphones
        h_SM = dsp.Filter(
            size=(rir_length, n_M, n_S),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        h_SM.assign_value(rirs_SM)

        # Loudspeakers to Audience
        h_LM = dsp.Filter(
            size=(rir_length, n_M, n_L),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        h_LM.assign_value(rirs_LM)

        # Loudspeakers to Microphones
        h_LA = dsp.Filter(
            size=(rir_length, n_A, n_L),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        h_LA.assign_value(rirs_LA)

        return h_SA, h_SM, h_LA, h_LM
    
    def plot_setup(self) -> None:
        r"""
        Plots the room setup.
        """
        plot_room_setup(self.transducer_positions)
        return None
    
    def plot_coupling(self) -> None:
        r"""
        Plots the room coupling.
        """
        plot_coupling(energy_values=self.energy_coupling)
        return None
    
    def plot_DRR(self) -> None:
        r"""
        Plots the direct-to-reverberant ratio (DRR).
        """
        plot_DRR(direct_to_reverb_ratios=self.direct_to_reverb_ratio)
        return None

    def plot_h_LM_distributions(self, db_scale: bool=False) -> None:
        r"""
        Plots the distributions of the room impulse responses between system emitters and system receivers.

            **Args**:
                - db_scale (bool): If True, the imaginary part is converted to dB scale.
        """
        h_LM = self.get_rirs()['LM']
        H_LM = torch.fft.rfft(h_LM, n=self.nfft, dim=0)
        real = torch.real(H_LM[1:-1, :, :])  # Exclude the DC component and Nyquist frequency
        imag = torch.imag(H_LM[1:-1, :, :])
        if db_scale:
            real = mag2db(real + 1e-10)
            imag = mag2db(imag + 1e-10)

        distributions = torch.stack((real.flatten(), imag.flatten()), dim=1)

        plot_distributions(distributions=distributions, n_bins=self.nfft//100, labels=['Real', 'Imaginary'])

        return None

    def only_direct_path(self, window_duration_ms: float = 1.0) -> None:
        r"""
        Applies a hard time window (rectangular) to each impulse response so that the
        direct path is preserved while early reflections and late reverberation are
        suppressed. Uses hard cuts to preserve frequency content of kept samples.

            **Args**:
                - window_duration_ms (float): Duration of the window after the direct path
                  in milliseconds. Defaults to 5.0 ms. The direct-path onset and this
                  window are preserved unchanged, everything after is hard-zeroed.

            **Modifies**:
                - h_SA, h_SM, h_LA, h_LM: The impulse responses are modified in-place to
                  contain only the direct path with a hard cut (no frequency modification).
        """
        window_samples = int(window_duration_ms * self.fs / 1000.0)
        if window_samples <= 0:
            return None

        # Store originals (once) so that windowing is non-destructive w.r.t. the
        # raw dataset / generated RIRs.
        if not hasattr(self, "h_SA_original"):
            self.h_SA_original = self.h_SA.param.clone().detach()
        if not hasattr(self, "h_SM_original"):
            self.h_SM_original = self.h_SM.param.clone().detach()
        if not hasattr(self, "h_LA_original"):
            self.h_LA_original = self.h_LA.param.clone().detach()
        if not hasattr(self, "h_LM_original"):
            self.h_LM_original = self.h_LM.param.clone().detach()

        def _apply_hard_cut(rirs: torch.Tensor) -> torch.Tensor:
            """
            For each [receiver, emitter] pair, keep everything up to direct_path_idx + window_samples,
            then hard-zero everything after. This preserves frequency content of kept samples.
            """
            n_samples = rirs.shape[0]
            for i in range(rirs.shape[1]):  # receivers
                for j in range(rirs.shape[2]):  # emitters
                    rir = rirs[:, i, j]
                    try:
                        direct_path_idx = find_direct_path(rir, fs=self.fs)
                    except RuntimeError:
                        # If direct path cannot be found, start window from sample 0
                        direct_path_idx = 0

                    if direct_path_idx >= n_samples:
                        continue

                    # Hard cut: keep everything up to direct_path_idx + window_samples
                    cut_end = min(direct_path_idx + window_samples, n_samples)
                    
                    # Keep everything before cut_end unchanged, zero everything after
                    rirs[cut_end:, i, j] = 0.0

            return rirs

        # Process all RIR blocks
        self.h_SA.assign_value(_apply_hard_cut(self.h_SA.param.clone()))
        self.h_SM.assign_value(_apply_hard_cut(self.h_SM.param.clone()))
        self.h_LA.assign_value(_apply_hard_cut(self.h_LA.param.clone()))
        self.h_LM.assign_value(_apply_hard_cut(self.h_LM.param.clone()))

        return None

    def only_early_reflections(self, window_duration_ms: float = 50.0, suppress_direct_path_ms: int = 1.0) -> None:
        r"""
        Keeps only the early-reflection portion of each impulse response, starting
        from the direct-path onset, using hard cuts to preserve frequency content.

            **Args**:
                - window_duration_ms (float): Duration of the early-reflection window
                  starting at the direct path in milliseconds. Defaults to 50 ms.
                - suppress_direct_path_samples (int): Number of samples to zero at the start
                  to suppress the direct path. Defaults to 10 samples (~0.2 ms at 48 kHz).

            **Modifies**:
                - h_SA, h_SM, h_LA, h_LM: The impulse responses are modified in-place to
                  contain only early reflections with hard cuts (no frequency modification).
        """
        window_samples = int(window_duration_ms * self.fs / 1000.0)
        if window_samples <= 0:
            return None

        # Store originals (once) so that windowing is non-destructive w.r.t. the
        # raw dataset / generated RIRs.
        if not hasattr(self, "h_SA_original"):
            self.h_SA_original = self.h_SA.param.clone().detach()
        if not hasattr(self, "h_SM_original"):
            self.h_SM_original = self.h_SM.param.clone().detach()
        if not hasattr(self, "h_LA_original"):
            self.h_LA_original = self.h_LA.param.clone().detach()
        if not hasattr(self, "h_LM_original"):
            self.h_LM_original = self.h_LM.param.clone().detach()

        def _keep_early(rirs: torch.Tensor) -> torch.Tensor:
            """
            For each [receiver, emitter] pair, keep a window of length `window_samples`
            starting at the direct path. Zero the direct path samples, keep early reflections
            unchanged, then hard-zero everything after the window.
            """
            n_samples = rirs.shape[0]
            for i in range(rirs.shape[1]):  # receivers
                for j in range(rirs.shape[2]):  # emitters
                    rir = rirs[:, i, j]
                    try:
                        direct_path_idx = find_direct_path(rir, fs=self.fs)
                    except RuntimeError:
                        # If direct path cannot be found, just keep a window at the start
                        direct_path_idx = 0

                    start = max(0, direct_path_idx)
                    end = min(start + window_samples, n_samples)
                    win_len = end - start
                    if win_len <= 0:
                        rirs[:, i, j] = 0.0
                        continue

                    # Hard cut: zero before start, suppress direct path, keep rest of window, zero after
                    rirs[:start, i, j] = 0.0
                    rirs[end:, i, j] = 0.0
                    
                    # Suppress direct path samples (zero first few samples of the window)
                    samples_from_direct_path = int(suppress_direct_path_ms * self.fs / 1000.0)
                    suppress_end = min(start + samples_from_direct_path, end)
                    rirs[start:suppress_end, i, j] = 0.0
                    
                    # Rest of window is kept unchanged (preserves frequency content)

            return rirs

        self.h_SA.assign_value(_keep_early(self.h_SA.param.clone()))
        self.h_SM.assign_value(_keep_early(self.h_SM.param.clone()))
        self.h_LA.assign_value(_keep_early(self.h_LA.param.clone()))
        self.h_LM.assign_value(_keep_early(self.h_LM.param.clone()))

        return None

    def only_late_reverb(self, transition_duration_ms: float = 50.0) -> None:
        r"""
        Keeps only the late reverberation tail of each impulse response, starting
        after the early-reflection window, with a hard cut to preserve frequency content.

            **Args**:
                - transition_duration_ms (float): Duration after direct path before
                  keeping the late reverb, in milliseconds. Defaults to 50 ms.
                  Everything before this point is hard-zeroed, everything after is kept unchanged.

            **Modifies**:
                - h_SA, h_SM, h_LA, h_LM: The impulse responses are modified in-place to
                  contain only the late reverberation with a hard cut (no frequency modification).
        """
        transition_samples = int(transition_duration_ms * self.fs / 1000.0)
        if transition_samples <= 0:
            return None

        # Store originals (once) so that windowing is non-destructive w.r.t. the
        # raw dataset / generated RIRs.
        if not hasattr(self, "h_SA_original"):
            self.h_SA_original = self.h_SA.param.clone().detach()
        if not hasattr(self, "h_SM_original"):
            self.h_SM_original = self.h_SM.param.clone().detach()
        if not hasattr(self, "h_LA_original"):
            self.h_LA_original = self.h_LA.param.clone().detach()
        if not hasattr(self, "h_LM_original"):
            self.h_LM_original = self.h_LM.param.clone().detach()

        def _keep_late(rirs: torch.Tensor) -> torch.Tensor:
            """
            For each [receiver, emitter] pair, hard-zero everything up to an onset
            (direct path + transition region), then keep the late tail unchanged.
            """
            n_samples = rirs.shape[0]
            for i in range(rirs.shape[1]):  # receivers
                for j in range(rirs.shape[2]):  # emitters
                    rir = rirs[:, i, j]
                    try:
                        direct_path_idx = find_direct_path(rir, fs=self.fs)
                    except RuntimeError:
                        # If direct path cannot be found, treat onset as 0
                        direct_path_idx = 0

                    onset = min(direct_path_idx + transition_samples, n_samples)
                    if onset >= n_samples:
                        # No late tail available
                        rirs[:, i, j] = 0.0
                        continue

                    # Hard cut: zero everything before onset, keep everything after unchanged
                    rirs[:onset, i, j] = 0.0
                    # Everything from onset onward is already correct (unchanged)

            return rirs

        self.h_SA.assign_value(_keep_late(self.h_SA.param.clone()))
        self.h_SM.assign_value(_keep_late(self.h_SM.param.clone()))
        self.h_LA.assign_value(_keep_late(self.h_LA.param.clone()))
        self.h_LM.assign_value(_keep_late(self.h_LM.param.clone()))

        return None

    def suppress_feedback(self, gain: float = 0.0) -> None:
        r"""
        Applies a gain (in linear scale) to the h_LM impulse responses to suppress
        feedback. The original h_LM is saved as h_LM_original if it doesn't already exist.

            **Args**:
                - gain (float): Gain in linear scale to apply to h_LM. Defaults to 0.0
                  (complete suppression). A value of 1.0 means no change, 0.5 means
                  half amplitude, etc.

            **Modifies**:
                - h_LM: The impulse responses are multiplied by the gain value in-place.
        """
        # Save original h_LM if not already saved
        if not hasattr(self, "h_LM_original"):
            self.h_LM_original = self.h_LM.param.clone().detach()

        # Apply gain to h_LM
        rirs_LM = self.h_LM.param.clone()
        rirs_LM = rirs_LM * gain
        self.h_LM.assign_value(rirs_LM)

        return None

    def return_to_original(self) -> None:
        r"""
        Restores the original (unwindowed) room impulse responses if they were
        previously cached in the `h_*_original` attributes.

            **Modifies**:
                - h_SA, h_SM, h_LA, h_LM: Restored to the values stored in
                  `h_SA_original`, `h_SM_original`, `h_LA_original`, `h_LM_original`
                  when available. If a given `*_original` attribute does not exist,
                  that block is left unchanged.
        """

        if hasattr(self, "h_SA_original"):
            self.h_SA.assign_value(self.h_SA_original.clone())
        if hasattr(self, "h_SM_original"):
            self.h_SM.assign_value(self.h_SM_original.clone())
        if hasattr(self, "h_LA_original"):
            self.h_LA.assign_value(self.h_LA_original.clone())
        if hasattr(self, "h_LM_original"):
            self.h_LM.assign_value(self.h_LM_original.clone())

        return None


# ==================================================================
# ========================== DATASET CLASS =========================

class PhRoom_dataset(PhRoom):
    r"""
    Subclass of PhRoom that loads the room impulse responses from the dataset.
    """
    def __init__(
            self,
            fs: int,
            nfft: int,
            alias_decay_db: float,
            dataset_directory: str,
            room_name: str,
            stg_idx: list[int] = None,
            mcs_idx: list[int] = None,
            lds_idx: list[int] = None,
            aud_idx: list[int] = None
        ) -> None:
        r"""
        Initializes the PhRoom_dataset object.

            **Args**:
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - dataset_directory (str): Path to the dataset.
                - room_name (str): Name of the room.
                - stg_idx (list[int]): List of indices of the requested stage emitters.
                - mcs_idx (list[int]): List of indices of the requested system receivers.
                - lds_idx (list[int]): List of indices of the requested system emitters.
                - aud_idx (list[int]): List of indices of the requested audience receivers.

            **Attributes**:
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - room_name (str): Name of the room.
                - high_level_info (dict): High-level information of the room.
                - low_level_info (dict): Low-level information of the room.
                - transducer_number (OrderedDict): Number of transducers in the room.
                - transducer_indices (OrderedDict): Indices of the requested stage emitters, system receivers, system emitters and audience receivers.
                - transducer_positions (OrderedDict): Positions of the requested stage emitters, system receivers, system emitters and audience receivers.
                - rir_length (int): Length of the room impulse responses in samples.
                - h_SA (nn.Module): Room impulse responses bewteen stage emitters and audience receivers.
                - h_SM (nn.Module): Room impulse responses bewteen stage emitters and system receivers.
                - h_LA (nn.Module): Room impulse responses bewteen system emitters and audience receivers.
                - h_LM (nn.Module): Room impulse responses bewteen system emitters and system receivers.
        """
        super().__init__(
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )

        self.room_name = room_name

        self.high_level_info = get_hl_info(
            ds_dir=dataset_directory,
            room=self.room_name
        )

        self.room_directory = self.high_level_info['RoomDirectory']

        self.low_level_info = get_ll_info(
            ds_dir=dataset_directory,
            room_dir=self.room_directory
        )

        self.transducer_number, self.transducer_indices = get_transducer_number(
            ll_info=self.low_level_info,
            stg_idx=stg_idx,
            mcs_idx=mcs_idx,
            lds_idx=lds_idx,
            aud_idx=aud_idx
        )

        self.transducer_positions = get_transducer_positions(
            ll_info=self.low_level_info,
            stg_idx=self.transducer_indices['stg'],
            mcs_idx=self.transducer_indices['mcs'],
            lds_idx=self.transducer_indices['lds'],
            aud_idx=self.transducer_indices['aud']
        )

        self.h_SA, self.h_SM, self.h_LA, self.h_LM, self.rir_length = self.__load_rirs(
            ds_dir=dataset_directory
        )

        self.energy_coupling = self.compute_energy_coupling()
        self.direct_to_reverb_ratio = self.compute_direct_to_reverb_ratio()

    def __load_rirs(self, ds_dir: str) -> tuple[dsp.Filter, dsp.Filter, dsp.Filter, dsp.Filter, int]:
        r"""
        Loads all the room impulse responses from the dataset and returns them in processing modules.

            **Args**:
                - ds_dir (str): Path to the dataset.
            
            **Returns**:
                - dsp.Filter: Room impulse responses bewteen stage emitters and audience receivers.
                - dsp.Filter: Room impulse responses bewteen stage emitters and system receivers.
                - dsp.Filter: Room impulse responses bewteen system emitters and audience receivers.
                - dsp.Filter: Room impulse responses bewteen system emitters and system receivers.
                - int: Length of the room impulse responses in samples.
        """
        # Load RIRs
        rirs, rir_length = get_rirs(
            ds_dir=ds_dir,
            room_dir=self.room_directory,
            transducer_indices=self.transducer_indices,
            target_fs=self.fs
        )

        # Energy normalization
        rirs_norm = normalize_rirs(
            fs=self.fs,
            stg_to_aud=rirs["stg_to_aud"],
            stg_to_sys=rirs["stg_to_sys"],
            sys_to_aud=rirs["sys_to_aud"],
            sys_to_sys=rirs["sys_to_sys"]
        )

        # Create processing modules
        h_SA, h_SM, h_LA, h_LM = self.create_modules(
            rirs_SA=rirs_norm["stg_to_aud"],
            rirs_SM=rirs_norm["stg_to_sys"],
            rirs_LA=rirs_norm["sys_to_aud"],
            rirs_LM=rirs_norm["sys_to_sys"],
            rir_length=rir_length
        )

        return h_SA, h_SM, h_LA, h_LM, rir_length
    
# ==================================================================
# =================== WHITE GAUSSIAN NOISE CLASS ===================

class PhRoom_wgn(PhRoom):
    r"""
    Subclass of PhRoom that synthetize the room impulse responses of a RES setup in a shoebox room.
    The room is defined by its size and reverberation time.
    The setup includes one stage emitter and one audience receiver only.
    The room impulse responses are approximated to late reverberation only as exponentially-decaying white-Gaussian-noise sequences.
    """
    def __init__(
            self,
            fs: int,
            nfft: int,
            alias_decay_db: float,
            room_dims: tuple[float, float, float],
            room_RT: float,
            n_M: int,
            n_L: int,
            method: str = 'Poletti'
        ) -> None:
        r"""
        Initializes the PhRoom_wgn object.

            **Args**:
                - room_size (tuple[float, float, float]): Room size in meters.
                - room_RT (float): Room reverberation time [s].
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - n_L (int): Number of system loudspeakers.
                - n_M (int): Number of system microphones.
        """
        assert n_M > 0,  "The number of system microphones must be higher than 0."
        assert n_L > 0,  "The number of system loudspeakers must be higher than 0."

        super().__init__(
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )

        self.room_dims = torch.FloatTensor(room_dims)
        self.RT = room_RT

        self.transducer_number = OrderedDict()
        self.transducer_number.update({'stg': 1})
        self.transducer_number.update({'mcs': n_M})
        self.transducer_number.update({'lds': n_L})
        self.transducer_number.update({'aud': 1})
        
        self.transducer_positions = simulate_setup(
            room_dims=self.room_dims,
            mcs_n=n_M,
            lds_n=n_L
        )

        if method not in ['Poletti', 'Barron']:
            raise ValueError(f"Method '{method}' is not supported. Choose 'Poletti' or 'Barron'.")
        self.method = method

        self.h_SA, self.h_SM, self.h_LA, self.h_LM, self.rir_length = self.__generate_rirs()

        self.energy_coupling = self.compute_energy_coupling()
        self.direct_to_reverb_ratio = self.compute_direct_to_reverb_ratio()

    def __generate_rirs(self) -> tuple[dsp.Filter, dsp.Filter, dsp.Filter, dsp.Filter, int]:
        r"""
        Generates the room impulse responses of the RES setup.

            **Returns**:
                - dsp.Filter: Room impulse responses bewteen stage emitters and audience receivers.
                - dsp.Filter: Room impulse responses bewteen stage emitters and system receivers.
                - dsp.Filter: Room impulse responses bewteen system emitters and audience receivers.
                - dsp.Filter: Room impulse responses bewteen system emitters and system receivers.
                - int: Length of the room impulse responses in samples.
        """
        
        rirs_SA = self.__generate_rirs_of(
            n_emitters=self.transducer_number['stg'],
            pos_emitters=self.transducer_positions['stg'],
            n_receivers=self.transducer_number['aud'],
            pos_receivers=self.transducer_positions['aud']
        )
        rirs_SM = self.__generate_rirs_of(
            n_emitters=self.transducer_number['stg'],
            pos_emitters=self.transducer_positions['stg'],
            n_receivers=self.transducer_number['mcs'],
            pos_receivers=self.transducer_positions['mcs']
        )
        rirs_LA = self.__generate_rirs_of(
            n_emitters=self.transducer_number['lds'],
            pos_emitters=self.transducer_positions['lds'],
            n_receivers=self.transducer_number['aud'],
            pos_receivers=self.transducer_positions['aud']
        )
        rirs_LM = self.__generate_rirs_of(
            n_emitters=self.transducer_number['lds'],
            pos_emitters=self.transducer_positions['lds'],
            n_receivers=self.transducer_number['mcs'],
            pos_receivers=self.transducer_positions['mcs']
        )

        # Get the length of the RIRs
        sa_length = rirs_SA.shape[0]
        sm_length = rirs_SM.shape[0]
        lm_length = rirs_LM.shape[0]
        la_length = rirs_LA.shape[0]

        max_length = max(sa_length, sm_length, lm_length, la_length)

        # Pad the RIRs to the same length
        rirs_SA = torch.nn.functional.pad(
            input=rirs_SA,
            pad=(0, 0, 0, 0, 0, max_length - sa_length),
            mode='constant',
            value=0
        )
        rirs_SM = torch.nn.functional.pad(
            input=rirs_SM,
            pad=(0, 0, 0, 0, 0, max_length - sm_length),
            mode='constant',
            value=0
        )
        rirs_LA = torch.nn.functional.pad(
            input=rirs_LA,
            pad=(0, 0, 0, 0, 0, max_length - la_length),
            mode='constant',
            value=0
        )
        rirs_LM = torch.nn.functional.pad(
            input=rirs_LM,
            pad=(0, 0, 0, 0, 0, max_length - lm_length),
            mode='constant',
            value=0
        )

        # Create processing modules
        h_SA, h_SM, h_LA, h_LM = self.create_modules(
            rirs_SA=rirs_SA,
            rirs_SM=rirs_SM,
            rirs_LA=rirs_LA,
            rirs_LM=rirs_LM,
            rir_length=max_length
        )

        return h_SA, h_SM, h_LA, h_LM, max_length


    def __generate_rirs_of(self,
            n_emitters: int,
            pos_emitters: torch.FloatTensor,
            n_receivers: int,
            pos_receivers: torch.FloatTensor
        ) -> torch.Tensor:
        r"""
        Generates the room impulse responses between the emitters and receivers.

            **Args**:
                - n_emitters (int): Number of emitters.
                - pos_emitters (torch.FloatTensor): Positions of the emitters.
                - n_receivers (int): Number of receivers.
                - pos_receivers (torch.FloatTensor): Positions of the receivers.

            **Returns**:
                - torch.Tensor: Room impulse responses.
        """
        # Generate the room impulse responses as exponentially-decaying white-Gaussian-noise sequences
        rirs = WGN_reverb(
            matrix_size=(n_receivers, n_emitters),
            t60=self.RT,
            samplerate=self.fs,
        )

        # Compute the distances between the emitters and receivers
        pos_emitters = pos_emitters.unsqueeze(0).repeat(n_receivers,1,1)
        pos_receivers = pos_receivers.unsqueeze(1).repeat(1,n_emitters,1)
        distances = torch.linalg.norm(pos_emitters - pos_receivers, dim=2)

        # Compute the propagation times and convert them to samples
        speed_of_sound = 343  # m/s
        propagation_times = distances / speed_of_sound
        propagation_samples = torch.round(propagation_times * self.fs).long()

        # Zero-pad the RIRs
        rirs = torch.nn.functional.pad(
            input=rirs,
            pad=(0, 0, 0, 0, 0, propagation_samples.max().item()),
            mode='constant',
            value=0
        )

        # Shift the RIRs according to the propagation times
        for r in range(n_receivers):
            for e in range(n_emitters):
                rirs[:,r,e] = torch.roll(rirs[:,r,e], shifts=propagation_samples[r,e].item(), dims=0)

        return rirs
    
    def regenerate_h_LM(self) -> None:
        r"""
        Regenerates the room impulse responses between system emitters and system receivers.
        """
        new_rirs = self.__generate_rirs_of(
                n_emitters=self.transducer_number['lds'],
                pos_emitters=self.transducer_positions['lds'],
                n_receivers=self.transducer_number['mcs'],
                pos_receivers=self.transducer_positions['mcs']
            )
        self.h_LM.assign_value(new_rirs)

        self.energy_coupling['LM'] = energy_coupling(new_rirs, fs=self.fs)
        self.direct_to_reverb_ratio['LM'] = direct_to_reverb_ratio(new_rirs, fs=self.fs)

        return None