# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
from collections import OrderedDict
# PyTorch
import torch
import torch.nn as nn
# FLAMO
from flamo import dsp
from flamo.functional import WGN_reverb
# PyRES
from PyRES.dataset_api import (
    get_hl_info,
    get_ll_info,
    get_rirs,
    normalize_rirs,
    get_transducer_number,
    get_transducer_positions
)
from PyRES.plots import (
    plot_room_setup,
    plot_coupling,
    plot_DRR
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
        RIRs.update({'h_SM': self.get_stg_to_mcs().param.clone().detach()})
        RIRs.update({'h_SA': self.get_stg_to_aud().param.clone().detach()})
        RIRs.update({'h_LM': self.get_lds_to_mcs().param.clone().detach()})
        RIRs.update({'h_LA': self.get_lds_to_aud().param.clone().detach()})
        return RIRs
    
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
        stg = self.transducer_positions['stg']
        mcs = self.transducer_positions['mcs']
        lds = self.transducer_positions['lds']
        aud = self.transducer_positions['aud']

        plot_room_setup(stg=stg, mcs=mcs, lds=lds, aud=aud)
    
    def plot_coupling(self) -> None:
        r"""
        Plots the room coupling.
        """
        plot_coupling(rirs=self.get_rirs(), fs=self.fs)
    
    def plot_DRR(self) -> None:
        r"""
        Plots the direct-to-reverberant ratio (DRR).
        """
        plot_DRR(rirs=self.get_rirs(), fs=self.fs)


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