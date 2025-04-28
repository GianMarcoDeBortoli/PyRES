# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
from collections import OrderedDict
import json
# PyTorch
import torch
import torch.nn as nn
import torchaudio
# FLAMO
from flamo import dsp
# PyRES
from PyRES.metrics import energy_coupling
from PyRES.plots import plot_room_setup, plot_coupling, plot_DRR


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
                - n_S (int): Number of stage sources.
                - n_L (int): Number of system loudspeakers.
                - n_M (int): Number of system microphones.
                - n_A (int): Number of audience positions.
                - rir_length (int): Length of the room impulse responses in samples.
                - h_SA (nn.Module): Room impulse responses bewteen stage sources and audience positions.
                - h_SM (nn.Module): Room impulse responses bewteen stage sources and system microphones.
                - h_LA (nn.Module): Room impulse responses bewteen system loudspeakers and audience positions.
                - h_LM (nn.Module): Room impulse responses bewteen system loudspeakers and system microphones.
        """
        object.__init__(self)

        self.fs = fs
        self.nfft = nfft
        self.alias_decay_db = alias_decay_db

        self.n_S: int
        self.n_L: int
        self.n_M: int
        self.n_A: int

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
        return self.n_S, self.n_M, self.n_L, self.n_A
    
    def get_stg_to_aud(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between stage sources and audience positions.

            **Returns**:
                - torch.Tensor: Stage-to-Audience RIRs. shape = (samples, n_A, n_S).
        """
        return self.h_SA

    def get_stg_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between stage sources and system microphones.

            **Returns**:
                - torch.Tensor: Stage-to-Microphones RIRs. shape = (samples, n_M, n_S).
        """
        return self.h_SM
    
    def get_lds_to_aud(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between system loudspeakers and audience positions.

            **Returns**:
                - torch.Tensor: Loudspeakers-to-Audience RIRs. shape = (samples n_A, n_L).
        """
        return self.h_LA

    def get_lds_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the room impulse responses between system loudspeakers and system microphones.

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
    
    def create_modules(self, rirs_SA: torch.Tensor, rirs_SM: torch.Tensor, rirs_LA: torch.Tensor, rirs_LM: torch.Tensor, rir_length: int) -> tuple[dsp.Filter, dsp.Filter, dsp.Filter, dsp.Filter]:
        r"""
        Creates the processing modules for the room-impulse-response blocks.

            **Args**:
                - rirs_SA (torch.Tensor): Room impulse responses between stage sources and audience positions.
                - rirs_SM (torch.Tensor): Room impulse responses between stage sources and system microphones.
                - rirs_LA (torch.Tensor): Room impulse responses between system loudspeakers and audience positions.
                - rirs_LM (torch.Tensor): Room impulse responses between system loudspeakers and system microphones.
                - rir_length (int): Length of the room impulse responses in samples.

            **Returns**:
                - dsp.Filter: Room impulse responses bewteen stage sources and audience positions.
                - dsp.Filter: Room impulse responses bewteen stage sources and system microphones.
                - dsp.Filter: Room impulse responses bewteen system loudspeakers and audience positions.
                - dsp.Filter: Room impulse responses bewteen system loudspeakers and system microphones.
        """

        # Stage to Audience
        h_SA = dsp.Filter(
            size=(rir_length, self.n_A, self.n_S),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        h_SA.assign_value(rirs_SA)

        # Stage to Microphones
        h_SM = dsp.Filter(
            size=(rir_length, self.n_M, self.n_S),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        h_SM.assign_value(rirs_SM)

        # Loudspeakers to Audience
        h_LM = dsp.Filter(
            size=(rir_length, self.n_M, self.n_L),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        h_LM.assign_value(rirs_LM)

        # Loudspeakers to Microphones
        h_LA = dsp.Filter(
            size=(rir_length, self.n_A, self.n_L),
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
        plot_room_setup(self)
    
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
            room_name: str
        ) -> None:
        r"""
        Initializes the PhRoom_dataset object.

            **Args**:
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - dataset_directory (str): Path to the dataset.
                - room_name (str): Name of the room.

            **Attributes**:
                - fs (int): Sample rate [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - room_name (str): Name of the room.
                - high_level_info (dict): High-level information of the room.
                - low_level_info (dict): Low-level information of the room.
                - n_S (int): Number of stage sources.
                - n_L (int): Number of system loudspeakers.
                - n_M (int): Number of system microphones.
                - n_A (int): Number of audience positions.
                - rir_length (int): Length of the room impulse responses in samples.
                - h_SA (nn.Module): Room impulse responses bewteen stage sources and audience positions.
                - h_SM (nn.Module): Room impulse responses bewteen stage sources and system microphones.
                - h_LA (nn.Module): Room impulse responses bewteen system loudspeakers and audience positions.
                - h_LM (nn.Module): Room impulse responses bewteen system loudspeakers and system microphones.
        """
        super().__init__(
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )

        self.room_name = room_name
        self.high_level_info = self.__find_room_in_dataset(
            ds_dir=dataset_directory,
            room=room_name
        )

        self.low_level_info = self.__get_room_info(
            ds_dir=dataset_directory,
            room_dir=self.high_level_info['RoomDirectory']
        )

        self.n_S, self.n_M, self.n_L, self.n_A = self.__ems_rcs_number()
        self.h_SA, self.h_SM, self.h_LA, self.h_LM, self.rir_length = self.__load_rirs(
            ds_dir=dataset_directory
        )

    def __find_room_in_dataset(self, ds_dir: str, room: str) -> dict:
        r"""
        Finds the room in the dataset.

            **Args**:
                - ds_dir (str): Path to the dataset.
                - room (str): Name of the room.

            **Returns**:
                - dict: High-level information of the room.
        """
        ds_dir = ds_dir.rstrip('/')
        with open(f"{ds_dir}/datasetInfo.json", 'r') as file:
            data = json.load(file)
        
        return data['Rooms'][room]
    
    def __get_room_info(self, ds_dir: str, room_dir: str) -> dict:
        r"""
        Gets the room information.

            **Args**:
                - ds_dir (str): Path to the dataset.
                - room_dir (str): Path to the room in the dataset.

            **Returns**:
                - dict: Low-level information of the room.
        """
        ds_dir = ds_dir.rstrip('/')
        with open(f"{ds_dir}/data/{room_dir}/roomInfo.json", 'r') as file:
            data = json.load(file)

        return data
    
    def __ems_rcs_number(self) -> tuple[int, int, int, int]:
        r"""
        Scans the room information for the number of emitters and receivers.

            **Returns**:
                - int: Number of stage sources.
                - int: Number of system microphones.
                - int: Number of system loudspeakers.
                - int: Number of audience positions.
        """
        n_S = self.low_level_info['StageAndAudience']['StageEmitters']['Number']
        n_M = self.low_level_info['AudioSetup']['SystemReceivers']['Number']
        n_L = self.low_level_info['AudioSetup']['SystemEmitters']['Number']
        n_A = self.low_level_info['StageAndAudience']['AudienceReceivers']['MonochannelNumber']

        return n_S, n_M, n_L, n_A

    def __load_rirs(self, ds_dir: str) -> tuple[dsp.Filter, dsp.Filter, dsp.Filter, dsp.Filter, int]:
        r"""
        Loads all the room impulse responses from the dataset and returns them in processing modules.

            **Args**:
                - ds_dir (str): Path to the dataset.
            
            **Returns**:
                - dsp.Filter: Room impulse responses bewteen stage sources and audience positions.
                - dsp.Filter: Room impulse responses bewteen stage sources and system microphones.
                - dsp.Filter: Room impulse responses bewteen system loudspeakers and audience positions.
                - dsp.Filter: Room impulse responses bewteen system loudspeakers and system microphones.
                - int: Length of the room impulse responses in samples.
        """

        rir_info = self.low_level_info['RoomImpulseResponses']

        rir_fs = rir_info['SampleRate_Hz']
        rir_length = rir_info['LengthInSamples']

        if rir_fs != self.fs:
            rir_length = int(self.fs * rir_length/rir_fs)

        ds_dir = ds_dir.rstrip('/')
        path_root = f"{ds_dir}/data/{self.high_level_info['RoomDirectory']}/{rir_info['Directory']}"

        path = f"{path_root}/{rir_info['StageEmitters-AudienceReceivers']['Directory']}"
        stg_to_aud = self.__load_rir_matrix(path=f"{path}", n_emitters=self.n_S, n_receivers=self.n_A, fs=rir_fs, n_samples=rir_length)
        path = f"{path_root}/{rir_info['StageEmitters-SystemReceivers']['Directory']}"
        stg_to_sys = self.__load_rir_matrix(path=f"{path}", n_emitters=self.n_S, n_receivers=self.n_M, fs=rir_fs, n_samples=rir_length)
        path = f"{path_root}/{rir_info['SystemEmitters-AudienceReceivers']['Directory']}"
        sys_to_aud = self.__load_rir_matrix(path=f"{path}", n_emitters=self.n_L, n_receivers=self.n_A, fs=rir_fs, n_samples=rir_length)
        path = f"{path_root}/{rir_info['SystemEmitters-SystemReceivers']['Directory']}"
        sys_to_sys = self.__load_rir_matrix(path=f"{path}", n_emitters=self.n_L, n_receivers=self.n_M, fs=rir_fs, n_samples=rir_length)

        h_SA, h_SM, h_LA, h_LM = self.create_modules(
            rirs_SA=stg_to_aud,
            rirs_SM=stg_to_sys,
            rirs_LA=sys_to_aud,
            rirs_LM=sys_to_sys,
            rir_length=rir_length
        )

        return h_SA, h_SM, h_LA, h_LM, rir_length
    
    def __load_rir_matrix(self, path: str, n_emitters: int, n_receivers: int, fs: int, n_samples: int) -> torch.Tensor:
        r"""
        Loads the room impulse responses from the dataset and returns them in a matrix.

            **Args**:
                - path (str): Path to the room impulse responses in the dataset.
                - n_sources (int): Number of emitters.
                - n_receivers (int): Number of receivers.
                - fs (int): Sample rate [Hz].
                - n_samples (int): Length of the room impulse responses in samples.

            **Returns**:
                - torch.Tensor: Room-impulse-response matrix as a torch tensor [n_samples, n_receivers, n_emitters].
        """
        matrix = torch.zeros(n_samples, n_receivers, n_emitters)
        for i in range(n_receivers):
            for j in range(n_emitters):
                w = torchaudio.load(f"{path}/E{j+1:03d}_R{i+1:03d}_M01.wav")[0]
                if self.fs != fs:
                    w = torchaudio.transforms.Resample(fs, self.fs)(w)
                matrix[:,i,j] = w.permute(1,0).squeeze()

        # Energy normalization
        ec = energy_coupling(rir=matrix, fs=self.fs, decay_interval='T30')
        norm_factor = torch.max(torch.tensor([self.n_L, self.n_M])) * torch.sqrt(torch.median(ec))
        matrix = matrix/norm_factor

        return matrix
    
