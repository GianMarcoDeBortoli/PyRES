# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
from collections import OrderedDict
import json
# PyTorch
import torch, torchaudio
# PyRES
from PyRES.metrics import energy_coupling


# ==================================================================
# ==================== ROOM INFO DICTIONARIES ======================

def get_hl_info(
        ds_dir: str,
        room: str
    ) -> dict:
    r"""
    Retrieves the high level information about a room given the dataset directory and the room name.

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

def get_ll_info(
        ds_dir: str,
        room_dir: str
    ) -> dict:
    r"""
    Retrieves the low level information about a room given the dataset directory and the room name.

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

# ==================================================================
# ==================== ROOM IMPULSE RESPONSES ======================

def get_rir_metadata(
        ds_dir: str,
        room_dir: str,
    ) -> tuple[int, int, OrderedDict]:
    r"""
    Retrieves the paths to the RIRs of all transducers in the room.

        **Args**:
            - ds_dir (str): Path to the dataset.
            - room_dir (str): Path to the room in the dataset.
            - stg_idx (list[int]): Indices of the requested stage emitters.
            - mcs_idx (list[int]): Indices of the requested system receivers.
            - lds_idx (list[int]): Indices of the requested system emitters.
            - aud_idx (list[int]): Indices of the requested audience receivers.

        **Returns**:
            - OrderedDict: Paths to the RIRs of all transducers in the room.
    """
    # Get the low-level information of the room
    ll_info = get_ll_info(ds_dir=ds_dir, room_dir=room_dir)

    # Sample rate and length of the RIRs
    samplerate = ll_info['RoomImpulseResponses']['SampleRate_Hz']
    rir_length = ll_info['RoomImpulseResponses']['LengthInSamples']

    # Path root
    ds_dir = ds_dir.rstrip('/')
    root = f"{ds_dir}/data/{room_dir}"

    # Part of the path common to all RIRs
    rir_path_1 = ll_info['RoomImpulseResponses']['Directory']
    common_path = f"{root}/{rir_path_1}"

    # Specific paths for each type of transducer
    stg_to_aud = get_rir_foldername_of(ll_info=ll_info, emitter_type='stg', receiver_type='aud')
    stg_to_sys = get_rir_foldername_of(ll_info=ll_info, emitter_type='stg', receiver_type='mcs')
    sys_to_aud = get_rir_foldername_of(ll_info=ll_info, emitter_type='lds', receiver_type='aud')
    sys_to_sys = get_rir_foldername_of(ll_info=ll_info, emitter_type='lds', receiver_type='mcs')

    paths = OrderedDict()
    paths.update({'stg_to_aud': f"{common_path}/{stg_to_aud}"})
    paths.update({'stg_to_sys': f"{common_path}/{stg_to_sys}"})
    paths.update({'sys_to_aud': f"{common_path}/{sys_to_aud}"})
    paths.update({'sys_to_sys': f"{common_path}/{sys_to_sys}"})

    return samplerate, rir_length, paths

def get_rir_foldername_of(
        ll_info: dict,
        emitter_type: str,
        receiver_type: str,
    ) -> str:
    r"""
    Retrieves the path to the RIR folder of the requested transducers.
    """
    match emitter_type:
        case 'stg':
            emitter = 'StageEmitters'
        case 'lds':
            emitter = 'SystemEmitters'
        case _:
            raise ValueError(f"Emitter type {emitter_type} is not valid. Must be 'stg' or 'lds'.")
        
    match receiver_type:
        case 'aud':
            receiver = 'AudienceReceivers-Mono' # TODO: add for -Array and -Binaural
        case 'mcs':
            receiver = 'SystemReceivers'
        case _:
            raise ValueError(f"Receiver type {receiver_type} is not valid. Must be 'aud' or 'mcs'.")
        
    json_field = f"{emitter}-{receiver}"
    folder_name = ll_info['RoomImpulseResponses'][json_field]['Directory']
    return folder_name

def get_rirs(
        ds_dir: str,
        room_dir: str,
        transducer_indices: OrderedDict,
        target_fs: int,
    ) -> tuple[OrderedDict, OrderedDict]:
    r"""
    Loads
    """
    # Get RIR metadata
    rir_fs, rir_length, paths = get_rir_metadata(
        ds_dir=ds_dir,
        room_dir=room_dir
    )

    # Stage to audience
    stg_to_aud, _ = get_rirs_of(
        path=paths["stg_to_aud"],
        emitter_idx=transducer_indices['stg'],
        receiver_idx=transducer_indices['aud'],
        origin_fs=rir_fs,
        target_fs=target_fs,
        origin_len=rir_length
    )
    # Stage to system
    stg_to_sys, _ = get_rirs_of(
        path=paths['stg_to_sys'],
        emitter_idx=transducer_indices['stg'],
        receiver_idx=transducer_indices['mcs'],
        origin_fs=rir_fs,
        target_fs=target_fs,
        origin_len=rir_length
    )
    # System to audience
    sys_to_aud, _ = get_rirs_of(
        path=paths['sys_to_aud'],
        emitter_idx=transducer_indices['lds'],
        receiver_idx=transducer_indices['aud'],
        origin_fs=rir_fs,
        target_fs=target_fs,
        origin_len=rir_length
    )
    # System to system
    sys_to_sys, rir_length = get_rirs_of(
        path=paths['sys_to_sys'],
        emitter_idx=transducer_indices['lds'],
        receiver_idx=transducer_indices['mcs'],
        origin_fs=rir_fs,
        target_fs=target_fs,
        origin_len=rir_length
    )

    rirs = OrderedDict()
    rirs.update({'stg_to_aud': stg_to_aud})
    rirs.update({'stg_to_sys': stg_to_sys})
    rirs.update({'sys_to_aud': sys_to_aud})
    rirs.update({'sys_to_sys': sys_to_sys})

    return rirs, rir_length

def get_rirs_of(path: str, emitter_idx: int, receiver_idx: int, origin_fs: int, target_fs: int, origin_len: int) -> torch.Tensor:
        r"""
        Loads the room impulse responses from the dataset and returns them in a matrix.

            **Args**:
                - path (str): Path to the room impulse responses in the dataset.
                - emitter_idx (list[int]): Indices of the emitters.
                - receiver_idx (list[int]): Indices of the receivers.
                - fs (int): Sample rate [Hz].
                - n_samples (int): Length of the room impulse responses in samples.

            **Returns**:
                - torch.Tensor: Room-impulse-response matrix as a torch tensor [n_samples, n_receivers, n_emitters].
        """
        n_emitters = len(emitter_idx)
        n_receivers = len(receiver_idx)

        n_samples = origin_len
        resample = False
        if target_fs != origin_fs:
            n_samples = int(origin_len * target_fs / origin_fs)
            resample = True

        matrix = torch.zeros(n_samples, n_receivers, n_emitters)
        for i,r in enumerate(receiver_idx):
            for j,e in enumerate(emitter_idx):
                filename = f"{path}/E{e+1:03d}_R{r+1:03d}_M01.wav"
                w = torchaudio.load(filename)[0]
                if resample:
                    w = torchaudio.transforms.Resample(origin_fs, target_fs)(w)
                matrix[:,i,j] = w.permute(1,0).squeeze()

        return matrix, n_samples

def normalize_rirs(
        fs: int,
        stg_to_aud: torch.Tensor,
        stg_to_sys: torch.Tensor,
        sys_to_aud: torch.Tensor,
        sys_to_sys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Normalizes the room impulse responses.
        **Args**:
            - stg_to_aud (torch.Tensor): Room impulse responses bewteen stage emitters and audience receivers.
            - stg_to_sys (torch.Tensor): Room impulse responses bewteen stage emitters and system receivers.
            - sys_to_aud (torch.Tensor): Room impulse responses bewteen system emitters and audience receivers.
            - sys_to_sys (torch.Tensor): Room impulse responses bewteen system emitters and system receivers.

        **Returns**:
            - torch.Tensor: Normalized room impulse responses bewteen stage emitters and audience receivers.
            - torch.Tensor: Normalized room impulse responses bewteen stage emitters and system receivers.
            - torch.Tensor: Normalized room impulse responses bewteen system emitters and audience receivers.
            - torch.Tensor: Normalized room impulse responses bewteen system emitters and system receivers.
    """
    # Energy coupling of the feedback path
    ec_lm = energy_coupling(rir=sys_to_sys, fs=fs, decay_interval='T30')
    dim = torch.mean(torch.tensor([ec_lm.shape[0], ec_lm.shape[1]], dtype=torch.float32)).int()

    # Norm factor: the feedback path is normalized to a power gain equal to the mean dimension of the system
    norm_factor = torch.sqrt(torch.sum(ec_lm)) / torch.sqrt(dim)

    # Normalization
    rirs_norm = OrderedDict()
    rirs_norm.update({'stg_to_aud': stg_to_aud / norm_factor})
    rirs_norm.update({'stg_to_sys': stg_to_sys / norm_factor})
    rirs_norm.update({'sys_to_aud': sys_to_aud / norm_factor})
    rirs_norm.update({'sys_to_sys': sys_to_sys / norm_factor})

    return rirs_norm


# ==================================================================
# ======================= TRANSDUCER NUMBER ========================

def get_transducer_number(
        ll_info: dict,
        stg_idx: list[int]=None,
        mcs_idx: list[int]=None,
        lds_idx: list[int]=None,
        aud_idx: list[int]=None
    ) -> tuple[OrderedDict, OrderedDict]:
    r"""
    Retrieves the number of all transducers in the room.

        **Args**:
            - ll_info (dict): Low-level information of the room.
            - stg_idx (list[int]): Indices of the requested stage emitters.
            - mcs_idx (list[int]): Indices of the requested system receivers.
            - lds_idx (list[int]): Indices of the requested system emitters.
            - aud_idx (list[int]): Indices of the requested audience receivers.

        **Returns**:
            - OrderedDict: Number of all transducers in the room.
            - OrderedDict: Indices of all transducers in the room.
    """

    stg_n, stg_idx = get_number_of(ll_info=ll_info, str1='StageAndAudience', str2='StageEmitters', idx=stg_idx)
    mcs_n, mcs_idx = get_number_of(ll_info=ll_info, str1='AudioSetup', str2='SystemReceivers', idx=mcs_idx)
    lds_n, lds_idx = get_number_of(ll_info=ll_info, str1='AudioSetup', str2='SystemEmitters', idx=lds_idx)
    aud_n, aud_idx = get_number_of(ll_info=ll_info, str1='StageAndAudience', str2='AudienceReceivers-Mono', idx=aud_idx) # TODO: add for -Array and -Binaural

    number = OrderedDict()
    number.update({'stg': stg_n})
    number.update({'mcs': mcs_n})
    number.update({'lds': lds_n})
    number.update({'aud': aud_n})

    indices = OrderedDict()
    indices.update({'stg': stg_idx})
    indices.update({'mcs': mcs_idx})
    indices.update({'lds': lds_idx})
    indices.update({'aud': aud_idx})

    return number, indices
    
def get_number_of(
        ll_info: dict,
        str1: str,
        str2: str,
        idx: list[int]=None
    ) -> int:
    r"""
    Retrieves the number of the requested transducers.

        **Args**:
            - ll_info (dict): Low-level information of the room.
            - str1 (str): Type of the first transducer.
            - str2 (str): Type of the second transducer.
            - idx (list[int]): Indices of the requested transducers.

        **Returns**:
            - int: Number of the requested transducers.
    """

    number = ll_info[str1][str2]['Number']
    if idx is None:
        if number == 0:
            Warning(f"For the requested room, the number of {str2} is zero.")
        return number, list(range(number))
    else:
        check_requested_indices(type=str2, number=number, idx=idx)
        return len(idx), idx

# ==================================================================
# ===================== TRANSDUCER POSITIONS =======================

def get_transducer_positions(
        ll_info: dict,
        stg_idx: list[int]=None,
        mcs_idx: list[int]=None,
        lds_idx: list[int]=None,
        aud_idx: list[int]=None
    ) -> OrderedDict:
    r"""
    Retrieves the positions of all transducers in the room.

        **Args**:
            - ll_info (dict): Low-level information of the room.
            - stg_idx (list[int]): Indices of the requested stage emitters.
            - mcs_idx (list[int]): Indices of the requested system receivers.
            - lds_idx (list[int]): Indices of the requested system emitters.
            - aud_idx (list[int]): Indices of the requested audience receivers.

        **Returns**:
            - OrderedDict: Positions of all transducers in the room.
    """
      
    stg = get_positions_of(ll_info=ll_info, str1='StageAndAudience', str2='StageEmitters', idx=stg_idx)
    mcs = get_positions_of(ll_info=ll_info, str1='AudioSetup', str2='SystemReceivers', idx=mcs_idx)
    lds = get_positions_of(ll_info=ll_info, str1='AudioSetup', str2='SystemEmitters', idx=lds_idx)
    aud = get_positions_of(ll_info=ll_info, str1='StageAndAudience', str2='AudienceReceivers-Mono', idx=aud_idx) # TODO: add for -Array and -Binaural

    pos = OrderedDict()
    pos.update({'stg': stg})
    pos.update({'mcs': mcs})
    pos.update({'lds': lds})
    pos.update({'aud': aud})

    return pos

def get_positions_of(
        ll_info: dict,
        str1: str,
        str2: str,
        idx: list[int]=None
    ) -> tuple:
    r"""
    Retrieves the positions of the requested transducers.

        **Args**:
            - ll_info (dict): Low-level information of the room.
            - str1 (str): Type of the first position.
            - str2 (str): Type of the second position.

        **Returns**:
            - tuple: Positions of the first and second type.
    """
    pos = ll_info[str1][str2]['Position_m']

    if idx is None:
        if len(pos) == 0:
            return None
        return pos
    else:
        check_requested_indices(type=str2, number=len(pos), idx=idx)
        p = []
        for i in idx:
            p.append(pos[i])
        return p
    
# ==================================================================
# ===================== CHECK REQUESTED INDICES ====================

def check_requested_indices(
        type: str,
        number: int,
        idx: list[int],
    ) -> None:
    r"""
    Checks if the requested indices are valid.

        **Args**:
            - type (str): Type of the transducer.
            - number (int): Number of the requested transducers.
            - idx (list[int]): Indices of the requested transducers.

        **Returns**:
            - None
    """
    if idx is None:
        return None
    assert isinstance(idx, list), f"Requested indices of {type} must be a list."
    assert all(isinstance(i, int) for i in idx), f"Requested indices of {type} must be integers."
    assert all(i >= 0 for i in idx), f"{type} indices start from 0. You cannot request a negative index."
    assert len(idx) == len(set(idx)), f"Requested indices of {type} must be unique."
    assert max(idx) <= number, f"For the requested room, the maximum index of {type} is {number-1}. You cannot request the index {max(idx)}." 
    return None