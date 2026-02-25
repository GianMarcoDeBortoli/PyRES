# ==================================================================
# ============================ IMPORTS =============================
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# PyRES
from PyRES.physical_room import PhRoom_dataset
from PyRES.dataset_api import get_hl_info, get_ll_info, get_transducer_number


###########################################################################################
# In this example, we again create a physical room by loading it from the dataset DataRES.
# This time, we will only load some of the impulse responses by selecting specific transducers.
###########################################################################################


if __name__ == '__main__':

    # Time-frequency
    samplerate = 48000              # Sampling frequency
    nfft = samplerate*3             # FFT size
    alias_decay_db = 0              # Anti-time-aliasing decay in dB

    # Physical room
    dataset_directory = './dataRES'
    room_name = 'ImmersiveLab'

    print(f"\nHaving downloaded DataRES, you can also inspect the information contained in the dataset using the api provided in PyRES: dataset_api.py.")
    high_level_info = get_hl_info(
        ds_dir=dataset_directory,
        room=room_name
    )
    room_directory = high_level_info['RoomDirectory']
    low_level_info = get_ll_info(
        ds_dir=dataset_directory,
        room_dir=room_directory
    )
    transducer_number,_ = get_transducer_number(
        ll_info=low_level_info
    )
    print(f"Now we can inspect the number of transducers in the room '{room_name}':")
    print(f"  Stage emitters: {transducer_number['stg']}")
    print(f"  System microphones: {transducer_number['mcs']}")
    print(f"  System loudspeakers: {transducer_number['lds']}")
    print(f"  Audience receivers: {transducer_number['aud']}")

    print(f"We can select only some of the transducers by giving the desired indices to PhRoom_dataset.")
    mcs_indices = [0, 1, 3]         # Load only some of the microphones
    lds_indices = [0, 2, 8, 11]     # Load only some of the loudspeakers

    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name,
        mcs_idx=mcs_indices,
        lds_idx=lds_indices
    )

    print(f"\nThe number of transducers loaded is:")
    print(f"  Stage emitters: {physical_room.transducer_number['stg']}")
    print(f"  System microphones: {physical_room.transducer_number['mcs']}")
    print(f"  System loudspeakers: {physical_room.transducer_number['lds']}")
    print(f"  Audience receivers: {physical_room.transducer_number['aud']}")
    print(f"And the transducer positions are:")
    print(f"  Stage emitters: \n{physical_room.transducer_positions['stg']}")
    print(f"  System microphones: \n{physical_room.transducer_positions['mcs']}")
    print(f"  System loudspeakers: \n{physical_room.transducer_positions['lds']}")
    print(f"  Audience receivers: \n{physical_room.transducer_positions['aud']}")

    physical_room.plot_setup()
    physical_room.plot_coupling()
    physical_room.plot_DRR()

    exit(0)