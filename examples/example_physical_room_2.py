# ==================================================================
# ============================ IMPORTS =============================

# PyTorch
from flamo.functional import mag2db
# PyRES
from pyres.physical_room import PhRoom_dataset


###########################################################################################
# In this example, we create a physical room by loading it from the dataset DataRES.
# DataRES is an accompanying dataset for the PyRES library, which contains a collection of
# measured physical spaces hosting a Reverberation Enhancement System. 

# Reference:
#   De Bortoli, G., Prawda, K., Coleman, P., and Schlecht, S. J.
#   DataRES: Dataset for research on Reverberation Enhancement Systems, Zenodo, 2025.
#   https://doi.org/10.5281/zenodo.15165524
###########################################################################################


if __name__ == '__main__':

    # Time-frequency
    samplerate = 48000              # Sampling frequency
    nfft = samplerate*3             # FFT size
    alias_decay_db = 0              # Anti-time-aliasing decay in dB

    # Physical room
    print(f"\nPhRoom_dataset interfaces with DataRES loading number and positions of all transducers and all RIRs between them.")
    print(f"PhRoom_dataset need to be given the dataset directory and the room name to load the data.")
    print(f"By downloading DataRES and inspecting the json file called 'datasetInfo.json', you can find the available rooms.")
    dataset_directory = './dataRES'
    room_name = 'Otala'

    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )

    print(f"\nThe PhRoom_dataset class is another subclass of the {type(physical_room).__bases__[0].__name__} class.")
    print(f"Transducer numbers:")
    print(f"  Stage emitters: {physical_room.transducer_number['stg']}")
    print(f"  System microphones: {physical_room.transducer_number['mcs']}")
    print(f"  System loudspeakers: {physical_room.transducer_number['lds']}")
    print(f"  Audience receivers: {physical_room.transducer_number['aud']}")
    print(f"Transducer positions:")
    print(f"  Stage emitters: \n{physical_room.transducer_positions['stg']}")
    print(f"  System microphones: \n{physical_room.transducer_positions['mcs']}")
    print(f"  System loudspeakers: \n{physical_room.transducer_positions['lds']}")
    print(f"  Audience receivers: \n{physical_room.transducer_positions['aud']}")

    physical_room.plot_setup()

    print(f"\nAll PhRoom subclasses host further useful information about the physical space.")
    print(f"The energy and the direct-to-reverberant ratio (DRR) of the RIRs are contained in the 'energy_coupling' and 'direct_to_reverb_ratio' attributes, respectively.")
    print(f"Energy coupling:")
    print(f"  Stage emitters to audience receivers: {mag2db(physical_room.energy_coupling['SA'])} dB")
    print(f"  Stage emitters to system microphones: {mag2db(physical_room.energy_coupling['SM'])} dB")
    print(f"  System loudspeakers to system microphones: {mag2db(physical_room.energy_coupling['LM'])} dB")
    print(f"  System loudspeakers to audience receivers: {mag2db(physical_room.energy_coupling['LA'])} dB")
    print(f"Direct-to-reverberant ratio (DRR):")
    print(f"  Stage emitters to audience receivers: {mag2db(physical_room.direct_to_reverb_ratio['SA'])} dB")
    print(f"  Stage emitters to system microphones: {mag2db(physical_room.direct_to_reverb_ratio['SM'])} dB")
    print(f"  System loudspeakers to system microphones: {mag2db(physical_room.direct_to_reverb_ratio['LM'])} dB")
    print(f"  System loudspeakers to audience receivers: {mag2db(physical_room.direct_to_reverb_ratio['LA'])} dB")
    print(f"You can easily plot the coupling and the DRR with the related methods.")

    physical_room.plot_coupling()
    physical_room.plot_DRR()

    exit(0)