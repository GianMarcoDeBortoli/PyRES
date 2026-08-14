# ==================================================================
# ============================ IMPORTS =============================

# PyRES
from pyres.physical_room import PhRoom_wgn


###########################################################################################
# In PyRES, the physical space where the Reverberation Enhancement System (RES) operates is
# the physical room.
# In this example, we create a physical room by simulating the room impulse responses (RIRs)
# using exponentially decaying white Gaussian noise.
###########################################################################################


if __name__ == '__main__':

    # Time-frequency parameters
    samplerate = 48000              # Sampling frequency
    nfft = samplerate*3             # FFT size
    alias_decay_db = 0              # Anti-time-aliasing decay in dB

    # Physical room
    room_dims = (12.1, 8.5, 3.2)    # Room dimensions in meters (length, width, height)
    room_RT = 0.7                   # Reverberation time in seconds
    n_M = 4                         # Number of microphones
    n_L = 8                         # Number of loudspeakers
    
    physical_room = PhRoom_wgn(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        room_dims=room_dims,
        room_RT=room_RT,
        n_M=n_M,
        n_L=n_L
    )

    print(f"\nThe PhRoom_wgn class is a subclass of the {type(physical_room).__bases__[0].__name__} class, which contains the common features of all physical room classes in PyRES.")
    print(f"The physical room was created with {n_M} microphones and {n_L} loudspeakers, and the simulations were performed considering 1 stage emitter and 1 audience receiver.")
    print(f"The information about all emitters and receivers are contained in the 'transducer_number' and 'transducer_positions' attributes, respectively.")
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
    print(f"You can easily plot the room setup with the related method.")

    physical_room.plot_setup()

    print(f"\nThe RIRs are contained in the 'h_SA', 'h_SM', 'h_LM', and 'h_LA' attributes, which are the RIRs between, respectively:")
    print(f"  stage emitters and audience receivers: {type(physical_room.get_stg_to_aud())}, with shape {physical_room.get_stg_to_aud().param.shape}")
    print(f"  stage emitters and system microphones: {type(physical_room.get_stg_to_mcs())}, with shape {physical_room.get_stg_to_mcs().param.shape}")
    print(f"  system loudspeakers and system microphones: {type(physical_room.get_lds_to_mcs())}, with shape {physical_room.get_lds_to_mcs().param.shape}")
    print(f"  system loudspeakers and audience receivers: {type(physical_room.get_lds_to_aud())}, with shape {physical_room.get_lds_to_aud().param.shape}")

    exit(0)