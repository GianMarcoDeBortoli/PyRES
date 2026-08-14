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

    exit(0)