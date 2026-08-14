# ==================================================================
# ============================ IMPORTS =============================

# PyRES
from pyres.virtual_room import FDN
from pyres.physical_room import PhRoom_wgn
from pyres.res import RES


###########################################################################################
# In PyRES, the Reverberation Enhancement System (RES) is created by combining a physical room
# and a virtual room.
# The class RES receives a PhRoom and a VrRoom as input arguments, and controls the
# interaction between them.
# In this example, we create a RES by combining the physical room we created in the example E2
# and the virtual room we created in the example E1.
###########################################################################################


if __name__ == '__main__':

    # Time-frequency
    samplerate = 48000               # Sampling frequency
    nfft = samplerate*3              # FFT size
    alias_decay_db = 0               # Anti-time-aliasing decay in dB

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
    
    # Virtual room
    virtual_room = FDN(
        n_M = n_M,
        n_L = n_L,
        fs = samplerate,
        nfft = nfft,
        alias_decay_db = alias_decay_db,
        order = 16,
        t60_DC = 1.0,
        t60_NY = 0.2,
    )

    # Reverberation Enhancement System
    res = RES(
        physical_room = physical_room,
        virtual_room = virtual_room
    )
    res.set_G_to_GBI(gbi_estimation_criterion='eigenvalue_real_part')
    res.system_simulation()
    res.open_loop_eigenvalues()

    exit(0)