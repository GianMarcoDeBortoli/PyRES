# ==================================================================
# ============================ IMPORTS =============================
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# FLAMO
from flamo.functional import mag2db
# PyRES
from PyRES.virtual_room import FDN
from PyRES.physical_room import PhRoom_wgn
from PyRES.res import RES
from PyRES.plots import plot_spectrograms_compare, plot_evs_distribution


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

    print(f"\nThe RES class is a container for the PhRoom and VrRoom class instances:")
    print(f"  Physical room: {type(res.phroom).__name__}")
    print(f"  Virtual room: {type(res.vrroom).__name__}")
    print(f"The RES class also hosts the system gain G, which is the master gain of the audio setup.")
    print(f"At first the system gain is set to GBI - 3 dB, where GBI is an estimation of the gain before instability.")
    print(f"  System gain G: {type(res.G)}, with value {mag2db(res.G.param[0])} dB")

    print(f"\nThis means that the system is initially instantiated in a stable state.")
    print(f"You can simulated the RES with the related method, which we will compare to the natural acoustics of the physical room.")
    print(f"The system simulation returns the room impulse response between the stage emitters and the audience receivers for three conditions: system off, only ea setup (neglecting natural path h_SA), and system on.")

    nat_rirs, _, res_rirs = res.system_simulation()

    plot_spectrograms_compare(
        ir_1 = nat_rirs[:2*samplerate,0],
        ir_2 = res_rirs[:2*samplerate,0],
        fs = samplerate,
        nfft = 2**11,
        noverlap = 2**10,
        label1 = 'RES off',
        label2 = 'RES on'
    )

    print(f"\nYou can estimate the GBI with the related method:")
    print(f"  GBI: {mag2db(res.compute_GBI(criterion='eigenvalue_magnitude'))} dB")
    print(f"You can set the system gain G to any value you desire with the method '.set_G()'.")
    print(f"Let's set the system gain G to the GBI. There is a method dedicated to this:")

    res.set_G_to_GBI(gbi_estimation_criterion='eigenvalue_magnitude')
    nat_rirs, _, res_rirs = res.system_simulation()

    plot_spectrograms_compare(
        ir_1 = nat_rirs[:2*samplerate,0],
        ir_2 = res_rirs[:2*samplerate,0],
        fs = samplerate,
        nfft = 2**11,
        noverlap = 2**10,
        label1 = 'Microphone 1',
        label2 = 'Loudspeaker 1'
    )

    print(f"\nThe system is still stable because the estimation methods tend to be conservative. You can obtain a better estimate of the true GBI by using the estimation criterion 'eigenvalue_real_part', instead of 'eigenvalue_magnitude'.")

    print(f"\nAdditionally, the RES class provides functionalities to check other important metrics of the system.")
    print(f"  The method '.open_loop()' will return the open-loop as a processing chain. You can see that the open loop is the series of the virtual room, the system gain, and the physical room:")
    for module in res.open_loop()._modules.values():
        print(type(module))
    print(f"  The method '.open_loop_responses()' will return the impulse responses and the frequency responses of the open-loop matrix")
    print(f"  Impulse responses of shape: {res.open_loop_responses()[0].shape}, frequency responses of shape: {res.open_loop_responses()[1].shape}")
    print(f"  The method '.closed_loop()' will return the closed-loop as a processing chain. You can see that the closed loop is a recursion:")
    print(type(res.closed_loop()))
    print(f"  The closed loop has the virtual room and the system gain in the feedforward path, and the physical room in the feedback path:")
    for module in res.closed_loop().feedforward._modules.values():
        print(type(module))
    print(type(res.closed_loop().feedback))
    print(f"  The method '.closed_loop_responses()' will return the impulse responses and the frequency responses of the closed-loop matrix")
    print(f"  Impulse responses of shape: {res.closed_loop_responses()[0].shape}, frequency responses of shape: {res.closed_loop_responses()[1].shape}")
    print(f"  The method '.open_loop_eigenvalues()' will return the complex eigenvalues of the open-loop matrix, with shape {res.open_loop_eigenvalues().shape}")
    print(f"  The open-loop eigenvalues are a metric used to evaluate the energy flow in the RES.")
    print(f"The flatter the magnitude distribution of the open-loop eigenvalues is, the more the energy is evenly distributed across frequencies and eigenchannels.")

    evs = res.open_loop_eigenvalues()

    plot_evs_distribution(
        evs=evs,
        fs=samplerate,
        nfft=nfft,
        lower_f_lim=20,
        higher_f_lim=20000
    )

    exit(0)