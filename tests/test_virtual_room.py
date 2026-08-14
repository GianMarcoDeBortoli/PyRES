# ==================================================================
# ============================ IMPORTS =============================

# FLAMO
from flamo import dsp
from flamo.functional import signal_gallery
# PyRES
from pyres.virtual_room import (
    unitary_parallel_connections,
    unitary_mixing_matrix,
    random_FIRs,
    phase_cancellation,
    FDN,
    unitary_reverberator
)


###########################################################################################
# In PyRES, the DSP filtering the system microphone signals to produce the system loudspeaker
# signals is the virtual room.
# In this example, we create a virtual room using the FDN class.
# This class implements a static Feedback Delay Network, having input connections, output 
# connections, and order independent of each other.
# You can find other virtual room classes in the file 'virtual_room.py'.
# Use this file to inspect the other virtual room classes, by changing which class is
# instantiated in the main.
# Each virtual room class has its own set of input arguments, which are described in the
# class constructor.
###########################################################################################


def virtual_room_processing(virtual_room, nfft, samplerate, n_inputs):
    """
    Process an impulse signal through the virtual room and return the output.
    """
    vrroom = virtual_room.get_v_ML()
    x = signal_gallery(
        batch_size=1,
        n_samples=nfft,
        n=n_inputs,
        signal_type="impulse",
        fs=samplerate
    )
    X = dsp.FFT(nfft=nfft)(x)
    Y = vrroom(X)
    y = dsp.iFFT(nfft=nfft)(Y)
    return y

if __name__ == '__main__':

    # Time-frequency parameters
    samplerate = 48000           # Sampling frequency
    nfft = samplerate*3          # FFT size
    alias_decay_db = 0           # Anti-time-aliasing decay
    

    # Virtual room 1
    n_inputs = 4
    n_outputs = 4
    virtual_room = unitary_parallel_connections(
        n_M = n_inputs,
        n_L = n_outputs,
        fs = samplerate,
        nfft = nfft,
        alias_decay_db = alias_decay_db,
    )
    virtual_room_processing(virtual_room, nfft, samplerate, n_inputs)

    # Virtual room 2
    n_inputs = 8
    n_outputs = 8
    virtual_room = unitary_mixing_matrix(
        n_M = n_inputs,
        n_L = n_outputs,
        fs = samplerate,
        nfft = nfft,
        alias_decay_db = alias_decay_db,
    )
    virtual_room_processing(virtual_room, nfft, samplerate, n_inputs)

    # Virtual room 3
    n_inputs = 8
    n_outputs = 8
    virtual_room = random_FIRs(
        n_M = n_inputs,
        n_L = n_outputs,
        fs = samplerate,
        nfft = nfft,
        alias_decay_db = alias_decay_db,
        FIR_order = 100,
    )
    virtual_room_processing(virtual_room, nfft, samplerate, n_inputs)

    # Virtual room 4
    n_inputs = 8
    n_outputs = 8
    MR_n_modes = 10                 # Modal reverb number of modes
    MR_f_low = 50                   # Modal reverb lowest mode frequency
    MR_f_high = 450                 # Modal reverb highest mode frequency
    MR_t60 = 1.00                   # Modal reverb reverberation time
    virtual_room = phase_cancellation(
        n_M=n_inputs,
        n_L=n_outputs,
        fs=samplerate,
        nfft=nfft,
        n_modes=MR_n_modes,
        low_f_lim=MR_f_low,
        high_f_lim=MR_f_high,
        t60=MR_t60,
        requires_grad=True,
        alias_decay_db=alias_decay_db
    )
    virtual_room_processing(virtual_room, nfft, samplerate, n_inputs)

    # Virtual room 5
    n_inputs = 4
    n_outputs = 8
    virtual_room = FDN(
        n_M = n_inputs,
        n_L = n_outputs,
        fs = samplerate,
        nfft = nfft,
        alias_decay_db = alias_decay_db,
        order = 16,
        t60_DC = 1.0,
        t60_NY = 0.2,
    )
    virtual_room_processing(virtual_room, nfft, samplerate, n_inputs)

    # Virtual room 6
    n_inputs = 4
    n_outputs = 8
    virtual_room = unitary_reverberator(
        n_M = n_inputs,
        n_L = n_outputs,
        fs = samplerate,
        nfft = nfft,
        alias_decay_db = alias_decay_db,
        order = 16,
        t60 = 1.0
    )
    virtual_room_processing(virtual_room, nfft, samplerate, n_inputs)

    exit(0)