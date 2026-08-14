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
from pyres.plots import plot_irs_compare, plot_spectrograms_compare


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


if __name__ == '__main__':

    # Time-frequency parameters
    samplerate = 48000           # Sampling frequency
    nfft = samplerate*3          # FFT size
    alias_decay_db = 0           # Anti-time-aliasing decay
    
    # Virtual room
    n_inputs = 4                 # Number of inputs (system microphones)
    n_outputs = 8                # Number of outputs (system loudspeakers)

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

    print(f"\nThe {type(virtual_room).__name__} class is a subclass of the {type(virtual_room).__bases__[0].__name__} class, which contains the common features of all virtual room classes in PyRES.")

    print(f"The virtual room was created with {virtual_room.n_M} microphone inputs and {virtual_room.n_L} loudspeaker outputs.")
    print(f"Thus, the DSP has {virtual_room.get_v_ML().input_channels} input connections and {virtual_room.get_v_ML().output_channels} output connections.")

    print(f"The DSP architecture is a {type(virtual_room.get_v_ML())} instance.")
    print(f"And it containes the following modules:")
    for module in virtual_room.get_v_ML()._modules.values():
        print(type(module))

    print("Without instantiating a physical room, you can check how the DSP architectures filters the microphone signals by accessing the 'v_ML' attribute of the virtual room instance.")
    vrroom = virtual_room.get_v_ML()

    print("But remember that the processing is always done in the frequency domain.")
    x = signal_gallery(
        batch_size = 1,
        n_samples = nfft,
        n = n_inputs,
        signal_type = "impulse",
        fs = samplerate
    )
    X = dsp.FFT(nfft=nfft)(x)
    Y = vrroom(X)
    y = dsp.iFFT(nfft=nfft)(Y)

    # Plot the results
    plot_irs_compare(
        ir_1 = x.squeeze(0)[:samplerate,0],
        ir_2 = y.squeeze(0)[:samplerate,0],
        fs = samplerate,
        label1 = 'Microphone 1',
        label2 = 'Loudspeaker 1'
    )
    plot_spectrograms_compare(
        ir_1 = x.squeeze(0)[:,0],
        ir_2 = y.squeeze(0)[:,0],
        fs = samplerate,
        nfft = 2**11,
        noverlap = 2**10,
        label1 = 'Microphone 1',
        label2 = 'Loudspeaker 1'
    )

    exit(0)