import sys
import argparse
import os
import time

import torch

from flamo import system, dsp
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer
from PyRES.loss_functions import MSE_evs_idxs, colorless_reverb
from flamo.functional import db2mag, mag2db

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PyRES.res import RES
from PyRES.physical_room import PhRoom_dataset
from PyRES.virtual_room import *
from PyRES.plots import *
from PyRES.utils import *

torch.manual_seed(130297)


def all_dsps(n_mcs, n_lds, samplerate, nfft, alias_decay_db) -> tuple[random_FIRs, FDN, unitary_mixing_matrix, phase_canceling_modal_reverb, unitary_reverberator]:

    # Virtual rooms
    fir_order = 2**8                   # FIR filter order
    firs = random_FIRs(
        n_M=n_mcs,
        n_L=n_lds,
        nfft=nfft,
        FIR_order=fir_order,
        alias_decay_db=alias_decay_db,
        requires_grad=True
    )

    t60_DC = 1.50                       # FDN reverberation time at 0 Hz
    t60_Ny = 0.20                       # FDN reverberation time at Nyquist frequency
    fdn = FDN(
        n_M = n_mcs,
        n_L = n_lds,
        fs = samplerate,
        nfft = nfft,
        t60_DC = t60_DC,
        t60_NY = t60_Ny,
        alias_decay_db = alias_decay_db
    )

    mm = unitary_mixing_matrix(
        n_M=n_mcs,
        n_L=n_lds,
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db
    )

    MR_n_modes = 150                   # Modal reverb number of modes
    MR_f_low = 50                      # Modal reverb lowest mode frequency
    MR_f_high = 450                    # Modal reverb highest mode frequency
    MR_t60 = 1.50                      # Modal reverb reverberation time
    mr = phase_canceling_modal_reverb(
        n_M=n_mcs,
        n_L=n_lds,
        fs=1000,
        nfft=2000,
        n_modes=MR_n_modes,
        low_f_lim=MR_f_low,
        high_f_lim=MR_f_high,
        t60=MR_t60,
        requires_grad=True,
        alias_decay_db=alias_decay_db
    )

    t60 = 1.50                          # Unitary allpass reverb reverberation time
    ur = unitary_reverberator(
        n_M = n_mcs,
        n_L = n_lds,
        fs = samplerate,
        nfft = nfft,
        t60 = t60,
        alias_decay_db = alias_decay_db,
    )

    return firs, fdn, mm, mr, ur

def dafx_figures_PhRoom(args) -> None:

    # --------------------- Parameters ------------------------
    # Time-frequency
    samplerate = 48000                 # Sampling frequency
    nfft = samplerate*2                # FFT size
    alias_decay_db = 0                 # Anti-time-aliasing decay in dB

    # Physical room
    room_dataset = './dataRES'      # Path to the dataset
    room = 'Otala'                  # Path to the room impulse responses
    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=room_dataset,
        room_name=room
    )
    plot_room_setup(physical_room)

    ph_rirs = physical_room.get_rirs()
    plot_coupling_pro_version(rirs=ph_rirs, fs=samplerate)
    plot_DRR_pro_version(rirs=ph_rirs, fs=samplerate)

    return None


def dafx_figures_VrRoom(args) -> None:

    # --------------------- Parameters ------------------------
    # Time-frequency
    samplerate = 48000                 # Sampling frequency
    nfft = samplerate*2                # FFT size
    alias_decay_db = 0                 # Anti-time-aliasing decay in dB

    firs, fdn, mm, mr, ur = all_dsps(16, 16, samplerate, nfft, alias_decay_db)

    firs_rirs = firs.param.clone().detach()

    test = system.Shell(core=fdn)
    fdn_rirs = test.get_time_response(identity=True).squeeze()

    test = system.Shell(core=mm)
    mm_rirs = test.get_time_response(identity=True).squeeze()

    test = system.Shell(core=mr)
    mr_rirs = test.get_time_response(identity=True).squeeze()

    test = system.Shell(core=ur)
    ur_rirs = test.get_time_response(identity=True).squeeze()

    plot_DAFx(mm_rirs, firs_rirs, mr_rirs, fdn_rirs, ur_rirs, samplerate, nfft)
    
    return None


def dafx_big_figure(args) -> None:

    # --------------------- Parameters ------------------------
    # Time-frequency
    samplerate = 48000                 # Sampling frequency
    nfft = samplerate*2                # FFT size
    alias_decay_db = 0                 # Anti-time-aliasing decay in dB

    idx1 = int(nfft/samplerate*20)
    idx2 = int(nfft/samplerate*8000)

    # Dataset directory
    room_dataset = './dataRES'      # Path to the dataset

    # Physical room 1
    room1 = 'Otala'                  # Path to the room impulse responses
    physical_room1 = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=room_dataset,
        room_name=room1
    )
    n_mcs = physical_room1.transducer_number['mcs']
    n_lds = physical_room1.transducer_number['lds']

    # Virtual rooms per room 1
    firs1, fdn1, mm1, mr1, ur1 = all_dsps(n_mcs, n_lds, samplerate, nfft, alias_decay_db)

    res11 = RES(physical_room=physical_room1, virtual_room=mm1)
    aur11 = res11.system_simulation()[:,0].squeeze()
    evs11 = mag2db(res11.open_loop_eigenvalues())[idx1:idx2,:]

    res21 = RES(physical_room=physical_room1, virtual_room=firs1)
    aur21 = res21.system_simulation()[:,0].squeeze()
    evs21 = mag2db(res21.open_loop_eigenvalues())[idx1:idx2,:]

    res31 = RES(physical_room=physical_room1, virtual_room=fdn1)
    aur31 = res31.system_simulation()[:,0].squeeze()
    evs31 = mag2db(res31.open_loop_eigenvalues())[idx1:idx2,:]

    res41 = RES(physical_room=physical_room1, virtual_room=ur1)
    aur41 = res41.system_simulation()[:,0].squeeze()
    evs41 = mag2db(res41.open_loop_eigenvalues())[idx1:idx2,:]

    physical_room1_resampled = PhRoom_dataset(
        fs=1000,
        nfft=2000,
        alias_decay_db=alias_decay_db,
        dataset_directory=room_dataset,
        room_name=room1
    )
    res51 = RES(physical_room=physical_room1_resampled, virtual_room=mr1)
    aur51 = res51.system_simulation()[:,0].squeeze()
    evs51 = mag2db(res51.open_loop_eigenvalues())[2*50:2*450,:]

    # Physical room 2
    room2 = 'ImmersiveLab'                  # Path to the room impulse responses
    physical_room2 = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=room_dataset,
        room_name=room2
    )
    n_mcs = physical_room2.transducer_number['mcs']
    n_lds = physical_room2.transducer_number['lds']

    # Virtual rooms per room 2
    firs2, fdn2, mm2, mr2, ur2 = all_dsps(n_mcs, n_lds, samplerate, nfft, alias_decay_db)

    res12 = RES(physical_room=physical_room2, virtual_room=mm2)
    aur12 = res12.system_simulation()[:,0].squeeze()
    evs12 = mag2db(res12.open_loop_eigenvalues())[idx1:idx2,:]

    res22 = RES(physical_room=physical_room2, virtual_room=firs2)
    aur22 = res22.system_simulation()[:,0].squeeze()
    evs22 = mag2db(res22.open_loop_eigenvalues())[idx1:idx2,:]

    res32 = RES(physical_room=physical_room2, virtual_room=fdn2)
    aur32 = res32.system_simulation()[:,0].squeeze()
    evs32 = mag2db(res32.open_loop_eigenvalues())[idx1:idx2,:]

    res42 = RES(physical_room=physical_room2, virtual_room=ur2)
    aur42 = res42.system_simulation()[:,0].squeeze()
    evs42 = mag2db(res42.open_loop_eigenvalues())[idx1:idx2,:]

    physical_room2_resampled = PhRoom_dataset(
        fs=1000,
        nfft=2000,
        alias_decay_db=alias_decay_db,
        dataset_directory=room_dataset,
        room_name=room2
    )
    res52 = RES(physical_room=physical_room2_resampled, virtual_room=mr2)
    aur52 = res52.system_simulation()[:,0].squeeze()
    evs52 = mag2db(res52.open_loop_eigenvalues())[2*50:2*450,:]

    # Physical room 3
    room3 = 'GLiveLab-Tampere'                  # Path to the room impulse responses
    physical_room3 = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=room_dataset,
        room_name=room3
    )
    n_mcs = physical_room3.transducer_number['mcs']
    n_lds = physical_room3.transducer_number['lds']

    # Virtual rooms per room 3
    firs3, fdn3, mm3, mr3, ur3 = all_dsps(n_mcs, n_lds, samplerate, nfft, alias_decay_db)

    res13 = RES(physical_room=physical_room3, virtual_room=mm3)
    aur13 = res13.system_simulation()[:,0].squeeze()
    evs13 = mag2db(res13.open_loop_eigenvalues())[idx1:idx2,:]

    res23 = RES(physical_room=physical_room3, virtual_room=firs3)
    aur23 = res23.system_simulation()[:,0].squeeze()
    evs23 = mag2db(res23.open_loop_eigenvalues())[idx1:idx2,:]

    res33 = RES(physical_room=physical_room3, virtual_room=fdn3)
    aur33 = res33.system_simulation()[:,0].squeeze()
    evs33 = mag2db(res33.open_loop_eigenvalues())[idx1:idx2,:]

    res43 = RES(physical_room=physical_room3, virtual_room=ur3)
    aur43 = res43.system_simulation()[:,0].squeeze()
    evs43 = mag2db(res43.open_loop_eigenvalues())[idx1:idx2,:]

    physical_room3_resampled = PhRoom_dataset(
        fs=1000,
        nfft=2000,
        alias_decay_db=alias_decay_db,
        dataset_directory=room_dataset,
        room_name=room3
    )
    res53 = RES(physical_room=physical_room3_resampled, virtual_room=mr3)
    aur53 = res53.system_simulation()[:,0].squeeze()
    evs53 = mag2db(res53.open_loop_eigenvalues())[2*50:2*450,:]

    fs = [samplerate, samplerate, samplerate,
          samplerate, samplerate, samplerate,
          samplerate, samplerate, samplerate,
          samplerate, samplerate, samplerate,
          1000, 1000, 1000,]
    
    N = [2**11, 2**11, 2**11,
            2**11, 2**11, 2**11,
            2**11, 2**11, 2**11,
            2**11, 2**11, 2**11,
            2**6, 2**6, 2**6]
    
    noverlap = [2**10, 2**10, 2**10,
                2**10, 2**10, 2**10,
                2**10, 2**10, 2**10,
                2**10, 2**10, 2**10,
                2**5, 2**5, 2**5]
    
    tensor_pairs = [(evs11, aur11),(evs12, aur12),(evs13, aur13),
                    (evs21, aur21),(evs22, aur22),(evs23, aur23),
                    (evs31, aur31),(evs32, aur32),(evs33, aur33),
                    (evs41, aur41),(evs42, aur42),(evs43, aur43),
                    (evs51, aur51),(evs52, aur52),(evs53, aur53)]

    plot_grid_boxplot_spectrogram(
        fs=fs,
        nfft=N,
        noverlap=noverlap,
        tensor_pairs=tensor_pairs,
        rows=5,
        cols=3,
        row_labels=['Unitary mixing matrix', 'FIRs', 'FDN', 'Unitary reverb', 'Modal reverb'],
        col_labels=['Otala', 'ImmersiveLab', 'GLivelab-Tampere'],
        figsize=(8,9),
        cmap='magma'
    )

    return None

def dafx_figures_dafx24(args) -> None:

    # --------------------- Parameters ------------------------
    # Time-frequency
    samplerate = 48000                 # Sampling frequency
    nfft = samplerate*2                # FFT size
    alias_decay_db = 0                 # Anti-time-aliasing decay in dB

    # Physical room
    room_dataset = './dataRES'      # Path to the dataset
    room = 'Otala'                  # Path to the room impulse responses
    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=room_dataset,
        room_name=room
    )
    _, n_mcs, n_lds, _ = physical_room.get_ems_rcs_number()

    # Virtual rooms
    fir_order = 2**8                   # FIR filter order
    virtual_room = random_FIRs(
        n_M=n_mcs,
        n_L=n_lds,
        nfft=nfft,
        FIR_order=fir_order,
        alias_decay_db=alias_decay_db,
        requires_grad=True
    )

    # Reverberation enhancement system
    res = RES(
        physical_room=physical_room,
        virtual_room=virtual_room
    )
    gbi = mag2db(res.compute_GBI())
    res.set_G(db2mag(gbi-1))

    # ------------------- Initialization ----------------------
    evs_init = res.open_loop_eigenvalues()
    irs_init = res.system_simulation()

    # -------------------- Optimization -----------------------
    virtual_room.load_state_dict(torch.load('./model_states/FIRs_Otala.pt'))
    # gbi = mag2db(res.compute_GBI())
    # res.set_G(db2mag(gbi-2))
    evs_opt = res.open_loop_eigenvalues()
    irs_opt = res.system_simulation()

    # plot_evs(evs_init=evs_init, evs_opt=evs_opt, fs=samplerate, nfft=nfft, lower_f_lim=20, higher_f_lim=8000)
    # plot_spectrograms(y_1=irs_init.squeeze()[:,0], y_2=irs_opt.squeeze()[:,0], fs=samplerate, nfft = 2**9, noverlap=2**8)
    curve = system_equalization_curve(
        evs=evs_init,
        fs=samplerate,
        nfft=nfft,
        f_c=9000
    )
    plot_eq_curve(curve, samplerate, nfft)
    plot_combined_figure(samplerate, nfft, evs_init, evs_opt, [20,10000], irs_init.squeeze(), irs_opt.squeeze(), [20,20000], cmap='magma')

    return None

def dafx_figures_jaes24(args):

    # --------------------- Parameters ------------------------
    # Time-frequency
    samplerate = 1000                  # Sampling frequency
    nfft = samplerate*3                # FFT size
    alias_decay_db = -20                 # Anti-time-aliasing decay in dB

    # Physical room
    room_dataset = './dataRES'      # Path to the dataset
    room = 'Otala'                  # Path to the room impulse responses
    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=room_dataset,
        room_name=room
    )
    n_mcs = physical_room.transducer_number['mcs']
    n_lds = physical_room.transducer_number['lds']

    # Virtual rooms
    MR_n_modes = 120                   # Modal reverb number of modes
    MR_f_low = 50                      # Modal reverb lowest mode frequency
    MR_f_high = 450                    # Modal reverb highest mode frequency
    MR_t60 = 1.00                      # Modal reverb reverberation time
    virtual_room = phase_canceling_modal_reverb(
        n_M=n_mcs,
        n_L=n_lds,
        fs=samplerate,
        nfft=nfft,
        n_modes=MR_n_modes,
        low_f_lim=MR_f_low,
        high_f_lim=MR_f_high,
        t60=MR_t60,
        requires_grad=True,
        alias_decay_db=alias_decay_db
    )
    init = system.Shell(core=virtual_room)
    tfs_init = init.get_freq_response(identity=True).squeeze()

    # Reverberation enhancement system
    res = RES(
        physical_room=physical_room,
        virtual_room=virtual_room
    )
    gbi = mag2db(res.compute_GBI())
    print(f'Initial GBI: {gbi:.2f} dB')
    res.set_G(db2mag(gbi - 2))

    # ------------------- Initialization ----------------------
    evs_init = res.open_loop_eigenvalues()
    irs_init = res.system_simulation()

    # -------------------- Optimization -----------------------
    virtual_room.load_state_dict(torch.load('./model_states/modalReverb_Otala.pt'))
    opt = system.Shell(core=virtual_room)
    tfs_opt = opt.get_freq_response(identity=True).squeeze()
    gbi = mag2db(res.compute_GBI())
    print(f'Optimized GBI: {gbi:.2f} dB')
    # res.set_G(db2mag(gbi - 3))
    evs_opt = res.open_loop_eigenvalues()
    irs_opt = res.system_simulation()

    # plot_evs(evs_init=evs_init, evs_opt=evs_opt, fs=samplerate, nfft=nfft, lower_f_lim=MR_f_low-10, higher_f_lim=MR_f_high+10)
    plot_combined_figure(samplerate, nfft, evs_init, evs_opt, [30,470], irs_init[:,0].squeeze(), irs_opt[:,0].squeeze(), [20,500], cmap='magma')

    return None


###########################################################################################

if __name__ == '__main__':

    # Define system parameters and pipeline hyperparameters
    parser = argparse.ArgumentParser()
    
    #----------------------- Dataset ----------------------
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--num', type=int, default=2**8,help = 'dataset size')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for computation')
    parser.add_argument('--split', type=float, default=0.8, help='split ratio for training and validation')
    #---------------------- Training ----------------------
    parser.add_argument('--train_dir', type=str, help='directory to save training results')
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs')
    parser.add_argument('--patience_delta', type=float, default=0.005, help='Minimum improvement in validation loss to be considered as an improvement')
    #---------------------- Optimizer ---------------------
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    #----------------- Parse the arguments ----------------
    args = parser.parse_args()

    # make output directory
    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join('training_output', time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.train_dir)

    # save arguments 
    with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # Run examples
    dafx_figures_jaes24(args)