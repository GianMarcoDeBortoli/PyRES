# ==================================================================
# ============================ IMPORTS =============================
import os
import argparse
import time
import soundfile as sf
# PyTorch
import torch
# FLAMO
from flamo import system, dsp
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.functional import db2mag, mag2db, get_magnitude, get_eigenvalues
# PyRES
from pyres.res import RES
from pyres.physical_room import PhRoom_dataset
from pyres.virtual_room import random_FIRs
from pyres.loss_functions import MSE_evs_mod, MSE_max_evs
from pyres.functional import system_equalization_curve
from pyres.plots import plot_evs_compare, plot_spectrograms_compare

import matplotlib.pyplot as plt

###########################################################################################
# In this example, we train a virtual room to equalize the RES.
# The physical room is simulated with the PhRoom_wgn class.
# The virtual room is a mixing matrix of finite-impulse-response (FIR) filters.
# For more information about the classes Dataset and Trainer, please refer to the FLAMO
# documentation.
# The training pipeline is as follows:
# 1. Initialize the physical room and the virtual room.
# 2. Initialize the RES with the physical and virtual rooms.
# 3. Define the model as the open loop of the RES.
# 4. Initialize the dataset with the input and target signals.
#    The input signal is a batch of unit impulses.
#    The target signal is the equalization curve of the RES.
# 5. Initialize the trainer with the model and the dataset.
# 6. Define the loss function as the mean squared error between the target and the eigenvalues
#    of the RES open loop.
# 7. Train the model with the trainer.
# 8. Plot the eigenvalues and the spectrograms to compare the system responses before and
#    after optimization.
# 9. Save the model parameters (optional).
# This example, with the current training parameters, is meant as a proof of concept. 
# By increasing the size of the dataset (see hyperparameter `--num`), the optimizer will
# iterate over more training examples (more unit impulses), and the result will improve.

# Reference:
#     De Bortoli, G., Dal Santo, G., Prawda, K., Lokki, T., Välimäki, V., and Schlecht, S. J.
#     "Differentiable Active Acoustics: Optimizing Stability via Gradient Descent"
#     Proceedings of the International Conference on Digital Audio Effects, pp. 254-261, 2024.
###########################################################################################

torch.manual_seed(141122)

def train_virtual_room(args) -> None:

    # -------------------- Initialize RES ---------------------
    # Time-frequency
    samplerate = 48000                 # Sampling frequency in Hz
    nfft = samplerate*2                # FFT size

    # Physical room
    dataset_directory = '/Users/deborg1/Library/CloudStorage/OneDrive-AaltoUniversity/Documents/DataRES/data'
    room_name = 'Otala'
    subpath = 'RIRs/SystemSystem'
    # room_name = 'AmadeusHUB'
    # subpath = 'data'

    rir_length = 18000
    n_L = 13
    n_M = 4
    rirs = torch.zeros(rir_length, n_M, n_L)
    for e in range(n_L):
        for r in range(n_M):
            filename = f"{dataset_directory}/{room_name}/{subpath}/E{e+1:03d}_R{r+1:03d}_M01.wav"
            w, samplerate = sf.read(filename)
            rirs[:,r,e] = torch.tensor(w.squeeze())

    # rirs = torch.cat((rirs[:, :, :30], rirs[:, :, 31:]), dim=2)
    # n_L = 35

    physical_room = dsp.Filter(
        size = (rir_length, n_M, n_L),
        nfft = nfft,
        requires_grad = False,
    )
    physical_room.assign_value(rirs)

    # Virtual room
    fir_order = 2**8                   # FIR filter order
    
    filters = dsp.Filter(
        size=(fir_order, n_L, n_M),
        nfft=nfft,
        requires_grad=True,
    )
    eqs_params = filters.param.clone().detach()
    frob_norm_squared = torch.sum(torch.square(eqs_params))
    gain_value = torch.sqrt(n_M / frob_norm_squared)
    compensation_gain = dsp.parallelGain(
        size=(n_L,),
        nfft=nfft,
        requires_grad=False
    )
    compensation_gain.assign_value(torch.ones(n_L) * gain_value)
    
    virtual_room = system.Series(
        filters, compensation_gain
    )

    open_loop = system.Shell(
        core = system.Series(virtual_room, physical_room)
    )

    model = system.Shell(
        core=open_loop,
        input_layer=system.Series(
            dsp.FFT(nfft=nfft),
            dsp.Transform(lambda x: x.diag_embed())
        )
    )
    
    # ------------- Performance at initialization -------------
    open_loop_tfs = open_loop.get_freq_response(fs=samplerate, identity=True).squeeze()
    evs_init = get_eigenvalues(open_loop_tfs)
    
    # ----------------- Initialize dataset --------------------
    dataset_input = torch.zeros(1, samplerate, n_M)
    dataset_input[:,0,:] = 1
    feedback_rtfs = dsp.FFT(nfft=nfft)(rirs.unsqueeze(0)).squeeze(0)
    dataset_target, average_max_evs_curve, max_evs_curve_realization = system_equalization_curve(
        feedback=feedback_rtfs, samplerate=samplerate, nfft=nfft//2+1, mc_reps=1000, smoothing_fraction=1/6
    )
    f_axis = torch.linspace(torch.tensor(0.0), torch.tensor(samplerate//2), int(nfft/2)+1)
    plt.figure()
    plt.plot(f_axis, max_evs_curve_realization)
    plt.plot(f_axis, average_max_evs_curve)
    plt.plot(f_axis, dataset_target.squeeze())
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Magnitude')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['Open loop - max eigenvalue, one realization', 'Open loop - Average max eigenvalue over realizations', 'Target - smooth average max eigenvalue over realizations'], loc='best')
    plt.title('Target curve generation')
    plt.show()

    dataset_target = dataset_target.view(1,-1,1)

    dataset = Dataset(
        input = dataset_input,
        target = dataset_target,
        expand = args.num,
        device = args.device
        )
    train_loader, valid_loader  = load_dataset(dataset, batch_size=1, split=args.split, shuffle=False)

    # ------------------- Initialize Trainer ------------------
    trainer = Trainer(
        net=model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        patience_delta=args.patience_delta,
        train_dir=args.train_dir,
        device=args.device
    )

    # ---------------- Initialize Loss Function ---------------
    criterion = MSE_evs_mod(
        iter_num = args.num,
        freq_points = nfft//2+1,
        samplerate = samplerate,
        flim_l = 20,
        flim_r = 20000
    )
    trainer.register_criterion(criterion, 1.0)
    
    # ------------------- Train the model --------------------
    trainer.train(train_loader, valid_loader)

    # ------------ Performance after optimization ------------
    open_loop_tfs = open_loop.get_freq_response(fs=samplerate, identity=True).squeeze()
    evs_opt = get_eigenvalues(open_loop_tfs)
    
    # ------------------------ Plots -------------------------
    plot_evs_compare(evs_init, evs_opt, samplerate, nfft, 20, 8000)

    evs_init_max = torch.max(torch.abs(evs_init), dim=1).values
    evs_opt_max = torch.max(torch.abs(evs_opt), dim=1).values
    plt.figure()
    plt.plot(evs_init_max)
    plt.plot(evs_opt_max)
    plt.plot(dataset_target.squeeze())
    plt.xlabel('Frequency bin')
    plt.ylabel('Magnitude in dB')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(['Init', 'Opt', 'Target'], loc='best')
    plt.show(block=True)

    # ---------------- Save the model parameters -------------
    # If desired, you can use the following line to save the virtual room model state.
    # res.save_state_to(directory='./model_states/')
    # The model state can be then loaded in another instance of the same virtual room to skip the training.
    data = filters.param.clone().detach()
    frob_norm_squared = torch.sum(torch.square(eqs_params))
    norm_factor = torch.sqrt(n_M / frob_norm_squared)
    data_normalized = data * norm_factor
    for i in range(n_M):
        mic_data = data_normalized[:,:,i]
        sf.write(
            f"Mic{i+1}.wav",
            mic_data.cpu().numpy(),
            48000,
        )

    return None


###########################################################################################

if __name__ == '__main__':

    # Define training pipeline hyperparameters
    parser = argparse.ArgumentParser()
    
    #----------------------- Dataset ----------------------
    parser.add_argument('--num', type=int, default=2**8,help = 'dataset size')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for computation')
    parser.add_argument('--split', type=float, default=0.8, help='split ratio for training and validation')
    #---------------------- Training ----------------------
    parser.add_argument('--train_dir', type=str, help='directory to save training results')
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs')
    parser.add_argument('--patience_delta', type=float, default=1e-8, help='Minimum improvement in validation loss to be considered as an improvement')
    #---------------------- Optimizer ---------------------
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    #----------------- Parse the arguments ----------------
    args = parser.parse_args()

    # make training output directory
    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join('training_output', time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.train_dir)

    # save arguments 
    with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # Run script
    train_virtual_room(args)