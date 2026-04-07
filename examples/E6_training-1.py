# ==================================================================
# ============================ IMPORTS =============================
import argparse
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# PyTorch
import torch
# FLAMO
from flamo import system, dsp
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer
# PyRES
from PyRES.res import RES
from PyRES.physical_room import PhRoom_wgn
from PyRES.virtual_room import random_FIRs
from PyRES.loss_functions import MSE_evs_mod
from PyRES.functional import system_equalization_curve
from PyRES.plots import plot_evs_compare, plot_spectrograms_compare

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
    nfft = samplerate*3                # FFT size
    alias_decay_db = 0                 # Anti-time-aliasing decay in dB

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
    fir_order = 2**8                   # FIR filter order
    virtual_room = random_FIRs(
        n_M=n_M,
        n_L=n_L,
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        FIR_order=fir_order,
        requires_grad=True
    )

    # Reverberation Enhancement System
    res = RES(
        physical_room = physical_room,
        virtual_room = virtual_room
    )

    # ------------------- Model Definition --------------------
    model = system.Shell(
        core=res.open_loop(),
        input_layer=system.Series(
            dsp.FFT(nfft=nfft),
            dsp.Transform(lambda x: x.diag_embed())
        )
    )
    
    # ------------- Performance at initialization -------------
    evs_init = res.open_loop_eigenvalues()
    _,_,ir_init = res.system_simulation()
    
    # ----------------- Initialize dataset --------------------
    dataset_input = torch.zeros(1, samplerate, n_M)
    dataset_input[:,0,:] = 1
    dataset_target = system_equalization_curve(evs=evs_init, fs=samplerate, nfft=nfft, f_c=8000)
    dataset_target = dataset_target.view(1,-1,1).expand(1, -1, n_M)

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
        lowest_f = 20,
        highest_f = 15000
    )
    trainer.register_criterion(criterion, 1.0)
    
    # ------------------- Train the model --------------------
    trainer.train(train_loader, valid_loader)

    # ------------ Performance after optimization ------------
    evs_opt = res.open_loop_eigenvalues()
    _,_,ir_opt = res.system_simulation()
    
    # ------------------------ Plots -------------------------
    plot_evs_compare(evs_init, evs_opt, samplerate, nfft, 20, 8000)
    plot_spectrograms_compare(ir_init[:,0], ir_opt[:,0], samplerate, nfft=2**11, noverlap=2**10)

    # ---------------- Save the model parameters -------------
    # If desired, you can use the following line to save the virtual room model state.
    # res.save_state_to(directory='./model_states/')
    # The model state can be then loaded in another instance of the same virtual room to skip the training.

    return None


###########################################################################################

if __name__ == '__main__':

    # Define training pipeline hyperparameters
    parser = argparse.ArgumentParser()
    
    #----------------------- Dataset ----------------------
    parser.add_argument('--num', type=int, default=2**5,help = 'dataset size')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for computation')
    parser.add_argument('--split', type=float, default=0.8, help='split ratio for training and validation')
    #---------------------- Training ----------------------
    parser.add_argument('--train_dir', type=str, help='directory to save training results')
    parser.add_argument('--max_epochs', type=int, default=10, help='maximum number of epochs')
    parser.add_argument('--patience_delta', type=float, default=1e-4, help='Minimum improvement in validation loss to be considered as an improvement')
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