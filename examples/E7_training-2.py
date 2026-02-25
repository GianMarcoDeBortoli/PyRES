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
from PyRES.physical_room import PhRoom_dataset
from PyRES.virtual_room import phase_cancellation
from PyRES.loss_functions import MSE_evs_idxs, colorless_reverb
from PyRES.plots import plot_evs_compare, plot_spectrograms_compare

############################################################################################
# In this example, we train a virtual room to generate feedback cancellation in the RES.
# The physical room is loaded from the DataRES dataset. with the PhRoom_dataset class.
# The virtual room is a phase-canceling modal reverb.
# For more information about the classes Dataset and Trainer, please refer to the FLAMO
# documentation.
# The training pipeline is as follows:
# 1. Initialize the physical room and the virtual room.
# 2. Initialize the RES with the physical and virtual rooms.
# 3. Define the model as the open loop of the RES.
# 4. Initialize the dataset with the input and target signals.
#    The input signal is a batch of unit impulses.
#    The target signal is a batch of zeros.
# 5. Initialize the trainer with the model and the dataset.
# 6. Define the loss function as the mean squared error between the target and the eigenvalues
#    of the RES open loop. In addition, we register a second loss that maitains the energy 
#    of the modal reverberator as high as at initialization.
# 7. Train the model with the trainer.
# 8. Plot the eigenvalues and the spectrograms of the system responses before and
#    after optimization.
# 9. Save the model parameters.
# This example, with the current training parameters, is meant as a proof of concept. 
# By increasing the size of the dataset (see hyperparameter `--num`), the optimizer will
# iterate over more training examples (more unit impulses), and the result will improve.

# Reference:
#   De Bortoli, G., Prawda, K., and Schlecht, S. J.
#   "Active Acoustics with a Phase Cancelling Modal Reverberator"
#   Journal of the Audio Engineering Society, Vol. 72, No. 10, pp. 705-715, 2024.
############################################################################################

torch.manual_seed(141122)

def train_virtual_room(args) -> None:

    # --------------------- Parameters ------------------------
    # Time-frequency
    samplerate = 1000               # Sampling frequency
    nfft = samplerate*3             # FFT size
    alias_decay_db = -20            # Anti-time-aliasing decay in dB

    # Physical room
    dataset_directory = './dataRES'
    room_name = 'Otala'

    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )
    n_M = physical_room.transducer_number['mcs']  # Number of microphones
    n_L = physical_room.transducer_number['lds']  # Number of loudspeakers

    # Virtual room
    MR_n_modes = 120                # Modal reverb number of modes
    MR_f_low = 50                   # Modal reverb lowest mode frequency
    MR_f_high = 450                 # Modal reverb highest mode frequency
    MR_t60 = 1.00                   # Modal reverb reverberation time
    virtual_room = phase_cancellation(
        n_M=n_M,
        n_L=n_L,
        fs=samplerate,
        nfft=nfft,
        n_modes=MR_n_modes,
        low_f_lim=MR_f_low,
        high_f_lim=MR_f_high,
        t60=MR_t60,
        requires_grad=True,
        alias_decay_db=alias_decay_db
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
    dataset_input = torch.zeros(1, nfft//2+1, n_M)
    dataset_input[:,0,:] = 1
    dataset_target = torch.zeros(1, nfft//2+1, n_M)
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
    MR_freqs = virtual_room.get_v_ML()[0].resonances[:,0,0].clone().detach()
    criterion1 = MSE_evs_idxs(
        iter_num = args.num,
        freq_points = nfft//2+1,
        samplerate = samplerate,
        freqs = MR_freqs
    )
    trainer.register_criterion(criterion1, 1.0)

    criterion2 = colorless_reverb(
        samplerate = samplerate,
        freq_points = nfft//2+1,
        freqs = MR_freqs
    )
    trainer.register_criterion(criterion2, 0.2, requires_model=True)
    
    # -------------------- Train the model --------------------
    trainer.train(train_loader, valid_loader)

    # ------------- Performance after optimization ------------
    evs_opt = res.open_loop_eigenvalues()
    _,_,ir_opt = res.system_simulation()
    
    # ------------------------- Plots -------------------------
    plot_evs_compare(evs_init, evs_opt, samplerate, nfft, 40, 460)
    plot_spectrograms_compare(ir_init, ir_opt, samplerate, nfft=2**4, noverlap=2**3)

    # ---------------- Save the model parameters -------------
    # If desired, you can use the following line to save the virtual room model state.
    # res.save_state_to(directory='./model_states/')
    # The model state can be then load in another instance of the same virtual room to skip the training.

    return None


###########################################################################################

if __name__ == '__main__':

    # Define training pipeline hyperparameters
    parser = argparse.ArgumentParser()
    
    #----------------------- Dataset ----------------------
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--num', type=int, default=2**4,help = 'dataset size')
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