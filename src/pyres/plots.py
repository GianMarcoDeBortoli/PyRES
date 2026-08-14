# ==================================================================
# ============================ IMPORTS =============================
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import colors
import seaborn as sns
import numpy as np
# PyTorch
import torch
# FLAMO
from flamo.functional import mag2db, get_magnitude


# ==================================================================
# ========================== PHYSICAL ROOM =========================

def plot_room_setup(positions: OrderedDict):

    stg = positions['stg']
    mcs = positions['mcs']
    lds = positions['lds']
    aud = positions['aud']

    if stg == None: stg = torch.tensor([])
    else: stg = torch.tensor(positions['stg'])
    if mcs == None: mcs = torch.tensor([])
    else: mcs = torch.tensor(positions['mcs'])
    if lds == None: lds = torch.tensor([])
    else: lds = torch.tensor(positions['lds'])
    if aud == None: aud = torch.tensor([])
    else: aud = torch.tensor(positions['aud'])

    if torch.sum(torch.tensor([len(stg), len(mcs), len(lds), len(aud)])) == 0:
        print("Audio setup data is not present for this room.")
        return None

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    colorPalette = [
        "#E3C21C",
        "#3364D7",
        "#1AB759",
        "#D51A43"
    ]

    # Use constrained layout
    fig = plt.figure(figsize=(9,4))

    # 3D Plot
    ax_3d = fig.add_subplot(111, projection='3d')
    ax_3d.xaxis.set_pane_color('white')
    ax_3d.yaxis.set_pane_color('white')
    ax_3d.zaxis.set_pane_color('white')

    if len(stg) != 0: ax_3d.scatter(*zip(*stg), marker='s', color=colorPalette[0], edgecolors='k', s=100, label='Stage emitters')
    else: stg = torch.tensor([[0, 0, 0]])
    if len(lds) != 0: ax_3d.scatter(*zip(*lds), marker='s', color=colorPalette[1], edgecolors='k', s=100, label='System loudspeakers')
    else: lds = torch.tensor([[0, 0, 0]])
    if len(mcs) != 0: ax_3d.scatter(*zip(*mcs), marker='o', color=colorPalette[2], edgecolors='k', s=100, label='System microphones')
    else: mcs = torch.tensor([[0, 0, 0]])
    if len(aud) != 0: ax_3d.scatter(*zip(*aud), marker='o', color=colorPalette[3], edgecolors='k', s=100, label='Audience receivers')
    else: aud = torch.tensor([[0, 0, 0]])

    # Labels
    ax_3d.set_xlabel('x in meters', labelpad=15)
    ax_3d.set_ylabel('y in meters', labelpad=15)
    ax_3d.set_zlabel('z in meters', labelpad=2)
    ax_3d.set_zlim(0,)

    # Equal scaling
    room_x = torch.max(torch.cat((stg[:, 0], lds[:, 0], mcs[:, 0], aud[:, 0]))).item() - torch.min(torch.cat((stg[:, 0], lds[:, 0], mcs[:, 0], aud[:, 0]))).item()
    room_y = torch.max(torch.cat((stg[:, 1], lds[:, 1], mcs[:, 1], aud[:, 1]))).item() - torch.min(torch.cat((stg[:, 1], lds[:, 1], mcs[:, 1], aud[:, 1]))).item()
    room_z = torch.max(torch.cat((stg[:, 2], lds[:, 2], mcs[:, 2], aud[:, 2]))).item()
    ax_3d.set_box_aspect([room_x, room_y, room_z])

    # Plot orientation
    ax_3d.view_init(30, 150)

    # Legend Plot
    ax_3d.legend(
        loc='center right',  # Center the legend in the legend plot
        bbox_to_anchor=(2, 0.5),  # Position the legend outside the plot
        handletextpad=0.1,
        borderpad=0.2,
        columnspacing=1.0,
        borderaxespad=0.1,
        handlelength=1
    )

    # Adjust layout
    fig.tight_layout()
    fig.subplots_adjust(left=0.00, top=1.3, right=0.5, bottom=-0.1)
    plt.show(block=True)

    return None

def plot_coupling(energy_values: OrderedDict):

    ec_SA = energy_values["SA"]
    ec_SM = energy_values["SM"]
    ec_LM = energy_values["LM"]
    ec_LA = energy_values["LA"]

    n_stg = ec_SA.shape[1]
    n_aud = ec_SA.shape[0]
    n_mcs = ec_LM.shape[0]
    n_lds = ec_LM.shape[1]

    ecs = torch.cat((torch.cat((ec_LM, ec_SM), dim=1), torch.cat((ec_LA, ec_SA), dim=1)), dim=0)
    ecs_db = 10*torch.log10(ecs + 1e-10)

    ecs_plot = [ecs_db[:n_mcs, :n_lds],
                ecs_db[:n_mcs, n_lds:],
                ecs_db[n_mcs:, :n_lds],
                ecs_db[n_mcs:, n_lds:]]

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    colorPalette = plt.get_cmap("viridis")

    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        layout="constrained",
        width_ratios=[n_lds, n_stg],
        height_ratios=[n_mcs, n_aud],
        gridspec_kw={'wspace':0.05, 'hspace':0.1},
        figsize=(9, 4)
    )
    fig.suptitle('Energy coupling')

    max_value = torch.max(ecs_db)
    min_value = torch.min(ecs_db)
    norm = colors.Normalize(vmin=min_value, vmax=max_value)
    
    images = []
    for ax, data in zip(axs.flat, ecs_plot):
        images.append(ax.imshow(data, norm=norm, cmap=colorPalette))

    fig.colorbar(mappable=images[0], ax=axs, label='Magnitude in dB', aspect=10, pad=0.03, ticks=[-40, -35, -30, -25, -20, -15, -10, -5, 0])

    labelpad = 20 if n_mcs<10 else 10
    axs[0,0].set_ylabel('Mic', labelpad=labelpad)
    ticks = torch.arange(start=0, end=n_mcs, step=int(torch.ceil(torch.sqrt(torch.tensor(n_mcs)))) if n_mcs>2 else 1).numpy()
    axs[0,0].set_yticks(ticks=ticks, labels=ticks+1)
    axs[0,0].set_xticks([])
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])
    labelpad = 20 if n_aud<10 else 10
    axs[1,0].set_ylabel('Aud', labelpad=labelpad)
    ticks = torch.arange(start=0, end=n_aud, step=int(torch.ceil(torch.sqrt(torch.tensor(n_aud)))) if n_aud>2 else 1).numpy()
    axs[1,0].set_yticks(ticks=ticks, labels=ticks+1)
    axs[1,0].set_xlabel('Ldsp', labelpad=5)
    ticks = torch.arange(start=0, end=n_lds, step=int(torch.ceil(torch.sqrt(torch.tensor(n_lds)))) if n_lds>2 else 1).numpy()
    axs[1,0].set_xticks(ticks=ticks, labels=ticks+1)
    axs[1,1].set_xlabel('Stage', labelpad=5)
    ticks = torch.arange(start=0, end=n_stg, step=int(torch.ceil(torch.sqrt(torch.tensor(n_stg)))) if n_stg>2 else 1).numpy()
    axs[1,1].set_xticks(ticks=ticks, labels=ticks+1)
    axs[1,1].set_yticks([])

    plt.show(block=True)

    return None

def plot_DRR(direct_to_reverb_ratios: OrderedDict):

    drr_SA = direct_to_reverb_ratios["SA"]
    drr_SM = direct_to_reverb_ratios["SM"]
    drr_LM = direct_to_reverb_ratios["LM"]
    drr_LA = direct_to_reverb_ratios["LA"]

    n_stg = drr_SA.shape[1]
    n_aud = drr_SA.shape[0]
    n_mcs = drr_LM.shape[0]
    n_lds = drr_LM.shape[1]

    drrs = torch.cat((torch.cat((drr_LM, drr_SM), dim=1), torch.cat((drr_LA, drr_SA), dim=1)), dim=0)
    drrs_db = 10*torch.log10(drrs + 1e-10)

    ecs_plot = [drrs_db[:n_mcs, :n_lds],
                drrs_db[:n_mcs, n_lds:],
                drrs_db[n_mcs:, :n_lds],
                drrs_db[n_mcs:, n_lds:]]

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        layout="constrained",
        width_ratios=[n_lds, n_stg],
        height_ratios=[n_mcs, n_aud],
        gridspec_kw={'wspace':0.05, 'hspace':0.1},
        figsize=(9,4)
    )
    fig.suptitle('Direct to reverberant ratio')

    max_value = torch.max(drrs_db)
    min_value = torch.min(drrs_db)
    norm = colors.Normalize(vmin=min_value, vmax=max_value)
    
    images = []
    for ax, data in zip(axs.flat, ecs_plot):
        images.append(ax.imshow(data, norm=norm))

    fig.colorbar(mappable=images[0], ax=axs, label='Magnitude in dB', aspect=10, pad=0.03, ticks=[-20, -15, -10, -5, 0, 5, 10, 15, 20])

    labelpad = 20 if n_mcs<10 else 10
    axs[0,0].set_ylabel('Mic', labelpad=labelpad)
    ticks = torch.arange(start=0, end=n_mcs, step=int(torch.ceil(torch.sqrt(torch.tensor(n_mcs)))) if n_mcs>2 else 1).numpy()
    axs[0,0].set_yticks(ticks=ticks, labels=ticks+1)    
    axs[0,0].set_xticks([])
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])
    labelpad = 20 if n_aud<10 else 10
    axs[1,0].set_ylabel('Aud', labelpad=labelpad)
    ticks = torch.arange(start=0, end=n_aud, step=int(torch.ceil(torch.sqrt(torch.tensor(n_aud)))) if n_aud>2 else 1).numpy()
    axs[1,0].set_yticks(ticks=ticks, labels=ticks+1)
    axs[1,0].set_xlabel('Ldsp', labelpad=10)
    ticks = torch.arange(start=0, end=n_lds, step=int(torch.ceil(torch.sqrt(torch.tensor(n_lds)))) if n_lds>2 else 1).numpy()
    axs[1,0].set_xticks(ticks=ticks, labels=ticks+1)
    axs[1,1].set_xlabel('Stage', labelpad=10)
    ticks = torch.arange(start=0, end=n_stg, step=int(torch.ceil(torch.sqrt(torch.tensor(n_stg)))) if n_stg>2 else 1).numpy()
    axs[1,1].set_xticks(ticks=ticks, labels=ticks+1)
    axs[1,1].set_yticks([])

    plt.show(block=True)

    return None

def plot_distributions(distributions: torch.Tensor, n_bins: int, labels: list[str] = None, log_scale: bool = False):
    
    if labels is None:
        labels = [f'Distribution {i+1}' for i in range(distributions.shape[1])]

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    colorPalette = sns.color_palette("muted", n_colors=distributions.shape[1])
    
    plt.figure(figsize=(7, 5))
    for i in range(distributions.shape[1]):
        plt.hist(
            distributions[:,i].squeeze(),
            bins=n_bins,
            label=labels[i],
            color=colorPalette[i],
            alpha=0.7,
            density=True,
            histtype='stepfilled',
            edgecolor='black',
            log=log_scale
        )
    plt.legend(loc='upper right')
    plt.xlabel('Value in dB')
    plt.ylabel('Density')
    plt.tight_layout()

    plt.show(block=True)

    return None

# ==================================================================
# ========================== SINGLE DATA ===========================

def plot_evs_distribution(evs, fs: int, nfft: int, lower_f_lim: float, higher_f_lim: float, label='Data') -> None:
    """
    Plot the magnitude distribution of the given eigenvalues.

    Args:
        evs (_type_): _description_
    """

    idx1 = int(nfft/fs * lower_f_lim)
    idx2 = int(nfft/fs * higher_f_lim)
    evs = mag2db(get_magnitude(evs[idx1:idx2,:].flatten()))

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    colorPalette = sns.color_palette("pastel", n_colors=1)

    plt.figure(figsize=(3,5))
    ax = plt.subplot(1,1,1)
    evs_max = torch.max(evs, 0)[0]
    data = dict({'evs': evs})
    sns.boxplot(data=data, positions=[0], width=0.6, showfliers=False,  patch_artist=True,
                boxprops=dict(edgecolor='k', facecolor=colorPalette[0]), medianprops=dict(color="k", linewidth=1.5), whiskerprops=dict(color="k"), capprops=dict(color='k'))
    ax.scatter([0], [evs_max], marker="o", s=20, edgecolors='black', facecolors='black')

    ax.yaxis.grid(True)
    plt.ylabel('Magnitude in dB')
    plt.title(label)
    plt.tight_layout()

    plt.show(block=True)

    return None

# ==================================================================
# ==================== OPTIMIZATION COMPARISON =====================

def plot_evs_compare(evs_init, evs_opt, fs: int, nfft: int, lower_f_lim: float, higher_f_lim: float, label1='Initialized', label2='Optimized') -> None:
    """
    Plot the magnitude distribution of the given eigenvalues.

    Args:
        evs (_type_): _description_
    """

    idx1 = int(nfft/fs * lower_f_lim)
    idx2 = int(nfft/fs * higher_f_lim)
    evs = mag2db(get_magnitude(torch.cat((evs_init.unsqueeze(-1), evs_opt.unsqueeze(-1)), dim=2)[idx1:idx2,:,:]))

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    colorPalette = sns.color_palette("pastel", n_colors=2)

    plt.figure(figsize=(5,5))
    ax = plt.subplot(1,1,1)
    for i in range(evs.shape[2]):
        evst = evs[:,:,i].flatten()
        evst_max = torch.max(evst, 0)[0]
        sns.boxplot(data=evst.numpy(), positions=[i], width=0.6, showfliers=False, patch_artist=True,
                    boxprops=dict(edgecolor='k', facecolor=colorPalette[i]), medianprops=dict(color="k", linewidth=1.5), whiskerprops=dict(color="k"), capprops=dict(color='k'))
        ax.scatter([i], [evst_max], marker="o", s=20, edgecolors='black', facecolors='black')

    ax.yaxis.grid(True)
    plt.xticks([0,1], [label1, label2])
    plt.ylabel('Magnitude in dB')
    plt.tight_layout()

    plt.show(block=True)

    return None


def plot_irs_compare(ir_1: torch.Tensor, ir_2: torch.Tensor, fs: int, label1='Initialized', label2='Optimized') -> None:
    r"""
    Plot the system impulse responses at initialization and after optimization.
    
        **Args**:
            - ir_1 (torch.Tensor): First impulse response to plot.
            - ir_2 (torch.Tensor): Second impulse response to plot.
            - fs (int): Sampling frequency.
            - label1 (str, optional): Label for the first impulse response. Defaults to 'Initialized'.
            - label2 (str, optional): Label for the second impulse response. Defaults to 'Optimized'.
            - title (str, optional): Title of the plot. Defaults to 'System Impulse Responses'.
    """
    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 4), constrained_layout=True)

    time = torch.arange(ir_1.shape[0]) / fs

    plt.subplot(2, 1, 1)
    plt.plot(time.numpy(), ir_1.detach().squeeze().numpy())
    plt.title(label1)
    plt.grid(True)

    time = torch.arange(ir_2.shape[0]) / fs

    plt.subplot(2, 1, 2)
    plt.plot(time.numpy(), ir_2.detach().squeeze().numpy())
    plt.title(label2)
    plt.grid(True)

    fig.supxlabel('Time in seconds')
    fig.supylabel('Amplitude')

    plt.show(block=True)

def plot_spectrograms_compare(ir_1: torch.Tensor, ir_2: torch.Tensor, fs: int, nfft: int=2**10, noverlap: int=2**8, label1='Initialized', label2='Optimized') -> None:
    r"""
    Plot the spectrograms of the system impulse responses at initialization and after optimization.
    
        **Args**:
            - y_1 (torch.Tensor): First signal to plot.
            - y_2 (torch.Tensor): Second signal to plot.
            - fs (int): Sampling frequency.
            - nfft (int, optional): FFT size. Defaults to 2**10.
            - label1 (str, optional): Label for the first signal. Defaults to 'Initialized'.
            - label2 (str, optional): Label for the second signal. Defaults to 'Optimized'.
            - title (str, optional): Title of the plot. Defaults to 'System Impulse Response Spectrograms'.
    """
    Spec_init,f,t = mlab.specgram(ir_1.detach().squeeze().numpy(), NFFT=nfft, Fs=fs, noverlap=noverlap)
    Spec_opt,_,_ = mlab.specgram(ir_2.detach().squeeze().numpy(), NFFT=nfft, Fs=fs, noverlap=noverlap)

    max_val = max(Spec_init.max(), Spec_opt.max())
    Spec_init = torch.tensor(Spec_init)/max_val
    Spec_opt = torch.tensor(Spec_opt)/max_val
    

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    fig,axes = plt.subplots(2,1, sharex=False, sharey=True, figsize=(8,5), constrained_layout=True)
    
    plt.subplot(2,1,1)
    plt.pcolormesh(t, f, 10*torch.log10(Spec_init), cmap='magma', vmin=-100, vmax=0)
    plt.xlim(0, ir_1.shape[0]/fs)
    plt.ylim(20, fs//2)
    plt.yscale('log')
    plt.title(label1)
    plt.grid(False)

    plt.subplot(2,1,2)
    im = plt.pcolormesh(t, f, 10*torch.log10(Spec_opt), cmap='magma', vmin=-100, vmax=0)
    plt.xlim(0, ir_1.shape[0]/fs)
    plt.ylim(20, fs//2)
    plt.yscale('log')
    plt.title(label2)
    plt.grid(False)

    fig.supxlabel('Time in seconds')
    fig.supylabel('Frequency in Hz')

    cbar = fig.colorbar(im, ax=axes[:], aspect=20)
    cbar.set_label('Magnitude in dB')
    ticks = torch.arange(start=-100, end=1, step=20)
    cbar.ax.set_ylim(-100, 0)
    cbar.ax.set_yticks(ticks, ['-100','-80','-60','-40','-20','0'])

    plt.show(block=True)
