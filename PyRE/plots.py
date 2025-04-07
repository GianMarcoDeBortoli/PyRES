# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import mlab
from matplotlib import colors
import seaborn as sns
import numpy as np
# PyTorch
import torch
import torchaudio
# FLAMO
from flamo.functional import mag2db, get_magnitude
# PyRES
from PyRES.metrics import energy_coupling, direct_to_reverb_ratio

# ==================================================================
# ========================== PHYSICAL ROOM =========================

def plot_room_setup(room) -> None:

    stage = torch.tensor(room.low_level_info['StageAndAudience']['StageEmitters']['Position_m'])
    loudspeakers = torch.tensor(room.low_level_info['AudioSetup']['SystemEmitters']['Position_m'])
    microphones = torch.tensor(room.low_level_info['AudioSetup']['SystemReceivers']['Position_m'])
    audience = torch.tensor(room.low_level_info['StageAndAudience']['AudienceReceivers']['MonochannelPosition_m'])

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})

    # Use constrained layout
    fig = plt.figure(figsize=(9,4))

    # 3D Plot
    ax_3d = fig.add_subplot(111, projection='3d')
    ax_3d.scatter(*zip(*stage), marker='s', color='g', s=100, label='Stage emitters')
    ax_3d.scatter(*zip(*loudspeakers), marker='s', color='b', s=100, label='System loudspeakers')
    ax_3d.scatter(*zip(*microphones), marker='o', color='r', s=100, label='System microphones')
    ax_3d.scatter(*zip(*audience), marker='o', color='y', s=100, label='Audience receivers')

    # Labels
    ax_3d.set_xlabel('x in meters', labelpad=15)
    ax_3d.set_ylabel('y in meters', labelpad=15)
    ax_3d.set_zlabel('z in meters', labelpad=2)
    ax_3d.set_zlim(0,)

    # Equal scaling
    room_x = torch.max(torch.cat((stage[:, 0], loudspeakers[:, 0], microphones[:, 0], audience[:, 0]))).item() - torch.min(torch.cat((stage[:, 0], loudspeakers[:, 0], microphones[:, 0], audience[:, 0]))).item()
    room_y = torch.max(torch.cat((stage[:, 1], loudspeakers[:, 1], microphones[:, 1], audience[:, 1]))).item() - torch.min(torch.cat((stage[:, 1], loudspeakers[:, 1], microphones[:, 1], audience[:, 1]))).item()
    room_z = torch.max(torch.cat((stage[:, 2], loudspeakers[:, 2], microphones[:, 2], audience[:, 2]))).item()
    ax_3d.set_box_aspect([room_x, room_y, room_z])

    # Plot orientation
    ax_3d.view_init(28, 150)

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

def plot_coupling(rirs: torch.Tensor, fs: int, decay_interval: str='T30', **kwargs) -> torch.Tensor:

    rirs_SA = rirs["h_SA"]
    ec_SA = energy_coupling(rirs_SA, fs=fs, decay_interval=decay_interval)
    rirs_SM = rirs["h_SM"]
    ec_SM = energy_coupling(rirs_SM, fs=fs, decay_interval=decay_interval)
    rirs_LM = rirs["h_LM"]
    ec_LM = energy_coupling(rirs_LM, fs=fs, decay_interval=decay_interval)
    rirs_LA = rirs["h_LA"]
    ec_LA = energy_coupling(rirs_LA, fs=fs, decay_interval=decay_interval)

    ecs = torch.cat((torch.cat((ec_LM, ec_SM), dim=1), torch.cat((ec_LA, ec_SA), dim=1)), dim=0)
    norm_value = torch.max(ecs)
    ecs_norm = ecs/norm_value
    ecs_db = 10*torch.log10(ecs_norm)

    ecs_plot = [ecs_db[:ec_LM.shape[0], :ec_LM.shape[1]],
                ecs_db[:ec_LM.shape[0], ec_LM.shape[1]:],
                ecs_db[ec_LM.shape[0]:, :ec_LM.shape[1]],
                ecs_db[ec_LM.shape[0]:, ec_LM.shape[1]:]]

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        layout="constrained",
        width_ratios=[ec_LM.shape[1], ec_SM.shape[1]],
        height_ratios=[ec_LM.shape[0], ec_LA.shape[0]],
        gridspec_kw={'wspace':0.05, 'hspace':0.1},
        figsize=(ecs.shape[1]/2, ecs.shape[0]/2)
    )
    # fig.suptitle('Energy coupling')

    max_value = torch.max(ecs_db)
    min_value = torch.min(ecs_db)
    norm = colors.Normalize(vmin=min_value, vmax=max_value)
    
    images = []
    for ax, data in zip(axs.flat, ecs_plot):
        images.append(ax.imshow(data, norm=norm))

    fig.colorbar(mappable=images[0], ax=axs, label='Magnitude in dB', aspect=10, pad=0.1)

    labelpad = 20 if rirs_LM.shape[0]<10 else 10
    axs[0,0].set_ylabel('Mic', labelpad=labelpad)
    ticks = torch.arange(start=0, end=rirs_LM.shape[1], step=int(torch.ceil(torch.sqrt(torch.tensor(rirs_LM.shape[1])))) if rirs_LM.shape[1]>2 else 1).numpy()
    axs[0,0].set_yticks(ticks=ticks, labels=ticks+1)
    axs[0,0].set_xticks([])
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])
    labelpad = 20 if rirs_LA.shape[0]<10 else 10
    axs[1,0].set_ylabel('Aud', labelpad=labelpad)
    ticks = torch.arange(start=0, end=rirs_LA.shape[1], step=int(torch.ceil(torch.sqrt(torch.tensor(rirs_LA.shape[1])))) if rirs_LA.shape[1]>2 else 1).numpy()
    axs[1,0].set_yticks(ticks=ticks, labels=ticks+1)
    axs[1,0].set_xlabel('Ldsp', labelpad=5)
    ticks = torch.arange(start=0, end=rirs_LM.shape[2], step=int(torch.ceil(torch.sqrt(torch.tensor(rirs_LM.shape[2])))) if rirs_LM.shape[2]>2 else 1).numpy()
    axs[1,0].set_xticks(ticks=ticks, labels=ticks+1)
    axs[1,1].set_xlabel('Stage', labelpad=5)
    ticks = torch.arange(start=0, end=rirs_SA.shape[2], step=int(torch.ceil(torch.sqrt(torch.tensor(rirs_SA.shape[2])))) if rirs_SA.shape[2]>2 else 1).numpy()
    axs[1,1].set_xticks(ticks=ticks, labels=ticks+1)
    axs[1,1].set_yticks([])

    plt.show(block=True)

    return None

def plot_DRR(rirs: torch.Tensor, fs: int, decay_interval: str='T30', **kwargs) -> torch.Tensor:

    rirs_SA = rirs["h_SA"]
    drr_SA = direct_to_reverb_ratio(rirs_SA, fs=fs, decay_interval=decay_interval)
    rirs_SM = rirs["h_SM"]
    drr_SM = direct_to_reverb_ratio(rirs_SM, fs=fs, decay_interval=decay_interval)
    rirs_LM = rirs["h_LM"]
    drr_LM = direct_to_reverb_ratio(rirs_LM, fs=fs, decay_interval=decay_interval)
    rirs_LA = rirs["h_LA"]
    drr_LA = direct_to_reverb_ratio(rirs_LA, fs=fs, decay_interval=decay_interval)

    drrs = torch.cat((torch.cat((drr_LM, drr_SM), dim=1), torch.cat((drr_LA, drr_SA), dim=1)), dim=0)
    norm_value = torch.max(drrs)
    drrs_norm = drrs/norm_value
    drrs_db = 10*torch.log10(drrs_norm)

    ecs_plot = [drrs_db[:drr_LM.shape[0], :drr_LM.shape[1]],
                drrs_db[:drr_LM.shape[0], drr_LM.shape[1]:],
                drrs_db[drr_LM.shape[0]:, :drr_LM.shape[1]],
                drrs_db[drr_LM.shape[0]:, drr_LM.shape[1]:]]

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        layout="constrained",
        width_ratios=[drr_LM.shape[1], drr_SM.shape[1]],
        height_ratios=[drr_LM.shape[0], drr_LA.shape[0]],
        gridspec_kw={'wspace':0.05, 'hspace':0.1},
        figsize=(drrs.shape[1]/2, drrs.shape[0]/2)
    )
    # fig.suptitle('Direct to reverberant ratio')

    max_value = torch.max(drrs_db)
    min_value = torch.min(drrs_db)
    norm = colors.Normalize(vmin=min_value, vmax=max_value)
    
    images = []
    for ax, data in zip(axs.flat, ecs_plot):
        images.append(ax.imshow(data, norm=norm))

    fig.colorbar(mappable=images[0], ax=axs, label='Magnitude in dB', aspect=10, pad=0.1)

    labelpad = 20 if rirs_LM.shape[0]<10 else 10
    axs[0,0].set_ylabel('Mic', labelpad=labelpad)
    ticks = torch.arange(start=0, end=rirs_LM.shape[1], step=int(torch.ceil(torch.sqrt(torch.tensor(rirs_LM.shape[1])))) if rirs_LM.shape[1]>2 else 1).numpy()
    axs[0,0].set_yticks(ticks=ticks, labels=ticks+1)    
    axs[0,0].set_xticks([])
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])
    labelpad = 20 if rirs_LA.shape[0]<10 else 10
    axs[1,0].set_ylabel('Aud', labelpad=labelpad)
    ticks = torch.arange(start=0, end=rirs_LA.shape[1], step=int(torch.ceil(torch.sqrt(torch.tensor(rirs_LA.shape[1])))) if rirs_LA.shape[1]>2 else 1).numpy()
    axs[1,0].set_yticks(ticks=ticks, labels=ticks+1)
    axs[1,0].set_xlabel('Ldsp', labelpad=10)
    ticks = torch.arange(start=0, end=rirs_LM.shape[2], step=int(torch.ceil(torch.sqrt(torch.tensor(rirs_LM.shape[2])))) if rirs_LM.shape[2]>2 else 1).numpy()
    axs[1,0].set_xticks(ticks=ticks, labels=ticks+1)
    axs[1,1].set_xlabel('Stage', labelpad=10)
    ticks = torch.arange(start=0, end=rirs_SA.shape[2], step=int(torch.ceil(torch.sqrt(torch.tensor(rirs_SA.shape[2])))) if rirs_SA.shape[2]>2 else 1).numpy()
    axs[1,1].set_xticks(ticks=ticks, labels=ticks+1)
    axs[1,1].set_yticks([])

    plt.show(block=True)

    return None


# ==================================================================
# ======================= EVALUATION METRICS =======================

def plot_evs(evs_init, evs_opt, fs: int, nfft: int, lower_f_lim: float, higher_f_lim: float):
    """
    Plot the magnitude distribution of the given eigenvalues.

    Args:
        evs (_type_): _description_
    """

    idx1 = int(nfft/fs * lower_f_lim)
    idx2 = int(nfft/fs * higher_f_lim)
    evs = mag2db(get_magnitude(torch.cat((evs_init.unsqueeze(-1), evs_opt.unsqueeze(-1)), dim=2)[idx1:idx2,:,:]))

    colors = ['xkcd:sky', 'coral', 'coral', "xkcd:mint green", "xkcd:mint green", "xkcd:light magenta", "xkcd:light magenta"]

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    plt.figure(figsize=(5,5))
    # plt.figure()
    ax = plt.subplot(1,1,1)
    for i in range(evs.shape[2]):
        evst = torch.reshape(evs[:,:,i], (evs.shape[0]*evs.shape[1], -1)).squeeze()
        evst_max = torch.max(evst, 0)[0]
        sns.boxplot(evst.numpy(), positions=[i], width=0.7, showfliers=False, patch_artist=True, boxprops=dict(edgecolor='k', facecolor=colors[i]), medianprops=dict(color="k", linewidth=2))
        ax.scatter([i], [evst_max], marker="o", s=35, edgecolors='black', facecolors=colors[i])

    ax.yaxis.grid(True)
    plt.xticks([0,1], ['Initialization', 'Optimized'], rotation=60, horizontalalignment='right')
    # plt.yticks(np.arange(-30, 1, 10), ['-30','-20', '-10','0'])
    plt.ylabel('Magnitude in dB')
    # plt.title(f'Eigenvalue Magnitude Distribution\nbetween {lower_f_lim} Hz and {higher_f_lim} Hz')
    plt.tight_layout()

    plt.show(block=True)


def plot_spectrograms(y_1: torch.Tensor, y_2: torch.Tensor, fs: int, nfft: int=2**10, noverlap: int=2**8, label1='Initialized', label2='Optimized', title='System Impulse Response Spectrograms') -> None:
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
    Spec_init,f,t = mlab.specgram(y_1.detach().squeeze().numpy(), NFFT=nfft, Fs=fs, noverlap=noverlap)
    Spec_opt,_,_ = mlab.specgram(y_2.detach().squeeze().numpy(), NFFT=nfft, Fs=fs, noverlap=noverlap)

    max_val = max(Spec_init.max(), Spec_opt.max())
    Spec_init = Spec_init/max_val
    Spec_opt = Spec_opt/max_val
    

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    fig,axes = plt.subplots(2,1, sharex=False, sharey=True, figsize=(7,5), constrained_layout=True)
    
    plt.subplot(2,1,1)
    plt.pcolormesh(t, f, 10*np.log10(Spec_init), cmap='magma', vmin=-100, vmax=0)
    plt.xlim(0, y_1.shape[0]/fs)
    plt.ylim(20, fs//2)
    plt.yscale('log')
    plt.title(label1)
    plt.grid(False)

    plt.subplot(2,1,2)
    im = plt.pcolormesh(t, f, 10*np.log10(Spec_opt), cmap='magma', vmin=-100, vmax=0)
    plt.xlim(0, y_1.shape[0]/fs)
    plt.ylim(20, fs//2)
    plt.yscale('log')
    plt.title(label2)
    plt.grid(False)

    fig.supxlabel('Time in seconds')
    fig.supylabel('Frequency in Hz')
    fig.suptitle(title)

    cbar = fig.colorbar(im, ax=axes[:], aspect=20)
    cbar.set_label('Magnitude in dB')
    ticks = np.arange(-100, 1, 20)
    cbar.ax.set_ylim(-100, 0)
    cbar.ax.set_yticks(ticks, ['-100','-80','-60','-40','-20','0'])

    plt.show(block=True)
