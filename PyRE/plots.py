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

def unpack_kwargs(kwargs):
    for k, v in kwargs.items():
        match k:
            case 'fontsize':
                plt.rcParams.update({'font.size':v})
            case 'fontweight':
                plt.rcParams.update({'font.weight':v})
            case 'fontfamily':
                plt.rcParams.update({'font.family':v})
            case 'usetex':
                plt.rcParams.update({'text.usetex':v})
            case 'linewidth':
                plt.rcParams.update({'lines.linewidth':v})
            case 'markersize':
                plt.rcParams.update({'lines.markersize':v})
            case 'color':
                colors = v
            case 'title':
                title = v

def plot_coupling(rirs: torch.Tensor, fs: int, decay_interval: str='T30', **kwargs) -> torch.Tensor:

    rirs = rirs["h_LM"]

    ec = energy_coupling(rirs, fs=fs, decay_interval=decay_interval)

    ec_norm = ec/torch.max(ec)
    ec_db = 10*torch.log10(ec_norm)

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})

    # plt.figure(figsize=(7,6))
    plt.figure()

    image = plt.imshow(ec_db)
    plt.ylabel('Microphone')
    plt.yticks(torch.arange(start=0, end=rirs.shape[1], step=5 if rirs.shape[1]>10 else 1).numpy(), labels=torch.arange(start=0, end=rirs.shape[1], step=5 if rirs.shape[1]>10 else 1).numpy())
    plt.xlabel('Loudspeaker')
    plt.xticks(torch.arange(start=0, end=rirs.shape[2], step=5 if rirs.shape[2]>10 else 1).numpy(), labels=torch.arange(start=0, end=rirs.shape[2], step=5 if rirs.shape[2]>10 else 1).numpy())
    plt.colorbar(mappable=image, label='Magnitude in dB')
    plt.title('Energy coupling')
    plt.tight_layout()

    plt.show(block=True)


def plot_coupling_pro_version(rirs: torch.Tensor, fs: int, decay_interval: str='T30', **kwargs) -> torch.Tensor:

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


def plot_DRR(rirs: torch.Tensor, fs: int, decay_interval: str='T30') -> torch.Tensor:

    rirs = rirs["h_LM"]

    drr = direct_to_reverb_ratio(rirs, fs=fs, decay_interval=decay_interval)

    drr_norm = drr/torch.max(drr)
    drr_db = 10*torch.log10(drr_norm)

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    # plt.figure(figsize=(7,6))
    plt.figure()
    image = plt.imshow(drr_db)
    plt.ylabel('Microphone')
    plt.yticks(torch.arange(start=0, end=rirs.shape[1], step=5 if rirs.shape[1]>10 else 1).numpy(), labels=torch.arange(start=0, end=rirs.shape[1], step=5 if rirs.shape[1]>10 else 1).numpy())
    plt.xlabel('Loudspeaker')
    plt.xticks(torch.arange(start=0, end=rirs.shape[2], step=5 if rirs.shape[2]>10 else 1).numpy(), labels=torch.arange(start=0, end=rirs.shape[2], step=5 if rirs.shape[2]>10 else 1).numpy())
    plt.colorbar(mappable=image, label='Magnitude in dB')
    plt.title('Direct-to-Reverberant Ratio')
    plt.tight_layout()

    plt.show(block=True)

    return drr_norm

def plot_DRR_pro_version(rirs: torch.Tensor, fs: int, decay_interval: str='T30', **kwargs) -> torch.Tensor:

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

# ==================================================================
# ========================== VIRTUAL ROOM ==========================

def plot_virtualroom_ir(ir, fs, nfft, **kwargs):

    ir = ir/torch.max(ir)

    n_samples = ir.shape[0]
    t_axis = torch.linspace(0, n_samples/fs, n_samples)
    f_axis = torch.linspace(0, fs/2, nfft//2+1)

    ir_squared = torch.square(ir)
    bwint = torch.zeros_like(ir_squared)
    for n in range(bwint.shape[0]):
        bwint[n] = torch.sum(ir_squared[n:])
    ir_db = mag2db(ir_squared)
    bwing_db = mag2db(bwint/torch.max(bwint))
    tf = torch.fft.rfft(ir, nfft, dim=0)
    tf_db = mag2db(get_magnitude(tf))

    Spec,f,t = mlab.specgram(ir.numpy(), NFFT=2**10, Fs=fs, noverlap=2**7)
    Spec = Spec/Spec.max()

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    plt.figure(figsize=(7,6))

    plt.subplot(2,2,1)
    plt.plot(t_axis, ir)
    plt.xlabel('Time in seconds')
    plt.ylabel('Amplitude')
    plt.title('Impulse Response')
    plt.grid()

    plt.subplot(2,2,3)
    plt.plot(t_axis, ir_db)
    plt.plot(t_axis, bwing_db)
    plt.xlabel('Time in seconds')
    plt.ylabel('Magnitude in dB')
    plt.title('Squared Impulse Response and Backward Integration')
    plt.grid()

    plt.subplot(2,2,2)
    plt.plot(f_axis, tf_db)
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Magnitude in dB')
    plt.title('Transfer Function')
    plt.grid()
    plt.xlim(20,20000)
    plt.ylim(-40,40)
    plt.xscale('log')

    plt.subplot(2,2,4)
    plt.pcolormesh(t, f, 10*np.log10(Spec), cmap='magma', vmin=-100, vmax=0)
    plt.ylim(20, fs//2)
    plt.xlabel('Time in seconds')
    plt.ylabel('Frequency in Hz')
    plt.yscale('log')
    plt.title('Spectrogram')
    cbar = plt.colorbar(aspect=20)
    cbar.set_label('Magnitude in dB')
    ticks = np.arange(-100, 1, 20)
    cbar.ax.set_ylim(-100, 0)
    cbar.ax.set_yticks(ticks, ['-100','-80','-60','-40','-20','0'])

    plt.tight_layout()
    plt.show(block=True)

# ==================================================================
# ======================= EVALUATION METRICS =======================

# def plot_evs_distributions(evs_1: torch.Tensor, evs_2: torch.Tensor, fs: int, nfft: int, lower_f_lim: float, higher_f_lim: float, label1: str='Initialized', label2: str='Optimized') -> None:
#     r"""
#     Plot the magnitude distribution of the given eigenvalues.

#         **Args**:
#             evs_init (torch.Tensor): First set of eigenvalues to plot.
#             evs_opt (torch.Tensor): Second set of eigenvalues to plot.
#             fs (int): Sampling frequency.
#             nfft (int): FFT size.
#             label1 (str, optional): Label for the first set of eigenvalues. Defaults to 'Initialized'.
#             label2 (str, optional): Label for the second set of eigenvalues. Defaults to 'Optimized'.
#     """

#     idx1 = int(nfft/fs * lower_f_lim)
#     idx2 = int(nfft/fs * higher_f_lim)
#     evs = mag2db(get_magnitude(torch.cat((evs_1.unsqueeze(-1), evs_2.unsqueeze(-1)), dim=len(evs_1.shape))[idx1:idx2,:,:]))
#     plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
#     plt.figure(figsize=(7,6))
#     ax = plt.subplot(1,1,1)
#     colors = ['tab:blue', 'tab:orange']
#     for i in range(evs.shape[2]):
#         evst = torch.reshape(evs[:,:,i], (evs.shape[0]*evs.shape[1], -1)).squeeze()
#         evst_max = torch.max(evst, 0)[0]
#         ax.boxplot(evst.numpy(), positions=[i], widths=0.7, showfliers=False, notch=True, patch_artist=True, boxprops=dict(edgecolor='k', facecolor=colors[i]), medianprops=dict(color='k'))
#         ax.scatter([i], [evst_max], marker="o", s=35, edgecolors='black', facecolors=colors[i])
#     plt.ylabel('Magnitude in dB')
#     plt.xticks([0,1], [label1, label2])
#     plt.xticks(rotation=90)
#     ax.yaxis.grid(True)
#     plt.title(f'Eigenvalue Magnitude Distribution\nbetween {lower_f_lim} Hz and {higher_f_lim} Hz')
#     plt.tight_layout()
#     plt.show(block=True)


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


def plot_ptmr(evs, fs, nfft):
    
    f_axis = torch.linspace(0, fs//2, nfft//2+1)
    evs_peak = torch.max(torch.abs(evs), dim=1)[0]
    evs_mean = torch.mean(torch.abs(evs), dim=1)
    evs_ptmr = evs_peak/evs_mean

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    plt.figure(figsize=(7,6))
    plt.plot(f_axis, mag2db(evs_peak))
    plt.plot(f_axis, mag2db(evs_mean))
    plt.plot(f_axis, mag2db(evs_ptmr))
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Magnitude in dB')
    plt.ylim(-50,10)
    plt.xlim(20,20000)
    plt.xscale('log')
    plt.grid()
    plt.legend(['Peak value', 'Mean value', 'Peak-to-mean ratio'])
    # plt.ylim(-20,30)
    # plt.xlim(20,20000)
    # plt.xscale('log')
    # plt.grid()
    plt.tight_layout()
    plt.show(block=True)




# ==================================================================

def plot_DAFx(unitary, firs, modal_reverb, fdn, poletti, fs, nfft):

    n_samples = torch.max(torch.tensor([unitary.shape[0], firs.shape[0], modal_reverb.shape[0], fdn.shape[0], poletti.shape[0]]))
    t_axis = torch.linspace(0, n_samples/fs, n_samples)
    y1 = torch.zeros(n_samples,)
    y1[:unitary.shape[0]] = unitary[:,0,0].squeeze()
    y1 = y1/torch.max(torch.abs(unitary))
    y2 = torch.zeros(n_samples,)
    y2[:firs.shape[0]] = firs[:,0,0].squeeze()
    y2 = y2/torch.max(torch.abs(firs))
    y3 = torch.zeros(n_samples,)
    modal_reverb_resample = torchaudio.transforms.Resample(orig_freq=1000, new_freq=fs)(modal_reverb[:,0,0])
    y3[:modal_reverb_resample.shape[0]] = modal_reverb_resample.squeeze()
    y3 = y3/torch.max(torch.abs(modal_reverb_resample))
    y4 = torch.zeros(n_samples,)
    y4[:fdn.shape[0]] = fdn[:,0,0].squeeze()
    y4 = y4/torch.max(torch.abs(fdn))
    y5 = torch.zeros(n_samples,)
    y5[:poletti.shape[0]] = poletti[:,0,0].squeeze()
    y5 = y5/torch.max(torch.abs(poletti))


    # bwi_3 = torch.zeros_like(y3)
    # for i in range(1, y3.shape[0]):
    #     bwi_3[i] = torch.sum(torch.pow(y3[i:], 2))
    # bwi_3 = bwi_3/torch.max(bwi_3)
    # bwi_4 = torch.zeros_like(y4)
    # for i in range(1, y4.shape[0]):
    #     bwi_4[i] = torch.sum(torch.pow(y4[i:], 2))
    # bwi_4 = bwi_4/torch.max(bwi_4)
    # bwi_5 = torch.zeros_like(y5)
    # for i in range(1, y5.shape[0]):
    #     bwi_5[i] = torch.sum(torch.pow(y5[i:], 2))
    # bwi_5 = bwi_5/torch.max(bwi_5)

    # plt.figure()
    # plt.plot(t_axis, 10*torch.log10(bwi_3))
    # plt.plot(t_axis, 10*torch.log10(bwi_4))
    # plt.plot(t_axis, 10*torch.log10(bwi_5))
    # plt.ylim(-100,0)
    # plt.xlabel('Time in seconds')
    # plt.ylabel('Magnitude in dB')
    # plt.title('Backward Integration')
    # plt.legend(['Modal reverb', 'FDN', 'Poletti'])
    # plt.grid()
    # plt.show(block=True)

    plt.rcParams.update({'font.family':'serif', 'font.size':16, 'font.weight':'heavy', 'text.usetex':True})
    
    fig, axs = plt.subplots(
        nrows=5,
        ncols=1,
        layout="constrained",
        gridspec_kw={'hspace':0.1},
        figsize=(6,10)
    )
    
    axs[0].plot(t_axis, y1)
    # plt.xlabel('Time in seconds')
    axs[0].set_xlim(-0.001, 0.02)
    axs[0].tick_params(axis='both', which='both', labelsize=14)
    # axs[0].set_ylabel('Amplitude', labelpad=17)
    axs[0].set_title('Unitary mixing matrix')
    axs[0].grid()

    axs[1].plot(t_axis, y2)
    # plt.xlabel('Time in seconds')
    axs[1].set_xlim(-0.001, 0.02)
    axs[1].tick_params(axis='both', which='both', labelsize=14)
    # axs[1].set_ylabel('Amplitude')
    axs[1].set_title('FIRs')
    axs[1].grid()

    axs[2].plot(t_axis, y3)
    # plt.xlabel('Time in seconds')
    axs[2].set_xlim(-0.03, 1)
    axs[2].tick_params(axis='both', which='both', labelsize=14)
    # axs[2].set_ylabel('Amplitude')
    axs[2].set_title('Modal reverberator')
    axs[2].grid()

    axs[3].plot(t_axis, y4)
    # plt.xlabel('Time in seconds')
    axs[3].set_xlim(-0.03, 1)
    axs[3].tick_params(axis='both', which='both', labelsize=14)
    # axs[3].set_ylabel('Amplitude')
    axs[3].set_title('FDN')
    axs[3].grid()

    axs[4].plot(t_axis, y5)
    # axs[4].set_xlabel('Time in seconds')
    axs[4].set_xlim(-0.03, 1)
    axs[4].tick_params(axis='both', which='both', labelsize=14)
    # axs[4].set_ylabel('Amplitude')
    axs[4].set_title('Unitary reverberator')
    axs[4].grid()

    fig.supxlabel('Time in seconds')
    fig.supylabel('Amplitude')

    plt.show(block=True)

    return None

def plot_combined_figure(fs, nfft, evs_init, evs_opt, evs_f_range, ir_init, ir_opt, ir_f_range, cmap='magma') -> None:
    """
    Produces a figure with:
    - A seaborn boxplot on the left (two boxplots).
    - Two spectrograms stacked vertically on the right, sharing a common colorbar.

    Parameters:
        tensor_2d_1, tensor_2d_2: torch.Tensor (2D) -> Data for the two boxplots.
        tensor_1d_1, tensor_1d_2: torch.Tensor (1D) -> Data for the two spectrograms.
        cmap: str -> Colormap for the spectrograms.
    """

    # Compute the frequency axis
    lower_f_lim = int(nfft/fs * evs_f_range[0])
    higher_f_lim = int(nfft/fs * evs_f_range[1])
    evs_init = mag2db(get_magnitude(evs_init[lower_f_lim:higher_f_lim,:]))
    evs_opt = mag2db(get_magnitude(evs_opt[lower_f_lim:higher_f_lim,:]))

    plt.rcParams.update({'font.family': 'serif', 'font.size': 16, 'font.weight': 'heavy', 'text.usetex': True})
    colors = ['xkcd:sky', 'coral', 'coral', "xkcd:mint green", "xkcd:mint green", "xkcd:light magenta", "xkcd:light magenta"]

    # Create the figure and gridspec
    fig = plt.figure(figsize=(6.5, 3.5))
    gs = gridspec.GridSpec(2, 5, width_ratios=[0.7, 0.7, 2, 0.1, 0.1], height_ratios=[1, 1], wspace=0, hspace=0.5)

    # Left subplot: Boxplot
    ax_box = fig.add_subplot(gs[:, 0])  # Use both rows for the boxplot
    data = [evs_init.flatten().numpy(), evs_opt.flatten().numpy()]
    max_vals = [torch.max(evs_init.flatten()), torch.max(evs_opt.flatten())]
    sns.boxplot(data=data, ax=ax_box, showfliers=False, palette=colors[0:2], boxprops=dict(edgecolor='k'),
                medianprops=dict(color="k", linewidth=2))
    ax_box.scatter([0,1], max_vals, marker="o", s=35, edgecolors='black', facecolors=colors[0:2])
    ax_box.set_xticks([0,1],["Init", "Opt"])
    ax_box.set_ylabel("Magnitude in dB")
    ax_box.set_ylim(-42, 0)
    ax_box.set_yticks(ticks=[-40, -30, -20, -10, 0], labels=['-40', '-30', '-20', '-10', '0'])
    ax_box.tick_params(axis='y', labelsize=14)
    ax_box.grid(True)

    # Right subplot: Spectrograms
    ax_spec1 = fig.add_subplot(gs[0, 2])  # Top spectrogram
    ax_spec2 = fig.add_subplot(gs[1, 2])  # Bottom spectrogram

    # Compute spectrograms
    spec1, f1, t1 = mlab.specgram(ir_init.numpy(), NFFT=2**6, Fs=fs, noverlap=2**5)
    spec2, f2, t2 = mlab.specgram(ir_opt.numpy(), NFFT=2**6, Fs=fs, noverlap=2**5)

    # Normalize spectrograms
    max_val = max(spec1.max(), spec2.max())
    spec1 /= max_val
    spec2 /= max_val

    # Plot spectrograms
    if fs == 48000:
        ticks = [20, 100, 1000, 5000, 20000]
        labels = ['20', '100', '1k', '5k', '20k']
    else:
        ticks = [0, 100, 200, 300, 400, 500]
        labels = ['0', '100', '200', '300', '400', '500']

    im1 = ax_spec1.pcolormesh(t1, f1, 10 * np.log10(spec1), cmap=cmap, shading='gouraud', vmin=-100, vmax=0)
    ax_spec1.set_xlim(0, 2)
    ax_spec1.set_ylim(ir_f_range[0], ir_f_range[1])
    # ax_spec1.set_yscale('log')
    ax_spec1.set_yticks(ticks=ticks, labels=labels)
    ax_spec1.tick_params(axis='both', which='both', labelsize=14)
    im2 = ax_spec2.pcolormesh(t2, f2, 10 * np.log10(spec2), cmap=cmap, shading='gouraud', vmin=-100, vmax=0)
    ax_spec2.set_xlim(0, 2)
    # ax_spec2.set_yscale('log')
    ax_spec2.set_ylim(ir_f_range[0], ir_f_range[1])
    ax_spec2.set_yticks(ticks=ticks, labels=labels)
    ax_spec2.tick_params(axis='both', which='both', labelsize=14)

    # Set labels
    ax_spec1.set_ylabel("")
    ax_spec1.set_title("Init", fontsize=16)
    ax_spec2.set_ylabel("")
    ax_spec2.set_xlabel("Time in seconds")
    ax_spec2.set_title("Opt", fontsize=16)
    fig.supylabel("Frequency in Hz", fontsize=16, x=0.33, y=0.5, va='center', ha='center')

    # Common colorbar
    cbar_ax = fig.add_subplot(gs[:, 4])  # Use both rows for the colorbar
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='vertical', label="Magnitude in dB")
    cbar.ax.tick_params(axis='both', which='both', length=0.5, width=0.5, pad=0.5, labelsize=14)
    cbar.ax.get_yaxis().labelpad = 0
    # cbar.ax.set_position([0.85, 0.1, 0.02, 0.8])

    fig.subplots_adjust(left=0.10, right=0.90, top=0.92, bottom=0.15)

    # Show the plot
    plt.show(block=True)

    return None

def plot_eq_curve(curve, fs, nfft):


    f_axis = torch.linspace(0, fs//2, curve.shape[0])
    curve_db = mag2db(curve)

    plt.rcParams.update({'font.family': 'serif', 'font.size': 14, 'font.weight': 'heavy', 'text.usetex': True})

    plt.figure(figsize=(6,2))
    plt.plot(f_axis, curve_db)
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Magnitude in dB')
    plt.xlim(20,20000)
    plt.ylim(-30,-15)
    plt.xscale('log')
    plt.grid()
    # plt.title('Equalization Curve')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)

    plt.show(block=True)

def plot_boxplot_spectrogram(subplot_spec, fig, nfft, fs, noverlap, evs, evs_label, rir, rir_time_label, rir_freq_label, rir_cbar_label, cmap, vmin, vmax, spec_y_scale='log'):
    """
    Plots a combined boxplot (top) and spectrogram with colorbar (bottom) within a single subplot.

    Parameters:
        subplot_spec: SubplotSpec -> Subplot specification for the combined plot
        fig: Matplotlib figure -> Figure to which the subplots belong
        tensor_2d: torch.Tensor (2D) -> Data for boxplot (flattened into 1D)
        tensor_1d: torch.Tensor (1D) -> Data for spectrogram
        cmap: str -> Colormap for spectrogram
        vmin, vmax: float -> Color scale limits for spectrogram
    """
    # Create a gridspec within the given subplot_spec
    gs = gridspec.GridSpecFromSubplotSpec(3, 5, subplot_spec=subplot_spec, height_ratios=[0.3, 2, 0.1], hspace=0.2, width_ratios=[4, 0.2, 0.2, 0.2, 0.3])
    
    # Boxplot
    ax_box = fig.add_subplot(gs[0, :4])
    sns.boxplot(x=evs.flatten().numpy(), ax=ax_box, showfliers=False, patch_artist=True, boxprops=dict(edgecolor='k', facecolor='xkcd:sky'), medianprops=dict(color="k", linewidth=2))
    max_outlier = evs.flatten().max().item()
    ax_box.scatter([max_outlier], [0], marker="o", s=25, facecolors='lightblue', edgecolors='black', zorder=3)
    ax_box.set_title("Magnitude in dB" if evs_label else "", fontsize=11)  # Move label above the boxplot
    ax_box.set_yticklabels([])
    ax_box.set_xlim(-55, 2)
    ax_box.set_xticks(ticks=[-50, -40, -30, -20, -10, 0], )
    ax_box.tick_params(axis='both', which='both', length=0.5, width=0.5, pad=0.5, labelsize=8, top=True, labeltop=True, bottom=False, labelbottom=False)
    ax_box.xaxis.grid(True)

    # Spectrogram
    ax_spec = fig.add_subplot(gs[1, :4])

    spec,f,t = mlab.specgram(rir.numpy(), NFFT=nfft, Fs=fs, noverlap=noverlap)
    max_val = max(spec.max(), spec.max())
    spec = spec/max_val

    im = ax_spec.pcolormesh(t, f, 10*np.log10(spec), shading='gouraud', cmap=cmap, vmin=-100, vmax=0)
    ax_spec.set_ylabel("Frequency in Hz" if rir_freq_label else "")
    ax_spec.set_yscale(spec_y_scale)
    ax_spec.set_ylim(20, 20000 if fs == 48000 else 500)
    if fs == 48000:
        ticks = [20, 100, 1000, 5000, 20000]
        labels = ['20', '100', '1k', '5k', '20k']
    else:
        ticks = [0, 100, 200, 300, 400, 500]
        labels = ['0', '100', '200', '300', '400', '500']
    ax_spec.set_yticks(ticks=ticks, labels=labels)
    ax_spec.set_xlabel("Time in seconds" if rir_time_label else "")
    ax_spec.set_xticks(ticks=[0, 0.5, 1.0, 1.5, 2.0], labels=[0, 0.5, 1.0, 1.5, 2.0])
    ax_spec.tick_params(axis='both', which='both', length=0.5, width=0.5, labelsize=8)
    ax_spec.tick_params(axis='x', pad=2)
    ax_spec.tick_params(axis='y', pad=0.5)

    # Colorbar
    cbar_ax = fig.add_subplot(gs[1, 4])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', label="Power in dB" if rir_cbar_label else "", ticks=[-100, -50, 0], aspect=10)
    cbar.ax.set_yticklabels(['-100', '-50', '0'])
    cbar.ax.tick_params(axis='both', which='both', length=0.5, width=0.5, pad=0.5, labelsize=8)
    
    return im  # Return the image for reference

def plot_grid_boxplot_spectrogram(nfft, fs, noverlap, tensor_pairs, rows, cols, row_labels, col_labels, figsize=(12, 8), cmap='magma'):
    """
    Plots a grid of combined boxplot-spectrogram pairs with row and column labels.

    Parameters:
        tensor_pairs: list of tuples [(2D tensor, 1D tensor), ...] -> Data for each subplot
        rows: int -> Number of rows in grid
        cols: int -> Number of columns in grid
        figsize: tuple -> Figure size
        cmap: str -> Colormap for spectrograms
    """
    plt.rcParams.update({'font.family':'serif', 'font.size':11, 'font.weight':'heavy', 'text.usetex':True})

    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(
        rows,
        cols,
        figure=fig,
        wspace=0.3,  # Space between columns
        hspace=0.2   # Space between rows
    )
    
    # Compute shared color scale for spectrograms
    all_specs = [tensor_1d.numpy() for _, tensor_1d in tensor_pairs]
    vmin = min(np.min(s) for s in all_specs)
    vmax = max(np.max(s) for s in all_specs)
    
    ims = []  # Store images for colorbar reference
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx >= len(tensor_pairs):
                continue  # Skip if there are fewer pairs than grid cells
            
            tensor_1, tensor_2 = tensor_pairs[idx]
            subplot_spec = spec[i, j]  # Get the SubplotSpec for this grid cell
            evs_label = False
            rir_time_label = False
            rir_freq_label = False
            rir_cbar_label = False
            if i == 0:
                evs_label = True
            if i == rows-1:
                rir_time_label = True
            if j == 0:
                rir_freq_label = True
            if j == cols-1:
                rir_cbar_label = True
            im = plot_boxplot_spectrogram(subplot_spec, fig, nfft[idx], fs[idx], noverlap[idx], tensor_1, evs_label, tensor_2, rir_time_label, rir_freq_label, rir_cbar_label, cmap, vmin, vmax, spec_y_scale='log' if i < 4 else 'linear')
            ims.append(im)
    
    # Add row labels using fig.text
    for i in range(rows):
        y = 0.86 - (i*0.93) / rows  # Calculate y position for each row
        fig.text(0.02, y, row_labels[i], va='center', ha='center', fontsize=14, rotation=90)
    
    # Add column labels using fig.text
    for j in range(cols):
        x = 0.23 + (j* 0.85) / cols  # Calculate x position for each column
        fig.text(x, 0.98, col_labels[j], va='center', ha='center', fontsize=14)
    
    # Adjust the layout to make space for labels
    fig.subplots_adjust(left=0.12, top=0.92, right=0.93, bottom=0.03)
    
    plt.show(block=True)
