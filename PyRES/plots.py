# ==================================================================
# ============================ IMPORTS =============================
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import mlab, colors, gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import numpy as np
# PyTorch
import torch
import torchaudio
# FLAMO
from flamo.functional import mag2db, get_magnitude


# ==================================================================
# ======================== PLOTTING UTILS ==========================

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

# ==================================================================
# ========================== PHYSICAL ROOM =========================

def plot_room_setup(positions: OrderedDict):

    # Always reset to default before updating
    plt.rcParams.update(plt.rcParamsDefault)

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
    plt.show()

    return None

# ==================================================================
# ====================== ACOUSTICAL ANALYSIS =======================

def plot_matrix_on_ax(
    ax,
    matrix_2d: torch.Tensor,
    norm,
    cmap,
    x_label: str = None,
    y_label: str = None,
    title: str = None
):
    im = ax.imshow(matrix_2d, norm=norm, cmap=cmap, aspect="auto")

    if title is not None:
        ax.set_title(title)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    return im

def plot_matrices(
    matrices: list[torch.Tensor] | OrderedDict[str, torch.Tensor],
    fig_x_label: str=None,
    fig_y_label: str=None,
    fig_title: str=None,
    common_colorbar: bool=True,
    fig_colorbar_label: str=None,
    matrix_colorbar_labels: list[str]=None,
    matrix_x_labels: list[str]=None,
    matrix_y_labels: list[str]=None,
    matrix_titles: list[str]=None,
    matrices_distribution: list[int]=None
):

    if isinstance(matrices, OrderedDict):
        matrices = list(matrices.values())
    n_matrices = len(matrices)

    # ---- Matplotlib style ----
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 18,
        'font.weight': 'heavy',
        'text.usetex': True
    })

    cmap = plt.get_cmap("viridis")

    # ---- Global normalization ----
    if common_colorbar:
        vmin = 1e10
        vmax = -1e10
        for matrix in matrices:
            matrix_vmin = torch.min(matrix)
            if matrix_vmin < vmin:
                vmin = matrix_vmin
            matrix_vmax = torch.max(matrix)
            if matrix_vmax > vmax:
                vmax = matrix_vmax
        vmin = torch.floor(vmin * 10) / 10
        vmax = torch.ceil(vmax * 10) / 10
        global_norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # ---- Layout ----
    if matrices_distribution is None:
        n_cols = int(torch.ceil(torch.sqrt(torch.tensor(n_matrices))))
        n_rows = int(torch.ceil(torch.tensor(n_matrices / n_cols)))
    else:
        assert len(matrices_distribution) == 2
        n_rows = matrices_distribution[0]
        n_cols = matrices_distribution[1]

    width_ratios = []
    fig_width = 0
    counter_width = 0
    for _ in range(n_rows):
        width = matrices[counter_width].shape[2]
        fig_width += width
        width_ratios.append(width)
        counter_width += 1
    height_ratios = []
    fig_height = 0
    counter_height = 0
    for _ in range(n_cols):
        height = matrices[counter_height].shape[1]
        fig_height += height
        height_ratios.append(height)
        counter_height += n_cols

    fig_size = [fig_width, fig_height]
    max_fig_size = 10
    min_idx = np.argmin(fig_size)
    fig_size[min_idx] = fig_size[min_idx] * (max_fig_size / fig_size[np.abs(min_idx-1)])
    fig_size[np.abs(min_idx-1)] = max_fig_size

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        gridspec_kw={'wspace':0.05, 'hspace':0.1},
        figsize=fig_size,
        constrained_layout=True
    )

    # 🔑 Normalize axes handling
    if n_matrices == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    images = []

    for matrix in range(n_matrices):
        if common_colorbar:
            norm=global_norm
        else:
            vmin = torch.min(matrices[matrix])
            vmax = torch.max(matrices[matrix])
            vmin = torch.floor(vmin * 10) / 10
            vmax = torch.ceil(vmax * 10) / 10
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

        im = plot_matrix_on_ax(
            ax=axs[matrix],
            matrix_2d=matrices[matrix].squeeze(0),
            norm=norm,
            cmap=cmap,
            x_label=matrix_x_labels[matrix] if matrix_x_labels is not None else None,
            y_label=matrix_y_labels[matrix] if matrix_y_labels is not None else None,
            title=matrix_titles[matrix] if matrix_titles is not None else None
        )
        images.append(im)

        # Remove unnecessary ticks
        row = matrix // n_cols
        col = matrix % n_cols
        if col != 0:            # not left column
            axs[matrix].set_yticks([])
        else:
            axs[matrix].set_yticks(ticks=np.arange(matrices[matrix].shape[1]), labels=np.arange(matrices[matrix].shape[1])+1)
        if row != n_rows - 1:   # not bottom row
            axs[matrix].set_xticks([])
        else:
            axs[matrix].set_xticks(ticks=np.arange(matrices[matrix].shape[2]), labels=np.arange(matrices[matrix].shape[2])+1)
        if not common_colorbar:
            cbar = fig.colorbar(
                im,
                ax=axs[matrix],
                label=matrix_colorbar_labels[matrix],
                aspect=15,
                pad=0.02
            )
            cbar.ax.yaxis.set_major_formatter(
                FormatStrFormatter('%.1f')
            )

    # Remove unused axes
    for ax in axs[n_matrices:]:
        ax.remove()

    # ---- Shared colorbar ----
    if common_colorbar:
        cbar = fig.colorbar(
            images[0],
            ax=axs[:n_matrices],
            label=fig_colorbar_label,
            aspect=15,
            pad=0.02
        )
        # Format tick labels to 1 decimal digit
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if fig_title is not None:
        fig.suptitle(fig_title)
    if fig_x_label is not None:
        fig.supxlabel(fig_x_label)
    if fig_y_label is not None:
        fig.supylabel(fig_y_label)

    plt.show(block=True)

# ==================================================================
# =========================== STATISTICS ===========================

def plot_distributions(distributions: torch.Tensor, n_bins: int, labels: list[str] = None, log_scale: bool = False):

    # Always reset to default before updating
    plt.rcParams.update(plt.rcParamsDefault)
    
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

    plt.show()

    return None

# ==================================================================
# ==================== OPTIMIZATION COMPARISON =====================

def plot_evs_compare(evs_init, evs_opt, fs: int, nfft: int, lower_f_lim: float, higher_f_lim: float, label1='Initialized', label2='Optimized') -> None:
    """
    Plot the magnitude distribution of the given eigenvalues.

    Args:
        evs (_type_): _description_
    """

    # Always reset to default before updating
    plt.rcParams.update(plt.rcParamsDefault)

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

    plt.show()

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

    # Always reset to default before updating
    plt.rcParams.update(plt.rcParamsDefault)

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

    plt.show()

def plot_spectrograms_compare(ir_1: torch.Tensor, ir_2: torch.Tensor, fs: int, nfft: int=2**8, noverlap: int=2**7, label1='Initialized', label2='Optimized') -> None:
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

    # Always reset to default before updating
    plt.rcParams.update(plt.rcParamsDefault)

    Spec_init,f,t = mlab.specgram(ir_1.detach().squeeze().numpy(), NFFT=nfft, Fs=fs, noverlap=noverlap)
    Spec_opt,_,_ = mlab.specgram(ir_2.detach().squeeze().numpy(), NFFT=nfft, Fs=fs, noverlap=noverlap)
    

    max_val = max(Spec_init.max(), Spec_opt.max())
    Spec_init = torch.tensor(Spec_init)/max_val
    Spec_opt = torch.tensor(Spec_opt)/max_val
    

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    fig,axes = plt.subplots(2,1, sharex=False, sharey=True, figsize=(8,5), constrained_layout=True)
    
    plt.subplot(2,1,1)
    plt.pcolormesh(t, f, 10*torch.log10(Spec_init), cmap='magma', shading='nearest', vmin=-100, vmax=0, rasterized=True)
    plt.xlim(0, ir_1.shape[0]/fs)
    plt.xticks([])
    plt.ylim(20, fs//2)
    plt.yscale('log')
    plt.title(label1)
    plt.grid(False)

    plt.subplot(2,1,2)
    im = plt.pcolormesh(t, f, 10*torch.log10(Spec_opt), cmap='magma', shading='nearest', vmin=-100, vmax=0, rasterized=True)
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


# ==================================================================
# ========================= ROOM GLIVELAB ==========================

def plot_edcs(edcs: torch.Tensor, fs: int) -> None:

    edcs_db = 10*torch.log10(edcs)
    edcs_db = edcs_db - torch.max(edcs_db)

    n_samples = edcs.shape[0]
    t_axis = torch.linspace(0, n_samples/fs, n_samples)
    n_curves = edcs.shape[1]

    # Always reset to default before updating
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    colorPalette = sns.color_palette("muted", n_colors=n_curves)

    fig, ax = plt.subplots(figsize=(7,3.9))

    # ---- Main plot ----
    for i in range(n_curves):
        ax.plot(
            t_axis,
            edcs_db[:, i],
            color=colorPalette[i],
            linewidth=2
        )

    ax.set_xlim([0,2.5])
    ax.set_ylim([-45, 0])
    ax.set_xlabel('Time in seconds')
    ax.set_ylabel('Amplitude in dB')
    ax.grid()

    # ---- Inset (zoomed) axis ----
    axins = inset_axes(
        ax,
        width="50%",     # relative to main axis
        height="60%",
        loc="upper right",
        borderpad=0.3
    )

    for i in range(n_curves):
        axins.plot(
            t_axis,
            edcs_db[:, i],
            color=colorPalette[i],
            linewidth=2
        )

    # ZOOM REGION — tweak these for your needs
    axins.set_xlim([-0.01, 0.4])   # seconds
    axins.set_ylim([-25, -5])     # dB

    axins.grid()
    axins.tick_params(labelsize=10)
    # fig.legend(["Setting 1", "Setting 2", "Setting 3", "Setting 4", "Setting 5"], loc='outside upper center', ncols=3)#, bbox_to_anchor=(0.6,0.4))

    plt.tight_layout(rect=(0,0,1,1))
    plt.show(block=True)

    return None