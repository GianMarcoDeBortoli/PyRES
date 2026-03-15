"""
PyRES: Python library for Reverberation Enhancement System development and simulation.
"""

from physical_room import PhRoom, PhRoom_dataset, PhRoom_wgn
from virtual_room import (
    VrRoom,
    unitary_parallel_connections,
    unitary_mixing_matrix,
    random_FIRs,
    phase_cancellation,
    FDN,
    unitary_reverberator,
)
from res import RES
from loss_functions import MSE_evs_mod, MSE_evs_idxs, colorless_reverb
from functional import (
    energy_coupling,
    direct_to_reverb_ratio,
    system_equalization_curve,
    one_pole_filter,
    resonance_filter,
    modal_reverb,
    reverb_time,
    simulate_setup,
)
from plots import (
    plot_evs_distribution,
    plot_evs_compare,
    plot_irs_compare,
    plot_spectrograms_compare,
    plot_room_setup,
    plot_coupling,
    plot_DRR,
    plot_distributions,
)
from dataset_api import (
    get_hl_info,
    get_ll_info,
    get_transducer_number,
    get_transducer_positions,
    get_rirs,
    get_rir_metadata,
)
from utils import find_direct_path, expand_to_dimension, limit_frequency_points, next_power_of_2

__all__ = [
    # Physical room
    'PhRoom', 'PhRoom_dataset', 'PhRoom_wgn',
    # Virtual room
    'VrRoom', 'unitary_parallel_connections', 'unitary_mixing_matrix',
    'random_FIRs', 'phase_cancellation', 'FDN', 'unitary_reverberator',
    # RES
    'RES',
    # Loss functions
    'MSE_evs_mod', 'MSE_evs_idxs', 'colorless_reverb',
    # Functional
    'energy_coupling', 'direct_to_reverb_ratio', 'system_equalization_curve',
    'one_pole_filter', 'resonance_filter', 'modal_reverb', 'reverb_time',
    'simulate_setup',
    # Plots
    'plot_evs_distribution', 'plot_evs_compare', 'plot_irs_compare',
    'plot_spectrograms_compare', 'plot_room_setup', 'plot_coupling',
    'plot_DRR', 'plot_distributions',
    # Dataset API
    'get_hl_info', 'get_ll_info', 'get_transducer_number',
    'get_transducer_positions', 'get_rirs', 'get_rir_metadata',
    # Utils
    'find_direct_path', 'expand_to_dimension', 'limit_frequency_points',
    'next_power_of_2',
]