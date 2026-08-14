# ==================================================================
# ============================ IMPORTS =============================
import numpy as np
# PyTorch
import torch
import torch.nn as nn
# FLAMO
from flamo import system
from flamo.functional import get_magnitude, get_eigenvalues
from flamo.optimize.loss import mse_loss


# ==================================================================

class MSE_evs_mod(nn.Module):
    def __init__(self, iter_num: int, freq_points: int, samplerate: int, lowest_f: float, highest_f: float):
        r"""
        Mean Squared Error (MSE) loss function for Active Acoustics.
        To reduce computational complexity (i.e. the number of eigendecompositions computed),
        the loss is applied only on a subset of the frequecy points at each iteration of an epoch.
        The subset is selected randomly ensuring that all frequency points are considered once and only once.

            **Args**:
                - iter_num (int): Number of iterations per epoch.
                - freq_points (int): Number of frequency points.
                - samplerate (int): Sampling rate of the signal [Hz].
                - lowest_f (float): Lowest frequency point [Hz].
                - highest_f (float): Highest frequency point [Hz].
        """
        super().__init__()

        assert(lowest_f >= 0)
        nyquist = samplerate//2
        assert(highest_f <= nyquist)

        min_freq_point = int(lowest_f/nyquist * freq_points)
        max_freq_point = int(highest_f/nyquist * freq_points)

        self.freq_points = max_freq_point - min_freq_point
        self.max_index = self.freq_points

        self.iter_num = iter_num
        self.idxs = torch.randperm(self.freq_points) + min_freq_point
        self.evs_per_iteration = torch.ceil(torch.tensor(self.freq_points / self.iter_num, dtype=torch.float))
        self.interval_count = 0

    def forward(self, y_pred, y_true):
        r"""
        Compute the MSE loss function.
            
            **Args**:
                - y_pred (torch.Tensor): Predicted eigenvalues.
                - y_true (torch.Tensor): True eigenvalues.

            **Returns**:
                - torch.Tensor: Mean Squared Error.
        """
        # Get the indexes of the frequency-point subset
        idxs = self.__get_indexes()
        # Get the eigenvalues
        evs_pred = get_magnitude(get_eigenvalues(y_pred[:,idxs,:,:]))
        evs_true = y_true[:,idxs,:]
        difference = evs_pred - evs_true
        mask = difference > 0.0
        difference[mask] = difference[mask] * 2
        mse = torch.mean(torch.square(torch.abs(difference)))
        return mse

    def __get_indexes(self):
        r"""
        Get the indexes of the frequency-point subset.

            **Returns**:
                - torch.Tensor: Indexes of the frequency-point subset.
        """
        # Compute indeces
        idx1 = np.min([int(self.interval_count*self.evs_per_iteration), self.max_index-1])
        idx2 = np.min([int((self.interval_count+1) * self.evs_per_iteration), self.max_index])
        idxs = self.idxs[torch.arange(idx1, idx2, dtype=torch.int)]
        # Update interval counter
        self.interval_count = (self.interval_count+1) % (self.iter_num)
        return idxs
    

class MSE_evs_idxs(nn.Module):
    def __init__(self, iter_num: int, freq_points: int, samplerate: int, freqs: torch.Tensor):
        r"""
        Mean Squared Error (MSE) loss function for Active Acoustics.
        To reduce computational complexity (i.e. the number of eigendecompositions computed),
        the loss is applied only on a subset of the frequecy points at each iteration of an epoch.
        The subset is selected randomly ensuring that all frequency points are considered once and only once.

            **Args**:
                - iter_num (int): Number of iterations per epoch.
                - freq_points (int): Number of frequency points.
                - samplerate (int): Sampling rate of the signal [Hz].
                - freqs (torch.Tensor): Frequencies to be considered [Hz].
        """
        super().__init__()

        freq_axis = torch.linspace(0, samplerate/2, freq_points)
        freqs = freqs.repeat(freq_points, 1)
        freq_axis = freq_axis.unsqueeze(1).repeat(1, freqs.shape[1])
        idxs = torch.argmin(torch.abs(freq_axis - freqs), dim=0)
        self.idxs = torch.cat((idxs-2, idxs-1, idxs, idxs+1, idxs+2), dim=0)
        self.freq_points = len(self.idxs)
        self.iter_num = iter_num

    def forward(self, y_pred, y_true):
        r"""
        Compute the MSE loss function.
            
            **Args**:
                - y_pred (torch.Tensor): Predicted eigenvalues.
                - y_true (torch.Tensor): True eigenvalues.

            **Returns**:
                - torch.Tensor: Mean Squared Error.
        """
        # Get the eigenvalues
        evs_pred = get_magnitude(get_eigenvalues(y_pred[:,self.idxs,:,:]))
        evs_true = y_true[:,self.idxs,:]
        difference = evs_pred - evs_true
        mse = torch.mean(torch.square(torch.abs(difference)))
        return mse

class colorless_reverb(mse_loss):
    def __init__(self, samplerate: int, freq_points: int, freqs: torch.Tensor):
        r"""
        Colorless reverb loss function for Active Acoustics.
        The loss is applied on a subset of the frequecy points at each iteration of an epoch.

            **Args**:
                - samplerate (int): Sampling rate of the signal [Hz].
                - freq_points (int): Number of frequency points.
                - freqs (torch.Tensor): Frequencies to be considered [Hz].
        """
        super().__init__()
        freq_axis = torch.linspace(0, samplerate/2, freq_points)
        freqs = freqs.repeat(freq_points, 1)
        freq_axis = freq_axis.unsqueeze(1).repeat(1, freqs.shape[1])
        idxs = torch.argmin(torch.abs(freq_axis - freqs), dim=0)
        self.idxs = torch.cat((idxs-2, idxs-1, idxs, idxs+1, idxs+2), dim=0)
        self.freq_points = len(self.idxs)

    def forward(self, y_pred, y_target, model):
        processor = system.Shell(core=model.get_core()[0])
        mag_response = get_magnitude(processor.get_freq_response(identity=True))
        prediction = mag_response[:,self.idxs,:,:]
        target = torch.ones_like(prediction)

        return self.mse_loss(prediction, target)
