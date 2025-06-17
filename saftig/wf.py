from typing import Union, Iterable

import numpy as np
from scipy.signal import correlate
import scipy

from .common import FilterBase, make_2D_array

from icecream import ic


def mean_cross_correlation_offset(A, B, N, offset):
    """ estimate the cross-correlation between A and B """
    assert len(A) == len(B)
    if offset < N-1:
        return correlate(A, B[offset:-N+1+offset], mode='valid')
    else:
        return correlate(A, B[offset:], mode='valid')

def wf_calculate(witness:Iterable[float]|Iterable[Iterable[float]], target:Iterable[float], N_filter:int, idx_target:int=0) -> Union[Iterable[Iterable[float]],bool]:
    """ caluclate the FIR coefficients for a wiener filter

    :param witness: Witness sensor data
    :param witness: Target sensor data
    :param N_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param idx_target: offset of the prediction relative to the end of the array

    :return: filter coefficients, full_rank (bool)
    """
    target = np.array(target)
    witness = make_2D_array(witness)
    assert witness.shape[1] == target.shape[0], "Missmatch between witness and target data shape"
    assert N_filter <= target.shape[0], "Input data must be at least as long as the filter"

    # calculate input autocorrelation and cross-correlation to target
    R_ws = [mean_cross_correlation_offset(target, A, N_filter, idx_target) for A in witness] # R_ws[channel, time]
    R_ws = np.array(R_ws).flatten(order='C')

    def calc_r_matrix(A, B, N_filter):
        """ calculate the cross correlation matrix of a and b """
        cc = correlate(A, B[N_filter:-N_filter], mode='valid')
        return np.array([[cc[N_filter + j - i] for j in range(N_filter)] for i in range(N_filter)])

    R_ww = np.block([[calc_r_matrix(A, B, N_filter) for B in witness] for A in witness])

    # calculate pseudo-inverse correlation matrix of inputs and the filter coefficients
    R_ww_inv, rank = scipy.linalg.pinv(R_ww, return_rank=True)
    full_rank = True
    ic(rank, R_ww_inv.shape, R_ws.shape)
    WFC = R_ww_inv.dot(R_ws)

    # unwrap into seperate FIR filters
    WFC = WFC.reshape((len(witness), N_filter))
    WFC = np.array([np.flip(i) for i in WFC])
    return WFC, full_rank

def wf_apply(WFC:Iterable[Iterable[float]], witness:Iterable[Iterable[float]]) -> Iterable[Iterable[float]]:
    """apply the WF to witness data

        :param witness: Witness sensor data
        :param target: Target sensor data

        :return: prediction
    """
    witness = np.array(witness).astype(np.longdouble)
    return np.sum([correlate(A, WF, mode='valid') for A, WF in zip(witness, WFC)], axis=0)

class WienerFilter(FilterBase):
    """Satic Wiener filter implementation

    :param N_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param idx_target: Position of the prediction
    :param N_channel: Number of witness sensor channels
    """

    #: The FIR coefficients of the WF
    filter_state:Iterable[Iterable[float]] = None

    def __init__(self, N_filter:int, idx_target:int, N_channel:int=1):
        super().__init__(N_filter, idx_target, N_channel)

    def condition(self, witness:Iterable[float]|Iterable[Iterable[float]], target:Iterable[float]) -> bool:
        """ Use an input dataset to condition the filter

        :param witness: Witness sensor data
        :param target: Target sensor data
        """
        witness, target = self.check_data_dimensions(witness, target)

        self.filter_state, full_rank = wf_calculate(witness, target, self.N_filter)

        if not full_rank:
            print("Warning: Filter is not of full rank")
        return self.filter_state, full_rank

    def apply(self, witness:Iterable[float]|Iterable[Iterable[float]], target:Iterable[float]=None, pad:bool=True, update_state:bool=False) -> Iterable[float]:
        """ Apply the filter to input data

        :param witness: Witness sensor data
        :param target: Target sensor data (is ignored)
        :param pad: if True, apply padding zeros so that the length matches the target signal
        :param update_state: ignored

        :return: prediction
        """
        witness, target = self.check_data_dimensions(witness, target)
        assert self.filter_state is not None, "The filter must be conditioned before calling apply()"

        pred = wf_apply(self.filter_state, witness)
        if pad:
            pred = np.concatenate([np.zeros(self.N_filter-1-self.idx_target), pred, np.zeros(self.idx_target)])
        return pred

