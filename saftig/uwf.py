from typing import Union, Iterable

import numpy as np
from scipy.signal import correlate
import scipy
import sys

from .common import FilterBase
from .wf import wf_calculate, wf_apply

from icecream import ic

class UpdatingWienerFilter(FilterBase):
    """Updating Wiener filter implementation

    :param N_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param idx_target: Position of the prediction
    :param N_channel: Number of witness sensor channels
    :param context_pre: how many additional samples before the current block are used to update the filters
    :param context_post: how many additional samples after the current block are used to update the filters
    """

    #: The FIR coefficients of the WF
    filter_state:Iterable[Iterable[float]] = None

    def __init__(self, N_filter:int, idx_target:int, N_channel:int=1, context_pre:int=0, context_post:int=0):
        super().__init__(N_filter, idx_target, N_channel)
        self.context_pre = context_pre
        self.context_post= context_post

    def condition(self, witness:Iterable[float]|Iterable[Iterable[float]], target:Iterable[float], hide_warning:bool=False) -> None:
        """ Placeholder for compatibility to other filters; does nothing!
        """
        if not hide_warning:
            print("Warning: UpdatingWienerFilter.condition() is just a placeholder, it has no effect.", file=sys.stderr)

    def apply(self, witness:Iterable[float]|Iterable[Iterable[float]], target:Iterable[float]=None, pad:bool=True, update_state:bool=False):
        """ Apply the filter to input data

        :param witness: Witness sensor data
        :param target: Target sensor data (is ignored)
        :param pad: if True, apply padding zeros so that the length matches the target signal
        :param update_state: ignored

        :return: prediction, bool indicating if all WF updates had full rank
        """
        witness, target = self.check_data_dimensions(witness, target)
        
        all_full_rank = True
        prediction = []
        for idx in range(self.N_filter, len(target), self.N_filter):
            # calculate filter coefficients
            selection_conditioning = np.arange(max(0, idx-self.context_pre), min(len(target), idx+self.N_filter+self.context_post))
            self.filter_state, full_rank = wf_calculate(witness[:,selection_conditioning], target[selection_conditioning], self.N_filter, idx_target=self.idx_target)
            all_full_rank &= full_rank

            # apply
            p = wf_apply(self.filter_state, witness[:,idx-self.N_filter+1:min(idx+self.N_filter, len(target))])
            ic(p.shape)
            prediction += list(p)

        if not all_full_rank:
            print('Warning: not all UWF calculations had full rank')

        if pad:
            ic(len(prediction))
            prediction = np.concatenate([np.zeros(self.N_filter-self.idx_target), prediction, np.zeros(self.idx_target)])
        return prediction

