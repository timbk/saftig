""" Updating Wiener Filter """
from typing import Iterable
from warnings import warn

import numpy as np

from .common import FilterBase
from .wf import wf_calculate, wf_apply

class UpdatingWienerFilter(FilterBase):
    """Updating Wiener filter implementation

    :param n_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param idx_target: Position of the prediction
    :param n_channel: Number of witness sensor channels
    :param context_pre: how many additional samples before the current block are used to update the filters
    :param context_post: how many additional samples after the current block are used to update the filters


    >>> import saftig as sg
    >>> n_filter = 128
    >>> witness, target = sg.TestDataGenerator(0.1).generate(int(1e5))
    >>> filt = sg.UpdatingWienerFilter(n_filter, 0, 1, context_pre=20*n_filter, context_post=20*n_filter)
    >>> prediction = filt.apply(witness, target) # check on the data used for conditioning
    >>> residual_rms = sg.RMS(target-prediction)
    >>> residual_rms > 0.05 and residual_rms < 0.15 # the expected RMS in this test scenario is 0.1
    True

    """

    #: The FIR coefficients of the WF
    filter_state:Iterable[Iterable[float]]|None = None
    filter_name = "UWF"

    def __init__(self,
                 n_filter:int,
                 idx_target:int,
                 n_channel:int=1,
                 context_pre:int=0,
                 context_post:int=0):
        super().__init__(n_filter, idx_target, n_channel)
        self.context_pre = context_pre
        self.context_post= context_post

    def condition(self,
                  witness:Iterable[float]|Iterable[Iterable[float]],
                  target:Iterable[float],
                  hide_warning:bool=False) -> None:
        """ Placeholder for compatibility to other filters; does nothing!
        """
        if not hide_warning:
            warn("Warning: UpdatingWienerFilter.condition() is just a placeholder, it has no effect.", RuntimeWarning)

    def apply(self,
              witness:Iterable[float]|Iterable[Iterable[float]],
              target:Iterable[float]=None,
              pad:bool=True,
              update_state:bool=False):
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
        for idx in range(self.n_filter-1, len(target), self.n_filter):
            # calculate filter coefficients
            selection_conditioning = np.arange(max(0, idx-self.context_pre), min(len(target), idx+self.n_filter+self.context_post))
            self.filter_state, full_rank = wf_calculate(witness[:,selection_conditioning],
                                                        target[selection_conditioning],
                                                        self.n_filter,
                                                        idx_target=self.idx_target)
            all_full_rank &= full_rank

            # apply
            p = wf_apply(self.filter_state, witness[:,idx-self.n_filter+1:min(idx+self.n_filter, len(target))])
            prediction += list(p)

        if not all_full_rank:
            print('Warning: not all UWF calculations had full rank')

        if pad:
            prediction = np.concatenate([np.zeros(self.n_filter-1-self.idx_target), prediction, np.zeros(self.idx_target)])
        return prediction
