from typing import Union, Iterable

import numpy as np
from scipy.signal import correlate

from .common import FilterBase, make_2D_array

from icecream import ic


class LMSFilter(FilterBase):
    """LMS filter implementation

    :param N_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param idx_target: Position of the prediction
    :param N_channel: Number of witness sensor channels
    :param normalized: if True: NLMS, else LMS
    :param step_scale: the learning rate of the LMS filter
    """

    #: The FIR coefficients of the WF
    filter_state:Iterable[Iterable[float]] = None

    def __init__(self, N_filter:int, idx_target:int, N_channel:int=1, normalized:bool=True, step_scale:float=0.5, coefficient_clipping:float|None=None):
        super().__init__(N_filter, idx_target, N_channel)
        self.normalized = normalized
        self.step_scale = step_scale
        self.coefficient_clipping = coefficient_clipping

        assert self.step_scale > 0, "Step scale must be positive"
        assert self.coefficient_clipping is None or self.coefficient_clipping > 0, "coefficient_clippen must be positive"

        self.reset()

    def reset(self) -> None:
        """ reset the filter to zero """
        self.filter_state = np.zeros((self.N_channel, self.N_filter))

    def condition(self, witness:Iterable[float]|Iterable[Iterable[float]], target:Iterable[float]) -> bool:
        """ Use an input dataset to condition the filter

        :param witness: Witness sensor data
        :param target: Target sensor data
        """
        self.apply(witness, target, update_state=True)

    def apply(self, witness:Iterable[float]|Iterable[Iterable[float]], target:Iterable[float]=None, pad:bool=True, update_state:bool=False) -> Iterable[float]:
        """ Apply the filter to input data

        :param witness: Witness sensor data
        :param target: Target sensor data (is ignored)
        :param pad: if True, apply padding zeros so that the length matches the target signal
        :param update_state: if True, the filter state will be changed. If false, the filter state will remain

        :return: prediction
        """
        witness, target = self.check_data_dimensions(witness, target)
        assert target is not None, "Target data must be supplied"

        pred_length = len(target)-max(self.N_filter, self.idx_target)

        # iterate over data (the python loop is very slow)
        prediction = []
        for idx in range(0, pred_length+1):
            # make prediction
            X = witness[:,idx:idx+self.N_filter] # input to predcition
            pred = np.einsum('ij,ij->', self.filter_state, X)
            err = target[idx+self.idx_target] - pred

            prediction.append(pred)

            # update filter
            if self.normalized:
                norm = np.einsum('ij,ij->', X, X)
                if norm < 0:
                    raise ValueError('Overflow! You are probably passing integers of insufficient precision to this function.')
                self.filter_state += 2*self.step_scale*err*X / norm
            else:
                self.filter_state += 2*self.step_scale*err*X

            if self.coefficient_clipping is not None:
                np.clip(self.filter_state, -self.coefficient_clipping, self.coefficient_clipping)

        prediction = np.array(prediction)
        if pad:
            prediction = np.concatenate([
                np.zeros(self.idx_target),
                prediction,
                np.zeros(len(target)-self.idx_target-pred_length-1)
                ])

        return prediction

