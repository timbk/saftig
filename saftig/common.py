from typing import Iterable, Union

import numpy as np

def RMS(A:Iterable[float]):
    """ Calculate the root mean square value of an array """
    return np.sqrt(np.mean(np.square(A)))

def make_2D_array(A:Iterable|Iterable[Iterable]):
    """add a dimension to 1D arrays and leave 2D arrays as they are
    This is intended to allow 1D array input for single channel application

    :param A: input array

    :return: exteneded array

    :raises: ValueError if the input shape is not compatible

    >>> make_2D_array([1, 2])
    array([[1, 2]])

    >>> make_2D_array([[1, 2], [3, 4]])
    array([[1, 2],
           [3, 4]])

    """
    A = np.array(A)
    if len(A.shape) == 1:
        return np.array([A])
    elif len(A.shape) == 2:
        return A
    else:
        raise ValueError("Input must be 1D or 2D array")


class FilterBase:
    """ common interface definition for Filter implementations

    :param N_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param idx_target: Position of the prediction
    :param N_channel: Number of witness sensor channels
    """
    filter_state = None

    def __init__(self, N_filter:int, idx_target:int, N_channel:int=1):
        self.N_filter = N_filter
        self.N_channel = N_channel
        self.idx_target = idx_target

        assert self.N_filter > 0, "N_filter must be a positive integer"
        assert self.N_channel > 0, "N_filter must be a positive integer"
        assert self.idx_target >= 0 and self.idx_target < self.N_filter, "idx_target must not be negative and smaller than N_filter"

    def condition(self, witness:Iterable[float]|Iterable[Iterable[float]], target:Iterable[float]):
        """ Use an input dataset to condition the filter
        
        :param witness: Witness sensor data
        :param target: Target sensor data
        """
        pass # this should be implemented by the child class

    def apply(self, witness:Iterable[float]|Iterable[Iterable[float]], target:Iterable[float], pad:bool=True, update_state:bool=False):
        """ Apply the filter to input data
        
        :param witness: Witness sensor data
        :param target: Target sensor data
        :param pad: if True, apply padding zeros so that the length matches the target signal
        :param update_state: if True, the filter state will be changed. If false, the filter state will remain

        :return: prediction
        """
        pass # this should be implemented by the child class

    def check_data_dimensions(self, witness:Iterable[float]|Iterable[Iterable[float]], target:Iterable[float]) -> Union[Iterable[Iterable[float]], Iterable[float]]:
        """Check the dimensions of the provided input data and apply make_2D_array()

        :param witness: Witness sensor data
        :param target: Target sensor data

        :return: target, witness

        :raises: AssertionError
        """
        witness = make_2D_array(witness)
        assert witness.shape[0] == self.N_channel, "witness data shape does not match configured channel count"
        assert target is None or target.shape[0] == witness.shape[1], "Missmatch between target and witness data shapes"

        return witness, target

