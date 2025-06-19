"""Collection of tools for the evaluation and testing of filters"""
from typing import Iterable

import numpy as np

class TestDataGenerator:
    """Generate simple test data for correlated noise mitigation techniques
    The channel count is implicitly defined by the shape of witness_noise_level

    :param witness_noise_level: amplitude ratio of the sensor noise to the correlated noise in the witness sensor
                 Scalar or 1D-vector for multiple sensors
    :param target_noise_level: amplitude ratio of the sensor noise to the correlated noise in the target sensor
    :param transfer_functon: ratio between the amplitude in the target and witness signals

    """

    def __init__(self,
                 witness_noise_level:float | Iterable[float]=0.1,
                 target_noise_level:float=0,
                 transfer_function:float=1,
                 sample_rate:float=1.):
        self.witness_noise_level = np.array(witness_noise_level)
        self.target_noise_level = np.array(target_noise_level)
        self.transfer_function = np.array(transfer_function)
        self.sample_rate = sample_rate

        if len(self.witness_noise_level.shape) == 0:
            self.witness_noise_level = np.array([self.witness_noise_level])

        assert len(self.witness_noise_level.shape) == 1, f"witness_noise_level.shape = {self.witness_noise_level.shape}"
        assert len(self.target_noise_level.shape) == 0
        assert len(self.transfer_function.shape) == 0
        assert self.sample_rate > 0

    def scaled_whitenoise(self, shape)->Iterable[float]:
        """Generate whitenoise with an ASD of one

        :param shape: shape of the new array

        :return: Array of white noise
        """
        return np.random.normal(0, 1/np.sqrt(self.sample_rate), shape)

    def generate(self, N:int):
        """Generate sequences of samples

        :param N: number of samples

        :return: witness signal, target signal

        """
        t_c = self.scaled_whitenoise(N)
        w_n = self.scaled_whitenoise((len(self.witness_noise_level), N)) * self.witness_noise_level[:,None]
        t_n = self.scaled_whitenoise(N) * self.target_noise_level

        return  (t_c + w_n) * self.transfer_function, \
                (t_c + t_n)
