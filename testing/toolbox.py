from typing import Iterable
import numpy as np
from scipy.signal import welch

def calc_mean_asd(A:Iterable[float], sample_rate:float=1.):
    return np.sqrt(np.mean(welch(A, fs=sample_rate)[1]))
