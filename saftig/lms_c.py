""" faster LMS filter implemented in c """
#from ._lms_c import lms_step_c
from ._lms_c import LMS_C
from .common import FilterBase

class LMSFilterC(FilterBase):
    """LMS filter implementation in C (faster but harder to adjust)

    :param n_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param idx_target: Position of the prediction
    :param n_channel: Number of witness sensor channels
    :param normalized: if True: NLMS, else LMS
    :param step_scale: the learning rate of the LMS filter

    # >>> import saftig as sg
    # >>> n_filter = 128
    # >>> witness, target = sg.TestDataGenerator(0.1).generate(int(1e5))
    # >>> filt = sg.LMSFilterC(n_filter, 0, 1)
    # >>> filt.condition(witness, target)
    # >>> prediction = filt.apply(witness, target) # check on the data used for conditioning
    # >>> residual_rms = sg.RMS(target-prediction)
    # >>> residual_rms > 0.05 and residual_rms < 0.15 # the expected RMS in this test scenario is 0.1
    # True

    """
