import saftig
from icecream import ic
import matplotlib.pyplot as plt
import numpy as np

# settings
N = int(1e5)
N_filter = 128
N_channel = 1

if __name__ == "__main__":
    w, t = saftig.TestDataGenerator([0.1]*N_channel).generate(N)
    
    #filt = saftig.WienerFilter(N_filter, 0, N_channel)
    filt = saftig.LMSFilter(N_filter, 0, N_channel, step_scale=0.1)

    filt.condition(w, t)
    fs_before = np.array(filt.filter_state)
    pred = filt.apply(w, t, pad=True, update_state=True)
    fs_after = np.array(filt.filter_state)

    ic((fs_before == fs_after).all())

    ic(filt.filter_state.shape)
    ic(pred.shape)
    ic(pred.shape[0] - t.shape[0])

    ic(saftig.RMS(t))
    ic(saftig.RMS(t-pred))


    plt.figure()
    plt.plot(t, label='target')
    plt.plot(pred, label='prediction')
    plt.plot(pred-t, label='residual')
    plt.legend()

    plt.show()

