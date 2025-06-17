import saftig
from icecream import ic

# settings
N = int(1e5)
N_filter = 128
N_channel = 2

if __name__ == "__main__":
    w, t = saftig.TestDataGenerator([0.1]*N_channel).generate(N)
    
    wf = saftig.WienerFilter(N_filter, 0, N_channel)

    wf.condition(w, t)
    pred = wf.apply(w, pad=True)

    ic(wf.filter_state.shape)
    ic(pred.shape)
    ic(pred.shape[0] - t.shape[0])

    ic(saftig.RMS(t))
    ic(saftig.RMS(t-pred))

